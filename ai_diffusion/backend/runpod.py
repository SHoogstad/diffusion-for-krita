"""Run ComfyUI on Runpod, billed to the user's own Runpod account.

GPU pod: the Diffusion for Krita image (scripts/runpod) runs ComfyUI behind a small proxy that
requires a per-pod token, reports start-up progress and terminates the pod when it is no
longer used. Models live on a network volume (kept between sessions) or on the container disk
(downloaded on every start).

Serverless: the same image runs as workers of a queue-based endpoint which scales to zero, see
runpod_serverless.py. Models are always on a network volume.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import urllib.request
from collections.abc import AsyncGenerator, Iterable
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from ..settings import RunpodMode, RunpodStorage
from ..util import client_logger as log
from .network import NetworkError, RequestManager

api_url = os.getenv("RUNPOD_API_URL", "https://api.runpod.io/v2")
image = os.getenv("RUNPOD_IMAGE", "ghcr.io/shoogstad/diffusion-for-krita-runpod:1.60.0")
pod_prefix = "diffusion-for-krita"
comfy_port = 8188

# Runpod list prices in USD, see https://docs.runpod.io/pods/pricing
network_volume_gb_month = 0.07
container_disk_gb_month = 0.10
assumed_download_mb_s = 100  # typical datacenter download speed from Hugging Face
container_base_gb = 20  # ComfyUI image, temp files and outputs on the container disk

workload_names = {
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "illu": "Illustrious",
    "flux": "Flux",
    "flux2": "Flux 2",
    "zimage": "Z-Image",
    "qwen": "Qwen Image",
    "qwen2": "Qwen Image 2.1",
    "krea2": "Krea 2",
    "moss": "MOSS-Image",
    "fibo2": "Fibo 2",
    "h3": "H3 Edit (experimental)",
}


class RunpodError(Exception):
    pass


class PodStopped(RunpodError):
    """The pod was stopped while it was still starting."""

    def __init__(self):
        super().__init__("The Runpod pod was stopped")


def include_checkpoints(storage: RunpodStorage, checkpoints: bool):
    # Without a network volume there is no way to bring your own checkpoints
    return checkpoints or storage is RunpodStorage.container


class GpuType(NamedTuple):
    id: str
    name: str
    memory: int  # GB VRAM
    price: float  # USD per hour, secure cloud pod
    availability: str  # NONE, LOW, MEDIUM, HIGH
    data_centers: list[tuple[str, str]]  # (id, availability)
    pool: str | None = None  # serverless GPU pool
    serverless_price: float = 0.0  # USD per hour of worker time

    @property
    def label(self):
        return f"{self.name} ({self.memory} GB) - ${self.price:.2f}/h"

    def hourly_price(self, mode: RunpodMode):
        return self.serverless_price if mode is RunpodMode.serverless else self.price

    @staticmethod
    def parse(data: dict):
        price = data.get("price") or {}
        return GpuType(
            id=data["id"],
            name=data.get("name") or data["id"],
            memory=int(data.get("memory", 0)),
            price=float(price.get("secure") or 0),
            availability=data.get("availability", "NONE"),
            data_centers=[
                (dc["id"], dc.get("availability", "NONE")) for dc in data.get("dataCenters", [])
            ],
            pool=data.get("pool"),
            serverless_price=float(price.get("serverless") or 0),
        )

    def best_data_center(self, allowed: set[str] | None = None):
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        available = [dc for dc in self.data_centers if rank.get(dc[1], 0) > 0]
        if allowed is not None:
            available = [dc for dc in available if dc[0] in allowed]
        if not available:
            return None
        return max(available, key=lambda dc: rank[dc[1]])[0]


class RunpodApi:
    """Thin client for the Runpod REST API v2."""

    def __init__(self, api_key: str, url: str = api_url):
        self.url = url.rstrip("/")
        self._key = api_key
        self._requests = RequestManager()

    retry_delays = (2.0, 5.0, 10.0, 20.0)  # for temporary errors of the Runpod API

    async def _call(self, method: str, path: str, data: dict | None = None):
        url = f"{self.url}/{path}"
        # Creating resources is not repeated, it could create a second pod or volume
        retries = self.retry_delays if method in ("GET", "DELETE") else ()
        for attempt in range(len(retries) + 1):
            try:
                return await self._requests.http(method, url, data, timeout=60, bearer=self._key)
            except NetworkError as e:
                if e.status == 404:
                    return None
                if attempt < len(retries) and _is_temporary(e):
                    log.warning(f"Runpod API {method} {path} failed ({e.message}), retrying")
                    await asyncio.sleep(retries[attempt])
                    continue
                detail = (e.data or {}).get("detail") or (e.data or {}).get("title") or e.message
                if e.status == 401:
                    detail = "The Runpod API key is invalid"
                raise RunpodError(f"Runpod: {detail}") from e

    async def gpus(self, mode=RunpodMode.pod) -> list[GpuType]:
        """GPU types with availability for pods or serverless workers."""
        product = "SERVERLESS" if mode is RunpodMode.serverless else "POD&cloud=SECURE"
        result = await self._call("GET", f"catalog/gpus?include=AVAILABILITY&product={product}")
        nvidia = [g for g in (result or {}).get("gpus", []) if g.get("manufacturer") == "NVIDIA"]
        gpus = [GpuType.parse(g) for g in nvidia]  # the image needs CUDA
        if mode is RunpodMode.serverless:
            gpus = [g for g in gpus if g.pool]
        gpus = [g for g in gpus if g.hourly_price(mode) > 0]
        return sorted(gpus, key=lambda g: (g.hourly_price(mode), g.memory))

    async def pods(self) -> list[dict]:
        return ((await self._call("GET", "pods")) or {}).get("pods", [])

    async def pod(self, id: str) -> dict | None:
        return await self._call("GET", f"pods/{id}")

    async def create_pod(self, data: dict) -> dict:
        return _ensure_result(await self._call("POST", "pods", data), "create pod")

    async def terminate_pod(self, id: str):
        await self._call("DELETE", f"pods/{id}")

    async def volume_data_centers(self) -> set[str]:
        """Data centers which support (standard) network volumes."""
        result = await self._call("GET", "catalog/datacenters?networkVolumeTypes=STANDARD")
        return {dc["id"] for dc in (result or {}).get("dataCenters", [])}

    async def volume(self, id: str) -> dict | None:
        return await self._call("GET", f"network-volumes/{id}")

    async def create_volume(self, name: str, size_gb: int, data_center: str) -> dict:
        data = {"name": name, "size": size_gb, "dataCenter": data_center}
        return _ensure_result(await self._call("POST", "network-volumes", data), "create volume")

    async def delete_volume(self, id: str):
        await self._call("DELETE", f"network-volumes/{id}")

    async def endpoint(self, id: str) -> dict | None:
        return await self._call("GET", f"serverless/{id}")

    async def create_endpoint(self, data: dict) -> dict:
        return _ensure_result(await self._call("POST", "serverless", data), "create endpoint")

    async def delete_endpoint(self, id: str):
        await self._call("DELETE", f"serverless/{id}")


def _is_temporary(e: NetworkError):
    # No status: connection failed or timed out
    return e.status is None or e.status in (408, 425, 429, 500, 502, 503, 504)


def _ensure_result(result: dict | None, operation: str) -> dict:
    if not isinstance(result, dict):
        raise RunpodError(f"Runpod: {operation} returned no result")
    return result


def terminate_pod_blocking(api_key: str, pod_id: str, timeout=10.0, url: str = api_url):
    """Used when Krita closes and the event loop no longer runs."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/pods/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception as e:
        log.warning(f"Failed to terminate Runpod pod {pod_id}: {e}")
        return False


# Model sizes ######################################################################################

_sizes_file = Path(__file__).parent / "runpod_models.json"
_sizes: dict | None = None


def model_size_gb(workloads: Iterable[str], checkpoints: bool) -> float:
    """Download size of the models for the selected workloads (shared files counted once)."""
    global _sizes
    sizes: dict = _sizes or json.loads(_sizes_file.read_text())
    _sizes = sizes
    files: set[str] = set()
    for w in workloads:
        files.update(sizes["workloads"].get(w, {}).get("files", []))
        if checkpoints:
            files.update(sizes["workloads"].get(w, {}).get("checkpoints", []))
    return sum(sizes["files"][f] for f in files) / 1024**3


def volume_size_gb(model_gb: float):
    """Network volume size: models plus room for LoRA and other additions, in steps of 10 GB."""
    return max(20, int((model_gb * 1.25 + 9.99) // 10 * 10))


class Estimate(NamedTuple):
    storage_month: float  # USD per month, also while no pod runs
    gpu_hour: float  # USD per hour while the pod runs
    start_minutes: float  # typical time until ready
    start_cost: float  # GPU cost of waiting for the start (download)

    def describe(self, storage: RunpodStorage, model_gb: float):
        gpu = f"${self.gpu_hour:.2f}/h" if self.gpu_hour > 0 else "GPU price per hour"
        if storage is RunpodStorage.network:
            return (
                f"≈ {model_gb:.0f} GB of models on a {volume_size_gb(model_gb)} GB network volume: "
                f"<b>${self.storage_month:.2f}/month</b>, also when no pod runs. "
                f"While generating: {gpu}. Start: ~{self.start_minutes:.0f} min "
                f"(the first start downloads the models once)."
            )
        return (
            f"No storage cost when stopped. While generating: {gpu}. Every start downloads "
            f"≈ {model_gb:.0f} GB: ~{self.start_minutes:.0f} min, costing about "
            f"<b>${self.start_cost:.2f}</b> of GPU time each time."
        )


def estimate(storage: RunpodStorage, model_gb: float, gpu_price: float) -> Estimate:
    boot_minutes = 2.0  # pull cached image, start ComfyUI
    download_minutes = model_gb * 1024 / assumed_download_mb_s / 60
    if storage is RunpodStorage.network:
        storage_month = volume_size_gb(model_gb) * network_volume_gb_month
        return Estimate(storage_month, gpu_price, boot_minutes, boot_minutes / 60 * gpu_price)
    start_minutes = boot_minutes + download_minutes
    return Estimate(0.0, gpu_price, start_minutes, start_minutes / 60 * gpu_price)


# Pod lifecycle ####################################################################################


async def ensure_volume(api: RunpodApi, volume_id: str, gpu: GpuType, model_gb: float) -> dict:
    """The existing network volume, or a new one where the GPU is available."""
    if volume_id:
        volume = await api.volume(volume_id)
        if volume is not None:
            return volume
        log.warning(f"Runpod network volume {volume_id} no longer exists")
    data_center = gpu.best_data_center(await api.volume_data_centers())
    if data_center is None:
        raise RunpodError(
            f"No {gpu.name} is available right now in a data center with network volumes. "
            "Choose another GPU, or download models on every start."
        )
    size = volume_size_gb(model_gb)
    log.info(f"Creating {size} GB Runpod network volume in {data_center}")
    return await api.create_volume(pod_prefix, size, data_center)


class PodPhase(Enum):
    creating = "creating"
    starting = "starting"
    downloading = "downloading"
    ready = "ready"


class PodProgress(NamedTuple):
    phase: PodPhase
    message: str
    progress: tuple[int, int] = (0, 0)


class PodInfo(NamedTuple):
    id: str
    url: str
    token: str


class PodConfig(NamedTuple):
    api_key: str
    gpu: str
    storage: RunpodStorage
    workloads: list[str]
    checkpoints: bool
    idle_minutes: int
    volume_id: str = ""
    pod_id: str = ""
    pod_token: str = ""


def pod_url(pod_id: str):
    return f"https://{pod_id}-{comfy_port}.proxy.runpod.net"


class RunpodPod:
    """Creates a pod for the plugin, waits until ComfyUI is ready and terminates it again."""

    poll_interval = 3.0
    pod_timeout = 20 * 60  # pulling the image on a new machine can take a while
    ready_timeout = 90 * 60  # downloading models on the container disk

    def __init__(self, config: PodConfig, api: RunpodApi | None = None):
        self.config = config
        self.api = api or RunpodApi(config.api_key)
        self.info: PodInfo | None = None
        self.volume_id = config.volume_id
        self._stopped = False
        self._status_requests = RequestManager()

    async def start(self) -> AsyncGenerator[PodProgress, None]:
        cfg = self.config
        if cfg.pod_id and cfg.pod_token:
            pod = await self.api.pod(cfg.pod_id)
            if pod and pod.get("status") in ("PROVISIONING", "STARTING", "RUNNING"):
                log.info(f"Reusing running Runpod pod {cfg.pod_id}")
                self.info = PodInfo(cfg.pod_id, pod_url(cfg.pod_id), cfg.pod_token)
        if self.info is None:
            await self._remove_stale_pods()
            yield PodProgress(PodPhase.creating, "Creating pod")
            info = await self._create_pod()
            if self._stopped:  # stopped while the pod was being created
                log.info(f"Terminating Runpod pod {info.id}")
                await self.api.terminate_pod(info.id)
                raise PodStopped()
            self.info = info
        async for progress in self._wait_until_ready():
            yield progress

    async def stop(self):
        self._stopped = True
        if self.info is not None:
            log.info(f"Terminating Runpod pod {self.info.id}")
            await self.api.terminate_pod(self.info.id)
            self.info = None

    async def _remove_stale_pods(self):
        # Pods stopped by the in-pod watchdog keep their machine reserved, clean them up
        for pod in await self.api.pods():
            if pod.get("name", "").startswith(pod_prefix) and pod.get("status") in (
                "EXITED",
                "ERROR",
            ):
                log.info(f"Removing stale Runpod pod {pod['id']}")
                await self.api.terminate_pod(pod["id"])

    async def _gpu(self) -> GpuType:
        gpus = await self.api.gpus()
        gpu = next((g for g in gpus if g.id == self.config.gpu), None)
        if gpu is None:
            raise RunpodError(f"GPU type '{self.config.gpu}' is not offered by Runpod")
        return gpu

    async def _ensure_volume(self, gpu: GpuType, model_gb: float) -> dict:
        volume = await ensure_volume(self.api, self.volume_id, gpu, model_gb)
        self.volume_id = volume["id"]
        return volume

    async def _create_pod(self) -> PodInfo:
        cfg = self.config
        gpu = await self._gpu()
        checkpoints = include_checkpoints(cfg.storage, cfg.checkpoints)
        model_gb = model_size_gb(cfg.workloads, checkpoints)
        token = secrets.token_urlsafe(32)
        data: dict = {
            "name": f"{pod_prefix}-{secrets.token_hex(3)}",
            "image": image,
            "gpu": {"id": gpu.id, "count": 1, "minCudaVersion": "12.8"},
            "cloud": "SECURE",
            "ports": [f"{comfy_port}/http"],
            "env": {
                "KRITA_TOKEN": token,
                "KRITA_STORAGE": cfg.storage.name,
                "KRITA_WORKLOADS": ",".join(cfg.workloads),
                "KRITA_CHECKPOINTS": "1" if checkpoints else "0",
                "KRITA_IDLE_MINUTES": str(cfg.idle_minutes + 5 if cfg.idle_minutes else 0),
            },
        }
        if cfg.storage is RunpodStorage.network:
            volume = await self._ensure_volume(gpu, model_gb)
            data["disk"] = container_base_gb
            data["mounts"] = {"network": [{"volumeId": volume["id"], "path": "/workspace"}]}
            data["dataCenterIds"] = [volume["dataCenter"]]
        else:
            data["disk"] = container_base_gb + int(model_gb * 1.2 + 1)
        try:
            pod = await self.api.create_pod(data)
        except RunpodError as e:
            if cfg.storage is RunpodStorage.network:
                raise RunpodError(
                    f"{e} - the network volume is in data center {data['dataCenterIds'][0]}, "
                    f"which may have no free {gpu.name} right now. Try again later or choose "
                    "another GPU."
                ) from e
            raise
        log.info(f"Created Runpod pod {pod['id']} ({gpu.name}, ${gpu.price:.2f}/h)")
        return PodInfo(pod["id"], pod_url(pod["id"]), token)

    def _running_info(self) -> PodInfo:
        if self.info is None:
            raise PodStopped()
        return self.info

    async def _check_pod_alive(self) -> dict:
        info = self._running_info()
        pod = await self.api.pod(info.id)
        if self._stopped:
            raise PodStopped()
        if pod is None or pod.get("status") in ("EXITED", "ERROR", "TERMINATED"):
            status = pod.get("status") if pod else "TERMINATED"
            self.info = None  # nothing left to terminate
            raise RunpodError(f"The Runpod pod stopped unexpectedly ({status})")
        return pod

    async def _wait_until_ready(self) -> AsyncGenerator[PodProgress, None]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.pod_timeout
        while True:
            pod = await self._check_pod_alive()
            if pod.get("status") == "RUNNING":
                break
            if loop.time() > deadline:
                raise RunpodError("The Runpod pod did not start in time")
            yield PodProgress(PodPhase.starting, "Starting pod")
            await asyncio.sleep(self.poll_interval)

        deadline = loop.time() + self.ready_timeout
        polls = 0
        while True:
            polls += 1
            if polls % 10 == 0:  # the pod can be terminated from the Runpod console meanwhile
                await self._check_pod_alive()
            status = await self._status()
            phase = status.get("phase", "starting") if status else "starting"
            message = status.get("message", "Starting") if status else "Starting pod"
            if phase == "ready":
                yield PodProgress(PodPhase.ready, "Ready")
                return
            if phase == "error":
                raise RunpodError(f"Runpod pod: {message}")
            if loop.time() > deadline:
                raise RunpodError("The Runpod pod did not become ready in time")
            if phase == "downloading" and status:
                done, total = status.get("progress", [0, 0])
                yield PodProgress(PodPhase.downloading, message, (int(done), int(total)))
            else:
                yield PodProgress(PodPhase.starting, message)
            await asyncio.sleep(self.poll_interval)

    async def _status(self) -> dict | None:
        info = self._running_info()
        url = f"{info.url}/krita/status"
        try:
            result = await self._status_requests.get(url, timeout=20, bearer=info.token)
            return result if isinstance(result, dict) else None
        except NetworkError:
            return None  # proxy not reachable yet while the container starts
