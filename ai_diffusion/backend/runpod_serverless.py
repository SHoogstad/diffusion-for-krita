"""ComfyUI on a Runpod serverless endpoint: pay per generated image instead of per pod hour.

The endpoint runs the same image as pods in serverless mode (scripts/runpod/krita_serverless.py).
There is no persistent connection, so `ServerlessComfyClient` keeps all of ComfyClient's logic
and only replaces the transport: HTTP requests become jobs, and a workflow job streams the
ComfyUI websocket messages and result images back.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import uuid
from collections.abc import AsyncGenerator, Iterable
from typing import Any, NamedTuple

from ..image import Image
from ..localization import translate as _
from ..settings import RunpodMode
from ..util import client_logger as log
from . import runpod
from .client import Client, ClientEvent, ServerError, loras_to_upload
from .comfy_client import ComfyClient
from .network import NetworkError, RequestManager
from .runpod import GpuType, PodPhase, PodProgress, RunpodApi, RunpodError

jobs_url = os.getenv("RUNPOD_JOBS_URL", "https://api.runpod.ai/v2")
max_job_input = 7_000_000  # bytes per job before base64; Runpod accepts 10 MB per /run request


class ServerlessJobs:
    """Job API of one queue-based endpoint."""

    poll_interval = 0.5
    terminal = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")
    stream_method = "GET"  # the runpod SDK's local test API (--rp_serve_api) uses POST

    def __init__(self, endpoint_id: str, api_key: str, url: str = jobs_url):
        self.url = f"{url.rstrip('/')}/{endpoint_id}"
        self._key = api_key
        self._requests = RequestManager()

    async def _call(self, method: str, path: str, data: dict | None = None):
        try:
            return await self._requests.http(method, f"{self.url}/{path}", data, bearer=self._key)
        except NetworkError as e:
            detail = (e.data or {}).get("error") or (e.data or {}).get("detail") or e.message
            raise RunpodError(f"Runpod: {detail}") from e

    async def submit(self, input: dict) -> str:
        result = await self._call("POST", "run", {"input": input})
        return result["id"]

    async def cancel(self, job_id: str):
        await self._call("POST", f"cancel/{job_id}", {})

    async def stream(self, job_id: str) -> AsyncGenerator[dict, None]:
        """Yields the items the worker produces, raises if it reports an error."""
        while True:
            result = await self._call(self.stream_method, f"stream/{job_id}") or {}
            for chunk in result.get("stream", []):
                item = chunk.get("output", chunk) if isinstance(chunk, dict) else chunk
                if isinstance(item, dict) and item.get("t") == "error":
                    raise RunpodError(item.get("error", "Unknown error on the Runpod worker"))
                yield item
            status = result.get("status", "")
            if status in self.terminal:
                if status != "COMPLETED":
                    error = result.get("error") or status.lower().replace("_", " ")
                    raise RunpodError(f"Runpod job {status.lower()}: {error}")
                return
            await asyncio.sleep(self.poll_interval)

    async def run(self, input: dict) -> AsyncGenerator[dict, None]:
        job_id = await self.submit(input)
        async for item in self.stream(job_id):
            yield item


class _ChunkedBody:
    def __init__(self):
        self.parts: dict[int, bytes] = {}
        self.count = 0
        self.status = 200
        self.content_type = ""

    def add(self, item: dict):
        self.parts[item["i"]] = base64.b64decode(item["d"])
        self.count = item["n"]
        self.status = item.get("status", self.status)
        self.content_type = item.get("ct", self.content_type)

    @property
    def complete(self):
        return self.count > 0 and len(self.parts) == self.count

    @property
    def data(self):
        return b"".join(self.parts[i] for i in range(self.count))


class _MessageQueue:
    """Stands in for the websocket ComfyClient reads messages from."""

    def __init__(self):
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.queue.get()


class ServerlessComfyClient(ComfyClient):
    def __init__(self, endpoint_id: str, api_key: str, jobs: ServerlessJobs | None = None):
        super().__init__(f"https://api.runpod.ai/v2/{endpoint_id}")
        self.endpoint_id = endpoint_id
        self.jobs = jobs or ServerlessJobs(endpoint_id, api_key)
        self._socket = _MessageQueue()
        self._images: dict[str, _ChunkedBody] = {}
        self._prompt_job: str | None = None
        self._prompt_task: asyncio.Task | None = None

    # Transport ####################################################################################

    async def _http(self, method: str, path: str, **kwargs):
        body = _ChunkedBody()
        async for item in self.jobs.run({"op": "http", "method": method, "path": path, **kwargs}):
            if item.get("t") == "body":
                body.add(item)
        if not body.complete:
            raise RunpodError(f"Incomplete response from the Runpod worker for {path}")
        url = f"{self.url}/{path}"
        if body.status >= 400:
            text = body.data.decode("utf-8", errors="replace")
            raise NetworkError(body.status, f"{text[:500]} ({body.status})", url, body.status)
        if "application/json" in body.content_type:
            return json.loads(body.data) if body.data else None
        return body.data

    async def _check_websocket(self):
        pass  # messages arrive with each workflow job

    async def _get(self, op: str, timeout: float | None = 60):
        return await self._http("GET", op)

    async def _post(self, op: str, data: dict):
        match op:
            case "prompt":
                return self._start_prompt(data)
            case "interrupt":
                return await self.interrupt()
            case "api/etn/workflow/subscribe" | "api/etn/workflow/unsubscribe":
                return {}  # no shared workflows without a permanent connection
        return await self._http("POST", op, json=data)

    async def _put(self, op: str, data: bytes):
        if len(data) <= max_job_input:
            return await self._http("PUT", op, data=base64.b64encode(data).decode())
        async for __ in self._upload(op, data):
            pass

    async def _upload(self, path: str, data: bytes) -> AsyncGenerator[float, None]:
        """Large uploads (LoRA) in parts, the worker collects them on the network volume."""
        upload_id = str(uuid.uuid4())
        count = math.ceil(len(data) / max_job_input)
        for i in range(count):
            part = data[i * max_job_input : (i + 1) * max_job_input]
            input = {"op": "upload", "path": path, "id": upload_id, "i": i, "n": count}
            input["data"] = base64.b64encode(part).decode()
            async for item in self.jobs.run(input):
                if item.get("t") == "body" and item.get("status", 200) >= 400:
                    raise RunpodError(f"Upload of {path} failed ({item['status']})")
            yield (i + 1) / count

    # Workflows ####################################################################################

    def _start_prompt(self, data: dict):
        self._prompt_task = asyncio.get_running_loop().create_task(self._run_prompt(data))
        return {"prompt_id": data["prompt_id"]}

    async def _run_prompt(self, data: dict):
        prompt_id = data["prompt_id"]
        try:
            job_id = await self.jobs.submit({"op": "prompt", "data": data})
            self._prompt_job = job_id
            async for item in self.jobs.stream(job_id):
                if item.get("t") == "img":
                    self._images.setdefault(item["id"], _ChunkedBody()).add(item)
                elif item.get("t") == "ws":
                    await self._socket.queue.put(item["m"])
        except Exception as e:
            log.error(f"Runpod serverless job for {prompt_id} failed: {e}")
            error = {"prompt_id": prompt_id, "exception_message": str(e), "traceback": ""}
            await self._socket.queue.put(json.dumps({"type": "execution_error", "data": error}))
        finally:
            self._prompt_job = None

    async def _listen(self):
        await self._socket.queue.put(json.dumps({"type": "status", "data": {}}))
        try:
            await self._listen_websocket(self._socket)  # type: ignore (duck-typed websocket)
        except asyncio.CancelledError:
            if self._prompt_task:
                self._prompt_task.cancel()

    async def _transfer_result_image(self, id: str):
        body = self._images.pop(id, None)
        if body is None or not body.complete:
            raise RunpodError(f"Result image {id} was not received from the Runpod worker")
        return Image.from_bytes(body.data)

    async def interrupt(self):
        if self._prompt_job:
            await self.jobs.cancel(self._prompt_job)

    async def cancel(self, job_ids: Iterable[str]):
        job_ids = list(job_ids)
        active = self._active_job
        if active and active.id in job_ids:
            await self.interrupt()
        if (job := self._waiting_job.peek()) and job.id in job_ids:
            self._waiting_job.clear()
        self._queue.remove_if(lambda j: j.id in job_ids)
        await asyncio.gather(*(self._report(ClientEvent.interrupted, id) for id in job_ids))

    # Models #######################################################################################

    async def try_inspect(self, folder_name: str):
        if "gguf" in folder_name and not self.features.gguf:
            return
        try:  # one job for the whole folder instead of pages of 8
            r: Any = await self._get(f"api/etn/model_info/{folder_name}?offset=0&limit=100000")
            r = dict(r or {})
            r.pop("_meta", None)
            yield (Client.DiscoverStatus(folder_name, len(r), len(r)), r)
        except NetworkError as e:
            log.error(f"Error while inspecting models in {folder_name}: {e!s}")

    async def upload_loras(self, work, local_job_id: str):
        for file in loras_to_upload(work, self.models):
            try:
                assert file.path is not None
                log.info(f"Uploading lora model {file.id} to the Runpod volume")
                data = file.path.read_bytes()
                async for progress in self._upload(f"api/etn/upload/loras/{file.id}", data):
                    await self._report(ClientEvent.upload, local_job_id, progress)
                await self.refresh()
            except Exception as e:
                raise ServerError(_("Error during upload of LoRA model") + f" {file.path}: {e!s}")


# Endpoint #########################################################################################


class ServerlessConfig(NamedTuple):
    api_key: str
    gpu: str
    workloads: list[str]
    checkpoints: bool
    keep_warm: int  # seconds a worker stays after a job
    volume_id: str = ""
    endpoint_id: str = ""


class RunpodServerless:
    """Sets up the network volume and endpoint, and makes sure the models are on the volume."""

    endpoint_name = f"{runpod.pod_prefix}-serverless"
    job_timeout_ms = 30 * 60 * 1000  # the first model download can take a while

    def __init__(self, config: ServerlessConfig, api: RunpodApi | None = None, jobs_url=jobs_url):
        self.config = config
        self.api = api or RunpodApi(config.api_key)
        self.volume_id = config.volume_id
        self.endpoint_id = config.endpoint_id
        self._jobs_url = jobs_url

    def client(self):
        jobs = ServerlessJobs(self.endpoint_id, self.config.api_key, self._jobs_url)
        return ServerlessComfyClient(self.endpoint_id, self.config.api_key, jobs)

    async def start(self) -> AsyncGenerator[PodProgress, None]:
        cfg = self.config
        yield PodProgress(PodPhase.creating, "Preparing endpoint")
        gpus = await self.api.gpus(RunpodMode.serverless)
        gpu = next((g for g in gpus if g.id == cfg.gpu), None)
        if gpu is None or not gpu.pool:
            raise RunpodError(f"GPU type '{cfg.gpu}' is not offered for Runpod serverless")
        model_gb = runpod.model_size_gb(cfg.workloads, cfg.checkpoints)
        volume = await runpod.ensure_volume(self.api, self.volume_id, gpu, model_gb)
        self.volume_id = volume["id"]
        await self._ensure_endpoint(gpu, volume)

        yield PodProgress(PodPhase.starting, "Waiting for a worker")
        jobs = ServerlessJobs(self.endpoint_id, cfg.api_key, self._jobs_url)
        input = {"op": "download", "workloads": cfg.workloads, "checkpoints": cfg.checkpoints}
        async for item in jobs.run(input):
            if item.get("t") == "progress":
                done, total = int(item.get("done", 0)), int(item.get("total", 0))
                yield PodProgress(PodPhase.downloading, item.get("message", ""), (done, total))
        yield PodProgress(PodPhase.ready, "Ready")

    def _endpoint_data(self, gpu: GpuType, volume: dict):
        return {
            "name": self.endpoint_name,
            "type": "QUEUE",
            "image": runpod.image,
            "env": {"KRITA_MODE": "serverless"},
            "disk": runpod.container_base_gb,
            "gpu": {"pools": [gpu.pool], "count": 1, "minCudaVersion": "12.8"},
            "scaling": {"type": "QUEUE_DELAY", "queueDelay": 4},
            "workers": {"min": 0, "max": 1, "idleTimeout": max(1, self.config.keep_warm)},
            "mounts": {"network": [{"volumeId": volume["id"]}]},
            "dataCenterIds": [volume["dataCenter"]],
            "flashboot": "FLASHBOOT",
            "timeout": self.job_timeout_ms,
        }

    async def _ensure_endpoint(self, gpu: GpuType, volume: dict):
        data = self._endpoint_data(gpu, volume)
        if self.endpoint_id:
            existing = await self.api.endpoint(self.endpoint_id)
            if existing and _endpoint_matches(existing, data):
                return
            if existing:
                log.info(f"Runpod endpoint {self.endpoint_id} has other settings, replacing it")
                await self.api.delete_endpoint(self.endpoint_id)
        endpoint = await self.api.create_endpoint(data)
        self.endpoint_id = endpoint["id"]
        log.info(f"Created Runpod serverless endpoint {self.endpoint_id} ({gpu.pool})")

    async def delete(self):
        if self.endpoint_id:
            await self.api.delete_endpoint(self.endpoint_id)
            self.endpoint_id = ""


def _endpoint_matches(existing: dict, wanted: dict):
    volumes = [m.get("volumeId") for m in (existing.get("mounts") or {}).get("network", [])]
    return (
        existing.get("image") == wanted["image"]
        and (existing.get("gpu") or {}).get("pools") == wanted["gpu"]["pools"]
        and volumes == [wanted["mounts"]["network"][0]["volumeId"]]
        and (existing.get("workers") or {}).get("idleTimeout") == wanted["workers"]["idleTimeout"]
    )


class ServerlessEstimate(NamedTuple):
    storage_month: float
    gpu_hour: float
    per_100_images: float  # at seconds_per_image
    keep_warm_cost: float  # after each burst of jobs

    seconds_per_image = 10

    def describe(self, model_gb: float, keep_warm: int):
        return (
            f"≈ {model_gb:.0f} GB of models on a {runpod.volume_size_gb(model_gb)} GB network "
            f"volume: <b>${self.storage_month:.2f}/month</b>. Generating costs "
            f"${self.gpu_hour:.2f}/h of worker time, about <b>${self.per_100_images:.2f} per 100 "
            f"images</b> (~{self.seconds_per_image} s each), plus ${self.keep_warm_cost:.3f} for "
            f"keeping the worker ready {keep_warm} s after use. The first image after a pause "
            "waits for a cold start (about a minute). No live painting."
        )


def estimate_serverless(model_gb: float, gpu_price: float, keep_warm: int):
    storage = runpod.volume_size_gb(model_gb) * runpod.network_volume_gb_month
    per_100 = 100 * ServerlessEstimate.seconds_per_image / 3600 * gpu_price
    return ServerlessEstimate(storage, gpu_price, per_100, keep_warm / 3600 * gpu_price)
