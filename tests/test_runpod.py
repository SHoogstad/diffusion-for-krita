import base64
import json
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from ai_diffusion.backend import runpod, runpod_serverless
from ai_diffusion.backend.network import NetworkError
from ai_diffusion.backend.runpod import (
    GpuType,
    PodConfig,
    PodPhase,
    PodStopped,
    RunpodApi,
    RunpodError,
    RunpodPod,
    estimate,
    model_size_gb,
    volume_size_gb,
)
from ai_diffusion.backend.runpod_serverless import (
    RunpodServerless,
    ServerlessConfig,
    ServerlessJobs,
    estimate_serverless,
)
from ai_diffusion.image import Extent, Image
from ai_diffusion.qt.QtCore import Qt
from ai_diffusion.settings import RunpodStorage
from ai_diffusion.style import Arch

from .conftest import qtapp

gpu_4090 = {
    "id": "NVIDIA GeForce RTX 4090",
    "name": "RTX 4090",
    "manufacturer": "NVIDIA",
    "memory": 24,
    "price": {"secure": 0.69, "community": 0.34, "serverless": 1.10},
    "pool": "ADA_24",
    "availability": "HIGH",
    "dataCenters": [
        {"id": "EU-RO-1", "name": "Romania", "availability": "LOW"},
        {"id": "US-TX-3", "name": "Texas", "availability": "HIGH"},
    ],
}


class FakeRunpod:
    """Runpod REST API v2 and the pod's /krita/status endpoint, enough for one pod."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict | None]] = []
        self.pods: dict[str, dict] = {}
        self.volumes: dict[str, dict] = {}
        self.pod_polls_until_running = 2
        self.status_sequence = [
            {"phase": "downloading", "message": "Downloading models", "progress": [1, 4]},
            {"phase": "starting", "message": "Starting ComfyUI", "progress": [4, 4]},
            {"phase": "ready", "message": "Ready", "progress": [4, 4]},
        ]
        self.create_pod_error: tuple[int, dict] | None = None
        self.volume_data_centers = ["EU-RO-1", "US-TX-3"]
        self.fail_next: dict[str, int] = {}  # "METHOD path-prefix" -> number of 504 replies
        self.endpoints: dict[str, dict] = {}
        self.endpoint_count = 0
        self.jobs: dict[str, dict] = {}
        # serverless worker: job input -> streamed items
        self.worker: Callable[[dict], Iterable[dict]] = lambda input: []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, body: dict | None = None):
                data = json.dumps(body or {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(length)) if length else None

            def _handle(self, method: str):
                body = self._body()
                fake.requests.append((method, self.path, body))
                auth = self.headers.get("Authorization", "")
                if self.path.startswith("/pod/"):  # the pod's proxy
                    pod_id = self.path.split("/")[2]
                    token = fake.pods.get(pod_id, {}).get("env", {}).get("KRITA_TOKEN")
                    if auth != f"Bearer {token}":
                        return self._reply(401, {"error": "unauthorized"})
                    status = fake.status_sequence[0]
                    if len(fake.status_sequence) > 1:
                        fake.status_sequence.pop(0)
                    return self._reply(200, status)
                if auth != "Bearer test-key":
                    return self._reply(401, {"title": "Unauthorized", "status": 401})
                for key, count in fake.fail_next.items():
                    key_method, prefix = key.split(" ", 1)
                    if count > 0 and method == key_method and self.path.startswith(prefix):
                        fake.fail_next[key] -= 1
                        return self._reply(504, {"title": "Gateway Timeout", "status": 504})
                return self._reply(*fake.handle(method, self.path, body))

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

            def log_message(self, format, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        Thread(target=self.server.serve_forever, daemon=True).start()

    def handle(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        match method, path.split("?", maxsplit=1)[0].strip("/").split("/"):
            case "GET", ["catalog", "gpus"]:
                amd = {**gpu_4090, "id": "AMD Instinct MI300X OAM", "manufacturer": "AMD"}
                return 200, {"gpus": [gpu_4090, amd]}
            case "GET", ["catalog", "datacenters"]:
                assert "networkVolumeTypes=STANDARD" in path
                return 200, {"dataCenters": [{"id": id} for id in self.volume_data_centers]}
            case "GET", ["pods"]:
                return 200, {"pods": list(self.pods.values())}
            case "POST", ["pods"]:
                if self.create_pod_error:
                    return self.create_pod_error
                assert body is not None
                pod = {**body, "id": f"pod{len(self.pods) + 1}", "status": "PROVISIONING"}
                self.pods[pod["id"]] = pod
                return 201, pod
            case "GET", ["pods", id] if id in self.pods:
                pod = self.pods[id]
                if pod["status"] != "RUNNING":
                    self.pod_polls_until_running -= 1
                    if self.pod_polls_until_running <= 0:
                        pod["status"] = "RUNNING"
                return 200, pod
            case "DELETE", ["pods", id] if id in self.pods:
                del self.pods[id]
                return 204, {}
            case "POST", ["network-volumes"]:
                assert body is not None
                volume = {**body, "id": f"vol{len(self.volumes) + 1}", "type": "STANDARD"}
                self.volumes[volume["id"]] = volume
                return 200, volume
            case "GET", ["network-volumes", id] if id in self.volumes:
                return 200, self.volumes[id]
            case "DELETE", ["network-volumes", id] if id in self.volumes:
                del self.volumes[id]
                return 204, {}
            case "POST", ["serverless"]:
                assert body is not None
                self.endpoint_count += 1
                endpoint = {**body, "id": f"ep{self.endpoint_count}"}
                self.endpoints[endpoint["id"]] = endpoint
                return 200, endpoint
            case "GET", ["serverless", id] if id in self.endpoints:
                return 200, self.endpoints[id]
            case "DELETE", ["serverless", id] if id in self.endpoints:
                del self.endpoints[id]
                return 204, {}
            case "POST", ["jobs", endpoint, "run"] if endpoint in self.endpoints:
                assert body is not None
                job_id = f"job{len(self.jobs) + 1}"
                try:
                    items, status = list(self.worker(body["input"])), "COMPLETED"
                except Exception as e:
                    items, status = [{"t": "error", "error": str(e)}], "FAILED"
                self.jobs[job_id] = {"input": body["input"], "items": items, "status": status}
                return 200, {"id": job_id, "status": "IN_QUEUE"}
            case "GET", ["jobs", _, "stream", job_id] if job_id in self.jobs:
                job = self.jobs[job_id]
                if job.get("streamed") is None:  # first poll: half of the items, still running
                    half = len(job["items"]) // 2
                    job["streamed"] = half
                    stream = [{"output": item} for item in job["items"][:half]]
                    return 200, {"status": "IN_PROGRESS", "stream": stream}
                stream = [{"output": item} for item in job["items"][job["streamed"] :]]
                job["streamed"] = len(job["items"])
                return 200, {"status": job["status"], "stream": stream}
            case "POST", ["jobs", _, "cancel", job_id] if job_id in self.jobs:
                self.jobs[job_id]["cancelled"] = True
                return 200, {"id": job_id, "status": "CANCELLED"}
        return 404, {"title": "Not Found", "status": 404, "detail": f"{method} {path}"}


@pytest.fixture()
def fake_runpod(monkeypatch):
    fake = FakeRunpod()
    monkeypatch.setattr(runpod, "pod_url", lambda id: f"{fake.url}/pod/{id}")
    monkeypatch.setattr(RunpodPod, "poll_interval", 0.01)
    monkeypatch.setattr(RunpodApi, "retry_delays", (0.01, 0.01, 0.01, 0.01))
    yield fake
    fake.server.shutdown()


def config(**kwargs):
    defaults = {
        "api_key": "test-key",
        "gpu": "NVIDIA GeForce RTX 4090",
        "storage": RunpodStorage.network,
        "workloads": ["sdxl"],
        "checkpoints": True,
        "idle_minutes": 15,
    }
    return PodConfig(**(defaults | kwargs))


def test_model_size():
    sdxl = model_size_gb(["sdxl"], checkpoints=False)
    sdxl_cp = model_size_gb(["sdxl"], checkpoints=True)
    both = model_size_gb(["sdxl", "illu"], checkpoints=False)
    assert 1 < sdxl < sdxl_cp
    assert sdxl <= both < sdxl + model_size_gb(["illu"], checkpoints=False) + 0.01
    assert model_size_gb([], checkpoints=True) == 0


def test_volume_size():
    assert volume_size_gb(1) == 20
    assert volume_size_gb(18.2) == 30
    assert volume_size_gb(40) == 50


def test_estimate():
    network = estimate(RunpodStorage.network, 18, 0.69)
    assert network.storage_month == pytest.approx(30 * runpod.network_volume_gb_month)
    assert network.gpu_hour == pytest.approx(0.69)
    container = estimate(RunpodStorage.container, 18, 0.69)
    assert container.storage_month == 0
    assert container.start_minutes > network.start_minutes
    assert container.start_cost == pytest.approx(container.start_minutes / 60 * 0.69)
    assert "$2.10/month" in network.describe(RunpodStorage.network, 18)
    assert f"${container.start_cost:.2f}" in container.describe(RunpodStorage.container, 18)


def test_gpu_type():
    gpu = GpuType.parse(gpu_4090)
    assert gpu.price == pytest.approx(0.69) and gpu.memory == 24
    assert gpu.best_data_center() == "US-TX-3"
    assert "$0.69/h" in gpu.label
    assert GpuType.parse({**gpu_4090, "dataCenters": []}).best_data_center() is None


@qtapp
async def test_start_network_volume(fake_runpod: FakeRunpod):
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    phases = [p async for p in pod.start()]

    assert phases[0].phase is PodPhase.creating
    assert PodPhase.starting in [p.phase for p in phases]
    downloading = [p for p in phases if p.phase is PodPhase.downloading]
    assert downloading and downloading[0].progress == (1, 4)
    assert phases[-1].phase is PodPhase.ready

    volume = fake_runpod.volumes[pod.volume_id]
    assert volume["dataCenter"] == "US-TX-3"  # best availability for the GPU
    assert volume["size"] == volume_size_gb(model_size_gb(["sdxl"], True))

    assert pod.info is not None
    created = fake_runpod.pods[pod.info.id]
    assert created["gpu"]["id"] == "NVIDIA GeForce RTX 4090"
    assert created["mounts"]["network"] == [{"volumeId": pod.volume_id, "path": "/workspace"}]
    assert created["dataCenterIds"] == ["US-TX-3"]
    assert created["ports"] == ["8188/http"]
    assert created["env"]["KRITA_STORAGE"] == "network"
    assert created["env"]["KRITA_WORKLOADS"] == "sdxl"
    assert created["env"]["KRITA_TOKEN"] == pod.info.token and len(pod.info.token) > 30

    pod_id = pod.info.id
    await pod.stop()
    assert pod_id not in fake_runpod.pods
    assert pod.volume_id in fake_runpod.volumes  # the volume stays


@qtapp
async def test_start_container_disk(fake_runpod: FakeRunpod):
    pod = RunpodPod(config(storage=RunpodStorage.container), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in pod.start()]
    assert pod.info is not None
    created = fake_runpod.pods[pod.info.id]
    assert "mounts" not in created and "dataCenterIds" not in created
    assert created["disk"] > runpod.container_base_gb
    assert created["env"]["KRITA_STORAGE"] == "container"
    assert not fake_runpod.volumes


@qtapp
async def test_container_disk_always_downloads_checkpoints(fake_runpod: FakeRunpod):
    # Nothing else could provide a checkpoint, the pod would be unusable without one
    cfg = config(storage=RunpodStorage.container, checkpoints=False)
    pod = RunpodPod(cfg, RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in pod.start()]
    assert pod.info is not None
    created = fake_runpod.pods[pod.info.id]
    assert created["env"]["KRITA_CHECKPOINTS"] == "1"
    assert created["disk"] > runpod.container_base_gb + model_size_gb(["sdxl"], True)

    volume_pod = RunpodPod(config(checkpoints=False), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in volume_pod.start()]
    assert volume_pod.info is not None
    assert fake_runpod.pods[volume_pod.info.id]["env"]["KRITA_CHECKPOINTS"] == "0"


@qtapp
async def test_reuse_running_pod(fake_runpod: FakeRunpod):
    first = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in first.start()]
    assert first.info is not None
    fake_runpod.status_sequence = [{"phase": "ready", "message": "Ready"}]

    cfg = config(pod_id=first.info.id, pod_token=first.info.token, volume_id=first.volume_id)
    second = RunpodPod(cfg, RunpodApi("test-key", fake_runpod.url))
    phases = [p async for p in second.start()]
    assert PodPhase.creating not in [p.phase for p in phases]
    assert second.info == first.info
    assert len(fake_runpod.pods) == 1


@qtapp
async def test_volume_only_in_supported_data_center(fake_runpod: FakeRunpod):
    fake_runpod.volume_data_centers = ["EU-RO-1"]  # US-TX-3 has more GPUs but no volumes
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in pod.start()]
    assert fake_runpod.volumes[pod.volume_id]["dataCenter"] == "EU-RO-1"


@qtapp
async def test_no_volume_data_center(fake_runpod: FakeRunpod):
    fake_runpod.volume_data_centers = ["EU-SE-1"]
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    with pytest.raises(RunpodError, match="data center with network volumes"):
        _ = [p async for p in pod.start()]


@qtapp
async def test_existing_volume_is_used(fake_runpod: FakeRunpod):
    fake_runpod.volumes["vol9"] = {"id": "vol9", "dataCenter": "EU-RO-1", "size": 50}
    pod = RunpodPod(config(volume_id="vol9"), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in pod.start()]
    assert pod.info is not None
    assert fake_runpod.pods[pod.info.id]["dataCenterIds"] == ["EU-RO-1"]
    assert list(fake_runpod.volumes) == ["vol9"]


@qtapp
async def test_stale_pods_removed(fake_runpod: FakeRunpod):
    fake_runpod.pods["old"] = {"id": "old", "name": "diffusion-for-krita-abc", "status": "EXITED"}
    fake_runpod.pods["other"] = {"id": "other", "name": "my-own-pod", "status": "EXITED"}
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    _ = [p async for p in pod.start()]
    assert "old" not in fake_runpod.pods
    assert "other" in fake_runpod.pods  # not ours


@qtapp
async def test_create_error(fake_runpod: FakeRunpod):
    fake_runpod.create_pod_error = (
        400,
        {"title": "Bad Request", "status": 400, "detail": "There are no instances available"},
    )
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    with pytest.raises(RunpodError, match=r"no instances available.*another GPU"):
        _ = [p async for p in pod.start()]


@qtapp
async def test_pod_reports_error(fake_runpod: FakeRunpod):
    fake_runpod.status_sequence = [{"phase": "error", "message": "Model download failed: 404"}]
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    with pytest.raises(RunpodError, match="Model download failed"):
        _ = [p async for p in pod.start()]


@qtapp
async def test_pod_disappears_while_starting(fake_runpod: FakeRunpod):
    fake_runpod.status_sequence = [{"phase": "starting", "message": "Starting ComfyUI"}]
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))

    async def start():
        async for progress in pod.start():
            if progress.message == "Starting ComfyUI":
                fake_runpod.pods.clear()  # terminated from the Runpod console

    with pytest.raises(RunpodError, match="stopped unexpectedly"):
        await start()
    assert pod.info is None


@qtapp
async def test_stop_while_starting(fake_runpod: FakeRunpod):
    fake_runpod.status_sequence = [{"phase": "starting", "message": "Starting ComfyUI"}]
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))

    async def start():
        async for progress in pod.start():
            if progress.message == "Starting ComfyUI":
                await pod.stop()

    with pytest.raises(PodStopped):
        await start()
    assert not fake_runpod.pods


@qtapp
async def test_stop_while_creating(fake_runpod: FakeRunpod):
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    create_pod = pod._create_pod

    async def create_and_stop():
        info = await create_pod()
        await pod.stop()  # pod.info is not set yet, start() has to clean up
        return info

    pod._create_pod = create_and_stop
    with pytest.raises(PodStopped):
        _ = [p async for p in pod.start()]
    assert not fake_runpod.pods


@qtapp
async def test_temporary_api_errors_are_retried(fake_runpod: FakeRunpod):
    fake_runpod.fail_next = {"GET /pods/": 2, "GET /catalog/gpus": 1}
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    phases = [p async for p in pod.start()]
    assert phases[-1].phase is PodPhase.ready
    assert all(count == 0 for count in fake_runpod.fail_next.values())


@qtapp
async def test_create_pod_is_not_retried(fake_runpod: FakeRunpod):
    fake_runpod.fail_next = {"POST /pods": 1}
    pod = RunpodPod(config(), RunpodApi("test-key", fake_runpod.url))
    with pytest.raises(RunpodError, match="Gateway Timeout"):
        _ = [p async for p in pod.start()]
    assert not fake_runpod.pods


@qtapp
async def test_retries_give_up(fake_runpod: FakeRunpod):
    fake_runpod.fail_next = {"GET /catalog/gpus": 10}
    with pytest.raises(RunpodError, match="Gateway Timeout"):
        await RunpodApi("test-key", fake_runpod.url).gpus()
    assert fake_runpod.fail_next["GET /catalog/gpus"] == 5  # 1 try + 4 retries


@qtapp
async def test_only_nvidia_gpus(fake_runpod: FakeRunpod):
    gpus = await RunpodApi("test-key", fake_runpod.url).gpus()
    assert [g.id for g in gpus] == ["NVIDIA GeForce RTX 4090"]


@qtapp
async def test_invalid_api_key(fake_runpod: FakeRunpod):
    with pytest.raises(RunpodError, match="API key is invalid"):
        await RunpodApi("wrong", fake_runpod.url).gpus()


def test_terminate_blocking(fake_runpod: FakeRunpod):
    fake_runpod.pods["pod1"] = {"id": "pod1", "name": "diffusion-for-krita-x", "status": "RUNNING"}
    assert runpod.terminate_pod_blocking("test-key", "pod1", url=fake_runpod.url)
    assert "pod1" not in fake_runpod.pods
    assert not runpod.terminate_pod_blocking("wrong", "pod2", url=fake_runpod.url)


# Serverless #######################################################################################


def serverless_config(**kwargs):
    defaults = {
        "api_key": "test-key",
        "gpu": "NVIDIA GeForce RTX 4090",
        "workloads": ["sdxl"],
        "checkpoints": True,
        "keep_warm": 60,
    }
    return ServerlessConfig(**(defaults | kwargs))


def serverless(fake: FakeRunpod, **kwargs):
    api = RunpodApi("test-key", fake.url)
    return RunpodServerless(serverless_config(**kwargs), api, jobs_url=f"{fake.url}/jobs")


def body_items(data: bytes, status=200, ct="application/json", chunk=10):
    parts = [data[i : i + chunk] for i in range(0, len(data), chunk)] or [b""]
    for i, part in enumerate(parts):
        yield {"t": "body", "i": i, "n": len(parts), "status": status, "ct": ct, "d": b64(part)}


def b64(data: bytes):
    return base64.b64encode(data).decode()


@pytest.fixture()
def fast_jobs(monkeypatch):
    monkeypatch.setattr(ServerlessJobs, "poll_interval", 0.01)


@qtapp
async def test_serverless_start(fake_runpod: FakeRunpod, fast_jobs):
    def worker(input):
        assert input == {"op": "download", "workloads": ["sdxl"], "checkpoints": True}
        yield {"t": "progress", "message": "Downloading models", "done": 1, "total": 4}
        yield {"t": "progress", "message": "Models ready", "done": 4, "total": 4}

    fake_runpod.worker = worker
    sl = serverless(fake_runpod)
    phases = [p async for p in sl.start()]
    assert phases[0].phase is PodPhase.creating
    assert (1, 4) in [p.progress for p in phases if p.phase is PodPhase.downloading]
    assert phases[-1].phase is PodPhase.ready

    endpoint = fake_runpod.endpoints[sl.endpoint_id]
    assert endpoint["type"] == "QUEUE" and endpoint["env"] == {"KRITA_MODE": "serverless"}
    assert endpoint["gpu"]["pools"] == ["ADA_24"]
    assert endpoint["workers"] == {"min": 0, "max": 1, "idleTimeout": 60}
    assert endpoint["mounts"]["network"] == [{"volumeId": sl.volume_id}]
    assert endpoint["dataCenterIds"] == [fake_runpod.volumes[sl.volume_id]["dataCenter"]]


@qtapp
async def test_serverless_endpoint_reused_or_replaced(fake_runpod: FakeRunpod, fast_jobs):
    first = serverless(fake_runpod)
    _ = [p async for p in first.start()]
    ids = {"volume_id": first.volume_id, "endpoint_id": first.endpoint_id}

    same = serverless(fake_runpod, **ids)
    _ = [p async for p in same.start()]
    assert same.endpoint_id == first.endpoint_id and len(fake_runpod.endpoints) == 1

    changed = serverless(fake_runpod, keep_warm=300, **ids)
    _ = [p async for p in changed.start()]
    assert changed.endpoint_id != first.endpoint_id
    assert list(fake_runpod.endpoints) == [changed.endpoint_id]
    assert fake_runpod.endpoints[changed.endpoint_id]["workers"]["idleTimeout"] == 300


@qtapp
async def test_serverless_download_error(fake_runpod: FakeRunpod, fast_jobs):
    def worker(input):
        raise RuntimeError("disk full")

    fake_runpod.worker = worker
    with pytest.raises(RunpodError, match="disk full"):
        _ = [p async for p in serverless(fake_runpod).start()]


async def connected_client(fake: FakeRunpod):
    sl = serverless(fake)
    _ = [p async for p in sl.start()]
    return sl.client()


@qtapp
async def test_serverless_http(fake_runpod: FakeRunpod, fast_jobs):
    stats = json.dumps({"system": {"python_version": "3.12"}, "devices": []}).encode()

    def worker(input):
        if input["op"] == "http" and input["path"] == "system_stats":
            yield from body_items(stats, chunk=7)  # reassembled from many chunks
        elif input["op"] == "http":
            yield from body_items(b"not found", status=404, ct="text/plain")

    client = await connected_client(fake_runpod)
    fake_runpod.worker = worker
    assert await client._get("system_stats") == json.loads(stats)
    with pytest.raises(NetworkError, match="not found"):
        await client._get("missing")


@qtapp
async def test_serverless_upload_in_parts(fake_runpod: FakeRunpod, fast_jobs, monkeypatch):
    monkeypatch.setattr(runpod_serverless, "max_job_input", 10)
    received: list[dict] = []

    def worker(input):
        received.append(input)
        yield {"t": "body", "i": 0, "n": 1, "status": 200, "ct": "text/plain", "d": ""}

    client = await connected_client(fake_runpod)
    fake_runpod.worker = worker
    await client._put("api/etn/image/small", b"0123456789")
    data = bytes(range(25))
    await client._put("api/etn/upload/loras/big", data)

    assert received[0]["op"] == "http" and base64.b64decode(received[0]["data"]) == b"0123456789"
    parts = received[1:]
    assert [p["op"] for p in parts] == ["upload"] * 3 and len({p["id"] for p in parts}) == 1
    assert b"".join(base64.b64decode(p["data"]) for p in parts) == data


@qtapp
async def test_serverless_prompt(fake_runpod: FakeRunpod, fast_jobs):
    png = Image.create(Extent(8, 4), Qt.GlobalColor.red).to_bytes().data()
    executed = {
        "prompt_id": "p1",
        "node": "9",
        "output": {"images": [{"source": "http", "id": "img1"}]},
    }

    def worker(input):
        assert input["op"] == "prompt" and input["data"]["prompt_id"] == "p1"
        yield {"t": "ws", "m": json.dumps({"type": "execution_start", "data": {"prompt_id": "p1"}})}
        for i, n, part in ((0, 2, png[:20]), (1, 2, png[20:])):
            yield {"t": "img", "id": "img1", "i": i, "n": n, "d": b64(part)}
        yield {"t": "ws", "m": json.dumps({"type": "executed", "data": executed})}

    client = await connected_client(fake_runpod)
    fake_runpod.worker = worker
    assert client._start_prompt({"prompt": {}, "prompt_id": "p1"}) == {"prompt_id": "p1"}
    assert client._prompt_task is not None
    await client._prompt_task

    messages = [json.loads(client._socket.queue.get_nowait()) for _ in range(2)]
    assert [m["type"] for m in messages] == ["execution_start", "executed"]
    image = await client._transfer_result_image("img1")
    assert image.extent == Extent(8, 4)


@qtapp
async def test_serverless_prompt_error(fake_runpod: FakeRunpod, fast_jobs):
    def worker(input):
        raise RuntimeError("CUDA out of memory")

    client = await connected_client(fake_runpod)
    fake_runpod.worker = worker
    client._start_prompt({"prompt": {}, "prompt_id": "p2"})
    assert client._prompt_task is not None
    await client._prompt_task
    message = json.loads(client._socket.queue.get_nowait())
    assert message["type"] == "execution_error"
    assert message["data"]["prompt_id"] == "p2"
    assert "CUDA out of memory" in message["data"]["exception_message"]


def test_serverless_estimate():
    estimate = estimate_serverless(18, 1.10, keep_warm=60)
    assert estimate.storage_month == pytest.approx(30 * runpod.network_volume_gb_month)
    assert estimate.per_100_images == pytest.approx(100 * 10 / 3600 * 1.10)
    assert estimate.keep_warm_cost == pytest.approx(60 / 3600 * 1.10)
    assert "per 100 images" in estimate.describe(18, 60)


@pytest.mark.parametrize("workload", ["qwen", "qwen2", "krea2", "moss", "fibo2"])
def test_new_workloads(workload: str):
    from ai_diffusion.backend.resources import ResourceKind, required_models

    assert workload in runpod.workload_names
    assert model_size_gb([workload], checkpoints=False) > 4  # text encoder at least
    assert model_size_gb([workload], checkpoints=True) > model_size_gb([workload], False) + 5
    archs = {
        "qwen": [Arch.qwen],
        "qwen2": [Arch.qwen2],
        "krea2": [Arch.krea2],
        "moss": [Arch.moss],
        "fibo2": [Arch.fibo2],
    }[workload]
    kinds = {m.kind for m in required_models if m.arch in archs}
    assert {ResourceKind.text_encoder, ResourceKind.vae} <= kinds


def test_shared_model_files_counted_once():
    together = model_size_gb(["krea2", "fibo2"], checkpoints=True)
    separate = model_size_gb(["krea2"], True) + model_size_gb(["fibo2"], True)
    assert together < separate - 4  # Qwen3-VL 4B text encoder is shared
