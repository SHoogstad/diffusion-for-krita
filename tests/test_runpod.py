import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from ai_diffusion.backend import runpod
from ai_diffusion.backend.runpod import (
    GpuType,
    PodConfig,
    PodPhase,
    RunpodApi,
    RunpodError,
    RunpodPod,
    estimate,
    model_size_gb,
    volume_size_gb,
)
from ai_diffusion.settings import RunpodStorage

from .conftest import qtapp

gpu_4090 = {
    "id": "NVIDIA GeForce RTX 4090",
    "name": "RTX 4090",
    "manufacturer": "NVIDIA",
    "memory": 24,
    "price": {"secure": 0.69, "community": 0.34},
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
        return 404, {"title": "Not Found", "status": 404, "detail": f"{method} {path}"}


@pytest.fixture()
def fake_runpod(monkeypatch):
    fake = FakeRunpod()
    monkeypatch.setattr(runpod, "pod_url", lambda id: f"{fake.url}/pod/{id}")
    monkeypatch.setattr(RunpodPod, "poll_interval", 0.01)
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
