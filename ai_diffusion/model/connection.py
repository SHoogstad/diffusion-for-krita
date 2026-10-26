from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from enum import Enum

from .. import eventloop, util
from ..backend.client import (
    Client,
    ClientEvent,
    ClientMessage,
    DeviceInfo,
    MissingResources,
    SharedWorkflow,
)
from ..backend.comfy_client import ComfyClient
from ..backend.network import NetworkError
from ..backend.runpod import PodConfig, PodPhase, RunpodError, RunpodPod, terminate_pod_blocking
from ..backend.runpod_serverless import RunpodServerless, ServerlessConfig
from ..localization import translate as _
from ..qt.QtCore import QObject, QTimer, pyqtSignal
from ..settings import PerformancePreset, RunpodMode, ServerMode, Settings, settings
from .properties import ObservableProperties, Property


class ConnectionState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3

    discover_models = 10

    pod_starting = 30  # Runpod: creating the pod, downloading models, starting ComfyUI


class Connection(QObject, ObservableProperties):
    state = Property(ConnectionState.disconnected)
    error = Property("")
    progress = Property((1, 1))
    pod_status = Property("")

    state_changed = pyqtSignal(ConnectionState)
    error_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(tuple)
    pod_status_changed = pyqtSignal(str)
    models_changed = pyqtSignal()
    message_received = pyqtSignal(ClientMessage)
    workflow_published = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self._client: Client | None = None
        self._task: asyncio.Task | None = None
        self._workflows: dict[str, dict] = {}
        self._temporary_disconnect = False
        self._pod: RunpodPod | None = None
        self._serverless: RunpodServerless | None = None
        self._last_activity = time.monotonic()
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(30_000)
        self._idle_timer.timeout.connect(self._check_idle)
        self.error_kind = ""
        self.missing_resources: MissingResources | None = None

        settings.changed.connect(self._handle_settings_changed)

    def __del__(self):
        if self._task is not None:
            self._task.cancel()

    async def _connect(self, client: Client):
        if self.state is ConnectionState.connecting:
            return
        if self.state is ConnectionState.connected:
            await self.disconnect(stop_pod=False)
        self.error = ""
        self.error_kind = ""
        self.state = ConnectionState.connecting
        try:
            await client.connect()

            self.state = ConnectionState.discover_models
            async for status in client.discover_models(refresh=False):
                self.progress = (status.current, status.total)
            self.missing_resources = client.missing_resources

            apply_performance_preset(settings, client.device_info)

            self._client = client
            if self._task is None:
                self._task = eventloop._loop.create_task(self._handle_messages())
            self.state = ConnectionState.connected
            self.models_changed.emit()
        except NetworkError as e:
            self.error = e.message
            self.error_kind = "network"
            self.state = ConnectionState.error
        except MissingResources as e:
            self.error = _(
                "Connection established, but the server is missing required custom nodes or models."
            )
            self.error_kind = "missing_resources"
            self.missing_resources = e
            self.state = ConnectionState.error
        except Exception as e:
            self.error = util.log_error(e)
            self.error_kind = "unknown"
            self.state = ConnectionState.error

    def create_client(self, settings: Settings) -> Client | None:
        match settings.server_mode:
            case ServerMode.external | ServerMode.managed:
                return ComfyClient(settings.server_url)
            case _:
                raise ValueError(f"Invalid server mode: {settings.server_mode}")

    def connect(self, client: Client):
        eventloop.run(self._connect(client))

    # Runpod ###########################################################################################

    @property
    def pod_running(self):
        return self._pod is not None or self._serverless is not None

    def start_pod(self):
        eventloop.run(self._start_pod())

    def stop_pod(self):
        eventloop.run(self.disconnect())

    async def _start_pod(self, reuse_only=False):
        if settings.runpod_mode is RunpodMode.serverless:
            if not reuse_only:  # connecting to serverless starts a worker, never automatically
                await self._start_serverless()
            return
        if self._pod is not None or self.state in (
            ConnectionState.pod_starting,
            ConnectionState.connecting,
        ):
            return
        if not settings.runpod_api_key:
            self.error = _("Enter your Runpod API key first.")
            self.state = ConnectionState.error
            return
        if reuse_only and not settings.runpod_pod_id:
            return
        config = PodConfig(
            api_key=settings.runpod_api_key,
            gpu=settings.runpod_gpu,
            storage=settings.runpod_storage,
            workloads=settings.runpod_workloads,
            checkpoints=settings.runpod_checkpoints,
            idle_minutes=settings.runpod_idle_minutes,
            volume_id=settings.runpod_volume_id,
            pod_id=settings.runpod_pod_id,
            pod_token=settings.runpod_pod_token,
        )
        pod = self._pod = RunpodPod(config)
        self.error = ""
        self.error_kind = ""
        self.state = ConnectionState.pod_starting
        try:
            async for status in pod.start():
                if pod.info and pod.info.id != settings.runpod_pod_id:
                    settings.runpod_pod_id = pod.info.id
                    settings.runpod_pod_token = pod.info.token
                    settings.runpod_volume_id = pod.volume_id
                    settings.save()
                if reuse_only and status.phase is PodPhase.creating:
                    self.state = ConnectionState.disconnected
                    return  # the saved pod is gone, don't start a new one automatically
                self.pod_status = _pod_status_text(status.phase, status.progress, serverless=False)
                done, total = status.progress
                self.progress = (done, total) if total else (0, 0)
            assert pod.info is not None
            self.pod_status = ""
            self._last_activity = time.monotonic()
            self._idle_timer.start()  # also on error, the pod is stopped when it isn't used
            await self._connect(ComfyClient(pod.info.url, pod.info.token))
        except Exception as e:
            self.pod_status = ""
            self.error = str(e) if isinstance(e, RunpodError) else util.log_error(e)
            self.error_kind = "runpod"
            self.state = ConnectionState.error
            await self._stop_pod()
        finally:
            if self._pod is pod and pod.info is None:
                self._pod = None

    async def _start_serverless(self):
        if self.pod_running or self.state in (
            ConnectionState.pod_starting,
            ConnectionState.connecting,
        ):
            return
        if not settings.runpod_api_key:
            self.error = _("Enter your Runpod API key first.")
            self.state = ConnectionState.error
            return
        config = ServerlessConfig(
            api_key=settings.runpod_api_key,
            gpu=settings.runpod_gpu,
            workloads=settings.runpod_workloads,
            checkpoints=settings.runpod_checkpoints,
            keep_warm=settings.runpod_keep_warm,
            volume_id=settings.runpod_volume_id,
            endpoint_id=settings.runpod_endpoint_id,
        )
        serverless = self._serverless = RunpodServerless(config)
        self.error = ""
        self.error_kind = ""
        self.state = ConnectionState.pod_starting
        try:
            async for status in serverless.start():
                if (serverless.volume_id, serverless.endpoint_id) != (
                    settings.runpod_volume_id,
                    settings.runpod_endpoint_id,
                ):
                    settings.runpod_volume_id = serverless.volume_id
                    settings.runpod_endpoint_id = serverless.endpoint_id
                    settings.save()
                self.pod_status = _pod_status_text(status.phase, status.progress, serverless=True)
                done, total = status.progress
                self.progress = (done, total) if total else (0, 0)
            self.pod_status = ""
            await self._connect(serverless.client())
            if self.state is not ConnectionState.connected:
                self._serverless = None
        except Exception as e:
            self._serverless = None
            self.pod_status = ""
            self.error = str(e) if isinstance(e, RunpodError) else util.log_error(e)
            self.error_kind = "runpod"
            self.state = ConnectionState.error

    async def delete_serverless_endpoint(self):
        """Removes the endpoint on Runpod (it costs nothing while idle, but keeps the volume)."""
        await self.disconnect()
        if settings.runpod_endpoint_id:
            config = ServerlessConfig(
                settings.runpod_api_key, settings.runpod_gpu, [], False, 0,
                endpoint_id=settings.runpod_endpoint_id,
            )  # fmt: skip
            await RunpodServerless(config).delete()
            settings.runpod_endpoint_id = ""
            settings.save()

    async def _stop_pod(self):
        self._idle_timer.stop()
        self._serverless = None  # scales to zero on its own
        if pod := self._pod:
            self._pod = None
            try:
                await pod.stop()
                settings.runpod_pod_id = ""  # else kept, the next start reuses or removes it
                settings.runpod_pod_token = ""
                settings.save()
            except Exception as e:
                message = util.log_error(e)
                self.error = (
                    _("Could not terminate the Runpod pod, check runpod.io") + f": {message}"
                )
        self.pod_status = ""

    def shutdown(self):
        """Krita is closing: terminate the pod without waiting for the event loop."""
        if self._pod and self._pod.info:
            if terminate_pod_blocking(settings.runpod_api_key, self._pod.info.id):
                settings.runpod_pod_id = ""
                settings.runpod_pod_token = ""
                settings.save()

    def _check_idle(self):
        minutes = settings.runpod_idle_minutes
        if self._pod is None or minutes <= 0:
            return
        idle = (time.monotonic() - self._last_activity) / 60
        if idle >= minutes:
            util.client_logger.info(f"Runpod pod idle for {idle:.0f} minutes, terminating")
            eventloop.run(self._stop_idle_pod(minutes))

    async def _stop_idle_pod(self, minutes: int):
        await self.disconnect()
        self.error = _("The Runpod pod was stopped after {minutes} minutes without use.").format(
            minutes=minutes
        )

    ####################################################################################################

    async def disconnect(self, stop_pod=True):  # type: ignore (hides QObject.disconnect)
        if self._task is not None:
            self._task.cancel()
            await self._task
            self._task = None

        self._client = None
        self.error = ""
        self.missing_resources = None
        self.state = ConnectionState.disconnected
        if stop_pod:
            await self._stop_pod()

    def interrupt(self):
        eventloop.run(self.client.interrupt())

    def cancel(self, job_ids: Iterable[str]):
        eventloop.run(self.client.cancel(job_ids))

    def refresh(self):
        async def _refresh():
            await self.client.refresh()
            self.models_changed.emit()

        if self.state is ConnectionState.connected:
            eventloop.run(_refresh())

    @property
    def client(self):
        assert self.state is ConnectionState.connected and self._client is not None
        return self._client

    @property
    def client_if_connected(self):
        return self._client

    @property
    def workflows(self):
        return self._workflows

    async def _handle_messages(self):
        try:
            client = self._client
            self._temporary_disconnect = False
            assert client is not None

            async with client:
                async for msg in client.listen():
                    try:
                        self._handle_message(msg)
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        util.client_logger.exception(e)
                        self.error = _("Error handling server message: ") + str(e)
        except asyncio.CancelledError:
            pass  # shutdown

    def _handle_message(self, msg: ClientMessage):
        self._last_activity = time.monotonic()
        match msg:
            case (ClientEvent.error, "", *_):
                self.error = _("Error communicating with server: ") + str(msg.error)
            case (ClientEvent.disconnected, *_):
                self._temporary_disconnect = True
                self.error = _("Disconnected from server, trying to reconnect...")
            case (ClientEvent.connected, *_):
                if self._temporary_disconnect:
                    self._temporary_disconnect = False
                    self.error = ""
            case (ClientEvent.published, *_):
                assert isinstance(msg.result, SharedWorkflow)
                self._workflows[msg.result.publisher] = msg.result.workflow
                self.workflow_published.emit(msg.result.publisher)
            case _:
                self.message_received.emit(msg)

    def _handle_settings_changed(self, key: str, value: object):
        if key == "server_mode" and self.pod_running and value is not ServerMode.runpod:
            self.error = ""
            eventloop.run(self.disconnect())


def _pod_status_text(phase: PodPhase, progress: tuple[int, int], serverless: bool):
    match phase:
        case PodPhase.creating if serverless:
            return _("Preparing Runpod endpoint...")
        case PodPhase.starting if serverless:
            return _("Waiting for a Runpod worker (cold start)...")
        case PodPhase.creating:
            return _("Creating Runpod pod...")
        case PodPhase.starting:
            return _("Starting pod...")
        case PodPhase.downloading:
            done, total = progress
            if total:
                return _("Downloading models") + f" {done / 1024**3:.1f} / {total / 1024**3:.1f} GB"
            return _("Checking models...")
        case PodPhase.ready:
            return _("Connecting...")


def apply_performance_preset(settings: Settings, device: DeviceInfo):
    if settings.performance_preset is PerformancePreset.auto:
        if device.type == "cpu":
            settings.apply_performance_preset(PerformancePreset.cpu)
        elif device.type.lower() == "cloud":
            settings.apply_performance_preset(PerformancePreset.cloud)
        elif device.vram <= 6:
            settings.apply_performance_preset(PerformancePreset.low)
        elif device.vram <= 12:
            settings.apply_performance_preset(PerformancePreset.medium)
        else:
            settings.apply_performance_preset(PerformancePreset.high)
