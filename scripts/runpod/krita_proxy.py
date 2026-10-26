"""Gatekeeper in front of ComfyUI on a Runpod pod.

Runpod exposes the pod's HTTP port on a public proxy URL, so every request must carry the
token the plugin generated when it created the pod (Authorization: Bearer <KRITA_TOKEN>).

* /krita/status reports start-up progress (model download, ComfyUI start) to the plugin.
* Everything else is forwarded to ComfyUI (HTTP and websocket) once it is ready.
* Safety net if Krita crashes or loses its connection: when no client has been connected
  for KRITA_ORPHAN_MINUTES, or no job was queued for KRITA_IDLE_MINUTES, the pod
  terminates itself so it stops costing money.
"""

import asyncio
import hmac
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp
from aiohttp import web

TOKEN = os.environ["KRITA_TOKEN"]
PORT = int(os.environ.get("KRITA_PORT", "8188"))
COMFY_URL = f"http://127.0.0.1:{os.environ.get('COMFY_PORT', '8189')}"
STATUS_FILE = Path(os.environ.get("KRITA_STATUS_FILE", "/tmp/krita_status.json"))
IDLE_MINUTES = float(os.environ.get("KRITA_IDLE_MINUTES", "15"))
ORPHAN_MINUTES = float(os.environ.get("KRITA_ORPHAN_MINUTES", "10"))

_hop_headers = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "authorization", "content-length",
}  # fmt: skip


class Activity:
    def __init__(self):
        self.last_job = time.monotonic()
        self.last_client = time.monotonic()
        self.clients = 0

    def job(self):
        self.last_job = time.monotonic()

    def idle_minutes(self):
        return (time.monotonic() - self.last_job) / 60

    def orphan_minutes(self):
        if self.clients > 0:
            return 0.0
        return (time.monotonic() - self.last_client) / 60


activity = Activity()


def read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {"phase": "starting", "message": "Starting"}


def authorized(request: web.Request):
    header = request.headers.get("Authorization", "")
    return hmac.compare_digest(header, f"Bearer {TOKEN}")


@web.middleware
async def auth_middleware(request: web.Request, handler):
    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return await handler(request)


async def status(request: web.Request):
    result = read_status()
    result["idle_minutes"] = round(activity.idle_minutes(), 1)
    return web.json_response(result)


async def forward(request: web.Request):
    if read_status().get("phase") != "ready":
        return web.json_response({"error": "ComfyUI is not ready yet"}, status=503)
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await forward_websocket(request)

    if request.method == "POST" and request.path in ("/prompt", "/api/prompt"):
        activity.job()
    session: aiohttp.ClientSession = request.app["session"]
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _hop_headers}
    async with session.request(
        request.method,
        COMFY_URL + request.path_qs,
        headers=headers,
        data=request.content if request.body_exists else None,
    ) as upstream:
        response = web.StreamResponse(status=upstream.status)
        for key, value in upstream.headers.items():
            if key.lower() not in _hop_headers:
                response.headers[key] = value
        await response.prepare(request)
        async for chunk in upstream.content.iter_chunked(1 << 16):
            await response.write(chunk)
        await response.write_eof()
        return response


async def forward_websocket(request: web.Request):
    client = web.WebSocketResponse(max_msg_size=0, heartbeat=30)
    await client.prepare(request)
    session: aiohttp.ClientSession = request.app["session"]
    activity.clients += 1
    try:
        async with session.ws_connect(
            COMFY_URL.replace("http", "ws", 1) + request.path_qs, max_msg_size=0
        ) as upstream:

            async def pipe(source, target):
                async for msg in source:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await target.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await target.send_bytes(msg.data)
                    else:
                        break

            tasks = [
                asyncio.create_task(pipe(client, upstream)),
                asyncio.create_task(pipe(upstream, client)),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                task.cancel()
    finally:
        activity.clients -= 1
        activity.last_client = time.monotonic()
        await client.close()
    return client


def terminate_pod(reason: str):
    print(f"[krita_proxy] Terminating pod: {reason}", flush=True)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if pod_id and api_key:
        try:
            req = urllib.request.Request(
                f"https://api.runpod.io/v2/pods/{pod_id}",
                method="DELETE",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            urllib.request.urlopen(req, timeout=30)
            return
        except Exception as e:
            print(f"[krita_proxy] API terminate failed: {e}", flush=True)
    if pod_id:  # fallback, at least stop the GPU from being billed (runpodctl v1 and v3 syntax)
        for args in (["pod", "stop", pod_id], ["stop", "pod", pod_id]):
            if subprocess.run(["runpodctl", *args], check=False).returncode == 0:
                break


async def watchdog(app: web.Application):
    error_since: float | None = None
    while True:
        await asyncio.sleep(30)
        phase = read_status().get("phase")
        if phase == "error":
            error_since = error_since or time.monotonic()
            if time.monotonic() - error_since >= 5 * 60:
                terminate_pod("start-up failed")
            continue
        error_since = None
        if phase != "ready":
            continue  # downloading models or starting, the plugin polls /krita/status
        if activity.orphan_minutes() >= ORPHAN_MINUTES:
            terminate_pod(f"no client connected for {ORPHAN_MINUTES:g} minutes")
        elif IDLE_MINUTES > 0 and activity.idle_minutes() >= IDLE_MINUTES:
            terminate_pod(f"no job queued for {IDLE_MINUTES:g} minutes")


async def on_startup(app: web.Application):
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    app["session"] = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)
    app["watchdog"] = asyncio.create_task(watchdog(app))


async def on_cleanup(app: web.Application):
    app["watchdog"].cancel()
    await app["session"].close()


def main():
    app = web.Application(middlewares=[auth_middleware], client_max_size=2**34)
    app.router.add_get("/krita/status", status)
    app.router.add_route("*", "/{path:.*}", forward)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)


if __name__ == "__main__":
    main()
