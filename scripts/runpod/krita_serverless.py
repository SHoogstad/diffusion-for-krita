"""Runpod serverless worker: runs ComfyUI and executes jobs from the Diffusion for Krita plugin.

ComfyUI starts once per worker; models are on the endpoint's network volume. Each job is one
of the operations below. Results are streamed back as small items (Runpod limits a streamed
chunk to 1 MB), the plugin reassembles them.

  {"op": "http", "method": "GET"|"POST"|"PUT", "path": "...", "json": {...}, "data": "<b64>"}
      Forward one HTTP request to ComfyUI. Yields {"t": "body", "i", "n", "status", "ct", "d"}.
  {"op": "upload", "path": "...", "id": "...", "i": 0, "n": 3, "data": "<b64>"}
      Part of a large PUT (e.g. a LoRA). Parts are kept on the volume until all have arrived.
  {"op": "prompt", "data": {"prompt": {...}, "prompt_id": "..."}}
      Queue a workflow and stream ComfyUI's websocket messages {"t": "ws", "m": "<json>"}.
      Result images are sent before the message announcing them: {"t": "img", "id", "i", "n", "d"}.
  {"op": "download", "workloads": [...], "checkpoints": true}
      Download models to the volume. Yields {"t": "progress", "message", "done", "total"}.
"""

import asyncio
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import aiohttp
import runpod

sys.path.insert(0, "/opt/krita/runpod")
import krita_download

COMFY_PORT = 8189
COMFY_URL = f"http://127.0.0.1:{COMFY_PORT}"
CHUNK = 700_000  # bytes of payload per streamed item, base64 grows it to ~930 KB
VOLUME = Path(os.environ.get("KRITA_VOLUME", "/runpod-volume")) / "krita"
UPLOADS = VOLUME / "uploads"
CLIENT_ID = str(uuid.uuid4())


def start_comfy():
    """Point ComfyUI's models folder to the network volume and start it in the background."""
    models = VOLUME / "models"
    models.mkdir(parents=True, exist_ok=True)
    comfy_models = Path("/opt/comfyui/models")
    if not comfy_models.is_symlink():
        shutil.copytree(comfy_models, models, dirs_exist_ok=True)
        shutil.rmtree(comfy_models)
        comfy_models.symlink_to(models)
    args = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(COMFY_PORT)]
    args += os.environ.get("COMFY_ARGS", "").split()
    return subprocess.Popen(args, cwd="/opt/comfyui")


comfy = start_comfy()


async def wait_for_comfy(session: aiohttp.ClientSession):
    while True:
        if comfy.poll() is not None:
            raise RuntimeError("ComfyUI failed to start, see the worker logs")
        try:
            async with session.get(f"{COMFY_URL}/system_stats") as r:
                if r.status == 200:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)


def chunks(data: bytes):
    n = max(1, math.ceil(len(data) / CHUNK))
    for i in range(n):
        yield i, n, base64.b64encode(data[i * CHUNK : (i + 1) * CHUNK]).decode()


async def http(session: aiohttp.ClientSession, job: dict):
    method, path = job.get("method", "GET"), job["path"].lstrip("/")
    kwargs = {}
    if "json" in job:
        kwargs["json"] = job["json"]
    elif "data" in job:
        kwargs["data"] = base64.b64decode(job["data"])
    async with session.request(method, f"{COMFY_URL}/{path}", **kwargs) as r:
        body = await r.read()
        ct = r.headers.get("Content-Type", "")
        for i, n, d in chunks(body):
            yield {"t": "body", "i": i, "n": n, "status": r.status, "ct": ct, "d": d}


async def upload(session: aiohttp.ClientSession, job: dict):
    folder = UPLOADS / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{job['i']:05d}").write_bytes(base64.b64decode(job["data"]))
    parts = sorted(folder.iterdir())
    if len(parts) < job["n"]:
        yield {"t": "body", "i": 0, "n": 1, "status": 202, "ct": "text/plain", "d": ""}
        return
    data = b"".join(p.read_bytes() for p in parts)
    shutil.rmtree(folder, ignore_errors=True)
    async with session.put(f"{COMFY_URL}/{job['path'].lstrip('/')}", data=data) as r:
        yield {"t": "body", "i": 0, "n": 1, "status": r.status, "ct": "text/plain", "d": ""}


async def prompt(session: aiohttp.ClientSession, job: dict):
    data = dict(job["data"], client_id=CLIENT_ID)
    prompt_id = data["prompt_id"]
    async with session.ws_connect(f"{COMFY_URL}/ws?clientId={CLIENT_ID}", max_msg_size=0) as ws:
        async with session.post(f"{COMFY_URL}/prompt", json=data) as r:
            if r.status != 200:
                text = await r.text()
                yield {"t": "error", "error": f"ComfyUI rejected the workflow: {text[:2000]}"}
                return
        try:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue  # binary previews are not forwarded
                message = json.loads(msg.data)
                kind, payload = message.get("type"), message.get("data") or {}
                if payload.get("prompt_id") not in (None, prompt_id):
                    continue
                if kind == "executed":
                    for image in (payload.get("output") or {}).get("images", []):
                        if image.get("source") == "http" and image.get("id"):
                            async with session.get(f"{COMFY_URL}/api/etn/image/{image['id']}") as r:
                                body = await r.read()
                            for i, n, d in chunks(body):
                                yield {"t": "img", "id": image["id"], "i": i, "n": n, "d": d}
                yield {"t": "ws", "m": msg.data}
                finished = kind == "executing" and payload.get("node") is None
                if finished or kind in ("execution_error", "execution_interrupted"):
                    return
        except (asyncio.CancelledError, GeneratorExit):
            await session.post(f"{COMFY_URL}/interrupt", json={})  # job cancelled in Runpod
            raise


async def download(job: dict):
    queue: asyncio.Queue = asyncio.Queue()

    def report(message: str, done: int, total: int):
        queue.put_nowait({"t": "progress", "message": message, "done": done, "total": total})

    task = asyncio.create_task(
        krita_download.download(VOLUME, job["workloads"], job.get("checkpoints", True), report)
    )
    while not task.done() or not queue.empty():
        try:
            yield await asyncio.wait_for(queue.get(), timeout=1)
        except TimeoutError:
            pass
    task.result()  # raise download errors


async def handler(job: dict):
    request = job["input"]
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            op = request.get("op")
            if op == "download":
                async for item in download(request):
                    yield item
                return
            await wait_for_comfy(session)
            if op == "http":
                generator = http(session, request)
            elif op == "upload":
                generator = upload(session, request)
            elif op == "prompt":
                generator = prompt(session, request)
            else:
                raise ValueError(f"Unknown operation {op!r}")
            async for item in generator:
                yield item
        except Exception as e:
            yield {"t": "error", "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})
