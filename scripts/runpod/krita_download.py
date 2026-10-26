"""Download the models the plugin asked for and report progress.

Pods run this as a script before ComfyUI starts: the selection comes from KRITA_WORKLOADS
(comma separated: sd15, sdxl, illu, flux, flux2, zimage, qwen, qwen2, krea2,
moss, fibo2, h3) and KRITA_CHECKPOINTS=1 to include the
default checkpoints of those workloads, progress goes to /krita/status. The serverless worker
calls download() directly. Files that already exist (network volume) are skipped.
"""

import asyncio
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/opt/krita/scripts")
sys.path.insert(0, "/opt/krita")
import download_models as dm

from ai_diffusion.backend.resources import Arch, ModelRequirements, default_checkpoints

STATUS_FILE = Path(os.environ.get("KRITA_STATUS_FILE", "/tmp/krita_status.json"))
_checkpoint_archs = {
    "sd15": [Arch.sd15],
    "sdxl": [Arch.sdxl],
    "illu": [Arch.illu, Arch.illu_v],
    "flux": [Arch.flux],
    "flux2": [Arch.flux2_4b],
    "zimage": [Arch.zimage],
    "qwen": [Arch.qwen, Arch.qwen_e, Arch.qwen_e_p],
    "qwen2": [Arch.qwen2],
    "krea2": [Arch.krea2],
    "moss": [Arch.moss],
    "fibo2": [Arch.fibo2, Arch.fibo2_e],
    "h3": [Arch.h3_e],
}

Progress = Callable[[str, int, int], None]  # message, bytes done, bytes total


def write_status(phase: str, message: str, done=0, total=0):
    data = {"phase": phase, "message": message, "progress": [done, total]}
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(STATUS_FILE)


def remote_size(url: str):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "krita"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("x-linked-size") or r.headers.get("content-length") or 0)
    except Exception:
        return 0


def local_size(path: Path):
    for candidate in (path, path.with_suffix(".part")):
        if candidate.exists():
            return candidate.stat().st_size
    return 0


def select_models(workloads: list[str], checkpoints: bool):
    unknown = [w for w in workloads if w not in _checkpoint_archs]
    if unknown:
        raise ValueError(f"Unknown workloads: {unknown}")
    checkpoint_ids = []
    if checkpoints:
        archs = [a for w in workloads for a in _checkpoint_archs[w]]
        checkpoint_ids = [m.id.identifier for m in default_checkpoints if m.arch in archs]
    return dm.list_models(
        **dict.fromkeys(workloads, True),
        checkpoints=checkpoint_ids,
        backend=ModelRequirements.cuda,
    )


async def download(destination: Path, workloads: list[str], checkpoints: bool, report: Progress):
    report("Checking models", 0, 0)
    models = select_models(workloads, checkpoints)
    files = [f for m in models for f in m.files]
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(remote_size, [dm._map_url(f.url) for f in files]))
    total = sum(sizes)

    async def report_progress():
        while True:
            done = sum(local_size(destination / f.path) for f in files)
            report("Downloading models", done, total)
            await asyncio.sleep(2)

    reporter = asyncio.create_task(report_progress())
    start = time.monotonic()
    try:
        # The download script only retries with continue_on_error, so check the result here
        for attempt in range(3):
            await dm.download_models(
                destination, models, parallel_downloads=6, continue_on_error=True
            )
            missing = [f.path.as_posix() for f in files if not (destination / f.path).exists()]
            if not missing:
                break
            print(f"[krita_download] attempt {attempt + 1}: {len(missing)} files missing, retrying")
        else:
            raise RuntimeError(f"{len(missing)} files could not be downloaded: {missing[:3]}")
    finally:
        reporter.cancel()
    print(f"[krita_download] {total / 1e9:.1f} GB checked in {time.monotonic() - start:.0f}s")
    report("Models ready", total, total)


async def main():
    workloads = [w for w in os.environ.get("KRITA_WORKLOADS", "sdxl").split(",") if w]
    checkpoints = os.environ.get("KRITA_CHECKPOINTS", "1") == "1"

    def report(message: str, done: int, total: int):
        write_status("downloading", message, done, total)

    await download(Path(sys.argv[1]), workloads, checkpoints, report)
    write_status("starting", "Starting ComfyUI")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        write_status("error", f"Model download failed: {e}")
        raise
