"""Download the models the plugin asked for and report progress to /krita/status.

The selection comes from KRITA_WORKLOADS (comma separated: sd15, sdxl, illu, flux, flux2,
zimage) and KRITA_CHECKPOINTS=1 to include the default checkpoints of those workloads.
Files that already exist (network volume) are skipped by the download script.
"""

import asyncio
import json
import os
import sys
import time
import urllib.request
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
}


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


def select_models():
    workloads = [w for w in os.environ.get("KRITA_WORKLOADS", "sdxl").split(",") if w]
    unknown = [w for w in workloads if w not in _checkpoint_archs]
    if unknown:
        raise ValueError(f"Unknown workloads: {unknown}")
    checkpoints = []
    if os.environ.get("KRITA_CHECKPOINTS", "1") == "1":
        archs = [a for w in workloads for a in _checkpoint_archs[w]]
        checkpoints = [m.id.identifier for m in default_checkpoints if m.arch in archs]
    return dm.list_models(
        **dict.fromkeys(workloads, True),
        checkpoints=checkpoints,
        backend=ModelRequirements.cuda,
    )


async def report_progress(destination: Path, files: list, total: int):
    while True:
        done = sum(local_size(destination / f.path) for f in files)
        write_status("downloading", "Downloading models", done, total)
        await asyncio.sleep(2)


async def main():
    destination = Path(sys.argv[1])
    write_status("downloading", "Checking models")
    models = select_models()
    files = [f for m in models for f in m.files]
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(remote_size, [dm._map_url(f.url) for f in files]))
    total = sum(sizes)
    reporter = asyncio.create_task(report_progress(destination, files, total))
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
    write_status("starting", "Starting ComfyUI", total, total)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        write_status("error", f"Model download failed: {e}")
        raise
