"""Write ai_diffusion/backend/runpod_models.json: download size of each model file and which files
each Runpod workload needs. The plugin uses it to estimate storage and download time.

Run from the repository root after changing model resources: python scripts/runpod/model_sizes.py
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(root))
import download_models as dm

from ai_diffusion.backend.resources import Arch, ModelRequirements, default_checkpoints

checkpoint_archs = {
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


def remote_size(url: str) -> int:
    for _ in range(3):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "krita"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return int(r.headers.get("x-linked-size") or r.headers.get("content-length") or 0)
        except Exception as e:
            error = e
    print(f"Could not get size of {url}: {error}", file=sys.stderr)
    return 0


def files_of(models) -> dict[str, str]:
    return {f.path.as_posix(): f.url for m in models for f in m.files}


def main():
    urls: dict[str, str] = {}
    workloads = {}
    for name, archs in checkpoint_archs.items():
        base = dm.list_models(**{name: True}, backend=ModelRequirements.cuda)
        cps = [m.id.identifier for m in default_checkpoints if m.arch in archs]
        full = dm.list_models(**{name: True}, checkpoints=cps, backend=ModelRequirements.cuda)
        base_files, full_files = files_of(base), files_of(full)
        urls.update(full_files)
        workloads[name] = {
            "files": sorted(base_files),
            "checkpoints": sorted(set(full_files) - set(base_files)),
        }
    with ThreadPoolExecutor(16) as pool:
        sizes = dict(zip(urls, pool.map(remote_size, urls.values())))
    result = {"files": dict(sorted(sizes.items())), "workloads": workloads}
    target = root / "ai_diffusion" / "backend" / "runpod_models.json"
    target.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Wrote {target} ({sum(sizes.values()) / 1024**3:.1f} GB in {len(sizes)} files)")


if __name__ == "__main__":
    main()
