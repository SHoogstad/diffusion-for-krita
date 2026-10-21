from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import ConditioningInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_client import ComfyClient
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.resources import ResourceKind, resource_id
from ai_diffusion.files import FileCollection, FileLibrary
from ai_diffusion.image import Extent, Image
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, Style

checkpoints = {
    Arch.moss: "moss_image_sft_bf16.safetensors",
    Arch.fibo2: "fibo2_dev_bf16.safetensors",
    Arch.fibo2_e: "fibo2_edit_turbo_bf16.safetensors",
}


def test_arch_from_string():
    # Model inspection doesn't know these models yet and reports "unknown"
    assert Arch.from_string("unknown", filename=checkpoints[Arch.moss]) is Arch.moss
    assert Arch.from_string("unknown", filename=checkpoints[Arch.fibo2]) is Arch.fibo2
    assert Arch.from_string("unknown", filename=checkpoints[Arch.fibo2_e]) is Arch.fibo2_e
    assert Arch.from_string("unknown", filename="something_else.safetensors") is None
    # Names used once inspection supports them
    assert Arch.from_string("moss-image") is Arch.moss
    assert Arch.from_string("fibo2", filename="Fibo2-Edit-Turbo.safetensors") is Arch.fibo2_e
    assert Arch.fibo2.supports_edit and not Arch.fibo2.is_edit
    assert Arch.fibo2_e.is_edit
    assert not Arch.moss.supports_edit


def node_info(clip_types: list[str]):
    clip_loader = {"input": {"required": {"type": [clip_types]}}}
    return ComfyObjectInfo({"CLIPLoader": clip_loader})


def client_models(arch: Arch, clip_types: list[str] | None = None):
    models = ClientModels()
    models.checkpoints = {
        checkpoints[arch]: CheckpointInfo(checkpoints[arch], arch, workflow.FileFormat.diffusion)
    }
    models.node_inputs = node_info(clip_types or ["flux2", arch.comfy_clip_type or ""])
    te = "qwen_3_5_9b" if arch is Arch.moss else "qwen_3vl_4b"
    te_file = "qwen3.5_9b_bf16.safetensors" if arch is Arch.moss else "qwen3vl_4b_bf16.safetensors"
    models.resources[resource_id(ResourceKind.text_encoder, Arch.all, te)] = te_file
    models.resources[resource_id(ResourceKind.vae, arch, "default")] = "flux2-vae.safetensors"
    return models


@pytest.mark.parametrize("arch", [Arch.moss, Arch.fibo2, Arch.fibo2_e])
def test_workload_requires_comfy_support(arch: Arch):
    def missing_for_arch(models: ClientModels):
        client = SimpleNamespace(models=models)
        missing = ComfyClient._check_workload(client, arch)  # type: ignore
        return [m for m in missing if m.arch is arch]  # ignore common models (upscalers etc.)

    assert missing_for_arch(client_models(arch)) == []

    missing = missing_for_arch(client_models(arch, clip_types=["flux2", "qwen_image"]))
    assert len(missing) == 1 and missing[0].kind is ResourceKind.node


def create(arch: Arch, kind: WorkflowKind, canvas: Image | Extent, cond: ConditioningInput):
    models = client_models(arch)
    style = Style(Path("default.json"))
    style.checkpoints = [checkpoints[arch]]
    style.cfg_scale = 3.0
    files = FileLibrary(FileCollection(), FileCollection())
    perf = PerformanceSettings()
    job = workflow.prepare(kind, canvas, cond, style, 1, models, files, perf, strength=1.0)
    w = workflow.create(job, models)
    return {id: node for id, node in w.root.items()}


def find(nodes: dict, class_type: str):
    return [n for n in nodes.values() if n["class_type"] == class_type]


def test_moss_generate():
    nodes = create(Arch.moss, WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("cat"))
    assert find(nodes, "CLIPLoader")[0]["inputs"]["type"] == "moss_image"
    assert len(find(nodes, "EmptyFlux2LatentImage")) == 1
    assert len(find(nodes, "CLIPTextEncode")) >= 1


def test_fibo2_generate():
    nodes = create(Arch.fibo2, WorkflowKind.generate, Extent(1024, 768), ConditioningInput("cat"))
    assert find(nodes, "CLIPLoader")[0]["inputs"]["type"] == "fibo2"
    assert len(find(nodes, "EmptyFlux2LatentImage")) == 1
    encode = find(nodes, "TextEncodeFibo2")
    assert len(encode) == 1
    assert encode[0]["inputs"]["prompt"] == "cat"
    assert "vae" not in encode[0]["inputs"]
    assert not any(k.startswith("images.") for k in encode[0]["inputs"])
    assert len(find(nodes, "ReferenceLatent")) == 0


def test_fibo2_edit():
    cond = ConditioningInput("make the cat orange")
    cond.edit_reference = True
    image = Image.create(Extent(768, 512))
    nodes = create(Arch.fibo2_e, WorkflowKind.refine, image, cond)
    encode = find(nodes, "TextEncodeFibo2")
    assert len(encode) == 1
    assert "images.image_1" in encode[0]["inputs"]  # text encoder sees the input image
    # reference latent of the input image for positive and negative
    assert len(find(nodes, "ReferenceLatent")) == 2
