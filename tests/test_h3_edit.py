from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import ConditioningInput, ControlInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_client import ComfyClient
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.resolution import prepare_diffusion_input
from ai_diffusion.backend.resources import ControlMode, ResourceKind, UpscalerName, resource_id
from ai_diffusion.files import FileCollection, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, SamplerPresets, Style

checkpoint = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
text_encoder = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
vae = "minimax_h3_video_vae_fp16.safetensors"
turbo_lora = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
h3_nodes = Arch.h3_e.comfy_nodes


def test_arch_from_string():
    # Model inspection reports H3 as "unknown"
    assert Arch.from_string("unknown", filename=checkpoint) is Arch.h3_e
    assert Arch.from_string("unknown", filename="DasiwaMinimaxH3_turbo.safetensors") is Arch.h3_e
    assert Arch.from_string("unknown", filename="MiniMax_H3_Ref2VA.safetensors") is Arch.h3_e
    assert Arch.from_string("unknown", filename="flux1-dev.safetensors") is None
    assert Arch.h3_e.is_edit
    assert "experimental" in Arch.h3_e.value


def node_info(nodes: list[str], still_decode=False):
    info: dict[str, dict] = {n: {"input": {"required": {}}} for n in nodes}
    info["CLIPLoader"] = {"input": {"required": {"type": [["minimax", "flux2"]]}}}
    if still_decode:
        info["H3StillSliceDecode"] = {"input": {"required": {}}}
    return ComfyObjectInfo(info)


def client_models(nodes: list[str] = h3_nodes, still_decode=False, lora=True):
    models = ClientModels()
    models.checkpoints = {
        checkpoint: CheckpointInfo(checkpoint, Arch.h3_e, workflow.FileFormat.diffusion)
    }
    models.node_inputs = node_info(nodes, still_decode)
    te_id = resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3vl_32b_h3")
    models.resources[te_id] = text_encoder
    models.resources[resource_id(ResourceKind.vae, Arch.h3_e, "default")] = vae
    if lora:
        models.resources[resource_id(ResourceKind.lora, Arch.h3_e, "turbo")] = turbo_lora
        models.loras = [turbo_lora]
    for upscaler in UpscalerName:
        models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = upscaler.value
    return models


def test_workload_requires_h3_nodes():
    def missing_for_arch(models: ClientModels):
        client = SimpleNamespace(models=models)
        missing = ComfyClient._check_workload(client, Arch.h3_e)  # type: ignore
        return [m for m in missing if m.arch is Arch.h3_e]

    assert missing_for_arch(client_models()) == []
    missing = missing_for_arch(client_models(nodes=["H3StudioEdit"]))
    assert {m.identifier for m in missing} == set(h3_nodes) - {"H3StudioEdit"}
    assert all(m.kind is ResourceKind.node for m in missing)


def create(
    kind: WorkflowKind,
    image: Image,
    cond: ConditioningInput,
    models: ClientModels | None = None,
    sampler="H3 - Turbo",
    batch=1,
    mask: Mask | None = None,
):
    models = models or client_models()
    style = Style(Path("default.json"))
    style.checkpoints = [checkpoint]
    style.sampler = sampler
    style.sampler_steps = 20
    style.cfg_scale = 1.0
    files = FileLibrary(FileCollection(), FileCollection())
    perf = PerformanceSettings()
    cond.edit_reference = True
    inpaint = None
    if mask is not None:
        inpaint = workflow.detect_inpaint(
            workflow.InpaintMode.fill, mask.bounds, Arch.h3_e, cond, 1.0
        )
    job = workflow.prepare(
        kind, image, cond, style, 7, models, files, perf, mask=mask, strength=1.0, inpaint=inpaint
    )
    job.batch_count = batch
    w = workflow.create(job, models)
    return {id: node for id, node in w.root.items()}


def find(nodes: dict, class_type: str):
    return [n for n in nodes.values() if n["class_type"] == class_type]


def test_keep_source_edit():
    image = Image.create(Extent(1024, 768))
    nodes = create(WorkflowKind.refine, image, ConditioningInput("make the jacket red"))
    assert find(nodes, "CLIPLoader")[0]["inputs"]["type"] == "minimax"
    assert find(nodes, "UNETLoader")[0]["inputs"]["unet_name"] == checkpoint
    prepare = find(nodes, "H3ImageToImagePrepare")
    assert len(prepare) == 1
    inputs = prepare[0]["inputs"]
    assert inputs["edit_instruction"] == "make the jacket red"
    assert inputs["width"] % 32 == 0 and inputs["height"] % 32 == 0
    sampling = find(nodes, "H3SamplingSettings")[0]["inputs"]
    assert sampling["sampler_name"] == "er_sde" and sampling["steps"] == 20
    # Turbo sampler preset loads Larry's v4 EMA LoRA at 0.38
    lora = find(nodes, "LoraLoader")[0]["inputs"]
    assert lora["lora_name"] == turbo_lora and lora["strength_model"] == pytest.approx(0.38)
    # Without H3StillSliceDecode on the server the stock decode + frame selector is used
    assert len(find(nodes, "H3ImageDecode")) == 1
    assert len(find(nodes, "H3StudioEdit")) == 0


def test_keep_source_still_decode_and_batch():
    image = Image.create(Extent(1024, 1024))
    models = client_models(still_decode=True)
    cond = ConditioningInput("add a hat")
    nodes = create(WorkflowKind.refine, image, cond, models, batch=2)
    decodes = find(nodes, "H3StillSliceDecode")
    assert len(decodes) == 2 and decodes[0]["inputs"]["slice"] == -1
    seeds = {n["inputs"]["noise_seed"] for n in find(nodes, "RandomNoise")}
    assert len(seeds) == 2
    assert len(find(nodes, "ImageBatch")) == 1
    assert len(find(nodes, "H3ImageDecode")) == 0


def test_creative_edit_with_references():
    image = Image.create(Extent(1024, 768))
    cond = ConditioningInput("put the outfit from <Picture 2> on the person")
    cond.control = [ControlInput(ControlMode.reference, Image.create(Extent(512, 512)))]
    nodes = create(WorkflowKind.refine, image, cond)
    assert len(find(nodes, "H3ImageToImagePrepare")) == 0
    models = find(nodes, "H3StudioModels")[0]["inputs"]
    assert models["diffusion_model"] == checkpoint
    assert models["text_encoder"] == text_encoder and models["video_vae"] == vae
    lora = find(nodes, "H3StudioLoRA")[0]["inputs"]
    assert lora["lora_name"] == turbo_lora and lora["strength_model"] == pytest.approx(0.38)
    edit = find(nodes, "H3StudioEdit")[0]["inputs"]
    assert edit["aspect_ratio"] == "custom pixels"
    assert "references.reference_image_2" in edit
    assert edit["instruction"].endswith("put the outfit from <Picture 2> on the person")
    render = find(nodes, "H3StudioRender")[0]["inputs"]
    assert render["steps"] == 20 and render["sampler_name"] == "er_sde"


def test_selection_edit():
    mask = Mask.rectangle(Bounds(256, 256, 512, 512), Bounds(128, 128, 768, 768))
    image = Image.create(Extent(768, 768))
    nodes = create(WorkflowKind.refine_region, image, ConditioningInput("a cat"), mask=mask)
    assert len(find(nodes, "H3ImageToImagePrepare")) == 1
    assert len(find(nodes, "KSampler")) == 0
    assert len(find(nodes, "DifferentialDiffusion")) == 0


def test_non_edit_workflows_are_rejected():
    with pytest.raises(ValueError, match="H3 Edit only edits images"):
        create(WorkflowKind.generate, Extent(512, 512), ConditioningInput(""))  # type: ignore


def test_large_canvas_generates_at_two_megapixels():
    style = Style(Path("default.json"))
    perf = PerformanceSettings()
    image = Image.create(Extent(3072, 3072))
    scaled, image, _ = prepare_diffusion_input(image.extent, image, Arch.h3_e, style, perf)
    assert scaled.initial.pixel_count <= 1472**2  # ~2 MP, rounded to multiples of 32
    assert scaled.initial.width % 32 == 0 and scaled.initial.height % 32 == 0
    assert scaled.target == Extent(3072, 3072)


def test_turbo_preset():
    preset = SamplerPresets.instance()["H3 - Turbo"]
    assert preset.lora == "turbo" and preset.lora_strength == pytest.approx(0.38)
    assert SamplerPresets.instance()["Flux - Turbo"].lora_strength == pytest.approx(1.0)


def test_download_workload():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import download_models as dm

    from ai_diffusion.backend.resources import ModelRequirements

    names = {m.name for m in dm.list_models(h3=True, backend=ModelRequirements.cuda)}
    assert {"Qwen3-VL 32B MiniMax H3 [nvfp4]", "Video VAE (MiniMax H3)"} <= names
    assert "Turbo LoRA v4 EMA (H3 Edit)" in names
    everything = dm.list_models(all=True, backend=ModelRequirements.cuda)
    assert not any(m.arch is Arch.h3_e for m in everything)  # experimental, opt-in only
