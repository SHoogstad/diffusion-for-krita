from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import ConditioningInput, ControlInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientModels, resolve_arch
from ai_diffusion.backend.comfy_client import ComfyClient
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.resolution import prepare_diffusion_input
from ai_diffusion.backend.resources import (
    ControlMode,
    ResourceKind,
    UpscalerName,
    is_minimax_h3_video,
    resource_id,
)
from ai_diffusion.files import FileCollection, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.model import prompt_templates
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, SamplerPresets, Style

checkpoint = "minimax_h3_image_pruned_int8_convrot.safetensors"
text_encoder = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
vae = "minimax_h3_image_vae_fp16.safetensors"
h3_nodes = Arch.h3.comfy_nodes


def test_arch_from_string():
    # Model inspection may report the image model as "unknown" or as MiniMax-H3
    assert Arch.from_string("unknown", filename=checkpoint) is Arch.h3
    assert Arch.from_string("minimax-h3", filename=checkpoint) is Arch.h3
    assert Arch.from_string("unknown", filename="MiniMax-H3-Image_bf16.safetensors") is Arch.h3
    assert Arch.from_string("minimax_h3_image", filename="whatever.safetensors") is Arch.h3
    # P-H3-Image and the video models stay P-H3-Image
    assert Arch.from_string("unknown", filename="p-h3-image_int8_convrot.safetensors") is Arch.ph3
    video = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert Arch.from_string("unknown", filename=video) is Arch.ph3
    assert is_minimax_h3_video(video) and not is_minimax_h3_video(checkpoint)
    assert not is_minimax_h3_video("p-h3-image_int8_convrot.safetensors")
    assert Arch.h3.supports_edit and not Arch.h3.is_edit
    assert Arch.h3.clip_loader_type == "minimax" and Arch.h3.comfy_clip_type == "minimax"
    assert Arch.h3.text_encoders == ["qwen_3vl_32b_h3"]
    assert "h3_e" not in Arch.__members__


def node_info(nodes: list[str]):
    info: dict[str, dict] = {n: {"input": {"required": {}}} for n in nodes}
    info["CLIPLoader"] = {"input": {"required": {"type": [["minimax", "flux2"]]}}}
    return ComfyObjectInfo(info)


def client_models(nodes: list[str] = h3_nodes):
    models = ClientModels()
    models.checkpoints = {
        checkpoint: CheckpointInfo(checkpoint, Arch.h3, workflow.FileFormat.diffusion),
    }
    models.node_inputs = node_info(nodes)
    te_id = resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3vl_32b_h3")
    models.resources[te_id] = text_encoder
    models.resources[resource_id(ResourceKind.vae, Arch.h3, "default")] = vae
    for upscaler in UpscalerName:
        models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = upscaler.value
    return models


def test_workload():
    def missing_for_arch(models: ClientModels):
        client = SimpleNamespace(models=models)
        missing = ComfyClient._check_workload(client, Arch.h3)  # type: ignore
        return [m for m in missing if m.arch is Arch.h3]

    assert missing_for_arch(client_models()) == []
    missing = missing_for_arch(client_models(nodes=[]))  # ComfyUI older than 0.45
    assert {m.identifier for m in missing} == {"MiniMaxH3ImageConditioning"}


def create(
    kind: WorkflowKind,
    image: Image | Extent,
    cond: ConditioningInput,
    batch=1,
    mask: Mask | None = None,
    edit=True,
):
    models = client_models()
    style = Style(Path("default.json"))
    style.checkpoints = [checkpoint]
    style.sampler = "MiniMax-H3 - 6 steps"
    style.sampler_steps = 6
    style.cfg_scale = 1.0
    files = FileLibrary(FileCollection(), FileCollection())
    perf = PerformanceSettings()
    cond.edit_reference = edit
    inpaint = None
    if mask is not None:
        inpaint = workflow.detect_inpaint(workflow.InpaintMode.fill, mask.bounds, Arch.h3, cond, 1)
    job = workflow.prepare(
        kind, image, cond, style, 7, models, files, perf, mask=mask, strength=1.0, inpaint=inpaint
    )
    job.batch_count = batch
    w = workflow.create(job, models)
    return {id: node for id, node in w.root.items()}


def find(nodes: dict, class_type: str):
    return [n for n in nodes.values() if n["class_type"] == class_type]


def check_sampling(nodes: dict):
    clip = find(nodes, "CLIPLoader")[0]["inputs"]
    assert clip["type"] == "minimax" and clip["clip_name"] == text_encoder
    assert find(nodes, "UNETLoader")[0]["inputs"]["unet_name"] == checkpoint
    assert find(nodes, "VAELoader")[0]["inputs"]["vae_name"] == vae
    # the model's sampling walks its distilled sigma grid: no shift node, stock 6-step schedule
    assert len(find(nodes, "MiniMaxH3SigmaShift")) == 0
    assert len(find(nodes, "ManualSigmas")) == 0
    scheduler = find(nodes, "BasicScheduler")[0]["inputs"]
    assert scheduler["scheduler"] == "simple" and scheduler["steps"] == 6
    assert scheduler["denoise"] == pytest.approx(1.0)
    assert find(nodes, "KSamplerSelect")[0]["inputs"]["sampler_name"] == "euler"
    assert len(find(nodes, "CFGGuider")) == 0 and len(find(nodes, "BasicGuider")) == 1
    assert len(find(nodes, "EmptyLatentImage")) == 0  # the latent comes from the conditioning
    assert len(find(nodes, "PH3ImageConditioning")) == 0


def test_text_to_image():
    nodes = create(WorkflowKind.generate, Extent(2048, 2048), ConditioningInput("a red fox"))
    check_sampling(nodes)
    cond = find(nodes, "MiniMaxH3ImageConditioning")[0]["inputs"]
    assert cond["prompt"] == "a red fox"
    assert cond["width"] == 2048 and cond["height"] == 2048
    assert not any(k.startswith("images.") for k in cond)


def test_text_to_image_small_canvas_keeps_its_size():
    nodes = create(WorkflowKind.generate, Extent(1024, 768), ConditioningInput("a fox"))
    cond = find(nodes, "MiniMaxH3ImageConditioning")[0]["inputs"]
    assert (cond["width"], cond["height"]) == (1024, 768)
    assert cond["width"] % 32 == 0 and cond["height"] % 32 == 0


def test_text_to_image_batch():
    nodes = create(WorkflowKind.generate, Extent(2048, 2048), ConditioningInput("a fox"), batch=2)
    seeds = {n["inputs"]["noise_seed"] for n in find(nodes, "RandomNoise")}
    assert len(seeds) == 2
    assert len(find(nodes, "MiniMaxH3ImageConditioning")) == 1
    assert len(find(nodes, "VAEDecode")) == 2 and len(find(nodes, "ImageBatch")) == 1


def test_edit_with_reference():
    image = Image.create(Extent(2048, 1536))
    cond = ConditioningInput("put the hat from <Picture 2> on the person in <Picture 1>")
    cond.control = [ControlInput(ControlMode.reference, Image.create(Extent(512, 512)))]
    nodes = create(WorkflowKind.refine, image, cond)
    check_sampling(nodes)
    inputs = find(nodes, "MiniMaxH3ImageConditioning")[0]["inputs"]
    assert "images.image_1" in inputs and "images.image_2" in inputs
    assert inputs["width"] % 32 == 0 and inputs["height"] % 32 == 0
    assert inputs["prompt"] == "put the hat from <Picture 2> on the person in <Picture 1>"
    assert len(find(nodes, "VAEEncode")) == 0  # the node encodes the pictures itself


def test_refine_without_edit_mode_still_edits():
    image = Image.create(Extent(2048, 2048))
    nodes = create(WorkflowKind.refine, image, ConditioningInput("make it night"), edit=False)
    assert "images.image_1" in find(nodes, "MiniMaxH3ImageConditioning")[0]["inputs"]


def test_selection_edit():
    mask = Mask.rectangle(Bounds(512, 512, 1024, 1024), Bounds(256, 256, 1536, 1536))
    image = Image.create(Extent(1536, 1536))
    for kind in (WorkflowKind.refine_region, WorkflowKind.inpaint):
        nodes = create(kind, image, ConditioningInput("a cat"), mask=mask)
        check_sampling(nodes)
        assert "images.image_1" in find(nodes, "MiniMaxH3ImageConditioning")[0]["inputs"]
        assert len(find(nodes, "DifferentialDiffusion")) == 0


def test_large_canvas_single_pass():
    style = Style(Path("default.json"))
    perf = PerformanceSettings()
    scaled, _, _ = prepare_diffusion_input(Extent(6144, 6144), None, Arch.h3, style, perf)
    assert scaled.initial == scaled.desired  # no second refine pass
    assert scaled.initial.pixel_count <= 2560**2
    assert scaled.initial.width % 32 == 0 and scaled.initial.height % 32 == 0
    assert scaled.target == Extent(6144, 6144)


def test_sampler_preset():
    preset = SamplerPresets.instance()["MiniMax-H3 - 6 steps"]
    assert preset.sampler == "euler" and preset.scheduler == "simple"
    assert preset.steps == 6 and preset.cfg == pytest.approx(1.0)
    assert "H3 - Turbo" not in SamplerPresets.instance().names()


def test_builtin_style():
    style = Style.load(Path(__file__).parent.parent / "ai_diffusion/styles/minimax-h3-image.json")
    assert style is not None
    assert style.sampler == "MiniMax-H3 - 6 steps"
    assert resolve_arch(style, client_models()) is Arch.h3


def test_download_workload():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import download_models as dm

    from ai_diffusion.backend.resources import ModelRequirements

    names = {m.name for m in dm.list_models(h3=True, backend=ModelRequirements.cuda)}
    assert {"Qwen3-VL 32B MiniMax H3 [nvfp4]", "Image VAE (MiniMax-H3-Image)"} <= names
    assert "MiniMax-H3-Image pruned [int8]" not in names
    with_cp = dm.list_models(
        h3=True, checkpoints=["minimax_h3_image"], backend=ModelRequirements.cuda
    )
    assert "MiniMax-H3-Image pruned [int8]" in {m.name for m in with_cp}
    everything = dm.list_models(all=True, backend=ModelRequirements.cuda)
    assert not any(m.arch is Arch.h3 for m in everything)  # very large, opt-in only


def test_improve_prompt_templates():
    text, source = prompt_templates.builtin(Arch.h3, False)
    assert source == "official" and "{prompt}" in text and "{input text}" not in text
    request = prompt_templates.build(Arch.h3, False, "a fox in the snow")
    assert "input_text: a fox in the snow" in request.user
    assert "expanded_caption" in request.system
    answer = '{"expanded_caption": "A red fox \\"Vulpes\\" in deep snow.", "aspect_ratio": "4:3"}'
    assert prompt_templates.parse(answer, request.options) == 'A red fox "Vulpes" in deep snow.'
    assert request.options.thinking and request.options.max_tokens == 8192

    text, source = prompt_templates.builtin(Arch.h3, True)
    assert source == "builtin" and "<Picture 1>" in text
    assert prompt_templates.key(Arch.h3, True) == "h3-edit"
