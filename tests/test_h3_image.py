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
from ai_diffusion.model.control import ControlLayer
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


def reference_extent(arch: Arch, reference: Extent):
    layer = SimpleNamespace(
        name="reference",
        bounds=Bounds(0, 0, *reference),
        get_pixels=lambda bounds, time: Image.create(reference),
    )
    control = SimpleNamespace(
        layer=layer,
        is_supported=True,
        mode=ControlMode.reference,
        strength=100,
        strength_multiplier=100,
        start=0.0,
        end=1.0,
        clip_vision_extent=Extent(224, 224),
        _model=SimpleNamespace(arch=arch, document=SimpleNamespace(extent=Extent(2000, 1088))),
    )
    return ControlLayer.to_api(control).image.extent  # type: ignore


@pytest.mark.parametrize("size", [(1707, 1280), (3840, 2160), (1080, 1080), (4096, 4096)])
def test_reference_images_at_4mp(size: tuple[int, int]):
    reference = Extent(*size)
    extent = reference_extent(Arch.h3, reference)
    assert abs(extent.pixel_count - 2048 * 2048) < 3 * 4096
    assert extent.width / extent.height == pytest.approx(reference.width / reference.height, 0.01)


def test_reference_images_other_edit_models():
    # Not MiniMax-H3: references are no taller than the canvas
    assert reference_extent(Arch.qwen2, Extent(1707, 1280)) == Extent(1450, 1088)


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


@pytest.mark.parametrize(
    "size", [(128, 160), (1000, 1000), (1448, 1448), (4096, 4096), (6144, 4096)]
)
def test_native_resolution(size):
    style = Style(Path("default.json"))
    perf = PerformanceSettings(max_pixel_count=0)  # no user limit
    extent = Extent(*size)
    image = Image.create(extent)
    scaled, out, _ = prepare_diffusion_input(extent, image, Arch.h3, style, perf)
    # generated at canvas size, no scaling down or up in the background
    assert scaled.input == scaled.initial == scaled.desired == scaled.target == extent
    assert out is image


def test_native_resolution_respects_max_pixel_count():
    style = Style(Path("default.json"))
    perf = PerformanceSettings(max_pixel_count=6)
    extent = Extent(4096, 4096)
    scaled, _, _ = prepare_diffusion_input(extent, Image.create(extent), Arch.h3, style, perf)
    assert scaled.initial == scaled.desired  # single pass, at the user's limit
    assert scaled.initial.pixel_count <= 6 * 10**6 * 1.01
    assert scaled.target == extent


@pytest.mark.parametrize("size", [(1024, 768), (1920, 1080), (1448, 1448)])
def test_preferred_resolution_is_a_minimum(size):
    style = Style(Path("default.json"))
    style.preferred_resolution = 2048
    perf = PerformanceSettings(max_pixel_count=0)
    extent = Extent(*size)
    image = Image.create(extent)
    scaled, out, _ = prepare_diffusion_input(extent, image, Arch.h3, style, perf)
    # generated at 4 MP, then scaled down to the canvas
    assert scaled.input == scaled.target == extent and out is image
    assert scaled.initial == scaled.desired
    assert abs(scaled.desired.pixel_count - 2048 * 2048) < 3 * 2048
    assert scaled.desired.width / scaled.desired.height == pytest.approx(size[0] / size[1], 0.01)


def test_preferred_resolution_keeps_larger_sizes():
    style = Style(Path("default.json"))
    style.preferred_resolution = 2048
    perf = PerformanceSettings(max_pixel_count=0)
    for extent in (Extent(2048, 2048), Extent(6144, 4096)):
        scaled, _, _ = prepare_diffusion_input(extent, Image.create(extent), Arch.h3, style, perf)
        assert scaled.input == scaled.initial == scaled.desired == scaled.target == extent

    perf = PerformanceSettings(max_pixel_count=2)  # the user's limit wins
    extent = Extent(1024, 768)
    scaled, _, _ = prepare_diffusion_input(extent, Image.create(extent), Arch.h3, style, perf)
    assert abs(scaled.desired.pixel_count - 2 * 10**6) < 3 * 2048


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
    # Default: the plugin's expert templates, written without thinking by a fast language model
    request = prompt_templates.build(Arch.h3, False, "a fox in the snow")
    assert request.source == "expert" and request.user == "a fox in the snow"
    assert prompt_templates.rules not in request.system
    assert not request.options.thinking and "qwen3.5_9b_w4a8" in request.options.llms

    assert prompt_templates.key(Arch.h3, True) == "h3-edit"
    edit, source = prompt_templates.builtin(Arch.h3, True, vision=True)
    assert source == "expert" and "<Subject 1>" in edit
    blind, source = prompt_templates.builtin(Arch.h3, True)  # vision is opt-in
    assert source == "expert" and blind != edit
    assert prompt_templates.options(Arch.h3, True).vision


def test_improve_prompt_official_templates():
    # MiniMax's own enhancers, with the settings of ComfyUI's templates
    assert prompt_templates.has_official(Arch.h3, False)
    assert prompt_templates.has_official(Arch.h3, True)
    assert not prompt_templates.has_official(Arch.ph3, False)

    text, source = prompt_templates.builtin(Arch.h3, False, official=True)
    assert source == "official" and "{prompt}" in text and "{input text}" not in text
    extent = Extent(1024, 768)
    request = prompt_templates.build(Arch.h3, False, "a fox in the snow", extent, official=True)
    assert "input_text: a fox in the snow\nFinal canvas: 1024x768 pixels." in request.user
    assert "expanded_caption" in request.system
    answer = '{"expanded_caption": "A red fox \\"Vulpes\\" in deep snow.", "aspect_ratio": "4:3"}'
    assert prompt_templates.parse(answer, request.options) == 'A red fox "Vulpes" in deep snow.'
    assert request.options.thinking and request.options.max_tokens == 8192

    request = prompt_templates.build(
        Arch.h3, True, 'a "red" hat', Extent(1707, 1280), official=True
    )
    assert request.source == "official" and request.options.vision
    assert 'User Editing Instruction: "a \\"red\\" hat"\nUser Aspect Ratio Selection: 4:3' in (
        request.user
    )
