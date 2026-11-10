from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import ConditioningInput, ControlInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_client import ComfyClient
from ai_diffusion.backend.comfy_workflow import PH3_SIGMAS, ComfyObjectInfo
from ai_diffusion.backend.resolution import prepare_diffusion_input
from ai_diffusion.backend.resources import ControlMode, ResourceKind, UpscalerName, resource_id
from ai_diffusion.files import FileCollection, FileLibrary
from ai_diffusion.image import Bounds, Extent, Image, Mask
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, SamplerPresets, Style

checkpoint = "p-h3-image_int8_convrot.safetensors"
text_encoder = "p-h3-image_text_encoder_int8_convrot.safetensors"
vae = "p-h3-image_vae_int8_convrot.safetensors"
upsampler = "p-h3-image_prompt_upsampler_int8_convrot.safetensors"
ph3_lora = "p-h3-image_lora_bf16.safetensors"
h3_checkpoint = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
ph3_nodes = Arch.ph3.comfy_nodes


def test_arch_from_string():
    # Model inspection reports MiniMax-H3 models as "unknown"
    assert Arch.from_string("unknown", filename=checkpoint) is Arch.ph3
    assert Arch.from_string("unknown", filename="p-h3-image_w4a8.safetensors") is Arch.ph3
    assert Arch.from_string("minimax-h3", filename="P-H3-Image_bf16.safetensors") is Arch.ph3
    # MiniMax-H3 video models only run P-H3-Image as a LoRA
    assert Arch.from_string("unknown", filename=h3_checkpoint) is Arch.ph3
    assert Arch.from_string("minimax-h3", filename="MiniMax_H3_Ref2VA.safetensors") is Arch.ph3
    assert Arch.from_string("unknown", filename="DasiwaMinimaxH3_turbo.safetensors") is Arch.ph3
    assert Arch.ph3.supports_edit and not Arch.ph3.is_edit
    assert Arch.ph3.clip_loader_type == "minimax"


def node_info(nodes: list[str]):
    info: dict[str, dict] = {n: {"input": {"required": {}}} for n in nodes}
    info["CLIPLoader"] = {"input": {"required": {"type": [["minimax", "flux2"]]}}}
    return ComfyObjectInfo(info)


def client_models(
    nodes: list[str] = ph3_nodes, with_upsampler=False, with_lora=False, with_ph3=True
):
    models = ClientModels()
    models.checkpoints = {
        checkpoint: CheckpointInfo(checkpoint, Arch.ph3, workflow.FileFormat.diffusion),
        h3_checkpoint: CheckpointInfo(h3_checkpoint, Arch.ph3, workflow.FileFormat.diffusion),
    }
    if not with_ph3:
        del models.checkpoints[checkpoint]
    if with_upsampler:
        models.resources[resource_id(ResourceKind.text_encoder, Arch.all, "p_h3_upsampler")] = (
            upsampler
        )
    if with_lora:
        models.resources[resource_id(ResourceKind.lora, Arch.ph3, "p_h3_image")] = ph3_lora
        models.loras = [ph3_lora]
    models.node_inputs = node_info(nodes)
    te_id = resource_id(ResourceKind.text_encoder, Arch.all, "p_h3_image")
    models.resources[te_id] = text_encoder
    models.resources[resource_id(ResourceKind.vae, Arch.ph3, "default")] = vae
    for upscaler in UpscalerName:
        models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = upscaler.value
    return models


def test_workload_requires_ph3_nodes():
    def missing_for_arch(models: ClientModels):
        client = SimpleNamespace(models=models)
        missing = ComfyClient._check_workload(client, Arch.ph3)  # type: ignore
        return [m for m in missing if m.arch is Arch.ph3]

    assert missing_for_arch(client_models()) == []
    missing = missing_for_arch(client_models(nodes=["MiniMaxH3SigmaShift"]))
    assert {m.identifier for m in missing} == {"PH3ImageConditioning"}


def test_workload_video_checkpoint_needs_lora():
    def missing_checkpoint(models: ClientModels):
        client = SimpleNamespace(models=models)
        missing = ComfyClient._check_workload(client, Arch.ph3)  # type: ignore
        return any(m.kind is ResourceKind.checkpoint for m in missing)

    assert missing_checkpoint(client_models(with_ph3=False))
    assert not missing_checkpoint(client_models(with_ph3=False, with_lora=True))
    assert not missing_checkpoint(client_models())


def create(
    kind: WorkflowKind,
    image: Image | Extent,
    cond: ConditioningInput,
    batch=1,
    mask: Mask | None = None,
    edit=True,
    models: ClientModels | None = None,
    style_checkpoint=checkpoint,
    use_upsampler=True,
):
    models = models or client_models()
    style = Style(Path("default.json"))
    style.checkpoints = [style_checkpoint]
    style.architecture = Arch.ph3
    style.prompt_upsampler = use_upsampler
    style.sampler = "P-H3 - 8 steps"
    style.sampler_steps = 8
    style.cfg_scale = 1.0
    files = FileLibrary(FileCollection(), FileCollection())
    perf = PerformanceSettings()
    cond.edit_reference = edit
    inpaint = None
    if mask is not None:
        inpaint = workflow.detect_inpaint(workflow.InpaintMode.fill, mask.bounds, Arch.ph3, cond, 1)
    job = workflow.prepare(
        kind, image, cond, style, 7, models, files, perf, mask=mask, strength=1.0, inpaint=inpaint
    )
    job.batch_count = batch
    w = workflow.create(job, models)
    return {id: node for id, node in w.root.items()}


def find(nodes: dict, class_type: str):
    return [n for n in nodes.values() if n["class_type"] == class_type]


def check_sampling(nodes: dict):
    assert find(nodes, "CLIPLoader")[0]["inputs"]["type"] == "minimax"
    assert find(nodes, "CLIPLoader")[0]["inputs"]["clip_name"] == text_encoder
    assert find(nodes, "UNETLoader")[0]["inputs"]["unet_name"] == checkpoint
    shift = find(nodes, "MiniMaxH3SigmaShift")[0]["inputs"]
    assert shift["shift_video"] == pytest.approx(1.0)
    assert shift["shift_audio"] == pytest.approx(1.0)
    assert find(nodes, "ManualSigmas")[0]["inputs"]["sigmas"] == PH3_SIGMAS
    assert find(nodes, "KSamplerSelect")[0]["inputs"]["sampler_name"] == "euler"
    assert len(find(nodes, "CFGGuider")) == 0 and len(find(nodes, "BasicGuider")) == 1
    assert len(find(nodes, "EmptyLatentImage")) == 0  # the latent comes from the conditioning


def test_text_to_image():
    nodes = create(WorkflowKind.generate, Extent(1536, 1536), ConditioningInput("a red fox"))
    check_sampling(nodes)
    cond = find(nodes, "PH3ImageConditioning")[0]["inputs"]
    assert cond["prompt"] == "a red fox"
    assert cond["width"] == 1536 and cond["height"] == 1536
    assert not any(k.startswith("images.") for k in cond)


def test_text_to_image_batch():
    nodes = create(WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("a fox"), batch=2)
    seeds = {n["inputs"]["noise_seed"] for n in find(nodes, "RandomNoise")}
    assert len(seeds) == 2
    assert len(find(nodes, "PH3ImageConditioning")) == 1
    assert len(find(nodes, "VAEDecode")) == 2 and len(find(nodes, "ImageBatch")) == 1


def test_edit_with_reference():
    image = Image.create(Extent(1024, 768))
    cond = ConditioningInput("put the hat from <Picture 2> on the person")
    cond.control = [ControlInput(ControlMode.reference, Image.create(Extent(512, 512)))]
    nodes = create(WorkflowKind.refine, image, cond)
    check_sampling(nodes)
    inputs = find(nodes, "PH3ImageConditioning")[0]["inputs"]
    assert "images.image_1" in inputs and "images.image_2" in inputs
    assert inputs["width"] % 32 == 0 and inputs["height"] % 32 == 0
    assert inputs["prompt"].endswith("put the hat from <Picture 2> on the person")


def test_refine_without_edit_mode_still_edits():
    image = Image.create(Extent(1024, 1024))
    nodes = create(WorkflowKind.refine, image, ConditioningInput("make it night"), edit=False)
    assert "images.image_1" in find(nodes, "PH3ImageConditioning")[0]["inputs"]
    assert len(find(nodes, "VAEEncode")) == 0


def test_selection_edit():
    mask = Mask.rectangle(Bounds(256, 256, 512, 512), Bounds(128, 128, 768, 768))
    image = Image.create(Extent(768, 768))
    for kind in (WorkflowKind.refine_region, WorkflowKind.inpaint):
        nodes = create(kind, image, ConditioningInput("a cat"), mask=mask)
        check_sampling(nodes)
        assert "images.image_1" in find(nodes, "PH3ImageConditioning")[0]["inputs"]
        assert len(find(nodes, "DifferentialDiffusion")) == 0


def test_large_canvas_single_pass():
    style = Style(Path("default.json"))
    perf = PerformanceSettings()
    scaled, _, _ = prepare_diffusion_input(Extent(4096, 4096), None, Arch.ph3, style, perf)
    assert scaled.initial == scaled.desired  # no second refine pass
    assert scaled.initial.pixel_count <= 2048**2
    assert scaled.initial.width % 32 == 0 and scaled.initial.height % 32 == 0
    assert scaled.target == Extent(4096, 4096)


def test_sampler_preset():
    preset = SamplerPresets.instance()["P-H3 - 8 steps"]
    assert preset.scheduler == "p_h3" and preset.steps == 8 and preset.cfg == pytest.approx(1.0)
    assert PH3_SIGMAS.count(",") == 8  # 8 steps


def test_download_workload():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import download_models as dm

    from ai_diffusion.backend.resources import ModelRequirements

    names = {m.name for m in dm.list_models(ph3=True, backend=ModelRequirements.cuda)}
    assert {"P-H3-Image Text Encoder [int8]", "P-H3-Image VAE [int8]"} <= names
    assert "P-H3-Image [int8]" not in names
    everything = dm.list_models(all=True, backend=ModelRequirements.cuda)
    assert "P-H3-Image [int8]" not in {m.name for m in everything}


@pytest.mark.parametrize("arch", [Arch.ph3])
def test_text_encoder(arch: Arch):
    assert arch.text_encoders == ["p_h3_image"]


def test_text_encoder_found_on_server():
    from ai_diffusion.backend.comfy_client import _find_text_encoder_models

    files = [text_encoder, "qwen3vl_8b_int8_convrot.safetensors", "t5xxl_fp16.safetensors"]
    found = _find_text_encoder_models(files)
    assert found[resource_id(ResourceKind.text_encoder, Arch.all, "p_h3_image")] == text_encoder
    # every architecture's text encoders are searched for
    for arch in Arch.list():
        for te in arch.text_encoders:
            assert resource_id(ResourceKind.text_encoder, Arch.all, te) in found


def test_prompt_upsampler_text_to_image():
    models = client_models(with_upsampler=True)
    nodes = create(
        WorkflowKind.generate, Extent(1536, 864), ConditioningInput("a fox"), models=models
    )
    loaders = {n["inputs"]["clip_name"]: n["inputs"]["type"] for n in find(nodes, "CLIPLoader")}
    assert loaders == {text_encoder: "minimax", upsampler: "minimax"}
    fmt = find(nodes, "StringFormat")[0]["inputs"]
    assert fmt["f_string"] == "Aspect ratio: 16:9\nRequest: {a}" and fmt["values.a"] == "a fox"
    gen = find(nodes, "TextGenerate")[0]["inputs"]
    assert gen["thinking"] is True and gen["sampling_mode"] == "off" and gen["max_length"] == 2048
    assert "image" not in gen
    cond = find(nodes, "PH3ImageConditioning")[0]["inputs"]
    assert nodes[cond["prompt"][0]]["class_type"] == "TextGenerate" and cond["prompt"][1] == 0


def test_prompt_upsampler_edit():
    models = client_models(with_upsampler=True)
    cond = ConditioningInput("put the hat from <Picture 2> on her")
    cond.control = [ControlInput(ControlMode.reference, Image.create(Extent(512, 512)))]
    image = Image.create(Extent(1024, 768))
    nodes = create(WorkflowKind.refine, image, cond, models=models)
    fmt = find(nodes, "StringFormat")[0]["inputs"]["f_string"]
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    assert fmt == (
        f"<|im_start|>user\n<Picture 1>: {vision}<Picture 2>: {vision}Edit <Picture 1>. {{a}}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    gen = find(nodes, "TextGenerate")[0]["inputs"]
    assert gen["thinking"] is False and gen["max_length"] == 3072
    assert nodes[gen["image"][0]]["class_type"] == "BatchImagesNode"
    assert len(find(nodes, "ResizeAndPadImage")) == 1
    scale = find(nodes, "ImageScaleToTotalPixels")[0]["inputs"]
    assert scale["megapixels"] == 1 and scale["resolution_steps"] == 1


def test_prompt_upsampler_off_or_missing():
    on, off = client_models(with_upsampler=True), client_models(with_upsampler=True)
    nodes = create(WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("a fox"), models=on)
    assert len(find(nodes, "TextGenerate")) == 1
    args = (WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("a fox"))
    assert len(find(create(*args, models=off, use_upsampler=False), "TextGenerate")) == 0
    assert len(find(create(*args), "TextGenerate")) == 0  # upsampler file not on the server


def test_h3_checkpoint_with_lora():
    models = client_models(with_lora=True)
    args = (WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("a fox"))
    nodes = create(*args, models=models, style_checkpoint=h3_checkpoint)
    assert find(nodes, "UNETLoader")[0]["inputs"]["unet_name"] == h3_checkpoint
    lora = find(nodes, "LoraLoaderBypassModelOnly")[0]["inputs"]
    assert lora["lora_name"] == ph3_lora and lora["strength_model"] == pytest.approx(1.0)
    assert nodes[lora["model"][0]]["class_type"] == "UNETLoader"
    shift = find(nodes, "MiniMaxH3SigmaShift")[0]["inputs"]
    assert nodes[shift["model"][0]]["class_type"] == "LoraLoaderBypassModelOnly"
    # P-H3-Image text encoder and VAE, not the MiniMax-H3 ones
    assert find(nodes, "CLIPLoader")[0]["inputs"]["clip_name"] == text_encoder
    assert find(nodes, "VAELoader")[0]["inputs"]["vae_name"] == vae
    # the real P-H3-Image model needs no LoRA
    nodes = create(*args, models=models)
    assert len(find(nodes, "LoraLoaderBypassModelOnly")) == 0


def test_h3_checkpoint_without_lora():
    args = (WorkflowKind.generate, Extent(1024, 1024), ConditioningInput("a fox"))
    with pytest.raises(ValueError, match="p-h3-image_lora"):
        create(*args, style_checkpoint=h3_checkpoint)


def test_video_checkpoint_resolves_to_ph3():
    from ai_diffusion.backend.client import resolve_arch

    style = Style(Path("default.json"))
    style.checkpoints = [h3_checkpoint]
    assert resolve_arch(style, client_models()) is Arch.ph3
    assert not Arch.is_compatible(Arch.h3, Arch.ph3)
    assert not Arch.is_compatible(Arch.ph3, Arch.qwen2)


def test_aspect_ratio_label():
    assert workflow.ph3_aspect_ratio(Extent(2048, 2048)) == "1:1"
    assert workflow.ph3_aspect_ratio(Extent(2720, 1536)) == "16:9"
    assert workflow.ph3_aspect_ratio(Extent(1664, 2496)) == "2:3"
    assert workflow.ph3_aspect_ratio(Extent(1000, 800)) == "4:3"
