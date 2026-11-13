from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import CheckpointInput
from ai_diffusion.backend.client import CheckpointInfo, Client, ClientModels
from ai_diffusion.backend.comfy_client import text_generate_node
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo, ComfyWorkflow
from ai_diffusion.backend.resources import ResourceKind, resource_id
from ai_diffusion.files import FileFormat
from ai_diffusion.image import Extent, Image
from ai_diffusion.model import prompt_enhance, prompt_templates
from ai_diffusion.qt.QtCore import Qt
from ai_diffusion.settings import settings
from ai_diffusion.style import Arch, Style, StyleSettings
from ai_diffusion.util import ensure

single_encoder_archs = [
    a for a in Arch.list() if a.supports_text_encoder_override and len(a.text_encoders) == 1
]


def models_for(arch: Arch):
    models = ClientModels()
    models.checkpoints = {
        "model.safetensors": CheckpointInfo("model.safetensors", arch, FileFormat.diffusion)
    }
    for te in ["clip_l", "clip_g", "t5"] + [a.main_text_encoder for a in single_encoder_archs]:
        models.resources[resource_id(ResourceKind.text_encoder, Arch.all, te)] = f"{te}.safetensors"
    models.resources[resource_id(ResourceKind.vae, arch, "default")] = "vae.safetensors"
    return models


def text_encoder_nodes(arch: Arch, text_encoder: str):
    w = ComfyWorkflow()
    checkpoint = CheckpointInput("model.safetensors", arch, text_encoder=text_encoder)
    workflow.load_checkpoint_with_lora(w, checkpoint, models_for(arch))
    loaders = ("CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader")
    return [n["inputs"] for n in w.root.values() if n["class_type"] in loaders]


@pytest.mark.parametrize("arch", single_encoder_archs + [Arch.flux, Arch.chroma])
def test_text_encoder_default(arch: Arch):
    default = StyleSettings.text_encoder.default
    nodes = text_encoder_nodes(arch, default)
    assert len(nodes) == 1
    files = [v for k, v in nodes[0].items() if k.startswith("clip_name")]
    assert f"{arch.main_text_encoder}.safetensors" in files


@pytest.mark.parametrize("arch", single_encoder_archs + [Arch.flux, Arch.chroma])
def test_text_encoder_override(arch: Arch):
    nodes = text_encoder_nodes(arch, "my-abliterated-te.safetensors")
    files = [v for k, v in nodes[0].items() if k.startswith("clip_name")]
    assert "my-abliterated-te.safetensors" in files
    assert f"{arch.main_text_encoder}.safetensors" not in files
    if arch.clip_loader_type and len(arch.text_encoders) == 1:
        assert nodes[0]["type"] == arch.clip_loader_type
    if arch is Arch.flux:
        assert nodes[0]["clip_name1"] == "clip_l.safetensors"  # only T5 is replaced


@pytest.mark.parametrize("arch", [Arch.sd15, Arch.sdxl, Arch.illu])
def test_text_encoder_override_ignored(arch: Arch):
    assert not arch.supports_text_encoder_override
    nodes = text_encoder_nodes(arch, "my-te.safetensors")
    assert all("my-te.safetensors" not in n.values() for n in nodes)


def test_clean_output():
    clean = prompt_enhance.clean_output
    assert clean("<think>hmm</think>\nA red fox.") == "A red fox."
    assert clean('"A red fox in snow"') == "A red fox in snow"
    assert clean("Prompt: a red fox") == "a red fox"
    assert clean("```\na red fox\n```") == "a red fox"
    assert clean("Improved prompt:  a red fox") == "a red fox"


def test_system_prompt():
    settings.prompt_enhancer_instructions = ""
    assert "Danbooru" in prompt_enhance.system_prompt(Arch.illu)
    assert "separated by commas" in prompt_enhance.system_prompt(Arch.sdxl)
    assert "Output ONLY the prompt" in prompt_enhance.system_prompt(Arch.sdxl)
    assert "Rewritten" in prompt_enhance.system_prompt(Arch.qwen_e)  # official, JSON answer
    assert "prompt engineer" in prompt_enhance.system_prompt(Arch.moss)  # official
    settings.prompt_enhancer_instructions = "keep it short"
    assert prompt_enhance.system_prompt(Arch.moss).endswith("keep it short")
    settings.prompt_enhancer_instructions = ""


@pytest.mark.parametrize("arch", Arch.list())
def test_template_per_arch(arch: Arch):
    generate, _ = prompt_templates.builtin(arch, editing=False)
    edit, _ = prompt_templates.builtin(arch, editing=True)
    assert generate and edit
    if arch.is_edit or not arch.supports_edit:
        assert generate == edit  # only one mode
    else:
        assert generate != edit
    if placeholder := workflow._layer_replace.get(arch):
        if prompt_templates.builtin(arch, True)[1] == "builtin":
            assert placeholder.format(2) in edit


def test_template_official_files():
    for file in set(prompt_templates._official.values()):
        assert (prompt_templates.official_dir / file).read_text(encoding="utf-8").strip()
    assert prompt_templates.builtin(Arch.krea2, False) == (
        (prompt_templates.official_dir / "krea2.txt").read_text(encoding="utf-8").strip(),
        "official",
    )
    assert prompt_templates.builtin(Arch.qwen_e_p, False)[1] == "official"
    assert prompt_templates.builtin(Arch.illu, False)[1] == "builtin"


def test_template_names_model():
    assert "Illustrious" in prompt_templates.builtin(Arch.illu_v, False)[0]
    assert "klein" in prompt_templates.builtin(Arch.flux2_9b, False)[0]
    assert "Kontext" in prompt_templates.builtin(Arch.flux_k, False)[0]
    assert "Edit <Picture 1>" in prompt_templates.builtin(Arch.ph3, True)[0]
    assert "@" in prompt_templates.builtin(Arch.anima, False)[0]
    assert "Layered" in prompt_templates.builtin(Arch.qwen_l, False)[0]


def test_request_user_template():
    # Krea 2's official enhancer has the prompt inside: one user message, extras as system
    req = prompt_templates.build(Arch.krea2, False, "a fox", extra="be brief")
    assert req.source == "official" and req.options.thinking
    assert req.user.endswith("User's Input:\na fox")
    assert req.system == "be brief"


def test_request_system_template():
    req = prompt_templates.build(Arch.sdxl, False, "a fox")
    assert req.user == "a fox" and req.source == "builtin"
    assert req.system.endswith(prompt_templates.rules)
    req = prompt_templates.build(Arch.moss, False, "a fox")
    assert req.user == "a fox" and prompt_templates.rules not in req.system  # official as-is


def test_request_placeholders():
    req = prompt_templates.build(Arch.ernie, False, 'a "red" fox', Extent(1024, 768))
    assert req.user == '{"prompt": "a \\"red\\" fox", "width": 1024, "height": 768}'
    enhancer = ensure(req.options.enhancer)
    request = prompt_templates.fill(enhancer.user_format, "a fox", Extent(768, 1024))
    assert request == "a fox"
    ph3 = ensure(prompt_templates.options(Arch.ph3, False).enhancer)
    request = prompt_templates.fill(ph3.user_format, "a {fox}", Extent(1600, 900))
    assert request == "Aspect ratio: 16:9\nRequest: a {fox}"
    assert prompt_templates.options(Arch.ph3, True).enhancer is None  # edits: card template


def test_template_style_prompt():
    style = "{prompt}, masterpiece, best quality, absurdres"
    prompt = prompt_enhance.system_prompt(Arch.illu, style_prompt=style)
    assert "do not repeat them: masterpiece, best quality, absurdres" in prompt
    assert "do not repeat" not in prompt_enhance.system_prompt(Arch.illu, style_prompt="{prompt}")  # noqa: RUF027


def test_parse_answer():
    opts = prompt_templates.options(Arch.qwen_e, True)
    answer = '```json\n{"Rewritten": "Add a gray cat on the sofa"}\n```'
    clean = prompt_enhance.clean_output(answer)
    assert prompt_templates.parse(clean, opts) == "Add a gray cat on the sofa"
    assert prompt_templates.parse("not json", opts) == "not json"
    opts = prompt_templates.options(Arch.qwen, False)
    result = prompt_templates.parse("A red fox\nin snow", opts)
    assert result == "A red fox in snow. Ultra HD, 4K, cinematic composition"


def test_template_user_file(tmp_path):
    (tmp_path / "illu.txt").write_text("my booru template\n", encoding="utf-8")
    (tmp_path / "qwen2-edit.txt").write_text("my qwen2 edit template", encoding="utf-8")
    (tmp_path / "moss.txt").write_text("  ", encoding="utf-8")  # empty: ignored
    assert prompt_templates.get(Arch.illu, False, tmp_path) == ("my booru template", "custom")
    assert prompt_templates.get(Arch.qwen2, True, tmp_path)[0] == "my qwen2 edit template"
    assert prompt_templates.get(Arch.qwen2, False, tmp_path)[1] == "official"
    assert prompt_templates.get(Arch.moss, False, tmp_path)[1] == "official"
    assert prompt_templates.user_file(Arch.qwen_e, False, tmp_path).name == "qwen_e-edit.txt"
    req = prompt_templates.build(Arch.illu, False, "x", folder=tmp_path)
    assert req.system == f"my booru template\n{prompt_templates.rules}"


def test_parse_json_without_key():
    opts = prompt_templates.options(Arch.sdxl, False)
    assert prompt_templates.parse('{"expanded_caption": "a red fox"}', opts) == "a red fox"
    assert prompt_templates.parse("{not json", opts) == "{not json"


def test_ratio():
    assert prompt_templates.ratio(Extent(1707, 1280)) == "4:3"  # scaled, a pixel off
    assert prompt_templates.ratio(Extent(1920, 1080)) == "16:9"
    assert prompt_templates.ratio(Extent(1024, 1024)) == "1:1"


def picture_context(has_canvas=True, selection=""):
    pictures = [
        prompt_enhance.Picture("the canvas being edited"),
        prompt_enhance.Picture("a pose to match"),
        prompt_enhance.Picture("a reference picture", layer="knight"),
    ]
    return prompt_enhance.PromptContext(pictures, has_canvas, selection)


def test_picture_text():
    context = picture_context(selection="the left side of <Picture 1>, about 30% of it")
    text, layers = prompt_enhance._picture_text("put <layer:knight> here", context)
    assert layers == {3: "knight"}
    assert text.splitlines() == [
        (
            "<Picture 1> is the canvas being edited; the change happens in the selected area, "
            "the left side of <Picture 1>, about 30% of it."
        ),
        "<Picture 2> is a pose to match.",
        "<Picture 3> is a reference picture.",
        "Request: put <Picture 3> here",
    ]
    text, __ = prompt_enhance._picture_text("<layer:knight>", context, describe=False)
    assert text == "<Picture 3>"

    no_canvas = prompt_enhance.PromptContext(context.pictures[2:], has_canvas=False)
    text, layers = prompt_enhance._picture_text("<layer:knight> <layer:other>", no_canvas)
    assert text.startswith("There is no canvas")
    assert text.endswith("Request: <Picture 1> <layer:other>")
    assert layers == {1: "knight"}


def test_restore_layers():
    fixed = prompt_enhance._fix_tokens("<Picture 3> runs in Picture 1. Subject 2> waves")
    assert fixed == "<Picture 3> runs in <Picture 1>. <Subject 2> waves"
    restored = prompt_enhance._restore_layers(fixed, {3: "knight"})
    assert restored == "<layer:knight> runs in <Picture 1>. <Subject 2> waves"


def test_prepare_image():
    prepared = prompt_enhance._prepare_image(Image.create(Extent(4000, 2000)))
    assert prepared.extent == Extent(2000, 1000)  # 1024 short edge would exceed 2 MP
    prepared = prompt_enhance._prepare_image(Image.create(Extent(2048, 1536)))
    assert prepared.extent == Extent(1365, 1024)

    image = Image.create(Extent(512, 384), Qt.GlobalColor.transparent)
    prepared = prompt_enhance._prepare_image(image)
    assert prepared.extent == Extent(512, 384)
    assert prepared.pixel(0, 0) == (255, 255, 255, 255)  # opaque on white
    assert image.pixel(0, 0) == (0, 0, 0, 0)  # the original is unchanged


def test_picture_labels():
    pictures = [
        prompt_enhance.Picture("the canvas", Image.create(Extent(1707, 1280))),
        prompt_enhance.Picture("a reference", Image.create(Extent(512, 512))),
    ]
    labels = prompt_enhance._picture_labels(pictures, official=False)
    assert labels == ["<Picture 1>:", "<Picture 2>:"]
    assert prompt_enhance._picture_labels(pictures, official=True) == [
        "Reference Media: <Picture 1>, aspect_ratio 4:3",
        "Reference Media: <Picture 2>, aspect_ratio 1:1",
    ]


h3_text_encoder = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
fast_llm = "qwen3.5_9b_w4a8_gptq.safetensors"


def llm_client(*files: str) -> Client:
    models = ClientModels()
    models.text_encoders = list(files)
    loader = {"input": {"required": {"type": [["stable_diffusion", "qwen_image", "minimax"]]}}}
    models.node_inputs = ComfyObjectInfo({"CLIPLoader": loader})
    te_id = resource_id(ResourceKind.text_encoder, Arch.all, Arch.h3.main_text_encoder)
    models.resources[te_id] = h3_text_encoder
    return SimpleNamespace(models=models)  # type: ignore


def test_llm_choices():
    client = llm_client(h3_text_encoder, fast_llm, "gemma-4-e4b-Q4.gguf", "t5xxl_fp16.safetensors")
    assert prompt_enhance.llm_choices(client) == [h3_text_encoder, fast_llm]
    assert prompt_enhance.llm_choices(None) == []
    assert prompt_enhance.llm_name("LLM/qwen3.5_9b_w4a8_gptq.safetensors") == "qwen3.5_9b_w4a8_gptq"


def test_select_llm():
    style = Style(Path("default.json"))
    opts = prompt_templates.options(Arch.h3, False)

    def select(client: Client, chosen=""):
        return prompt_enhance.select_llm(Arch.h3, style, client, opts, "expert", chosen)

    client = llm_client(h3_text_encoder, "qwen3.5_4b_prompt_upsampler.safetensors", fast_llm)
    assert select(client) == prompt_enhance.LanguageModel(fast_llm, "minimax")  # fast first
    assert select(client, chosen=h3_text_encoder) == (h3_text_encoder, "minimax", None)
    assert select(client, chosen="missing.safetensors").file == fast_llm
    assert select(llm_client(h3_text_encoder)).file == h3_text_encoder  # style's own


def test_text_generate_node():
    nodes = ComfyObjectInfo({"TextGenerate": {}, "TextGenerateV2": {}})
    assert text_generate_node(nodes, with_images=True) == "TextGenerateV2"
    old = ComfyObjectInfo({"TextGenerate": {}})
    assert text_generate_node(old, with_images=True) == "TextGenerate"  # without the pictures
    with pytest.raises(RuntimeError):
        text_generate_node(ComfyObjectInfo({}))


def test_can_see():
    client = llm_client(fast_llm)
    llm = prompt_enhance.LanguageModel(fast_llm, "minimax")
    assert not prompt_enhance._can_see(llm, client)  # ComfyUI without TextGenerateV2
    client.models.node_inputs.nodes["TextGenerateV2"] = {}
    assert prompt_enhance._can_see(llm, client)
    enhancer = ensure(prompt_templates.options(Arch.ph3, False).enhancer)
    assert not prompt_enhance._can_see(llm._replace(enhancer=enhancer), client)
    assert prompt_enhance._can_see(None, None)  # OpenAI-compatible server
