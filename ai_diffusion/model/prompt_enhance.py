"""Improve a text prompt with a language model.

Two backends: ComfyUI's TextGenerate node running a language model from the server's text encoders
(the style's own one, or the one chosen in the Improve Prompt menu), or any OpenAI-compatible chat
completions API (llama.cpp, Ollama, LM Studio, OpenAI...). Templates which want it also get the
pictures the prompt refers to: the canvas being edited and the reference layers.
"""

from __future__ import annotations

import math
import random
import re
from typing import NamedTuple

from ..backend.client import Client, resolve_arch
from ..backend.network import RequestManager
from ..backend.resources import Arch, ResourceId, ResourceKind
from ..image import Bounds, Extent, Image
from ..localization import translate as _
from ..settings import ImageFileFormat, PromptEnhancer, settings
from ..style import Style, StyleSettings
from ..text import pattern_layer
from ..util import client_logger as log
from ..util import user_data_dir
from . import prompt_templates


class Picture(NamedTuple):
    """A picture which the generation gets and the prompt refers to as <Picture N>."""

    role: str  # what it is, for the language model, eg. "the canvas being edited"
    image: Image | None = None
    layer: str = ""  # name of the reference layer, if the prompt mentions it as <layer:name>


class PromptContext(NamedTuple):
    """The pictures of the job, numbered like the model will see them."""

    pictures: list[Picture]
    has_canvas: bool = False
    selection: str = ""  # where the selected area is, eg. "the left side of <Picture 1>"


def extra_instructions(style_prompt: str = ""):
    """The parts the style adds by itself and the user's own instructions."""
    lines = []
    style_text = style_prompt.replace("{prompt}", "").strip(" ,.")
    if style_text:
        lines.append(
            "The style automatically adds these words around the prompt, do not repeat them: "
            + style_text
        )
    if extra := settings.prompt_enhancer_instructions.strip():
        lines.append(f"Additional instructions: {extra}")
    return "\n".join(lines)


def build_request(
    arch: Arch,
    prompt: str,
    editing=False,
    style_prompt="",
    extent: Extent | None = None,
    vision=False,
    official=False,
):
    extra = extra_instructions(style_prompt)
    folder = templates_dir()
    return prompt_templates.build(arch, editing, prompt, extent, extra, folder, vision, official)


def system_prompt(arch: Arch, editing: bool = False, style_prompt: str = ""):
    """The system prompt (or the whole request, for templates with the prompt inside)."""
    request = build_request(arch, "{prompt}", editing, style_prompt)
    return "\n".join(t for t in (request.system, request.user) if t != "{prompt}")


def templates_dir():
    return user_data_dir / "prompt_templates"


def uses_pictures(arch: Arch):
    """Whether Improve Prompt shows the canvas and reference images for this architecture."""
    return prompt_templates.options(arch, True).vision


def clean_output(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(improved |enhanced |rewritten )?prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


# Language models on the ComfyUI server

_llm_files = re.compile(
    r"qwen3\.?5|qwen3_5|qwen35|qwen3[-_]?vl|gemma[-_]?4|gemma[-_]?3[-_]?12b|prompt_upsampler",
    re.IGNORECASE,
)
_not_llm_files = re.compile(
    r"\.gguf$|text_encoder", re.IGNORECASE
)  # GGUF returns no text in ComfyUI
_qwen35_files = re.compile(r"qwen3\.?5|qwen3_5|qwen35|prompt_upsampler", re.IGNORECASE)
_task_llm_files = re.compile(r"_pe_|upsampler", re.IGNORECASE)  # fine-tuned for one model's prompts


def llm_choices(client: Client | None) -> list[str]:
    """Text encoder files on the server which are language models that can write a prompt."""
    if client is None:
        return []
    files = client.models.text_encoders
    return [f for f in files if _llm_files.search(f) and not _not_llm_files.search(f)]


def llm_name(file: str):
    return file.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".safetensors")


def _preferred_llm(opts: prompt_templates.Options, client: Client):
    """The first of the template's preferred language models which is on the server."""
    files = [f for f in llm_choices(client) if not _task_llm_files.search(f)]
    for name in opts.llms:
        if file := next((f for f in files if name in f.lower()), None):
            return file
    return None


def _style_text_encoder(arch: Arch, style: Style, client: Client):
    text_encoder = style.text_encoder
    if not text_encoder or text_encoder == StyleSettings.text_encoder.default:
        id = ResourceId(ResourceKind.text_encoder, arch, arch.main_text_encoder)
        return client.models.find(id)
    return text_encoder


def _find_enhancer(enhancer: prompt_templates.Enhancer, client: Client):
    files = client.models.text_encoders
    return next((f for f in files if enhancer.file in f.lower()), None)


class LanguageModel(NamedTuple):
    file: str
    clip_type: str
    enhancer: prompt_templates.Enhancer | None = None  # the model's dedicated prompt enhancer

    @property
    def is_qwen35(self):
        return _qwen35_files.search(self.file) is not None


def _generic_clip_type(client: Client):
    """A CLIPLoader type which loads Qwen3.5, Qwen3-VL and Gemma files as plain language models."""
    types = client.models.node_inputs.options("CLIPLoader", "type")
    return next((t for t in ("minimax", "qwen_image", "stable_diffusion") if t in types), "minimax")


def select_llm(
    arch: Arch,
    style: Style,
    client: Client,
    opts: prompt_templates.Options,
    source: str,
    chosen: str = "",
) -> LanguageModel:
    """The language model which writes the prompt: `chosen` in the menu, else the model's
    dedicated prompt enhancer, a fast model the template prefers, or the style's text encoder."""
    style_te = _style_text_encoder(arch, style, client) if arch is not Arch.auto else None
    files = client.models.text_encoders

    def model(file: str, enhancer: prompt_templates.Enhancer | None = None):
        if enhancer:
            return LanguageModel(file, enhancer.clip_type, enhancer)
        if file == style_te and arch.clip_loader_type:  # shares the loaded model with generation
            return LanguageModel(file, arch.clip_loader_type)
        return LanguageModel(file, _generic_clip_type(client))

    if chosen and chosen in files:
        return model(chosen)
    if chosen:
        log.warning(f"Improve Prompt: {chosen} is not on the server, choosing automatically")

    enhancer = opts.enhancer
    if enhancer and source != "custom":
        if file := _find_enhancer(enhancer, client):
            return model(file, enhancer)
    if file := _preferred_llm(opts, client):
        return model(file)

    if arch is Arch.auto:
        raise RuntimeError(_("The style's checkpoint was not found on the server."))
    if not arch.can_generate_text or arch.clip_loader_type is None:
        msg = _("The text encoder of {arch} models can't write text.").format(arch=arch.value)
        msg += " " + _("Choose a language model with the arrow next to the Improve Prompt button.")
        raise RuntimeError(msg)
    if style_te is None:
        id = ResourceId(ResourceKind.text_encoder, arch, arch.main_text_encoder)
        raise RuntimeError(_("Text encoder not found on the server") + f": {id.name}")
    return model(style_te)


def automatic_llm(style: Style, client: Client | None, editing=False):
    """Name of the language model which "Automatic" picks for the style, for the menu."""
    if client is None:
        return ""
    arch = resolve_arch(style, client)
    try:
        request = build_request(arch, "", editing)
        llm = select_llm(arch, style, client, request.options, request.source)
        return llm_name(llm.file)
    except Exception:
        return ""


# Pictures


def _picture_text(prompt: str, context: PromptContext, describe=True):
    """Describes the pictures (`describe`) and replaces <layer:name> with their <Picture N>.
    Returns the request for the language model and the numbers of the pictures which are layers."""
    layers = {p.layer: i + 1 for i, p in enumerate(context.pictures) if p.layer}

    def replace(m: re.Match[str]):
        index = layers.get(m[1])
        return f"<Picture {index}>" if index else m[0]

    prompt = pattern_layer.sub(replace, prompt)
    if not describe:
        return prompt, {index: name for name, index in layers.items()}
    lines = []
    if not context.has_canvas:
        lines.append("There is no canvas: the target image is a new picture.")
    for i, picture in enumerate(context.pictures):
        line = f"<Picture {i + 1}> is {picture.role}"
        if i == 0 and context.has_canvas and context.selection:
            line += f"; the change happens in the selected area, {context.selection}"
        lines.append(line + ".")
    lines.append(f"Request: {prompt}")
    return "\n".join(lines), {index: name for name, index in layers.items()}


def _restore_layers(text: str, layers: dict[int, str]):
    """The <Picture N> of reference layers back to <layer:name>, which adds the layer as image."""

    def replace(m: re.Match[str]):
        name = layers.get(int(m[1]))
        return f"<layer:{name}>" if name else m[0]

    return re.sub(r"<Picture (\d+)>", replace, text)


def _fix_tokens(text: str):
    """Small models sometimes drop the angle brackets of <Picture N> and <Subject N>."""
    return re.sub(r"(?<![<\w])(Picture|Subject) (\d+)\b>?", r"<\1 \2>", text)


def _prepare_image(image: Image, short_edge=1024, megapixels=2.0):
    """Scaled down like MiniMax-H3-Image's own text encoder sees its pictures (1024 short edge),
    at most `megapixels`, opaque on white."""
    pixels = image.width * image.height
    scale = min(short_edge / min(image.width, image.height), math.sqrt(megapixels * 1e6 / pixels))
    if scale < 1:
        extent = Extent(max(32, round(image.width * scale)), max(32, round(image.height * scale)))
        image = Image.scale(image, extent)
    else:
        image = Image.copy(image)
    image.make_opaque()
    return image


def _picture_labels(pictures: list[Picture], official: bool):
    """The text before each picture. MiniMax's edit enhancer wants its aspect ratio too."""
    labels = []
    for i, picture in enumerate(pictures):
        if official and picture.image is not None:
            ratio = prompt_templates.ratio(picture.image.extent)
            labels.append(f"Reference Media: <Picture {i + 1}>, aspect_ratio {ratio}")
        else:
            labels.append(f"<Picture {i + 1}>:")
    return labels


def _qwen_chat(system: str, user: str, labels: list[str], thinking: bool):
    """Qwen3.5 chat template which labels each picture, eg. with its <Picture N>."""
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    pictures = "".join(f"{label} {vision}\n" for label in labels)
    text = f"<|im_start|>user\n{pictures}{user}<|im_end|>\n<|im_start|>assistant\n"
    if system:
        text = f"<|im_start|>system\n{system}<|im_end|>\n{text}"
    return text + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


# Improve Prompt


async def improve_prompt(
    prompt: str,
    style: Style,
    client: Client | None,
    editing: bool = False,
    extent: Extent | None = None,
    context: PromptContext | None = None,
    rules: str = "",
):
    """`context`: the pictures the prompt refers to, for templates which use them. Then the
    request counts as an edit when there are pictures, whatever `editing` says.
    `rules`: added after the request (small models follow it there), eg. the answer format."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    arch = resolve_arch(style, client)
    pictures = context.pictures if context else []
    if context is not None:
        editing = len(pictures) > 0
    official = settings.prompt_enhancer_official and prompt_templates.has_official(arch, editing)
    source = prompt_templates.get(arch, editing, templates_dir(), official=official)[1]
    official = official and source == "official"  # a user template replaces both
    opts = prompt_templates.options(arch, editing, official)
    use_openai = settings.prompt_enhancer is PromptEnhancer.openai

    llm = None
    if not use_openai:
        if client is None:
            raise RuntimeError(_("Not connected to a server."))
        if arch is Arch.auto:
            raise RuntimeError(_("The style's checkpoint was not found on the server."))
        llm = select_llm(arch, style, client, opts, source, settings.prompt_enhancer_llm)

    images: list[Image] = []
    if opts.vision and pictures and settings.prompt_enhancer_vision and _can_see(llm, client):
        if all(p.image is not None for p in pictures):
            images = [_prepare_image(p.image) for p in pictures if p.image is not None]
    if official and pictures and not images:  # MiniMax's edit enhancer must see the pictures
        official = False
    labels = _picture_labels(pictures, official) if images else []
    layers: dict[int, str] = {}
    if context is not None and pictures:
        prompt, layers = _picture_text(prompt, context, describe=not official)
        if official and context.has_canvas and (canvas := pictures[0].image):
            extent = canvas.extent  # the target image is the edited area

    request = build_request(
        arch, prompt, editing, style.style_prompt, extent, bool(images), official
    )
    if rules:
        request = request._replace(user=f"{request.user}\n\n{rules}")
    key = prompt_templates.key(arch, editing)
    seeing = f", {len(images)} pictures" if images else ""
    log.info(f"Improve Prompt: template {key} ({request.source}{seeing})")
    if llm is None:
        result = await _openai_chat(request, images, labels)
    else:
        result = await _comfy_generate(request, prompt, extent, llm, images, labels, client)
    result = prompt_templates.parse(clean_output(result), request.options)
    result = clean_output(result)
    if pictures:
        result = _restore_layers(_fix_tokens(result), layers)
    if not result:
        raise RuntimeError(_("The language model returned an empty prompt."))
    return result


# Layouts: MOSS-Image gets the regions as boxes (workflow.apply_layout_prompt). Its official
# enhancer (moss.txt) takes pixel xyxy boxes and describes them as "index | description" lines.


class LayoutBox(NamedTuple):
    prompt: str  # the region's own prompt
    bounds: Bounds  # in canvas pixels


def uses_layout(arch: Arch):
    """Whether Improve Prompt also writes the region prompts, as boxes of a layout."""
    return prompt_templates.options(arch, False).layout


def _box_line(index: int, box: LayoutBox):
    b = box.bounds
    coords = f"[{b.x}, {b.y}, {b.x + b.width}, {b.y + b.height}]"
    text = " ".join(box.prompt.split()) or "(empty: describe what fits there in the scene)"
    return f"{index} | {coords} | {text}"


def layout_request(prompt: str, boxes: list[LayoutBox], extent: Extent):
    lines = [
        prompt.strip() or "(no overall description: derive the scene from the boxes)",
        "",
        f"Canvas: {extent.width}x{extent.height} pixels.",
        "Bounding boxes (index | absolute pixel xyxy | request for its content):",
    ]
    lines += [_box_line(i + 1, box) for i, box in enumerate(boxes)]
    return "\n".join(lines)


def layout_rules(count: int):
    return (
        "This request has bounding boxes. Answer in exactly this form, nothing else:\n"
        "First the caption of the whole picture, following the rules above. It describes the "
        "scene, setting, composition, light and style as a whole and says in words where the "
        "boxed parts are; leave the close detail of each box to its line.\n"
        "Then a line that is exactly 'Boxes:', followed by one line per box in the given order, "
        "formatted as 'index | English description' (no coordinates), following the bounding "
        f"box rules. There are {count} boxes, so write {count} lines. A box without a request "
        "gets a description of what fits there in the scene."
    )


_box_answer = re.compile(r"^\s*(?:box\s*)?\[?(\d+)\]?\s*[|:.)]\s*(.*\S)\s*$", re.IGNORECASE)
_boxes_header = re.compile(r"^\s*[#*]*\s*boxes\s*[:*]*\s*$", re.IGNORECASE | re.MULTILINE)
_coords = re.compile(r"^\[[\d\s,.]+\]\s*\|?\s*")


def parse_layout(text: str, count: int) -> tuple[str, list[str]]:
    """The caption and the description of each box ("" where the answer has none)."""
    header = _boxes_header.search(text)
    caption_part, boxes_part = (
        (text[: header.start()], text[header.end() :]) if header else ("", text)
    )
    descriptions = [""] * count
    rest = []
    for line in boxes_part.splitlines():
        m = _box_answer.match(line)
        if m and 1 <= int(m[1]) <= count:
            descriptions[int(m[1]) - 1] = _coords.sub("", m[2]).strip()
        elif not header:
            rest.append(line)
    caption = caption_part if header else "\n".join(rest)
    return clean_output(caption.strip()), descriptions


async def improve_layout(
    prompt: str, boxes: list[LayoutBox], style: Style, client: Client | None, extent: Extent
) -> tuple[str, list[str]]:
    """Improve the prompt and the region prompts together. Returns the caption for the whole
    image and one description per box ("" where the language model wrote none)."""
    request = layout_request(prompt, boxes, extent)
    rules = layout_rules(len(boxes))
    answer = await improve_prompt(request, style, client, False, extent, rules=rules)
    caption, descriptions = parse_layout(answer, len(boxes))
    if not caption and not any(descriptions):
        raise RuntimeError(_("The language model returned an empty prompt."))
    return caption, descriptions


max_planned_boxes = 6


def plan_rules(extent: Extent):
    """Answer format for a picture which needs a layout (needs_layout said so)."""
    w, h = extent.width, extent.height
    return (
        "This picture needs a layout: you plan the bounding boxes yourself, so the rule against "
        "inventing coordinates does not apply here.\n"
        "Answer in exactly this form, nothing else: first the caption of the whole picture "
        "(scene, setting, composition, light and style as a whole, saying in words where the "
        "boxed parts are). Then a line that is exactly 'Boxes:', followed by one line per box: "
        "'index | [x0, y0, x1, y1] | English description', with absolute pixel coordinates "
        f"(top-left, bottom-right) on the {w}x{h} canvas, x from 0 to {w}, y from 0 to {h}. "
        f"Example line: 1 | [{w // 8}, {h // 16}, {w * 7 // 8}, {h // 5}] | description. "
        f"Use at most {max_planned_boxes} boxes: one per text block, panel or subject whose "
        "place the request sets, each large enough to read; describe each following the "
        "bounding box rules, text boxes with the exact words in double quotes."
    )


_planned_box = re.compile(
    r"^\s*(?:box\s*)?\[?\d+\]?\s*[|:.)]\s*\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,"
    r"\s*(-?[\d.]+)\s*\]\s*\|?\s*(.*\S)\s*$",
    re.IGNORECASE,
)


def parse_plan(text: str, extent: Extent) -> tuple[str, list[LayoutBox]]:
    """The caption and the boxes the language model planned (none if it chose a caption)."""
    header = _boxes_header.search(text)
    if header is None:
        lines = text.splitlines()
        matches = [_planned_box.match(l) for l in lines]
        caption = "\n".join(l for l, m in zip(lines, matches) if not m)
    else:
        caption = text[: header.start()]
        matches = [_planned_box.match(l) for l in text[header.end() :].splitlines()]
    boxes: list[LayoutBox] = []
    min_size = max(16, min(extent.width, extent.height) // 32)
    for m in matches:
        if not m or len(boxes) >= max_planned_boxes:
            continue
        x0, y0, x1, y1 = (float(m[i]) for i in range(1, 5))
        x0, x1 = sorted((min(max(x0, 0), extent.width), min(max(x1, 0), extent.width)))
        y0, y1 = sorted((min(max(y0, 0), extent.height), min(max(y1, 0), extent.height)))
        if x1 - x0 < min_size or y1 - y0 < min_size:
            continue
        bounds = Bounds(int(x0), int(y0), math.ceil(x1) - int(x0), math.ceil(y1) - int(y0))
        boxes.append(LayoutBox(m[5].strip(), bounds))
    return clean_output(caption.strip()), boxes


_layout_question = (
    "Request: {prompt}\n\n"
    "Does this request put text or subjects at specific places in the picture (for example a "
    "title at the top, text in a corner, a sign or speech bubble with given words, something "
    "on the left or right), or ask for a poster, cover, flyer, comic page or several panels? "
    "A single subject or an ordinary scene without positions is NO. Answer only YES or NO."
)


async def needs_layout(prompt: str, style: Style, client: Client | None):
    """A short yes/no question first: small language models decide this badly while they also
    follow the long enhancer template."""
    arch = resolve_arch(style, client)
    opts = prompt_templates.Options(max_tokens=16, temperature=0.0, sampling=False)
    system = "You sort requests for an image generator. Answer only YES or NO."
    question = _layout_question.format(prompt=prompt.strip())
    request = prompt_templates.Request(system, question, opts, "builtin")
    if settings.prompt_enhancer is PromptEnhancer.openai:
        answer = await _openai_chat(request, [], [])
    else:
        if client is None:
            raise RuntimeError(_("Not connected to a server."))
        source = prompt_templates.get(arch, False, templates_dir())[1]
        template_opts = prompt_templates.options(arch, False)
        llm = select_llm(arch, style, client, template_opts, source, settings.prompt_enhancer_llm)
        answer = await _comfy_generate(request, "", None, llm, [], [], client, keep_loaded=True)
    answer = clean_output(answer).strip().upper()
    log.info(f"Improve Prompt: needs a layout? {answer[:20]!r}")
    return answer.startswith("YES")


async def plan_layout(
    prompt: str, style: Style, client: Client | None, extent: Extent
) -> tuple[str, list[LayoutBox]]:
    """Improve a prompt which has no regions yet. The language model first decides whether the
    picture needs a layout; returns the caption and the boxes (empty: no layout needed)."""
    if not prompt.strip():
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    if not await needs_layout(prompt, style, client):
        return await improve_prompt(prompt, style, client, False, extent), []
    request = f"{prompt.strip()}\n\nCanvas: {extent.width}x{extent.height} pixels."
    answer = await improve_prompt(request, style, client, False, extent, rules=plan_rules(extent))
    caption, boxes = parse_plan(answer, extent)
    if not caption:
        raise RuntimeError(_("The language model returned an empty prompt."))
    return caption, boxes


async def improve_box(
    prompt: str, box: LayoutBox, style: Style, client: Client | None, extent: Extent
):
    """Improve one region prompt: only its box goes to the language model (the enhancer's
    "one box" mode, a few sentences), the image prompt is context."""
    if not box.prompt.strip():
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    request = layout_request(prompt, [box], extent)
    rules = (
        "This request has exactly one bounding box. Answer only with the description of what "
        "is inside that box, one to three sentences, so it fits into the scene described "
        "above. No caption of the whole image, no coordinates, no index, no commentary."
    )
    answer = await improve_prompt(request, style, client, False, extent, rules=rules)
    lines = [l for l in answer.splitlines() if l.strip() and not _boxes_header.match(l)]
    if len(lines) == 1 and (m := _box_answer.match(lines[0])):
        return _coords.sub("", m[2]).strip()
    return answer


def _can_see(llm: LanguageModel | None, client: Client | None):
    """Whether pictures can go to the language model. None is the OpenAI-compatible server."""
    if llm is None:
        return True
    if llm.enhancer is not None or client is None:  # dedicated enhancers only take text
        return False
    if "TextGenerateV2" not in client.models.node_inputs:
        log.warning("Improve Prompt: ComfyUI is older than 0.43, writing without the pictures")
        return False
    return True


def _thinking(opts: prompt_templates.Options):
    return opts.thinking or settings.prompt_enhancer_thinking


def _max_tokens(opts: prompt_templates.Options):
    """Thinking needs room for the reasoning before the answer."""
    return opts.max_tokens if not _thinking(opts) else max(opts.max_tokens, 6144)


async def _comfy_generate(
    request: prompt_templates.Request,
    prompt: str,
    extent: Extent | None,
    llm: LanguageModel,
    images: list[Image],
    labels: list[str],
    client: Client | None,
    keep_loaded=False,
):
    assert client is not None
    opts = request.options
    seed = random.randint(0, 2**31 - 1)
    unload = settings.prompt_enhancer_unload and not keep_loaded

    # Dedicated prompt enhancer of the model (eg. P-H3-Image's upsampler)
    if enhancer := llm.enhancer:
        user = prompt_templates.fill(enhancer.user_format, prompt, extent)
        system = request.system if enhancer.system else ""
        raw = enhancer.raw_template
        if raw:
            user = raw.replace("{system}", system).replace("{user}", user)
            system = ""
        log.info(f"Improving prompt with {llm.file} ({enhancer.clip_type})")
        return await client.generate_text(
            llm.file,
            enhancer.clip_type,
            user,
            system,
            seed,
            max_length=opts.max_tokens,
            thinking=enhancer.thinking,
            temperature=opts.temperature,
            sampling=enhancer.sampling,
            use_default_template=raw is None,
            unload=unload,
        )

    thinking = _thinking(opts)
    user, system = request.user, request.system
    if images and llm.is_qwen35:
        user, system = _qwen_chat(system, user, labels, thinking), ""
    elif images:
        order = ", ".join(label.rstrip(":") for label in labels)
        user = f"The pictures are attached in this order: {order}.\n{user}"
    log.info(f"Improving prompt with {llm.file} ({llm.clip_type}, thinking {thinking})")
    return await client.generate_text(
        llm.file,
        llm.clip_type,
        user,
        system,
        seed,
        max_length=_max_tokens(opts),
        thinking=thinking,
        temperature=opts.temperature,
        sampling=opts.sampling,
        top_k=opts.top_k,
        repetition_penalty=opts.repetition_penalty,
        images=images or None,
        unload=unload,
    )


_requests: RequestManager | None = None


async def _openai_chat(request: prompt_templates.Request, images: list[Image], labels: list[str]):
    global _requests
    if _requests is None:
        _requests = RequestManager()
    base = settings.prompt_enhancer_url.strip().rstrip("/")
    if not base:
        raise RuntimeError(_("Set the LLM Server URL in Settings > Interface."))
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    content: str | list[dict] = request.user
    if images:
        content = []
        for label, image in zip(labels, images):
            data_url = f"data:image/jpeg;base64,{image.to_base64(ImageFileFormat.jpeg)}"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        content.append({"type": "text", "text": request.user})
    messages: list[dict] = [{"role": "user", "content": content}]
    if request.system:
        messages.insert(0, {"role": "system", "content": request.system})
    opts = request.options
    data = {
        "model": settings.prompt_enhancer_model.strip() or "default",
        "messages": messages,
        "temperature": opts.temperature,
        "max_tokens": max(_max_tokens(opts), 4096 if _thinking(opts) else 1024),
        "stream": False,
    }
    key = settings.prompt_enhancer_api_key.strip() or None
    log.info(f"Improving prompt with {url} (model {data['model']})")
    reply = await _requests.http("POST", url, data, timeout=600, bearer=key)
    try:
        message = reply["choices"][0]["message"]
        return message.get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(_("Unexpected answer from the LLM server") + f": {str(reply)[:200]}")
