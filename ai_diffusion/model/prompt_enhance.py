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
from ..image import Extent, Image
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
):
    """`context`: the pictures the prompt refers to, for templates which use them. Then the
    request counts as an edit when there are pictures, whatever `editing` says."""
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
):
    assert client is not None
    opts = request.options
    seed = random.randint(0, 2**31 - 1)
    unload = settings.prompt_enhancer_unload

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
            url = f"data:image/jpeg;base64,{image.to_base64(ImageFileFormat.jpeg)}"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": url}})
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
