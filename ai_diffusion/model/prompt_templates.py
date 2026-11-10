"""Templates for Improve Prompt, one per diffusion architecture.

Where a model maker publishes a prompt enhancer, it is used verbatim from
`ai_diffusion/prompt_templates/` (see SOURCES.md there). The other templates below are written
from the makers' prompting guides. Architectures that can edit images also have an edit template.

Users can replace any template with a text file in the user data folder:
`prompt_templates/<arch>.txt` or `prompt_templates/<arch>-edit.txt` (eg. `illu.txt`,
`qwen_e_p-edit.txt`), where <arch> is the internal name used in style files. `{prompt}` in a
template marks where the prompt goes; without it the template is used as system prompt.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import NamedTuple

from ..backend.resources import Arch
from ..image import Extent

official_dir = Path(__file__).parent.parent / "prompt_templates"

rules = """Rules:
- Keep every subject, detail and style the user asked for. Do not contradict the input.
- Write in English, even if the input is in another language. Text that should appear in the \
image stays in its original language.
- If the input is already detailed, only polish it; do not invent a different scene.
- Output ONLY the prompt. No preface, no explanation, no quotes around it, no Markdown."""


class Enhancer(NamedTuple):
    """A dedicated prompt enhancer model, used when its file is on the ComfyUI server."""

    file: str  # part of the text encoder file name
    clip_type: str
    user_format: str = "{prompt}"
    raw_template: str | None = None  # replaces the model's chat template: {system}, {user}
    system: bool = True  # whether the template text is sent as system prompt
    thinking: bool = False
    sampling: bool = True  # off: greedy, like the official workflow


class Options(NamedTuple):
    thinking: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    sampling: bool = True
    json_key: str | None = None  # the answer is a JSON object, the prompt is in this key
    suffix: str = ""  # appended to the answer
    system: str = ""  # system prompt for templates that are the user message
    user_format: str = "{prompt}"  # placeholders: {prompt} {prompt_json} {width} {height} ...
    enhancer: Enhancer | None = None


class Request(NamedTuple):
    system: str
    user: str
    options: Options
    source: str  # "custom", "official" or "builtin"


# Shared building blocks

_quoted_text = (
    "If the image should contain text (signs, titles, labels, speech), write the exact words "
    'in double quotes and say where they appear and in which lettering, eg. a neon sign reading "OPEN".'
)
_positive_only = (
    "The model has no negative prompt: describe what should be there, never write 'no X' or "
    "'without X'."
)
_edit_core = (
    "The user describes a change to an existing image. Rewrite it as a clear, specific "
    "instruction: what to change, where in the image, and what it should look like afterwards. "
    "You cannot see the image: refer to things only with what the user said (eg. 'the cat', "
    "'the person on the left', 'the sky') and never invent details like clothing, colors or "
    "names. Avoid vague pronouns like 'it' when it is unclear what they mean. Then say what "
    "must stay the same, eg. 'Keep the faces, poses and the rest of the image unchanged.' Only "
    "keep things that the change does not affect. Keep it to one to three sentences."
)


def _references(placeholder: str):
    return (
        f"Images are referred to as {placeholder.format(1)} (the canvas), "
        f"{placeholder.format(2)}, {placeholder.format(3)} and so on (reference images). "
        "Keep every such reference exactly as written, and use them to say which element comes "
        "from which image."
    )


def _natural(model: str, length: str, extra: str = ""):
    return (
        f"You improve text prompts for {model}, which understands natural language. Rewrite the "
        f"user's prompt into {length}: start with the main subject, then its appearance, "
        "clothing, pose and expression, the setting, the composition and camera angle, the "
        "lighting, the color palette and the medium or photographic style. Use concrete visual "
        "words; avoid empty praise like 'masterpiece', 'stunning' or '8k'. " + extra
    ).strip()


def _edit(model: str, extra: str = ""):
    return f"You improve instructions for {model}, an AI image editing model. {_edit_core} {extra}"


# Templates written for the plugin, for models without an official prompt enhancer.
# Keys are "<arch>" for text-to-image and "<arch>-edit" for edits.

_bfl_flux1 = _natural(
    "Flux.1",
    "one flowing paragraph of 40 to 100 words, written like a photo caption",
    _positive_only + " " + _quoted_text,
)
# docs.bfl.ai: [klein] has no prompt upsampling, "write in prose, not keyword lists: describe
# scenes like a novelist", give the light's source, direction and softness, end with
# "Style: ... Mood: ...". FLUX.2 supports hex colors and has no negative prompts.
_bfl_klein = (
    "You improve text prompts for FLUX.2 [klein], which renders exactly what it is given and "
    "has no prompt upsampling of its own. Rewrite the user's prompt as prose, not a keyword "
    "list: describe the scene like a novelist in one paragraph of 50 to 120 words, most "
    "important first: the subject and what it is doing, then the setting, the composition and "
    "camera, and the lighting with its source, direction and softness. Exact colors may be "
    "given as hex codes (eg. #1E90FF). " + _positive_only + " " + _quoted_text + " End with "
    "'Style: <style>. Mood: <mood>.'"
)
_bfl_klein_edit = _edit(
    "FLUX.2 [klein]",
    "Describe only what changes; the input image provides everything else. "
    + _references("image {}"),
)
# docs.bfl.ai, FLUX.1 Kontext: "Specify what should change: the input image provides all other
# visual context", quote text edits (Replace 'joy' with 'BFL'), state what stays the same,
# prefer specific verbs ("change the clothes" over "transform the person").
_bfl_kontext = _edit(
    "FLUX.1 Kontext",
    "Describe only the change: the input image provides all other visual context. Prefer "
    "specific verbs like 'change the clothes' over vague ones like 'transform the person'. For "
    "text use: Replace 'old text' with 'new text'. For a style change, name the style and its "
    "traits and say that the composition stays the same.",
)
# huggingface.co/PrunaAI/P-H3-Image: say what the image is in the first sentence (a real
# object or medium), describe one scene with a viewpoint, one main subject in a moment, a few
# supporting elements and light with a source and direction; quote every string of text, say
# where it sits and end with "no other text"; avoid hype words. Edits: "Edit <Picture 1>.
# <The change, described as the finished result>. Keep <everything that must stay> exactly as
# it is."
_ph3 = (
    "You improve text prompts for P-H3-Image, which renders full, specific prompts much better "
    "than short requests. Rewrite the user's prompt into one paragraph of 80 to 200 words. "
    "Open with one sentence that says what the image is, as a real object or medium, eg. 'A "
    "photograph by an experienced street photographer of ...' or 'A gouache illustration "
    "of ...'. Describe one scene: a viewpoint, one main subject caught in a moment, a few "
    "supporting elements, and the light with its source and direction. Only if the user asks "
    "for text in the image, put every string of "
    "text in double quotes, keep each short, say where it sits, and end with 'no other text'; "
    "otherwise never mention text, signs or labels at all. Never use hype words like "
    "'stunning', 'epic', 'cinematic', '8k' or 'masterpiece'."
)
_ph3_edit = (
    "You improve edit requests for P-H3-Image. The image to edit is <Picture 1>; reference "
    "images are <Picture 2>, <Picture 3> and so on, keep those names exactly. Rewrite the "
    "request in this form: 'Edit <Picture 1>. <The change, described as the finished "
    "result>. Keep <everything that must stay> exactly as it is.' Name the exact target and "
    "where it is, describe the result rather than the action, list what must stay unchanged, "
    "and put any new text in double quotes. You cannot see the images: refer to things only "
    "with what the user said and never invent details like clothing, colors or names."
)

# MiniMax's edit enhancer (ComfyUI template image_minimax_h3_image_edit) needs to see the images,
# which Improve Prompt can't send. Written from its output format: subjects are defined against
# <Picture N>, preserved parts are named, then the target image is described.
_h3_edit = (
    "You improve edit requests for MiniMax-H3-Image, which works best with a full description of "
    "the finished image. The image to edit is <Picture 1>; reference images are <Picture 2>, "
    "<Picture 3> and so on, keep those names exactly. Rewrite the request as one paragraph: "
    "first say what in <Picture 1> stays exactly the same, then describe the finished result of "
    "the change: what is added, removed or different, where it is, and how it fits the "
    "lighting and perspective of the scene. Say which element comes from which reference image. "
    "Put any new text in double quotes. You cannot see the images: refer to things only with "
    "what the user said and never invent details like clothing, colors or names."
)

_builtin: dict[str, str] = {
    "sd15": (
        "You improve text prompts for Stable Diffusion 1.5, which reads short, comma-separated "
        "keywords and only the first 75 tokens well. Rewrite the user's prompt as one line of "
        "concise keywords and short phrases, most important first: subject, appearance, pose, "
        "setting, composition, lighting, colors, then art style or photo terms (eg. 'oil "
        "painting', '35mm photo, shallow depth of field'). Keep it under 40 words. Weights "
        "like (red hair:1.2) are allowed for the one or two most important details."
    ),
    "sdxl": (
        "You improve text prompts for Stable Diffusion XL. It works best with short descriptive "
        "phrases separated by commas: a one-sentence description of the subject and scene "
        "first, then keywords for details, composition, lighting, colors and the medium or "
        "camera (eg. 'cinematic photo, 85mm, golden hour, film grain' or 'watercolor "
        "illustration, soft edges'). Keep it under 60 words; text inside the image rarely works, "
        "so only keep it if the user asks for it."
    ),
    "illu": (
        "You improve text prompts for an Illustrious anime model trained on Danbooru tags. "
        "Rewrite the user's prompt as comma-separated Danbooru tags, lowercase, with spaces "
        "instead of underscores, in this order: subject count (1girl, 2boys, no humans), "
        "character and series names if given (eg. 'hatsune miku, vocaloid'), artist tags if "
        "given, then hair, eyes, body, clothing, accessories, pose, expression, action, "
        "setting, composition (cowboy shot, from above, close-up) and lighting. Use real "
        "Danbooru tags only, no sentences. Keep it under 40 tags."
    ),
    # Stability AI's SD 3.5 guide: natural language, be specific about style, subject,
    # composition, lighting and color; text in quotes; negative prompts are not needed.
    "sd3": _natural(
        "Stable Diffusion 3.5",
        "two to four detailed sentences (at most 150 words): style first, then subject, "
        "composition, lighting and colors",
        _quoted_text,
    ),
    "flux": _bfl_flux1,
    "flux_k-edit": _bfl_kontext,
    "flux2_4b": _bfl_klein,
    "flux2_4b-edit": _bfl_klein_edit,
    "chroma": _natural(
        "Chroma, a Flux-based model that also knows art styles and anime",
        "one detailed paragraph of 50 to 120 words",
        "Name the medium precisely (photo, anime illustration, 3D render, oil painting). "
        + _quoted_text,
    ),
    # github.com/QwenLM/Qwen-Image-Layered: the prompt is a caption of the whole image,
    # including hidden or occluded elements, not instructions for single layers.
    "qwen_l-edit": (
        "You improve captions for Qwen-Image-Layered, which splits an existing image into "
        "separate layers. Rewrite the user's text into a caption of the whole image (two to "
        "four sentences) that names each distinct element: the background, each subject or "
        "object, partly hidden things behind them, and any text (in double quotes). The caption "
        "describes the overall scene; it does not give instructions for single layers."
    ),
    # huggingface.co/circlestone-labs/Anima: "[quality/meta/year/safety tags] [1girl/1boy/
    # 1other etc] [character] [series] [artist] [general tags]", lowercase tags with spaces,
    # "@" before artist names, natural language of at least two sentences, both can be mixed,
    # give each named character an appearance description.
    "anima": (
        "You improve text prompts for Anima, an anime and illustration model that reads Danbooru "
        "tags and natural language together. Rewrite the user's prompt as: comma-separated tags "
        "in the order subject count (1girl, 2boys, 1other), character, series, artist, general "
        "tags (appearance, clothing, pose, expression, setting), lowercase with spaces instead of "
        "underscores; artist names get an @ in front (eg. @artist name). Then add at least two "
        "plain English sentences that describe the scene, the composition and the lighting, "
        "with normal capitalization for character and series names. Give every named character "
        "an appearance description (eg. 'Fern from Sousou no Frieren, with long purple hair'). "
        "The model does not do realism well, so keep the image illustrated. Keep it under 100 "
        "words."
    ),
    "fibo2": _natural(
        "Fibo 2, which was trained on very detailed captions",
        "a detailed, factual paragraph of 80 to 180 words: list every object with its "
        "position, size, color and material, then the background, lighting, camera angle, "
        "depth of field and the medium or style",
        _quoted_text,
    ),
    # Official Fibo 2 edit enhancer (ComfyUI image_fibo2_edit_*_pe): needs the image and writes
    # a JSON caption; its "Edit Instruction Rewriting Rule" asks for one imperative instruction.
    "fibo2-edit": _edit(
        "Fibo 2",
        "Write the change as one instruction in the imperative, like 'Turn the image into the "
        "cartoon style.', 'Change the dog's color to black and white.', 'Add a wide-brimmed "
        "felt hat to the subject.' or 'Remove the book from the subject's hands.'. "
        + _references("<image_{}>"),
    ),
    "h3-edit": _h3_edit,
    "ph3": _ph3,
    "ph3-edit": _ph3_edit,
}
for _key, _same in [
    ("illu_v", "illu"),
    ("flux2_9b", "flux2_4b"),
    ("flux2_9b-edit", "flux2_4b-edit"),
]:
    _builtin[_key] = _builtin[_same]
_builtin["fibo2_e-edit"] = _builtin["fibo2-edit"]

# Official prompt enhancers, files in ai_diffusion/prompt_templates

_official: dict[str, str] = {
    "qwen": "qwen.txt",
    "qwen_e-edit": "qwen_e-edit.txt",
    "qwen_e_p-edit": "qwen_e-edit.txt",
    "qwen2": "qwen2.txt",
    "qwen2-edit": "qwen2-edit.txt",
    "krea2": "krea2.txt",
    "moss": "moss.txt",
    "ernie": "ernie.txt",
    "zimage": "zimage.txt",
    "h3": "h3.txt",
}

_options: dict[str, Options] = {
    "qwen": Options(max_tokens=1024, suffix=" Ultra HD, 4K, cinematic composition"),
    "qwen_e-edit": Options(max_tokens=1024, json_key="Rewritten"),
    "qwen_e_p-edit": Options(max_tokens=1024, json_key="Rewritten"),
    "qwen2": Options(thinking=True, max_tokens=4096, temperature=1.0),
    "qwen2-edit": Options(thinking=True, max_tokens=4096, temperature=1.0),
    "krea2": Options(thinking=True, max_tokens=2048),
    "moss": Options(max_tokens=4096),
    "zimage": Options(max_tokens=2048),
    "ernie": Options(
        max_tokens=2048,
        temperature=0.6,
        user_format='{{"prompt": {prompt_json}, "width": {width}, "height": {height}}}',
        enhancer=Enhancer(
            "ernie-image-prompt-enhancer",
            "flux2",
            raw_template="<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]",
        ),
    ),
    # ComfyUI template image_minimax_h3_image_t2i: TextGenerateV2 with thinking, temperature 1,
    # top-k 20, 8192 tokens, then the "expanded_caption" of the JSON answer
    "h3": Options(
        thinking=True,
        max_tokens=8192,
        temperature=1.0,
        json_key="expanded_caption",
        system=(
            "Reason in the thinking section first. The visible final content must be exactly one "
            "JSON object with exactly two keys: expanded_caption and aspect_ratio. Every ASCII "
            "double quotation mark used as caption content inside expanded_caption must be "
            'serialized as the JSON escape sequence \\". Do not output text outside that object.'
        ),
    ),
    "ph3": Options(
        max_tokens=1536,
        enhancer=Enhancer(
            "p-h3-image_prompt_upsampler",
            "minimax",
            user_format="Aspect ratio: {aspect_ratio}\nRequest: {prompt}",
            system=False,
            thinking=True,
            sampling=False,
        ),
    ),
}

_prompt = "{" + "prompt}"

_aspect_ratios = {"1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3}
_aspect_ratios.update({"16:9": 16 / 9, "9:16": 9 / 16})


def aspect_ratio(extent: Extent):
    ratio = extent.width / max(extent.height, 1)
    return min(_aspect_ratios, key=lambda k: abs(math.log(_aspect_ratios[k] / ratio)))


def is_edit(arch: Arch, editing: bool):
    """Edit-only models always get an instruction, text-to-image-only models never do."""
    return arch.is_edit or (editing and arch.supports_edit)


def key(arch: Arch, editing: bool):
    return f"{arch.name}-edit" if is_edit(arch, editing) else arch.name


def user_file(arch: Arch, editing: bool, folder: Path):
    return folder / f"{key(arch, editing)}.txt"


_fallback = _natural("an AI image generation model", "one detailed, vivid paragraph")
_fallback_edit = _edit("an AI image editing model")


def builtin(arch: Arch, editing: bool):
    """The template shipped with the plugin and where it comes from ("official"/"builtin")."""
    k = key(arch, editing)
    if file := _official.get(k):
        return (official_dir / file).read_text(encoding="utf-8").strip(), "official"
    if text := _builtin.get(k):
        return text, "builtin"
    return (_fallback_edit if is_edit(arch, editing) else _fallback), "builtin"


def get(arch: Arch, editing: bool, folder: Path | None = None):
    """The template for `arch` and its source, from a user file if there is one."""
    if folder is not None:
        file = user_file(arch, editing, folder)
        try:
            if file.exists() and (text := file.read_text(encoding="utf-8").strip()):
                return text, "custom"
        except OSError:
            pass
    return builtin(arch, editing)


def options(arch: Arch, editing: bool):
    return _options.get(key(arch, editing), Options())


def fill(template: str, prompt: str, extent: Extent | None):
    extent = extent or Extent(1024, 1024)
    values = {
        "prompt": prompt,
        "prompt_json": json.dumps(prompt, ensure_ascii=False),
        "width": str(extent.width),
        "height": str(extent.height),
        "aspect_ratio": aspect_ratio(extent),
    }

    def replace(m: re.Match[str]):
        if m.group(1) is None:
            return m.group(0)[0]  # {{ or }}
        return values.get(m.group(1), m.group(0))

    return re.sub(r"\{\{|\}\}|\{(\w+)\}", replace, template)


def build(
    arch: Arch,
    editing: bool,
    prompt: str,
    extent: Extent | None = None,
    extra: str = "",
    folder: Path | None = None,
):
    """The messages for the language model. `extra` is added to the system prompt."""
    text, source = get(arch, editing, folder)
    opts = options(arch, editing)
    user = fill(opts.user_format, prompt, extent)
    if _prompt in text:  # the template is the user message, with the prompt inside
        user = text.replace(_prompt, user)
        system = opts.system if source == "official" else ""
    else:
        system = text if source == "official" else f"{text}\n{rules}"
    if extra:
        system = f"{system}\n{extra}" if system else extra
    return Request(system, user, opts, source)


def parse(text: str, opts: Options):
    """The prompt from a model's answer, for templates that ask for JSON or add a suffix."""
    if opts.json_key:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and isinstance(data.get(opts.json_key), str):
                    text = data[opts.json_key]
            except json.JSONDecodeError:
                pass
    text = " ".join(text.split()) if opts.suffix else text
    if text and opts.suffix:
        text = text.rstrip() + ("" if text.rstrip().endswith((".", ",")) else ".") + opts.suffix
    return text
