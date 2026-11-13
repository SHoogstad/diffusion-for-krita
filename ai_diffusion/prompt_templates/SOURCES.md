# Improve Prompt templates

These are the official prompt enhancers published by the model makers, copied verbatim.
`{prompt}` marks where the user's prompt is inserted; files without it are system prompts.
Settings for each one (thinking, length, temperature, output format) are in
`ai_diffusion/model/prompt_templates.py`. Models without an official enhancer use templates
written for this plugin from the makers' prompting guides (also in that file).

| File | Model | Source | License |
| --- | --- | --- | --- |
| `qwen.txt` | Qwen-Image | `polish_prompt_en` in [QwenLM/Qwen-Image `src/examples/tools/prompt_utils.py`](https://github.com/QwenLM/Qwen-Image/blob/main/src/examples/tools/prompt_utils.py) | Apache-2.0 |
| `qwen_e-edit.txt` | Qwen-Image-Edit (2509/2511) | `polish_edit_prompt` in the same file | Apache-2.0 |
| `qwen2.txt`, `qwen2-edit.txt` | Qwen-Image 2.1 | ComfyUI templates `image_qwen_image_2_1_t2i` / `image_qwen_image_2_1_image_edit` ([Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates)) | MIT |
| `krea2.txt` | Krea 2 | "Prompt Enhancer system instruction from the Krea team", ComfyUI template `image_krea2_turbo_t2i` | MIT |
| `moss.txt` | MOSS-Image | ComfyUI template `image_moss_image_t2i` | MIT |
| `ernie.txt` | ERNIE-Image | `pe/chat_template.jinja` of [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image), as used in ComfyUI template `image_ernie_image` | Apache-2.0 / MIT |
| `h3.txt`, `h3-edit.txt` | MiniMax-H3-Image (optional, see below) | `task_prompt` of ComfyUI templates `image_minimax_h3_image_t2i` / `image_minimax_h3_image_edit` (system prompt in `prompt_templates.py`) | MIT |
| `zimage.txt` | Z-Image | `pe.py` of the [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) Space | Apache-2.0 |

P-H3-Image has no text template: Improve Prompt runs PrunaAI's own prompt upsampler with the
request format of the official ComfyUI template (`Aspect ratio: <ratio>` / `Request: <prompt>`).

MiniMax-H3-Image does not use `h3.txt` by default any more: MiniMax's enhancer thinks for up to
8192 tokens, which takes minutes on the 32B text encoder. The plugin's own templates are condensed
from the instructions of MiniMax's enhancers (ComfyUI templates `image_minimax_h3_image_t2i` and
`image_minimax_h3_image_edit`) for a small model which writes without thinking, by default
Qwen3.5-9B (`qwen3.5_9b_w4a8_gptq`, like ComfyUI's templates):

| File | Used for |
| --- | --- |
| `h3-expert.txt` | text to image: one detailed caption |
| `h3-expert-edit.txt` | canvas edits and reference pictures, when the language model sees the pictures: `<Subject N>` definitions against `<Picture N>`, one relationship per subject, then only the changes |
| `h3-expert-edit-blind.txt` | the same when the language model can't see the pictures |

MiniMax's own enhancers stay available with "Model maker's own instructions" in the Improve Prompt
menu (setting `prompt_enhancer_official`): `h3.txt` and `h3-edit.txt`, the `task_prompt` of the ComfyUI
templates `image_minimax_h3_image_t2i` / `image_minimax_h3_image_edit` (MIT), with their settings
(thinking, temperature 1, top-k 20, 8192 tokens, JSON answer). As those instructions expect, each
picture is labeled `Reference Media: <Picture N>, aspect_ratio W:H` and the request ends with
`User Aspect Ratio Selection: W:H` (the edited area or the canvas).
