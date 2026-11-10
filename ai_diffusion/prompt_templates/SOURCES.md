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
| `h3.txt` | MiniMax-H3-Image | `task_prompt` of ComfyUI template `image_minimax_h3_image_t2i` (system prompt in `prompt_templates.py`) | MIT |
| `zimage.txt` | Z-Image | `pe.py` of the [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo) Space | Apache-2.0 |

P-H3-Image has no text template: Improve Prompt runs PrunaAI's own prompt upsampler with the
request format of the official ComfyUI template (`Aspect ratio: <ratio>` / `Request: <prompt>`).
