<h1><img width="64px" src="ai_diffusion/icons/logo-128.png"> Diffusion <i>for Krita</i></h1>

Generative AI painting in **Krita 5 and Krita 6** with diffusion models: inpaint, outpaint, refine, live painting,
regions, control layers and upscaling, with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as the backend.

⭳ [Download](https://github.com/SHoogstad/diffusion-for-krita/releases) | ✨ [Features](#features) | 🛠️ [Installation](#installation) | ☁️ [Runpod](#runpod) | 🧩 [ComfyUI setup](#comfyui-setup) | 🧠 [Base models](#base-models) | ❓ [Common issues](#common-issues)

## <a name="features"></a> Features

* **Inpainting**: use selections for generative fill, expand the canvas, add or remove objects.
* **Live painting**: the AI interprets your canvas in real time while you paint.
* **Upscaling**: upscale and enrich images to 4k, 8k and beyond without running out of memory.
* **Diffusion models**: Flux 2, Z-Image, Qwen Image, MOSS-Image, Fibo 2, Anima, Stable Diffusion 1.5 / XL, Illustrious and more.
* **Edit models**: change images with text instructions.
* **Control layers**: scribble, line art, canny edge, pose, depth, normals, segmentation and more.
* **Reference images**: style and composition transfer, face swap.
* **Regions**: give image areas defined by layers their own text descriptions.
* **Job queue and history**: queue and cancel jobs, browse earlier results and prompts.
* **Styles**: presets with your own checkpoints, LoRA, samplers and more.
* **Krita 5 and Krita 6**: one download for both.
* **No GPU? Use Runpod**: rent a GPU with your own [Runpod](#runpod) account, started and stopped from Krita.

### MOSS-Image and Fibo 2
* **MOSS-Image**: text-to-image with a Qwen3.5-9B text encoder. **Regions** become MOSS layout boxes
  (bounding box + caption per region), and region prompts with quoted text (e.g. a neon sign saying `"OPEN"`)
  become text boxes. Needs ComfyUI 0.42 or newer.
* **Fibo 2** (Dev, generate + edit) and **Fibo 2 Edit** (Edit Turbo, 4 steps). The text encoder also sees the
  edit and reference images; reference images are called `<image_N>` in instructions. Large canvases are
  generated at about 1 MP (the training resolution) and then upscaled. Needs ComfyUI master.
* Both use the Flux 2 VAE and are detected by filename. They only show up if the connected ComfyUI
  supports them; otherwise you get an "update ComfyUI" hint.

## <a name="installation"></a> Installation

1. Install [Krita](https://krita.org/) **5.3** or **6** (5.2 may work).
2. Download `diffusion_for_krita-<version>.zip` from the [releases](https://github.com/SHoogstad/diffusion-for-krita/releases).
   Don't use GitHub's "Source code" archives, because they don't include the bundled `websockets` module.
3. In Krita, open *Tools ▸ Scripts ▸ Import Python Plugin from File...* and select the zip.
   It replaces Krita AI Diffusion if that's installed (same `ai_diffusion` folder). Your settings stay.
4. Restart Krita and create a new document or open an image.
5. Show the docker with *Settings ▸ Dockers ▸ Diffusion for Krita*.
6. Click **Configure** in the docker to install a local server, connect to one, or start a [Runpod](#runpod) GPU.

**Updates:** the plugin checks this repository's releases on startup and verifies each download against the
SHA-256 digest that GitHub publishes. *Configure ▸ Plugin ▸ Include Pre-releases* also offers pre-releases.

### Hardware
Windows, Linux and macOS. To run locally, a graphics card with at least 6 GB VRAM is recommended.

| Hardware | Support |
|---|---|
| NVIDIA GPU | CUDA (Windows/Linux) |
| AMD GPU | ROCm (Windows/Linux) |
| Intel GPU | XPU (Windows/Linux) |
| Apple Silicon | MPS on macOS 14+ |
| CPU | works, but very slow |

## <a name="runpod"></a> Runpod (rent a GPU)

Without a suitable GPU you can run ComfyUI on a [Runpod](https://www.runpod.io) GPU pod. It runs on your own
Runpod account; you pay Runpod directly, per second while the pod runs.

1. [Create a Runpod account](https://runpod.io?ref=y5f8nxcm), add credit, and create an API key (*Settings ▸ API Keys*).
   This is a referral link: it supports the development of Diffusion for Krita and costs you nothing extra.
2. In Krita: *Configure ▸ Connection ▸ Runpod*, paste the API key and click **Load GPUs**.
3. Choose a GPU (price per hour is shown), the workloads (base models) and where models are kept:
   * **Keep models (network volume)**: models are downloaded once to a network volume. It costs about
     $0.07 per GB per month, also when no pod runs, but every start after the first takes only a minute or two.
   * **Download models on every start**: no storage cost, but each start downloads the models again, which
     takes several minutes of paid GPU time.

   The settings page shows an estimate for both, based on the selected GPU and models.

   Workloads: Stable Diffusion 1.5, SDXL, Illustrious, Flux, Flux 2, Z-Image, Qwen Image (with Qwen Image
   Edit 2511), Qwen Image 2.1, Krea 2, MOSS-Image and Fibo 2 (with Fibo 2 Edit). The larger models are
   downloaded in fp8/int8 versions; Qwen Image needs a GPU with at least 24 GB, 48 GB is more comfortable.
4. Click **Start pod**. The plugin creates the pod, shows the download progress and connects.

**Stopping.** The pod is terminated when you click **Stop pod**, when Krita closes, and after the idle time
you set (default 15 minutes without a job). As a safety net the pod also terminates itself when Krita has
been disconnected for 10 minutes, for example after a crash. Still check
[your pods](https://www.runpod.io/console/pods) now and then. A network volume stays (and costs storage)
until you click **Delete network volume** or delete it on Runpod.

### Serverless (pay per image)
Choose **Serverless - pay per image** as mode to use a Runpod serverless endpoint instead of a pod. Workers
only run while they generate (plus a short "keep ready" time), so there's no cost while you're not
generating, apart from the network volume that holds the models. Serverless GPU time costs more per hour
than a pod, so it's cheapest for occasional use; for long sessions a pod is cheaper.

* **Connect** creates the network volume and endpoint if needed and downloads the models once.
* The first image after a pause waits for a worker to start (a cold start, about a minute).
* **Live painting is not practical**: every update is a separate job. Generate, inpaint, refine and upscale work.
* **Keep worker ready** sets how long a worker stays after a job: longer means fewer cold starts but more cost.
* The endpoint costs nothing while idle. **Delete endpoint** removes it; the network volume stays.

**Security.** Runpod makes the pod reachable on a public URL. The pod only accepts requests with a secret
token that the plugin generates for each pod.

Pods and serverless workers run the image built from [`scripts/runpod`](scripts/runpod): ComfyUI with the required custom
nodes, a small gatekeeper that checks the token and reports start-up progress, and the plugin's model
download script.

## <a name="comfyui-setup"></a> ComfyUI setup

The plugin uses ComfyUI as its backend. The **Local Managed Server** option installs and runs ComfyUI for you.
With **Custom Server** you connect to your own ComfyUI, locally or remote. If it's already running when Krita
starts, the plugin connects automatically.

A custom ComfyUI needs these custom nodes (install them with
[ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) or by hand):
* ControlNet preprocessors: [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
* IP-Adapter: [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
* Inpaint nodes: [Acly/comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes)
* Tooling nodes: [Acly/comfyui-tooling-nodes](https://github.com/Acly/comfyui-tooling-nodes)

Optional: [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) to load `.gguf` models.

**Models.** Some models are required, and you can't connect until they're found. You also need at least one
diffusion model (checkpoint) of a [supported base model](#base-models). The easiest way to get the right files
with the right names is the download script in the plugin folder (`ai_diffusion`):
```bash
python -m pip install aiohttp tqdm
python download_models.py --minimal /path/to/ComfyUI   # required models only
python download_models.py --all /path/to/ComfyUI       # everything the plugin supports
python download_models.py --help
```
It only downloads models; it doesn't install or change ComfyUI or custom nodes. Paths from
`extra_model_paths.yaml` work too, because the plugin asks ComfyUI which models it has.

**Missing models?** Open *Configure ▸ Connection ▸ View log files* and check `client.log`. It lists the models
that were found and the filename patterns the plugin searches for. Filenames must contain one of the patterns;
subfolders are fine, and files in a `krita` subfolder win when there are several matches.

## <a name="base-models"></a> Base models

A base model (architecture) and everything built on it, such as LoRA, ControlNet and IP-Adapter, must match.
**Extensions for one base model don't work with another.** You can install several base models side by side.
The style list shows each style's base model as an icon, and any LoRA you add must match it.

| Base model | Notes |
|---|---|
| Stable Diffusion 1.5, SDXL, Illustrious / NoobAI | Widest support for control layers and reference images |
| Flux, Flux Kontext, Flux 2 (Klein 4B / 9B) | Flux 2 also edits images |
| Z-Image, Qwen Image (incl. Edit), Chroma, Anima | |
| MOSS-Image, Fibo 2, Fibo 2 Edit | See [MOSS-Image and Fibo 2](#moss-image-and-fibo-2) |

When you install the managed server you choose "workloads": the minimal set of models a base model needs.

### <a name="custom-sampler-presets"></a> Custom sampler presets
Add your own sampler presets in `presets/samplers.json` in the plugin's settings folder (the file has examples).
Restart Krita for changes to take effect.

## <a name="common-issues"></a> Common issues

**Plugin is grayed out in the Python Plugin Manager.** Usually a "Source code" zip was installed instead of a
release zip. Hover over the entry to see the error.

**<a name="bind-address"></a>"Could not bind on address" when the server starts.** Another process is using
the port (8188): a ComfyUI you started yourself, a server from an earlier Krita session that didn't shut down,
or another program. Close that process, or log off / reboot. To use another port, close Krita, open
`settings.json` in the settings folder (`%APPDATA%\krita\ai_diffusion` on Windows) and set
`"server_arguments": "--port 25608"` (any free port).

**Updating on Windows fails with `PermissionError: [WinError 32] ... ai_diffusion`.** Close Krita, delete
`%APPDATA%\krita\pykrita\ai_diffusion`, then import the zip again. Settings live in
`%APPDATA%\krita\ai_diffusion` and aren't affected.

**Log files.** *Configure ▸ Connection ▸ View log files* opens the log folder. With *Configure ▸ Interface ▸
Dump Workflow* on, the last generated ComfyUI workflow is saved there as `workflow.json`.

Report problems [here](https://github.com/SHoogstad/diffusion-for-krita/issues). Official Krita channels are
**not** the right place for questions about this plugin.

### Object selection tools
For AI object selection and background removal, see the separate
[krita-ai-tools](https://github.com/Acly/krita-ai-tools) plugin.

## Versioning

Release tags look like `v<version>`, for example `v1.56.0`. 1.56.0 is the first Diffusion for Krita release.
Before that, the project was the Krita 5 only port
[krita-ai-diffusion-qt5](https://github.com/SHoogstad/krita-ai-diffusion-qt5) (1.54.1 to 1.55.3, tagged
`v<version>-krita5.3`). The updater still understands those tags, so those installs update to newer releases.

## Contributing

Contributions are welcome. See the [contributing guide](CONTRIBUTING.md).

## License

GPL-3.0, see [LICENSE](LICENSE). Diffusion for Krita is a modified version of
[Krita AI Diffusion](https://github.com/Acly/krita-ai-diffusion) © Acly and contributors, with changes since
2026-10-21 (see the git history). It is not affiliated with or endorsed by that project.
