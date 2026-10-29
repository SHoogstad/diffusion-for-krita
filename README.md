<h1><img width="64px" src="ai_diffusion/icons/logo-128.png"> Diffusion <i>for Krita</i></h1>

Generative AI painting in **Krita 5 and Krita 6** with diffusion models: inpaint, outpaint, refine, live painting,
regions, control layers and upscaling, with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as the backend.

⭳ [Download](https://github.com/SHoogstad/diffusion-for-krita/releases) | ✨ [Features](#features) | 🛠️ [Installation](#installation) | ☁️ [Runpod](#runpod) | 🧩 [ComfyUI setup](#comfyui-setup) | 🧠 [Base models](#base-models) | 🦴 [Poses](#poses) | ❓ [Common issues](#common-issues)

## <a name="features"></a> Features

* **Inpainting**: use selections for generative fill, expand the canvas, add or remove objects.
* **Live painting**: the AI interprets your canvas in real time while you paint.
* **Upscaling**: upscale and enrich images to 4k, 8k and beyond without running out of memory.
* **Diffusion models**: Flux 2, Z-Image, Qwen Image, MOSS-Image, Fibo 2, Anima, Stable Diffusion 1.5 / XL, Illustrious and more.
* **Edit models**: change images with text instructions.
* **Control layers**: scribble, line art, canny edge, pose, depth, normals, segmentation and more.
* **Pose transfer**: take a pose from any image and keep your character's body proportions with [Match Proportions](#poses).
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

### H3 Edit (experimental)
* Edits images with **MiniMax H3**, the same way [H3 Image Studio](https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio)
  does. Without reference images it uses **Keep source**: the canvas is anchored as the first frame of a
  short clip, which keeps faces, pose and background. With reference images it switches to **Creative**,
  which redraws the scene using all of them (`<Picture 1>` is the canvas, references are `<Picture 2>` and up).
* Needs ComfyUI master with H3 support and the H3 Image Studio custom nodes. `H3StillSliceDecode` (included
  in the Runpod image) removes the 16 px banding of the stock decode; without it the stock decode is used.
* The default style uses the **H3 - Turbo** sampler: Larry's v4 EMA turbo LoRA at 0.38, 20 steps, er_sde.
  **H3 - ER-SDE** runs 50 steps without the LoRA. Large canvases are edited at about 2 MP and then upscaled.
* The models are big: about 40 GB (FL2VA pruned int8, Qwen3-VL 32B text encoder in nvfp4, video VAE). On
  Runpod, pick a GPU with 48 GB or more.

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
   Edit 2511), Qwen Image 2.1, Krea 2, MOSS-Image, Fibo 2 (with Fibo 2 Edit) and H3 Edit (experimental,
   about 40 GB). The larger models are downloaded in fp8/int8 versions; Qwen Image needs a GPU with at least
   24 GB, 48 GB is more comfortable. H3 Edit needs 48 GB or more.
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
| H3 Edit (experimental) | See [H3 Edit](#h3-edit-experimental) |

When you install the managed server you choose "workloads": the minimal set of models a base model needs.

### <a name="custom-sampler-presets"></a> Custom sampler presets
Add your own sampler presets in `presets/samplers.json` in the plugin's settings folder (the file has examples).
Restart Krita for changes to take effect.

## <a name="poses"></a> Poses and pose transfer

A **pose layer** is a vector layer with an OpenPose skeleton: colored dots are **joints** (nose, neck,
shoulders, elbows, wrists, hips, knees, ankles, eyes, ears) and the lines between them are **bones**. A control
layer in **Pose** mode turns this skeleton into a pose ControlNet input, so the generated figure takes that pose.
You need a pose ControlNet for your base model. If it's missing, the control layer shows a "model not found"
warning.

This section covers making skeletons, editing them, and **pose transfer**, which puts the pose from one image
onto your own character.

### Making a pose layer

**From an image.** Add a control layer, set its mode to **Pose**, and click **From Image** (the generate icon
in the control layer bar, or the button under ⋯ when the docker is narrow). The plugin sends the **whole
visible canvas** to ComfyUI (DWPose), so hide every layer except the image you want the pose from. A selection
does not limit the area for poses. The result is a new vector layer, and the control layer switches to it.

* New layers are numbered: `[Control] Pose`, `[Control] Pose 2`, `[Control] Pose 3` and so on, so you can
  tell several pose layers apart in the Layers docker and in the control layer's layer dropdown.
* Only the body skeleton (18 joints per person) goes into the layer. Hands and face are detected but not used.
* Joints the detector isn't sure about are left out, together with their bones.
* Several people in the image become several skeletons in the same layer (person 1, 2, ...).

**By hand.** Create a vector layer (*Layer ▸ New ▸ Add Vector Layer*), pick it in a Pose control layer, and
click **Add Skeleton** (the add-pose icon). This adds a standing default figure on the left side of the canvas,
sized to the canvas. Every click adds another person. Then move the joints to the pose you want.

### Editing a skeleton

* **Select the pose layer in the Layers docker first.** The plugin only updates bones for the *active* layer,
  about twice per second.
* Use Krita's **Select Shapes Tool** (the vector selection arrow). Drag **joints**; their bones follow.
* **Move, scale or rotate several joints at once**: select them (drag a rectangle or Shift+click) and use the
  handles of the selection. Including the bones in the selection works too. When you *scale* bones along,
  their line width scales as well, so select only the joints if you scale a lot.
* **Groups work.** You can group a skeleton (Ctrl+G) and then move, scale or rotate the whole group.
* **Undo and redo** (Ctrl+Z / Ctrl+Shift+Z) step through your own edits. Bones following their joints don't
  add undo steps.
* **Add a second person** by copying and pasting a whole skeleton (joints and bones). The pasted joints are
  renumbered as a new person automatically.
* **Leave out a body part** by selecting its joints *and* bones and deleting them.
* **Bones always follow the joints.** If you drag a bone line by itself, it snaps back onto its joints. A
  bone line you deleted by accident comes back when you nudge one of its joints a little.

Things to avoid:

* **Don't rename shapes.** Joints and bones are recognized by their names (`P00_J04` = person 0, joint 4).
* **Scaling joints also scales the dots.** The joint position stays exact, but much bigger or smaller dots
  change the control image a bit. Scale a selection mostly for positions, not to resize single dots.

### Pose transfer with your character's proportions (Match Proportions)

When you take a pose from another image, its skeleton also carries **that person's body proportions**:
leg length, arm length, shoulder width, head size. If you use it as-is, your character is drawn with those
proportions. **Match Proportions** keeps the pose but uses your character's proportions.

You need two pose layers:

* a **character skeleton**: your character, ideally standing upright and facing the camera, so every bone is
  seen at full length;
* a **target pose**: the pose you want.

Step by step:

1. Show only the image of your character. In a Pose control layer, click **From Image**. You get
   `[Control] Pose`, the character skeleton. Check it and fix joints by hand if needed (see
   [Editing a skeleton](#poses)).
2. Hide the character image **and** the new `[Control] Pose` layer, and show only the image with the target
   pose. Click **From Image** again. You get `[Control] Pose 2`, the target pose. The control layer now uses
   this layer.
3. Make sure the control layer's layer dropdown shows the **target pose** (`[Control] Pose 2`).
4. Click **Match Proportions** (the pose icon next to *Add Skeleton*, or the button under ⋯ when the docker is
   narrow) and pick the **character skeleton** (`[Control] Pose`) from the menu.
5. A new layer `[Control] Pose 2 (proportions of [Control] Pose)` appears above the target pose, and the
   control layer switches to it. The original target pose is not changed.
6. Hide the character skeleton and the original target pose (or delete them). If the character skeleton is
   still used by **another** control layer, remove that control layer, otherwise both poses steer the image.
7. Generate.

To undo, press Ctrl+Z twice (the new layer's shapes, then the layer), or simply delete the new layer. Like
any control layer whose layer disappears, the control layer entry that used it is removed too. Add a Pose
control layer again and pick the original target pose in its dropdown.

**How it works.** The skeleton is rebuilt from the neck outwards. Every bone keeps the **direction** from the
target pose and gets its **length** from the character skeleton. The neck stays where it is in the target, and
the overall size is matched to the target (from the median ratio between the two skeletons' bones), so the
figure fills about the same space as the target pose. The face (nose, eyes, ears) is matched the same way.

**Foreshortening.** A limb that points towards the camera looks short in the target pose. Stretching it to the
character's full length would undo that depth. So bones that are shorter than **half** of their expected length
are treated as foreshortened and only partly stretched: a forearm showing 30% of its length becomes 60%
(30% ÷ 0.5). Bones longer than half are set to the character's full length.

**Several people.** Person 1 of the target gets the proportions of person 1 of the character layer, person 2
of person 2, and so on. If the character layer has fewer people, the remaining people get person 1's
proportions. For full control, keep one person per layer.

**Missing joints.**

* A joint missing in the target stays missing.
* A bone missing in the character skeleton keeps its length from the target.
* A joint whose parent joint is missing in the target stays where it was.

**Tips and limits.**

* The character skeleton is the "ruler". If your character sits, crouches or has legs towards the camera in
  its image, those bones are measured too short. Use a standing, front-facing image, or fix the character
  skeleton by hand before matching.
* Proportions are measured in 2D. Turning or leaning in either image changes the measured lengths.
* Very stylized proportions (chibi, very large heads) work, but they can look odd on poses with lots of
  foreshortening.
* You can run Match Proportions again on the result, for example with a corrected character skeleton.

### Troubleshooting poses

**"From Image" gives an empty layer (no skeleton).** DWPose first looks for people with a person detector.
When it finds no one, the result is empty. This happens with very stylized art, extreme foreshortening, tight
crops or small figures. The ComfyUI log then shows `DWPose: Bbox ...ms` with no `DWPose: Pose ...` line after
it. Try a clearer full-body image, make the figure bigger on the canvas, or place a skeleton by hand with
**Add Skeleton**.

**The first pose estimation is slow.** ComfyUI downloads the DWPose models (about 350 MB) on first use.

**Bones don't follow the joints.** Select the pose layer in the Layers docker; only the active layer is
updated.

**Match Proportions or Add Skeleton is missing or grayed out.** Both only show up in **Pose** mode, and they
only work when the control layer's layer is a **vector layer** (a pose layer from "From Image" is one).

**The Match Proportions menu says "No other vector layers with a skeleton".** There is no other vector layer
in the document. Create the character skeleton first.

**"Both layers must contain a pose skeleton."** The layer you picked, or the control layer's own layer, has no
joints (for example an empty result from "From Image").

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
