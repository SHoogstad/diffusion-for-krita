"""H3 Studio still decode: decodes one slice of a sampled H3 latent without the
stock 16 px "lone-token" banding, for the Keep source path of H3 Studio.

Same technique as ComfyUI-MiniMax-H3-Image-Studio's still_decode.py (adapted there
from ComfyUI-Fizgig-H3-Still): repeat the slice into a 5-latent group, decode the
group with the video VAE, and keep decoded frame 3.
"""

import comfy.model_management
import torch

GROUP_SIZE, KEEP = 5, 3


def decode_slice(vae, latent, index):
    fsm = vae.first_stage_model
    if not all(
        hasattr(fsm, n)
        for n in ("_adaptive_decode", "_finalize_pixels", "latents_mean", "latents_std")
    ):
        raise ValueError("Needs the official MiniMax H3 video VAE and a ComfyUI with H3 support.")
    shape = (1, latent.shape[1], GROUP_SIZE, *latent.shape[-2:])
    comfy.model_management.load_models_gpu(
        [vae.patcher],
        memory_required=vae.memory_used_decode(shape, vae.vae_dtype),
        force_full_load=getattr(vae, "disable_offload", False),
    )
    out = []
    with torch.no_grad():
        for b in range(latent.shape[0]):
            z = latent[b : b + 1, :, index : index + 1].to(device=vae.device, dtype=vae.vae_dtype)
            mean = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
            std = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
            raw = fsm._adaptive_decode((z * std + mean).repeat(1, 1, GROUP_SIZE, 1, 1))
            pixels = fsm._finalize_pixels(raw[:, :, KEEP : KEEP + 1])
            out.append(
                pixels[:, :, 0]
                .movedim(1, -1)
                .to(
                    device=comfy.model_management.intermediate_device(),
                    dtype=torch.float32,
                    copy=True,
                )
            )
            del z, mean, std, raw, pixels
    return torch.cat(out)


class H3StillSliceDecode:
    DESCRIPTION = (
        "Decodes one temporal slice of a sampled H3 latent as a still, without the 16 px banding of the "
        "stock decode. slice -1 is the last slice; in a 5-frame Keep source packet that is the edited image."
    )
    CATEGORY = "H3 Studio"
    FUNCTION = "decode"
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "slice": (
                    "INT",
                    {
                        "default": -1,
                        "min": -64,
                        "max": 64,
                        "tooltip": "Temporal latent slice; negative counts from the end.",
                    },
                ),
            }
        }

    def decode(self, samples, vae, slice):
        latent = samples["samples"]
        if latent.is_nested:
            latent = latent.unbind()[0]
        if latent.ndim != 5 or latent.shape[1] != 24:
            raise ValueError("Expects an H3 video latent [B, 24, T, H, W].")
        t = latent.shape[2]
        index = slice if slice >= 0 else t + slice
        if not 0 <= index < t:
            raise ValueError(f"slice {slice} is outside this latent's {t} temporal slices.")
        return (decode_slice(vae, latent, index),)


NODE_CLASS_MAPPINGS = {"H3StillSliceDecode": H3StillSliceDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3StillSliceDecode": "H3 Still Slice Decode (no banding)"}
