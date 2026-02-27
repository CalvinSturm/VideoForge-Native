"""
Swin2SR HuggingFace Loader — Loads Swin2SR models from HF-format state dicts.

Extracted from model_manager.py. Lazy-imports transformers at call time.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

log = logging.getLogger("videoforge")


class Swin2SRWrapper(nn.Module):
    """
    Wraps a HuggingFace Swin2SRForImageSuperResolution so its forward()
    returns a plain NCHW tensor instead of a dataclass.

    Also handles window padding and output cropping internally,
    so the adapter should be LightweightAdapter (pass-through).
    """

    def __init__(self, hf_model: nn.Module, window_size: int = 8, scale: int = 2) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.window_size = window_size
        self.scale = scale
        self._vf_spandrel = True  # use pass-through adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        # Pad to window_size multiple
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.hf_model(x).reconstruction
        # Crop back to original dims * scale
        return out[:, :, : h * self.scale, : w * self.scale].clamp(0.0, 1.0)


def load_swin2sr_hf(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> Tuple[nn.Module, int]:
    """Load Swin2SR from HuggingFace-format state dict (keys prefixed with 'swin2sr.')."""
    try:
        from transformers import Swin2SRForImageSuperResolution, Swin2SRConfig
    except ImportError:
        raise RuntimeError(
            "transformers package required for HuggingFace Swin2SR models. "
            "Install: pip install transformers"
        )

    # Infer architecture config from state dict
    embed_dim = state_dict[
        "swin2sr.embeddings.patch_embeddings.projection.weight"
    ].shape[0]

    stage_ids: set = set()
    for k in state_dict:
        if "encoder.stages." in k:
            stage_ids.add(int(k.split("encoder.stages.")[1].split(".")[0]))

    depths = []
    for s in sorted(stage_ids):
        layer_ids: set = set()
        for k in state_dict:
            if f"encoder.stages.{s}.layers." in k:
                layer_ids.add(
                    int(k.split(f"encoder.stages.{s}.layers.")[1].split(".")[0])
                )
        depths.append(len(layer_ids))

    # Detect num_heads from logit_scale
    num_heads = 6  # default
    for k in state_dict:
        if "logit_scale" in k:
            num_heads = state_dict[k].shape[0]
            break
    num_heads_list = [num_heads] * len(depths)

    # Detect scale from model key
    key_lower = model_key.lower()
    scale = 4
    for s in (2, 3, 4, 8):
        if f"x{s}" in key_lower:
            scale = s
            break

    # Detect upsampler type
    has_pixelshuffle = any("upsample.upsample" in k for k in state_dict)
    has_lightweight = any("upsample.conv.weight" in k for k in state_dict)
    if has_pixelshuffle:
        upsampler = "pixelshuffle"
    elif has_lightweight:
        upsampler = "pixelshuffledirect"
    else:
        upsampler = "pixelshuffle"

    config = Swin2SRConfig(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads_list,
        window_size=8,
        upscale=scale,
        img_range=1.0,
        upsampler=upsampler,
        num_channels=3,
    )

    hf_model = Swin2SRForImageSuperResolution(config)
    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info(f"Swin2SR HF load: {len(missing)} missing keys")

    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)

    wrapper = Swin2SRWrapper(hf_model, window_size=8, scale=scale)
    log.info(f"Loaded Swin2SR via HuggingFace transformers: "
        f"embed={embed_dim}, depths={depths}, scale={scale}x, upsampler={upsampler}",
    )
    return wrapper, scale
