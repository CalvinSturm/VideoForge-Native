"""
RealESRGAN — RRDBNet builder for RealESRGAN checkpoints.

Extracted from model_manager.py.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

log = logging.getLogger("videoforge")


def build_realesrgan(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """Build RRDBNet for RealESRGAN checkpoints."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        raise RuntimeError(
            "basicsr package required for RealESRGAN.  "
            "Install: pip install basicsr"
        )

    # Detect num_block from state-dict keys
    body_keys = [k for k in state_dict if k.startswith("body.") and ".rdb" in k]
    if body_keys:
        indices = {int(k.split(".")[1]) for k in body_keys if k.split(".")[1].isdigit()}
        num_block = max(indices) + 1 if indices else 23
    else:
        num_block = 23

    # Detect anime variant (6 blocks, different grow channels)
    is_anime = "anime" in model_key.lower() or num_block == 6
    num_grow_ch = 32 if not is_anime else 32

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
    )
    model.load_state_dict(state_dict, strict=True)
    return model
