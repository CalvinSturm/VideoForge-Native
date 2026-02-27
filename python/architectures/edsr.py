"""
EDSR — Enhanced Deep Residual Network for Image Super-Resolution.

Extracted from model_manager.py. Contains EDSRResBlock, EDSR classes,
remap_edsr_keys() utility, and the build_edsr() builder.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

log = logging.getLogger("videoforge")

_SUPPORTED_SCALES = [2, 3, 4, 8]


class EDSRResBlock(nn.Module):
    """Residual Block without BN — matches official EDSR-PyTorch key format."""
    def __init__(self, num_feat: int = 256, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    """
    Enhanced Deep Residual Network for Image Super-Resolution.
    Architecture matches the official EDSR-PyTorch repo for weight compatibility.
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 256,
        num_block: int = 32,
        res_scale: float = 0.1,
        scale: int = 4
    ):
        super().__init__()
        if scale not in _SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale {scale}. Valid scales: {_SUPPORTED_SCALES}")
        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.head = nn.Sequential(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True))
        body_modules = [EDSRResBlock(num_feat, res_scale) for _ in range(num_block)]
        body_modules.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True))
        self.body = nn.Sequential(*body_modules)
        upsampler = self._make_upsampler(num_feat, scale)
        self.tail = nn.Sequential(
            upsampler,
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)
        )

    def _make_upsampler(self, num_feat: int, scale: int) -> nn.Sequential:
        import math as _math
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(_math.log2(scale))):
                layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True))
                layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1, bias=True))
            layers.append(nn.PixelShuffle(3))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body_feat = self.body(feat)
        feat = feat + body_feat
        return self.tail(feat)


def remap_edsr_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap BasicSR-style EDSR keys (conv_first/conv_after_body/upsample/conv_last)
    to the official EDSR format (head/body/tail) used by our EDSR class.

    Also strips sub_mean/add_mean MeanShift keys.
    """
    has_head = any(k.startswith("head.") for k in state_dict)
    has_tail = any(k.startswith("tail.") for k in state_dict)
    if has_head and has_tail:
        return {k: v for k, v in state_dict.items()
                if not k.startswith("sub_mean.") and not k.startswith("add_mean.")}

    new_dict: Dict[str, torch.Tensor] = {}
    max_body_idx = -1
    for key in state_dict:
        if key.startswith("body.") and ".conv1." in key:
            parts = key.split(".")
            if parts[1].isdigit():
                max_body_idx = max(max_body_idx, int(parts[1]))

    for key, value in state_dict.items():
        if key.startswith("sub_mean.") or key.startswith("add_mean."):
            continue
        if key.startswith("conv_first."):
            new_key = key.replace("conv_first.", "head.0.")
        elif key.startswith("body.") and (".conv1." in key or ".conv2." in key):
            new_key = key.replace(".conv1.", ".body.0.").replace(".conv2.", ".body.2.")
        elif key.startswith("conv_after_body."):
            rest = key[len("conv_after_body."):]
            new_key = f"body.{max_body_idx + 1}.{rest}"
        elif key.startswith("upsample."):
            rest = key[len("upsample."):]
            new_key = f"tail.0.{rest}"
        elif key.startswith("conv_last."):
            rest = key[len("conv_last."):]
            new_key = f"tail.1.{rest}"
        else:
            new_key = key
        new_dict[new_key] = value

    log.info(f"Remapped {len(new_dict)} EDSR keys from BasicSR -> official format")
    return new_dict


def build_edsr(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """
    Build EDSR from local class definition.

    Handles two state-dict formats:
      - Official EDSR-PyTorch (head/body/tail) — loads directly
      - BasicSR (conv_first/conv_after_body/upsample/conv_last) — remapped first
    """
    # ── Remap BasicSR format if needed
    is_basicsr = any(k.startswith("conv_first.") for k in state_dict)
    if is_basicsr:
        log.info("Detected BasicSR EDSR format, remapping to official format")
        state_dict = remap_edsr_keys(state_dict)

    # ── Detect scale from model key
    scale = 4
    for s in (2, 3, 4):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    # ── Detect num_feat from head.0.weight shape
    num_feat = 256
    if "head.0.weight" in state_dict:
        num_feat = state_dict["head.0.weight"].shape[0]

    # ── Detect num_block from body keys
    block_indices = set()
    for k in state_dict:
        if k.startswith("body."):
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit() and parts[2] == "body":
                block_indices.add(int(parts[1]))
    num_block = len(block_indices) if block_indices else 32

    # ── res_scale: large=0.1, baseline=1.0
    res_scale = 0.1 if num_feat >= 128 else 1.0

    log.info(f"EDSR params: num_feat={num_feat}, num_block={num_block}, "
          f"res_scale={res_scale}, scale={scale}")

    model = EDSR(num_feat=num_feat, num_block=num_block,
                 res_scale=res_scale, scale=scale)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        log.info(f"EDSR strict load failed: {e}")
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        missing = model_keys - weight_keys
        unexpected = weight_keys - model_keys
        if missing:
            log.info(f"Missing keys ({len(missing)}): {sorted(missing)[:5]}...")
        if unexpected:
            log.info(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")
        model.load_state_dict(state_dict, strict=False)
        log.info("EDSR loaded with strict=False")
    return model
