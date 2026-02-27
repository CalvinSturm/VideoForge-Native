"""
RCAN — Residual Channel Attention Network for Image Super-Resolution.

Extracted from model_manager.py. Contains ChannelAttention, SpatialAttention,
CSAM, RCAB, ResidualGroup, RCAN classes and the build_rcan() builder.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

log = logging.getLogger("videoforge")

_SUPPORTED_SCALES = [2, 3, 4, 8]


class ChannelAttention(nn.Module):
    """Channel Attention with fc1/PReLU/fc2 structure (RCAN+ variant)."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False)
        self.relu1 = nn.PReLU(num_feat // squeeze_factor)
        self.fc2 = nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial Attention with a single 7x7 conv."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = self.conv1(avg_out)
        y = self.sigmoid(y)
        return x * y


class CSAM(nn.Module):
    """Combined Channel + Spatial Attention Module for RCAN."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.ca = ChannelAttention(num_feat, squeeze_factor)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class RCAB(nn.Module):
    """Residual Channel Attention Block with combined CA+SA."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            CSAM(num_feat, squeeze_factor)
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    """Residual Group containing multiple RCABs — matches official RCAN naming."""
    def __init__(self, num_feat: int, num_rcab: int = 20, squeeze_factor: int = 16, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(num_feat, squeeze_factor, res_scale) for _ in range(num_rcab)],
            nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN(nn.Module):
    """
    Residual Channel Attention Network for Image Super-Resolution.
    Architecture matches official implementation for weight compatibility.
    """
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_group: int = 10,
        num_rcab: int = 20,
        squeeze_factor: int = 16,
        scale: int = 4
    ):
        super().__init__()
        if scale not in _SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale {scale}. Valid scales: {_SUPPORTED_SCALES}")
        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.head = nn.Sequential(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body_modules = [ResidualGroup(num_feat, num_rcab, squeeze_factor, res_scale=0.1) for _ in range(num_group)]
        body_modules.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = nn.Sequential(*body_modules)
        upsampler = self._make_upsampler(num_feat, scale)
        self.tail = nn.Sequential(
            upsampler,
            nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        )

    def _make_upsampler(self, num_feat: int, scale: int) -> nn.Sequential:
        layers = []
        if scale == 2:
            layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        elif scale == 4:
            layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(nn.PixelShuffle(2))
        elif scale == 8:
            for _ in range(3):
                layers.append(nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        deep = self.body(shallow)
        deep = deep + shallow
        return self.tail(deep)


def build_rcan(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """
    Build RCAN from local class definition.

    Detects architecture parameters (num_feat, num_group, num_rcab) from the
    state dict keys and tensor shapes so we match the actual checkpoint.
    """
    # ── Detect scale from model key
    scale = 4
    for s in (2, 3, 4, 8):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    # ── Detect num_feat from head conv weight shape
    num_feat = 64
    if "head.0.weight" in state_dict:
        num_feat = state_dict["head.0.weight"].shape[0]

    # ── Detect num_group (residual groups)
    group_indices = set()
    plain_body_indices = set()
    for k in state_dict:
        if k.startswith("body."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                if len(parts) > 2 and parts[2] == "body":
                    group_indices.add(idx)
                else:
                    plain_body_indices.add(idx)

    num_group = len(group_indices) if group_indices else 10

    # ── Detect num_rcab (RCABs per group)
    num_rcab = 20
    if group_indices:
        first_group = min(group_indices)
        rcab_indices = set()
        for k in state_dict:
            prefix = f"body.{first_group}.body."
            if k.startswith(prefix):
                sub = k[len(prefix):]
                sub_idx = sub.split(".")[0]
                if sub_idx.isdigit():
                    rcab_indices.add(int(sub_idx))
        if rcab_indices:
            num_rcab = max(rcab_indices)

    log.info(f"RCAN params: num_feat={num_feat}, num_group={num_group}, "
          f"num_rcab={num_rcab}, scale={scale}")

    model = RCAN(num_feat=num_feat, num_group=num_group,
                 num_rcab=num_rcab, scale=scale)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        log.info(f"RCAN strict load failed: {e}")
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        missing = model_keys - weight_keys
        unexpected = weight_keys - model_keys
        if missing:
            log.info(f"Missing keys ({len(missing)}): {sorted(missing)[:5]}...")
        if unexpected:
            log.info(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")
        model.load_state_dict(state_dict, strict=False)
        log.info("RCAN loaded with strict=False")
    return model
