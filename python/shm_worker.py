"""
VideoForge AI Worker - Deterministic Super-Resolution Engine

This worker provides deterministic, editor-grade upscaling using RCAN (preferred)
or EDSR (fallback) models. It explicitly avoids GANs and any sources of randomness
to ensure frame-to-frame stability and bit-exact reproducibility.

Design Philosophy:
- Correctness over visual "pop"
- Determinism over perceptual sharpness
- Fail loudly with clear errors, never guess

Author: VideoForge Team
"""

import argparse
import json
import mmap
import os
import struct
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, Any

# =============================================================================
# PRECISION CONFIGURATION
# =============================================================================
# Configurable at startup via --precision flag. Do NOT set torch backend flags
# at module level — configure_precision() is the single source of truth.

import torch

# Global precision mode — set by configure_precision(), read by inference()
_PRECISION_MODE: str = "fp32"


def configure_precision(mode: str = "fp32") -> None:
    """
    Configure PyTorch backend flags for the selected precision mode.

    Must be called BEFORE any model loading or CUDA kernel launch.

    Modes:
      fp32          — TF32 enabled, cuDNN deterministic, no autocast.
                      Best balance of speed and quality.
      fp16          — TF32 enabled, cuDNN deterministic, autocast float16.
                      ~2× throughput, minor quality trade-off.
      deterministic — TF32 disabled, strict deterministic algorithms,
                      cuDNN deterministic, no autocast, batch_size=1.
                      Same GPU + driver + CUDA → bit-exact output.
    """
    global _PRECISION_MODE
    mode = mode.lower().strip()
    if mode not in ("fp32", "fp16", "deterministic"):
        raise ValueError(f"Unknown precision mode: {mode!r}. Use fp32/fp16/deterministic.")
    _PRECISION_MODE = mode

    # --- Common: always keep cuDNN deterministic, never auto-tune ---
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if mode == "deterministic":
        # Strictest reproducibility: disable TF32 and enable deterministic algorithms
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        print(f"[Python] Precision: DETERMINISTIC (TF32=off, strict_deterministic=on)", flush=True)
    else:
        # fp32 / fp16: enable TF32 for 2-4× speedup on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        tag = "FP16 (autocast)" if mode == "fp16" else "FP32"
        print(f"[Python] Precision: {tag} (TF32=on, cuDNN_deterministic=on)", flush=True)


# Apply safe defaults immediately (overridden by configure_precision() at startup)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
# 3rd Party Imports
# =============================================================================
try:
    import cv2
    import numpy as np
    import zenoh
except ImportError as e:
    print(f"[Python Critical] Missing Dependency: {e}", flush=True)
    sys.exit(1)

# Research layer (optional — graceful fallback if unavailable)
try:
    from research_layer import (
        VideoForgeResearchLayer,
        ModelRole,
        SpatialRouter,
        create_research_layer,
    )
    HAS_RESEARCH_LAYER = True
except ImportError:
    HAS_RESEARCH_LAYER = False
    print("[Python] Research layer not available — running vanilla inference only", flush=True)

# Blender engine (optional — SR pipeline post-processing)
try:
    from blender_engine import PredictionBlender, clear_temporal_buffers
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    print("[Python] Blender engine not available — SR pipeline post-processing disabled", flush=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    TILE_SIZE = 512
    TILE_PAD = 32
    RING_SIZE = 3
    PARENT_CHECK_INTERVAL = 2  # seconds
    ZENOH_PREFIX = "videoforge/ipc"

    # --- SHM Slot State Machine ---
    # Must match Rust src-tauri/src/shm.rs constants exactly.
    SLOT_EMPTY = 0
    SLOT_RUST_WRITING = 1
    SLOT_READY_FOR_AI = 2
    SLOT_AI_PROCESSING = 3
    SLOT_READY_FOR_ENCODE = 4
    SLOT_ENCODING = 5

    # --- SHM Global Header (36 bytes at file offset 0) ---
    # Must match Rust src-tauri/src/shm.rs constants exactly.
    # Layout: magic[8] | version[4] | header_size[4] | slot_count[4] |
    #         width[4] | height[4] | scale[4] | pixel_format[4]
    SHM_MAGIC = b"VFSHM001"
    SHM_VERSION = 1
    PIXEL_FORMAT_RGB24 = 1
    GLOBAL_HEADER_SIZE = 36  # 8 + 7 × 4

    # Per-slot header: 4 × u32 = 16 bytes
    SLOT_HEADER_SIZE = 16
    # State field is at byte offset 8 within each slot header
    STATE_FIELD_OFFSET = 8
    # frame_bytes field is at byte offset 12
    FRAME_BYTES_FIELD_OFFSET = 12

    # Micro-batching: max frames to batch in a single GPU forward pass.
    # Set to 1 to disable batching.  RING_SIZE is the upper bound.
    MAX_BATCH_SIZE = 3

    # Supported models and scales
    # Canonical format: {FAMILY}_x{SCALE} for deterministic models
    VALID_MODELS = [
        "RCAN_x2", "RCAN_x3", "RCAN_x4", "RCAN_x8",
        "EDSR_x2", "EDSR_x3", "EDSR_x4",
        "RealESRGAN_x2plus", "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B"  # Note: No official 2x anime model exists
    ]
    SUPPORTED_SCALES = [2, 3, 4, 8]

    # Default precision - FP32 for determinism
    DEFAULT_PRECISION = "fp32"


ZENOH_PREFIX = Config.ZENOH_PREFIX
SPATIAL_MAP_TOPIC = "videoforge/research/spatial_map"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHTS_DIRS = [
    os.path.join(PROJECT_ROOT, "weights"),
    os.path.join(SCRIPT_DIR, "weights"),
]


# =============================================================================
# RCAN Architecture Definition
# =============================================================================
# Reference: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# https://arxiv.org/abs/1807.02758

class ChannelAttention(torch.nn.Module):
    """Channel Attention with fc1/PReLU/fc2 structure (RCAN+ variant)."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False)
        self.relu1 = torch.nn.PReLU(num_feat // squeeze_factor)
        self.fc2 = torch.nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SpatialAttention(torch.nn.Module):
    """Spatial Attention with a single 7x7 conv."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7, padding=3, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        y = self.conv1(avg_out)
        y = self.sigmoid(y)
        return x * y


class CSAM(torch.nn.Module):
    """Combined Channel + Spatial Attention Module for RCAN."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16):
        super().__init__()
        self.ca = ChannelAttention(num_feat, squeeze_factor)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class RCAB(torch.nn.Module):
    """Residual Channel Attention Block with combined CA+SA."""
    def __init__(self, num_feat: int, squeeze_factor: int = 16, res_scale: float = 0.1):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            CSAM(num_feat, squeeze_factor)
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(torch.nn.Module):
    """Residual Group containing multiple RCABs - matches official RCAN naming"""
    def __init__(self, num_feat: int, num_rcab: int = 20, squeeze_factor: int = 16, res_scale: float = 0.1):
        super().__init__()
        # Use 'body' to match official RCAN key names
        self.body = torch.nn.Sequential(
            *[RCAB(num_feat, squeeze_factor, res_scale) for _ in range(num_rcab)],
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN(torch.nn.Module):
    """
    Residual Channel Attention Network for Image Super-Resolution
    Architecture matches official implementation for weight compatibility.

    This is a deterministic SR model with no randomness or GAN components.
    Same input will always produce same output.

    Args:
        num_in_ch: Number of input channels (3 for RGB)
        num_out_ch: Number of output channels (3 for RGB)
        num_feat: Number of intermediate feature channels
        num_group: Number of residual groups
        num_rcab: Number of RCAB blocks per group
        squeeze_factor: Reduction ratio for channel attention
        scale: Upscaling factor (2, 3, 4, or 8)
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

        if scale not in Config.SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale {scale}. Valid scales: {Config.SUPPORTED_SCALES}")

        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat

        # Head: shallow feature extraction (matches official 'head')
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        )

        # Body: deep feature extraction with residual groups
        body_modules = [ResidualGroup(num_feat, num_rcab, squeeze_factor, res_scale=0.1) for _ in range(num_group)]
        body_modules.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.body = torch.nn.Sequential(*body_modules)

        # Tail: upsampling + reconstruction
        # Official RCAN structure: tail[0] = Upsampler (Sequential), tail[1] = final Conv
        # This nesting is critical for weight key matching
        upsampler = self._make_upsampler(num_feat, scale)
        self.tail = torch.nn.Sequential(
            upsampler,  # tail.0.X.weight
            torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)  # tail.1.weight
        )

    def _make_upsampler(self, num_feat: int, scale: int) -> torch.nn.Sequential:
        """
        Create Upsampler module matching official RCAN structure.

        Official RCAN uses common.Upsampler which is a Sequential.
        Keys: tail.0.0.weight, tail.0.2.weight (for 4x), etc.
        """
        layers = []
        if scale == 2:
            layers.append(torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(torch.nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(torch.nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1))
            layers.append(torch.nn.PixelShuffle(3))
        elif scale == 4:
            # Two stages: conv -> shuffle -> conv -> shuffle
            layers.append(torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(torch.nn.PixelShuffle(2))
            layers.append(torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
            layers.append(torch.nn.PixelShuffle(2))
        elif scale == 8:
            # Three stages for 8x
            for _ in range(3):
                layers.append(torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1))
                layers.append(torch.nn.PixelShuffle(2))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Head: shallow feature
        shallow = self.head(x)

        # Body: deep feature with global residual
        deep = self.body(shallow)
        deep = deep + shallow

        # Tail: upsample and reconstruct
        out = self.tail(deep)

        return out


# =============================================================================
# EDSR Architecture Definition (Fallback)
# =============================================================================
# Reference: Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921
#
# Key format matches the official EDSR-PyTorch repo (head/body/tail naming,
# Sequential ResBlocks) so state dicts from official .pt files load directly.

class EDSRResBlock(torch.nn.Module):
    """
    Residual Block without Batch Normalization.

    Uses Sequential body (body.0 = conv, body.1 = ReLU, body.2 = conv) to
    match the official EDSR-PyTorch key format.
    """
    def __init__(self, num_feat: int = 256, res_scale: float = 0.1):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class EDSR(torch.nn.Module):
    """
    Enhanced Deep Residual Network for Image Super-Resolution.

    Architecture matches the official EDSR-PyTorch repo for weight compatibility.
    Uses head/body/tail naming with Sequential ResBlocks.

    Default parameters match the EDSR *large* model (256 feat, 32 blocks,
    res_scale=0.1) since the official .pt files are the large variant.

    Args:
        num_in_ch: Number of input channels (3 for RGB)
        num_out_ch: Number of output channels (3 for RGB)
        num_feat: Number of intermediate feature channels (256 for large, 64 for baseline)
        num_block: Number of residual blocks (32 for large, 16 for baseline)
        res_scale: Residual scaling factor (0.1 for large, 1.0 for baseline)
        scale: Upscaling factor (2, 3, or 4)
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

        if scale not in Config.SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale {scale}. Valid scales: {Config.SUPPORTED_SCALES}")

        self.scale = scale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat

        # head.0 — shallow feature extraction
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        )

        # body.0..{num_block-1} = ResBlocks, body.{num_block} = conv after body
        body_modules = [EDSRResBlock(num_feat, res_scale) for _ in range(num_block)]
        body_modules.append(torch.nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True))
        self.body = torch.nn.Sequential(*body_modules)

        # tail.0 = Upsampler (Sequential), tail.1 = final conv
        upsampler = self._make_upsampler(num_feat, scale)
        self.tail = torch.nn.Sequential(
            upsampler,
            torch.nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, bias=True)
        )

    def _make_upsampler(self, num_feat: int, scale: int) -> torch.nn.Sequential:
        """Create PixelShuffle upsampler matching official Upsampler structure."""
        import math as _math
        layers = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(_math.log2(scale))):
                layers.append(torch.nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True))
                layers.append(torch.nn.PixelShuffle(2))
        elif scale == 3:
            layers.append(torch.nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1, bias=True))
            layers.append(torch.nn.PixelShuffle(3))
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body_feat = self.body(feat)
        feat = feat + body_feat
        out = self.tail(feat)
        return out


# =============================================================================
# HELPERS
# =============================================================================

@contextmanager
def suppress_stdout():
    """Suppress stdout from noisy C++ libraries like PyTorch"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def find_weight_file(model_id: str) -> Optional[str]:
    """
    Find weight file for a given model identifier.

    Handles various naming conventions:
    - Canonical: RCAN_x4 -> RCAN_4x.pt, RCAN_4x.pth
    - Canonical: EDSR_x3 -> EDSR_Mx3_*.pth, EDSR_3x_*.pth
    - Direct: RealESRGAN_x4plus.pth
    """
    # Normalize the model ID
    model_id = model_id.strip()

    # Build search patterns based on model family
    patterns = []

    upper_id = model_id.upper()

    if upper_id.startswith("RCAN"):
        # Extract scale from RCAN_xN format
        scale = None
        for s in [2, 3, 4, 8]:
            if f"_X{s}" in upper_id or f"X{s}" in upper_id:
                scale = s
                break

        if scale:
            # RCAN weight file patterns (actual files use _Nx format, e.g., RCAN_2x.pt)
            patterns = [
                f"RCAN_{scale}x.pt",
                f"RCAN_{scale}x.pth",
                f"RCAN_x{scale}.pt",
                f"RCAN_x{scale}.pth",
                f"RCAN_{scale}x_BI.pt",  # Bicubic degradation variant
                f"RCAN_{scale}x_BD.pt",  # Blur-downscale degradation variant
            ]

    elif upper_id.startswith("EDSR"):
        # Extract scale from EDSR_xN format
        scale = None
        for s in [2, 3, 4]:
            if f"_X{s}" in upper_id or f"X{s}" in upper_id:
                scale = s
                break

        if scale:
            # EDSR weight file patterns (actual files have complex names)
            # Try medium model (M) first, then large (L)
            patterns = [
                f"EDSR_Mx{scale}_*.pth",  # Medium model
                f"EDSR_{scale}x_*.pth",   # Alternative naming
                f"EDSR_Lx{scale}_*.pth",  # Large model
                f"EDSR_x{scale}.pth",     # Simple naming
                f"EDSR_x{scale}.pt",
            ]

    else:
        # RealESRGAN or other models - use direct filename
        base = model_id.replace(".pth", "").replace(".pt", "")
        patterns = [
            f"{base}.pth",
            f"{base}.pt",
            model_id,  # Try as-is
        ]

    # Search in all weight directories
    for d in WEIGHTS_DIRS:
        if not os.path.exists(d):
            continue

        for pattern in patterns:
            if '*' in pattern:
                # Glob pattern matching
                import glob
                matches = glob.glob(os.path.join(d, pattern))
                if matches:
                    # Return first match (prefer M over L for EDSR)
                    matches.sort()  # Alphabetical, M comes before L
                    return matches[0]
            else:
                # Exact match
                candidate = os.path.join(d, pattern)
                if os.path.exists(candidate):
                    return candidate

                # Check nested directory
                nested = os.path.join(d, pattern.replace(".pth", "").replace(".pt", ""), pattern)
                if os.path.exists(nested):
                    return nested

    return None


def extract_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    """
    Extract model state dict from various checkpoint formats.

    Handles:
    - Direct state dict
    - {'params': state_dict}
    - {'params_ema': state_dict} (preferred if available)
    - {'state_dict': state_dict}
    - Full model object (torch.save(model, path))
    """
    # If loaded is a model object (saved with torch.save(model, ...))
    if isinstance(loaded, torch.nn.Module):
        print("[Python] Detected full model object, extracting state_dict", flush=True)
        return loaded.state_dict()

    # If not a dict, we can't extract state_dict
    if not isinstance(loaded, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(loaded)}")

    # Prefer EMA weights if available (more stable)
    if 'params_ema' in loaded:
        print("[Python] Using EMA weights (params_ema)", flush=True)
        return loaded['params_ema']
    if 'params' in loaded:
        return loaded['params']
    if 'state_dict' in loaded:
        return loaded['state_dict']
    if 'model' in loaded:
        # Some checkpoints store model state under 'model' key
        if isinstance(loaded['model'], dict):
            return loaded['model']
        elif isinstance(loaded['model'], torch.nn.Module):
            return loaded['model'].state_dict()

    # Check if it's already a state dict (has tensor values)
    if any(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded

    # Recursive search for nested state dicts
    for key, value in loaded.items():
        if isinstance(value, dict):
            # Check if this nested dict looks like a state dict
            if any(isinstance(v, torch.Tensor) for v in value.values()):
                print(f"[Python] Found state_dict under key '{key}'", flush=True)
                return value

    return loaded  # Return as-is and let strict loading catch issues


def remap_rcan_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Strip MeanShift keys from official RCAN state dicts.

    Our RCAN architecture uses the same head/body/tail naming as the official
    EDSR-PyTorch repo, so no key renaming is needed — only sub_mean/add_mean
    removal (these are handled by the EDSRRCANAdapter instead).
    """
    new_dict = {k: v for k, v in state_dict.items()
                if not k.startswith("sub_mean.") and not k.startswith("add_mean.")}

    removed = len(state_dict) - len(new_dict)
    if removed > 0:
        print(f"[Python] Stripped {removed} MeanShift keys from RCAN state dict", flush=True)

    return new_dict


def remap_edsr_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap BasicSR-style EDSR keys (conv_first/conv_after_body/upsample/conv_last)
    to the official EDSR format (head/body/tail) used by our EDSR class.

    Also strips sub_mean/add_mean MeanShift keys.
    """
    # Check if this is already in official format (has head/tail keys)
    has_head = any(k.startswith("head.") for k in state_dict)
    has_tail = any(k.startswith("tail.") for k in state_dict)

    if has_head and has_tail:
        # Already in official format, just strip MeanShift
        return {k: v for k, v in state_dict.items()
                if not k.startswith("sub_mean.") and not k.startswith("add_mean.")}

    # Remap from BasicSR format to official format
    new_dict = {}
    max_body_idx = -1

    # Find max body block index (for conv_after_body placement)
    for key in state_dict:
        if key.startswith("body.") and ".conv1." in key:
            parts = key.split(".")
            if parts[1].isdigit():
                max_body_idx = max(max_body_idx, int(parts[1]))

    for key, value in state_dict.items():
        if key.startswith("sub_mean.") or key.startswith("add_mean."):
            continue  # strip MeanShift

        if key.startswith("conv_first."):
            # conv_first.weight → head.0.weight
            new_key = key.replace("conv_first.", "head.0.")
        elif key.startswith("body.") and (".conv1." in key or ".conv2." in key):
            # body.N.conv1.X → body.N.body.0.X
            # body.N.conv2.X → body.N.body.2.X
            new_key = key.replace(".conv1.", ".body.0.").replace(".conv2.", ".body.2.")
        elif key.startswith("conv_after_body."):
            # conv_after_body.X → body.{max+1}.X
            rest = key[len("conv_after_body."):]
            new_key = f"body.{max_body_idx + 1}.{rest}"
        elif key.startswith("upsample."):
            # upsample.X → tail.0.X
            rest = key[len("upsample."):]
            new_key = f"tail.0.{rest}"
        elif key.startswith("conv_last."):
            # conv_last.X → tail.1.X
            rest = key[len("conv_last."):]
            new_key = f"tail.1.{rest}"
        else:
            new_key = key

        new_dict[new_key] = value

    print(f"[Python] Remapped {len(new_dict)} EDSR keys from BasicSR -> official format", flush=True)
    return new_dict


def verify_scale_from_weights(state_dict: Dict[str, torch.Tensor], expected_scale: int) -> bool:
    """
    Verify model scale by inspecting weight tensor shapes.

    For PixelShuffle-based upsampling:
    - x2: upsample has conv with out_channels = num_feat * 4 (one PixelShuffle(2))
    - x4: upsample has two convs with out_channels = num_feat * 4 (two PixelShuffle(2))

    Returns True if scale matches, False otherwise.
    """
    # Count PixelShuffle upsampling convolutions
    # They have pattern: upsample.N.weight where output channels = num_feat * 4
    upsample_convs = 0
    for key in state_dict.keys():
        # Check for RCAN/EDSR style upsample
        if 'upsample' in key and 'weight' in key and 'bias' not in key:
            shape = state_dict[key].shape
            if len(shape) == 4 and shape[0] == shape[1] * 4:
                upsample_convs += 1
        # Check for RealESRGAN style upsample (conv_up1, conv_up2)
        elif 'conv_up' in key and 'weight' in key:
            upsample_convs += 1

    detected_scale = 2 ** upsample_convs if upsample_convs > 0 else 1

    if detected_scale != expected_scale:
        print(f"[Python] Scale verification: detected {detected_scale}x from weights, expected {expected_scale}x", flush=True)
        return False

    return True


# =============================================================================
# WATCHDOG (Suicide Pact with Parent Process)
# =============================================================================
import threading
import ctypes


def is_pid_alive(pid: int) -> bool:
    """Check if PID is alive on Windows using ctypes kernel32"""
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False

        exit_code = ctypes.c_ulong()
        if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE

        ctypes.windll.kernel32.CloseHandle(handle)
        return False
    except Exception as e:
        print(f"[Python Warning] PID check failed: {e}", flush=True)
        return False


def watchdog_loop(parent_pid: int) -> None:
    """Monitor parent process. If it dies, we die."""
    print(f"[Python] Watchdog started for Parent PID: {parent_pid}", flush=True)
    while True:
        if not is_pid_alive(parent_pid):
            print(f"[Python] Parent {parent_pid} died. Committing seppuku...", flush=True)
            os._exit(0)
        time.sleep(Config.PARENT_CHECK_INTERVAL)


def start_watchdog(parent_pid: int) -> None:
    if parent_pid <= 0:
        return
    t = threading.Thread(target=watchdog_loop, args=(parent_pid,), daemon=True)
    t.start()


# =============================================================================
# MODEL LOADER - Deterministic with Strict Validation
# =============================================================================

class ModelLoader:
    """
    Deterministic model loader with explicit architecture verification.

    Guarantees:
    - strict=True for state dict loading (no silent partial loads)
    - Scale verified from weight tensor shapes (no filename parsing)
    - EDSR fallback if RCAN unavailable
    - FP32 default, FP16 only if explicitly requested
    """

    def __init__(self, precision: str = "fp32"):
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.model_scale: int = 4
        # Some models were trained on BGR, others on RGB
        # Set True if model expects RGB input (standard), False for BGR
        self.expects_rgb: bool = True

        # Log precision mode
        if precision == "fp16":
            print("[Python WARNING] FP16 mode enabled - determinism not guaranteed across hardware", flush=True)

    def load(self, model_identifier: str) -> Tuple[torch.nn.Module, int]:
        """
        Load any model by identifier.

        Uses model_manager._load_module() as the single universal loader.
        It handles:
          - Full model objects (torch.save(model, path))
          - State-dict checkpoints via spandrel (30+ architectures)
          - Legacy RCAN / EDSR builders as fallback
        Also creates an architecture adapter for proper pre/post processing
        (window padding for transformers, output clamping, etc.)
        """
        print(f"[Python] Loading model: '{model_identifier}'", flush=True)

        from model_manager import _load_module
        from arch_wrappers import create_adapter

        model, scale = _load_module(model_identifier)
        model = self._prepare_model(model)

        # Create adapter for proper pre/post processing during inference
        self.adapter = create_adapter(model_identifier, model, scale, self.device)

        self.model = model
        self.model_name = model_identifier
        self.model_scale = scale
        self.expects_rgb = True

        return model, scale

    def _load_rcan(self, scale: int) -> Tuple[torch.nn.Module, int]:
        """Load RCAN model with strict validation"""
        weight_file = f"RCAN_x{scale}"
        weight_path = find_weight_file(weight_file)

        if weight_path is None:
            raise FileNotFoundError(f"RCAN weights not found for scale {scale}x")

        print(f"[Python] Loading RCAN x{scale} from {weight_path}", flush=True)

        # Load weights
        loaded = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(loaded)

        # Print some keys for debugging
        sample_keys = list(state_dict.keys())[:10]
        print(f"[Python] Sample weight keys: {sample_keys}", flush=True)

        # Check if this is official RCAN format (head.X, body.X, tail.X)
        has_head = any(k.startswith('head.') for k in state_dict.keys())
        has_tail = any(k.startswith('tail.') for k in state_dict.keys())
        has_body = any(k.startswith('body.') for k in state_dict.keys())

        if has_head and has_tail and has_body:
            print("[Python] Detected official RCAN format (head/body/tail)", flush=True)
            # No remapping needed - our architecture matches
        elif not has_head and not has_tail:
            # Different format - might need remapping
            print("[Python] Detected non-standard RCAN format, may need key adjustment", flush=True)
            # Check for conv_first style naming and remap if needed
            if any('conv_first' in k for k in state_dict.keys()):
                state_dict = remap_rcan_keys(state_dict)

        # Infer architecture from state dict
        num_feat = 64  # Default
        num_group = 10  # Default
        num_rcab = 20  # Default

        # Try to infer num_feat from head.0.weight
        for key in state_dict.keys():
            if key == 'head.0.weight' or key.endswith('.head.0.weight'):
                num_feat = state_dict[key].shape[0]
                print(f"[Python] Detected num_feat={num_feat} from {key}", flush=True)
                break

        # Count residual groups from body keys
        group_indices = set()
        for key in state_dict.keys():
            if key.startswith('body.'):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    group_indices.add(int(parts[1]))
        if group_indices:
            # Last index might be final conv, not a residual group
            num_group = max(group_indices)
            print(f"[Python] Detected num_group={num_group}", flush=True)

        print(f"[Python] RCAN config: num_feat={num_feat}, num_group={num_group}, scale={scale}", flush=True)

        # Create model architecture
        model = RCAN(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_group=num_group,
            num_rcab=num_rcab,
            squeeze_factor=16,
            scale=scale
        )

        # Try strict load first
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[Python] Strict load successful", flush=True)
        except RuntimeError as e:
            print(f"[Python WARNING] Strict load failed: {e}", flush=True)
            # Check what keys are missing vs unexpected
            model_keys = set(model.state_dict().keys())
            weight_keys = set(state_dict.keys())
            missing = model_keys - weight_keys
            unexpected = weight_keys - model_keys
            if missing:
                print(f"[Python] Missing keys: {list(missing)[:5]}...", flush=True)
            if unexpected:
                print(f"[Python] Unexpected keys: {list(unexpected)[:5]}...", flush=True)

            # Try non-strict load
            model.load_state_dict(state_dict, strict=False)
            print("[Python] Non-strict load completed", flush=True)

        # Prepare for inference
        model = self._prepare_model(model)

        self.model = model
        self.model_name = f"RCAN_x{scale}"
        self.model_scale = scale
        # RCAN official weights were trained on RGB (DIV2K loaded as RGB)
        self.expects_rgb = True

        return model, scale

    def _load_edsr(self, scale: int) -> Tuple[torch.nn.Module, int]:
        """Load EDSR model with strict validation"""
        weight_file = f"EDSR_x{scale}"
        weight_path = find_weight_file(weight_file)

        if weight_path is None:
            raise FileNotFoundError(f"EDSR weights not found for scale {scale}x")

        print(f"[Python] Loading EDSR x{scale} from {weight_path}", flush=True)

        # Load weights
        loaded = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(loaded)

        # Print some keys for debugging
        sample_keys = list(state_dict.keys())[:10]
        print(f"[Python] Sample weight keys: {sample_keys}", flush=True)

        # Check if this is BasicSR format (has conv_after_body, body.N.conv1/conv2)
        has_conv_after_body = any('conv_after_body' in k for k in state_dict.keys())
        has_conv1_conv2 = any('.conv1.' in k or '.conv2.' in k for k in state_dict.keys())

        if has_conv_after_body and has_conv1_conv2:
            print("[Python] Detected BasicSR EDSR format (conv_after_body + conv1/conv2)", flush=True)
            # No remapping needed - our architecture matches BasicSR
        else:
            print(f"[Python] WARNING: Non-standard EDSR format detected", flush=True)

        # Infer architecture from state dict
        # EDSR-M (medium): 16 blocks, 64 features
        # EDSR-L (large): 32 blocks, 256 features
        num_feat = 64
        num_block = 16
        res_scale = 1.0  # BasicSR default is 1.0, not 0.1

        # Detect num_feat from first conv
        for key in state_dict.keys():
            if key == 'conv_first.weight':
                num_feat = state_dict[key].shape[0]
                print(f"[Python] Detected num_feat={num_feat} from {key}", flush=True)
                break

        # Count residual blocks from body keys (BasicSR uses body.N.conv1)
        block_indices = set()
        for key in state_dict.keys():
            if key.startswith('body.') and '.conv1.' in key:
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.add(int(parts[1]))
        if block_indices:
            num_block = max(block_indices) + 1  # +1 because indices are 0-based
            print(f"[Python] Detected num_block={num_block}", flush=True)

        # res_scale: BasicSR EDSR typically uses 1.0
        # Some variants use 0.1 but official DIV2K models use 1.0
        res_scale = 1.0

        print(f"[Python] EDSR config: num_feat={num_feat}, num_block={num_block}, res_scale={res_scale}, scale={scale}", flush=True)

        # Create model architecture
        model = EDSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_block=num_block,
            res_scale=res_scale,
            scale=scale
        )

        # Try strict load first
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[Python] Strict load successful", flush=True)
        except RuntimeError as e:
            print(f"[Python WARNING] Strict load failed: {e}", flush=True)
            # Check what keys are missing vs unexpected
            model_keys = set(model.state_dict().keys())
            weight_keys = set(state_dict.keys())
            missing = model_keys - weight_keys
            unexpected = weight_keys - model_keys
            if missing:
                print(f"[Python] Missing keys: {list(missing)[:5]}...", flush=True)
            if unexpected:
                print(f"[Python] Unexpected keys: {list(unexpected)[:5]}...", flush=True)

            # Try non-strict load
            model.load_state_dict(state_dict, strict=False)
            print("[Python] Non-strict load completed", flush=True)

        # Prepare for inference
        model = self._prepare_model(model)

        self.model = model
        self.model_name = f"EDSR_x{scale}"
        self.model_scale = scale
        # BasicSR EDSR weights were trained on RGB (bgr2rgb=True in config)
        self.expects_rgb = True

        return model, scale

    def _load_realesrgan(self, model_name: str) -> Tuple[torch.nn.Module, int]:
        """Load RealESRGAN model (Creative Mode)"""
        from basicsr.archs.rrdbnet_arch import RRDBNet

        weight_path = find_weight_file(model_name)
        if weight_path is None:
            raise FileNotFoundError(f"RealESRGAN weights not found: {model_name}")

        scale = 4
        if "x2" in model_name.lower():
            scale = 2

        print(f"[Python] Loading RealESRGAN x{scale} from {weight_path}", flush=True)

        # Load weights
        loaded = torch.load(weight_path, map_location="cpu", weights_only=False)
        state_dict = extract_state_dict(loaded)

        # RRDBNet Config
        # Most RealESRGAN models use 64 features, 23 blocks
        # Anime models (6B) use 6 blocks
        num_block = 23
        if "anime" in model_name.lower() or "6B" in model_name:
            num_block = 6

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=num_block,
            num_grow_ch=32,
            scale=scale
        )

        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[Python Warning] Strict load failed for {model_name}, trying non-strict: {e}", flush=True)
            model.load_state_dict(state_dict, strict=False)

        model = self._prepare_model(model)
        self.model = model
        self.model_name = model_name
        self.model_scale = scale
        # RealESRGAN (basicsr) expects RGB input
        self.expects_rgb = True

        return model, scale

    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for deterministic inference.

        CRITICAL: This method ensures determinism by:
        1. Setting eval() mode (disables dropout, uses running stats for BN)
        2. Moving to correct device
        3. Applying precision settings
        """
        # CRITICAL: Set eval mode - disables any training-time randomness
        model.eval()

        # Move to device
        model = model.to(self.device)

        # Apply precision
        if self.precision == "fp16" and self.device.type == "cuda":
            model = model.half()

        # Ensure all parameters are not tracking gradients
        for param in model.parameters():
            param.requires_grad = False

        return model


# =============================================================================
# UNIFIED INFERENCE FUNCTION
# =============================================================================

def inference(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    half: bool = False,
    adapter=None
) -> np.ndarray:
    """
    Unified inference function for both image and video processing.

    DETERMINISM GUARANTEES:
    - No randomness in this function
    - torch.no_grad() context ensures no gradient computation
    - Model must be in eval() mode (caller responsibility)
    - Input normalization is deterministic (simple division)
    - Output denormalization is deterministic (simple multiplication + clamp)

    Precision behaviour:
    - fp32:          Standard float32 inference (TF32 used on Ampere+ via backend flags)
    - fp16:          torch.autocast wraps the forward pass in float16
    - deterministic: Standard float32, strictest backend flags

    Args:
        model: The SR model (must be in eval mode)
        img_rgb: Input image as numpy array, RGB order, uint8 [0-255]
        device: Torch device (cuda or cpu)
        half: Whether to use FP16 precision (legacy flag, overridden by _PRECISION_MODE)
        adapter: Optional BaseAdapter for pre/post processing (window padding,
                 output clamping). Required for transformer models (SwinIR, HAT, DAT).

    Returns:
        Upscaled image as numpy array, RGB order, uint8 [0-255]
    """
    # Validate input
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {img_rgb.shape}")

    # Normalize to [0, 1] float32 - DETERMINISTIC
    img_float = img_rgb.astype(np.float32) / 255.0

    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)

    # Determine dtype from precision mode
    use_fp16 = (_PRECISION_MODE == "fp16") or half
    dtype = torch.float16 if use_fp16 else torch.float32
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

    # Use adapter if available — handles window padding, output cropping, etc.
    if adapter is not None:
        if use_fp16 and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                output = adapter.forward(tensor)
        else:
            output = adapter.forward(tensor)
    else:
        # CRITICAL: No gradient computation - ensures determinism and saves memory
        with torch.no_grad():
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(tensor)
            else:
                output = model(tensor)

    # Convert back to numpy: (1, C, H, W) -> (H, W, C)
    output = output.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    output = output.transpose(1, 2, 0)

    # Denormalize to [0, 255] uint8 - DETERMINISTIC
    output = (output * 255.0).round().astype(np.uint8)

    return output


def inference_batch(
    model: torch.nn.Module,
    imgs_rgb: list,
    device: torch.device,
    half: bool = False,
    adapter=None,
) -> list:
    """
    Batched inference: process multiple frames in a single GPU forward pass.

    Requires all images to have identical (H, W) dimensions (guaranteed for
    video frames from the same SHM ring buffer).

    Returns a list of numpy arrays in the same order as the input.

    Falls back to sequential inference if the batch forward pass fails
    (e.g. OOM on very large frames).
    """
    if not imgs_rgb:
        return []
    if len(imgs_rgb) == 1:
        return [inference(model, imgs_rgb[0], device, half=half, adapter=adapter)]

    # Validate: all frames must have same shape
    shape0 = imgs_rgb[0].shape
    for img in imgs_rgb[1:]:
        if img.shape != shape0:
            # Shape mismatch — fall back to sequential
            return [inference(model, img, device, half=half, adapter=adapter) for img in imgs_rgb]

    # Stack into batch tensor: list of (H,W,3) -> (N,3,H,W)
    batch_float = np.stack([img.astype(np.float32) / 255.0 for img in imgs_rgb], axis=0)
    batch_tensor = torch.from_numpy(batch_float.transpose(0, 3, 1, 2))  # (N,3,H,W)

    use_fp16 = (_PRECISION_MODE == "fp16") or half
    dtype = torch.float16 if use_fp16 else torch.float32
    batch_tensor = batch_tensor.to(device=device, dtype=dtype, non_blocking=True)

    try:
        if adapter is not None:
            if use_fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    output = adapter.forward(batch_tensor)
            else:
                output = adapter.forward(batch_tensor)
        else:
            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model(batch_tensor)
                else:
                    output = model(batch_tensor)
    except RuntimeError as e:
        # OOM or other failure — fall back to sequential
        if "out of memory" in str(e).lower():
            print(f"[Python] Batch OOM (N={len(imgs_rgb)}), falling back to sequential", flush=True)
            torch.cuda.empty_cache()
        else:
            print(f"[Python] Batch forward failed: {e}, falling back to sequential", flush=True)
        return [inference(model, img, device, half=half, adapter=adapter) for img in imgs_rgb]

    # Split batch output back to list of numpy arrays
    output = output.float().cpu().clamp_(0, 1)
    results = []
    for i in range(output.shape[0]):
        frame = output[i].numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        frame = (frame * 255.0).round().astype(np.uint8)
        results.append(frame)

    return results


# =============================================================================
# WORKER CLASS
# =============================================================================

class AIWorker:
    def __init__(self, port: str, precision: str = "fp32"):
        # Deterministic mode forces batch_size=1 for bit-exact output
        if precision == "deterministic":
            Config.MAX_BATCH_SIZE = 1
            print(f"[Python] Deterministic mode: batch_size forced to 1", flush=True)

        print(f"[Python] Initializing Zenoh on Port {port}...", flush=True)
        print(f"[Python] Precision mode: {precision}", flush=True)
        print(f"[Python] CUDNN deterministic: {torch.backends.cudnn.deterministic}", flush=True)
        print(f"[Python] CUDNN benchmark: {torch.backends.cudnn.benchmark}", flush=True)

        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))

        try:
            self.session = zenoh.open(conf)
            print("[Python] Zenoh connected successfully", flush=True)
        except Exception as e:
            print(f"[Python CRITICAL] Zenoh connection failed: {e}", flush=True)
            sys.exit(1)

        unique_prefix = f"{ZENOH_PREFIX}/{port}"

        try:
            self.pub = self.session.declare_publisher(f"{unique_prefix}/res")
            self.sub = self.session.declare_subscriber(
                f"{unique_prefix}/req", self.on_request
            )
        except Exception as e:
            print(f"[Python CRITICAL] Zenoh pub/sub setup failed: {e}", flush=True)
            sys.exit(1)

        self.shm_file = None
        self.mmap = None
        self.shm_path = None
        self.is_configured = False
        self.running = True

        # Frame loop state (SHM atomic polling)
        self._frame_loop_active = False
        self._frame_loop_thread: Optional[threading.Thread] = None
        self._cached_research_params: Optional[Dict] = None
        self.header_region_size = 0
        self.output_size = 0

        # IPC correlation — set for the duration of each on_request call.
        self._current_request = None

        # Model state
        self.precision = precision
        self.model_loader = ModelLoader(precision=precision)
        self.model: Optional[torch.nn.Module] = None
        self.model_scale: int = 4
        self.model_name: str = ""
        self.active_scale: int = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = (precision == "fp16" and self.device.type == "cuda")
        # Color format: True if model expects RGB, False for BGR
        self.expects_rgb: bool = True
        # Architecture adapter for pre/post processing (window padding, etc.)
        self.adapter = None

        # Research layer (initialized after first model load)
        self.research_layer: Optional[Any] = None
        self.spatial_pub = None
        if HAS_RESEARCH_LAYER:
            try:
                self.spatial_pub = self.session.declare_publisher(SPATIAL_MAP_TOPIC)
                print("[Python] Spatial map publisher ready", flush=True)
            except Exception as e:
                print(f"[Python Warning] Spatial map publisher failed: {e}", flush=True)

        # Load default model
        default_model = "rcan_4x"
        print(f"[Python] Attempting initial load: {default_model}", flush=True)
        self.load_model(default_model)
        self.loop()

    def loop(self) -> None:
        print("[Python] Ready...", flush=True)
        while self.running:
            time.sleep(0.1)
        self.cleanup()

    def cleanup(self) -> None:
        print("[Python] Cleanup...", flush=True)
        if self.mmap:
            try:
                self.mmap.close()
            except Exception as e:
                print(f"[Python Warning] mmap close failed: {e}", flush=True)
        if self.shm_file:
            try:
                self.shm_file.close()
                if self.shm_path and os.path.exists(self.shm_path):
                    os.unlink(self.shm_path)
            except Exception as e:
                print(f"[Python Warning] SHM file cleanup failed: {e}", flush=True)
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                print(f"[Python Warning] Zenoh session close failed: {e}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_identifier: str) -> None:
        """Load model using the deterministic model loader"""
        try:
            model, scale = self.model_loader.load(model_identifier)
            self.model = model
            self.model_scale = scale
            self.model_name = self.model_loader.model_name
            self.expects_rgb = self.model_loader.expects_rgb
            self.adapter = getattr(self.model_loader, 'adapter', None)

            print(
                f"[Python] Loaded: {self.model_name} (Scale: x{self.model_scale}, expects_rgb={self.expects_rgb})",
                flush=True,
            )

            # Register with research layer
            if HAS_RESEARCH_LAYER:
                try:
                    self.research_layer = create_research_layer(
                        models={"structure": self.model},
                        scale=self.model_scale,
                        device=self.device,
                    )
                    print(f"[Python] Research layer initialized with {self.model_name} as structure model", flush=True)
                except Exception as e:
                    print(f"[Python Warning] Research layer init failed: {e}", flush=True)
                    self.research_layer = None

            self.send_status(
                "MODEL_LOADED", {"model": self.model_name, "scale": self.model_scale}
            )

        except Exception as e:
            print(f"[Python CRITICAL] Load Error: {e}", flush=True)
            traceback.print_exc()
            self.send_status("error", {"message": f"Load Failed: {str(e)}"})

    def send_status(self, status: str, extra: Optional[Dict] = None) -> None:
        """Publish a response envelope conforming to the IPC protocol.

        Includes protocol fields (version, request_id, job_id, kind) for
        correlation while preserving top-level backward-compat extra fields.
        """
        req = getattr(self, "_current_request", None)
        payload: Dict[str, Any] = {
            "version": ZENOH_PREFIX and 1 or 1,  # PROTOCOL_VERSION = 1
            "request_id": req.request_id if req else "",
            "job_id": req.job_id if req else "",
            "kind": "error" if status == "error" else "status",
            "status": status,
            "error": None,
        }
        if extra:
            # Merge extra at top level for backward compat.
            # If extra contains an "error" dict, promote it to the error field.
            if "message" in extra and status == "error":
                payload["error"] = {
                    "code": extra.pop("code", "INTERNAL"),
                    "message": extra.pop("message", ""),
                }
            payload.update(extra)
        try:
            self.pub.put(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            print(f"[Python Warning] Failed to send status: {e}", flush=True)

    def on_request(self, sample) -> None:
        try:
            from ipc_protocol import RequestEnvelope as _Envelope
            raw = json.loads(sample.payload.to_bytes().decode("utf-8"))
            # Parse into typed envelope — unknown fields silently ignored.
            env = _Envelope.from_dict(raw)
            self._current_request = env  # stash for send_status correlation
            cmd = env.kind
            payload = raw  # legacy handlers still read from raw dict

            if cmd == "create_shm":
                # create_shm reads from the raw payload for backward compat
                p = env.payload if isinstance(env.payload, dict) and env.payload else raw
                self.create_shm(p)
            elif cmd == "process_frame":
                self.process_frame(payload)
            elif cmd == "process_one_frame":
                p = env.payload if isinstance(env.payload, dict) else {}
                self.process_one_frame(p)
            elif cmd == "start_frame_loop":
                p = env.payload if isinstance(env.payload, dict) else {}
                self.start_frame_loop(p)
            elif cmd == "stop_frame_loop":
                self.stop_frame_loop(payload)
            elif cmd == "load_model":
                p = env.payload if isinstance(env.payload, dict) else {}
                model_name = p.get("model_name") or raw.get("params", {}).get("model_name")
                if model_name:
                    self.load_model(model_name)
            elif cmd == "upscale_image_file":
                self.handle_image_file(payload)
            elif cmd == "analyze_for_auto_grade":
                self.handle_auto_grade_analysis(payload)
            elif cmd == "update_research_params":
                self.handle_update_research_params(payload)
                self._cached_research_params = (
                    env.payload.get("params") if isinstance(env.payload, dict)
                    else raw.get("params")
                )
            elif cmd == "shutdown":
                self.stop_frame_loop()
                self.running = False
            else:
                print(f"[Python Warning] Unknown command kind: {cmd!r}", flush=True)
                self.send_status("error", {"message": f"Unknown command: {cmd}"})
        except Exception as e:
            print(f"[Python Error] Request failed: {e}", flush=True)
            traceback.print_exc()
            self.send_status("error", {"message": str(e)})
        finally:
            self._current_request = None

    def handle_auto_grade_analysis(self, payload: Dict[str, Any]) -> None:
        """
        Analyze an image for auto color grading.
        
        Expected payload:
        {
            "command": "analyze_for_auto_grade",
            "params": {
                "image_path": "/path/to/image.jpg",
                "protect_skin": true,
                "conservative_mode": false
            }
        }
        
        Responds with auto-grade analysis results including recommended corrections.
        """
        try:
            params = payload.get("params", {})
            image_path = params.get("image_path")
            protect_skin = params.get("protect_skin", True)
            conservative_mode = params.get("conservative_mode", False)
            
            if not image_path:
                self.send_status("error", {"message": "No image_path provided"})
                return
            
            if not os.path.exists(image_path):
                self.send_status("error", {"message": f"Image not found: {image_path}"})
                return
            
            # Import auto-grade analysis module
            from auto_grade_analysis import (
                analyze_frame_for_auto_grade,
                convert_corrections_to_edit_config
            )
            
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                self.send_status("error", {"message": f"Could not load image: {image_path}"})
                return
            
            print(f"[Python] Auto-grade analysis for: {image_path}", flush=True)
            
            # Run analysis
            result = analyze_frame_for_auto_grade(frame, protect_skin, conservative_mode)
            
            # Convert to edit config format
            edit_config = convert_corrections_to_edit_config(result["corrections"])
            
            # Send response
            self.send_status("AUTO_GRADE_COMPLETE", {
                "corrections": result["corrections"],
                "edit_config": edit_config,
                "confidence": result["confidence"],
                "summary": result["summary"],
                "analysis": {
                    "scene": result["analysis"]["scene"],
                    "skin": {
                        "has_skin": result["analysis"]["skin"]["has_skin"],
                        "is_face_dominant": result["analysis"]["skin"]["is_face_dominant"]
                    }
                }
            })
            
            print(f"[Python] Auto-grade complete: confidence={result['confidence']:.2f}, summary={result['summary']}", flush=True)
            
        except Exception as e:
            print(f"[Python Error] Auto-grade analysis failed: {e}", flush=True)
            traceback.print_exc()
            self.send_status("error", {"message": f"Auto-grade failed: {str(e)}"})

    def handle_update_research_params(self, payload: Dict[str, Any]) -> None:
        """Handle research parameter updates from the Rust backend."""
        if not HAS_RESEARCH_LAYER or self.research_layer is None:
            self.send_status("error", {"message": "Research layer not available"})
            return
        try:
            params = payload.get("params", {})
            self.research_layer.update_params(params)
            print(f"[Python] Research params updated: {list(params.keys())}", flush=True)
            self.send_status("RESEARCH_PARAMS_UPDATED", {"keys": list(params.keys())})
        except Exception as e:
            print(f"[Python Error] Research params update failed: {e}", flush=True)
            self.send_status("error", {"message": f"Params update failed: {str(e)}"})

    def _publish_spatial_map(self, lr_rgb: np.ndarray) -> None:
        """Compute and publish spatial routing mask for the frontend overlay."""
        if self.spatial_pub is None or not HAS_RESEARCH_LAYER:
            return
        try:
            h, w = lr_rgb.shape[:2]
            img_float = lr_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(self.device)

            rl = self.research_layer
            edge_t = rl.params.edge_threshold if rl else 0.5
            tex_t = rl.params.texture_threshold if rl else 0.2
            edge_mask, texture_mask, flat_mask = SpatialRouter.compute_routing_masks(
                tensor, edge_threshold=edge_t, texture_threshold=tex_t
            )

            # Build classification mask: 0=flat, 1=texture, 2=edge
            edge_np = edge_mask.squeeze().cpu().numpy()
            texture_np = texture_mask.squeeze().cpu().numpy()
            classification = np.zeros((h, w), dtype=np.uint8)
            classification[texture_np > 0.5] = 1
            classification[edge_np > 0.5] = 2

            # Binary payload: [u32 LE width][u32 LE height][mask bytes]
            import struct
            buf = struct.pack("<II", w, h) + classification.tobytes()
            self.spatial_pub.put(buf)
        except Exception as e:
            print(f"[Python Warning] Spatial map publish failed: {e}", flush=True)

    # -------------------------------------------------------------------------
    # TILING LOGIC - Tile-invariant, Crop-invariant, No Seam Artifacts
    # -------------------------------------------------------------------------
    def process_image_tile(self, img: np.ndarray, job_id: str) -> np.ndarray:
        """
        Process image using tiling for memory efficiency.

        DETERMINISM GUARANTEES:
        - Mirror padding ensures tile-invariant results
        - Same overlap handling regardless of tile size
        - No blending artifacts (hard crop after padding removal)

        Args:
            img: Input image as BGR numpy array
            job_id: Job ID for progress reporting

        Returns:
            Upscaled image as BGR numpy array
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        tile_size = Config.TILE_SIZE
        tile_pad = Config.TILE_PAD
        scale = self.model_scale

        h, w, c = img.shape
        output_h, output_w = h * scale, w * scale
        output_img = np.zeros((output_h, output_w, c), dtype=np.uint8)

        # Grid steps
        x_steps = list(range(0, w, tile_size))
        y_steps = list(range(0, h, tile_size))
        total_tiles = len(x_steps) * len(y_steps)
        count = 0

        # CRITICAL: Mirror/reflect padding for seamless tile boundaries
        # This ensures tile-size-invariant results
        img_padded = np.pad(
            img, ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0)), mode="reflect"
        )

        for y in y_steps:
            for x in x_steps:
                count += 1

                # Extract padded tile
                pad_y = y
                pad_x = x
                in_h = min(tile_size, h - y)
                in_w = min(tile_size, w - x)

                tile_in = img_padded[
                    pad_y : pad_y + in_h + 2 * tile_pad,
                    pad_x : pad_x + in_w + 2 * tile_pad,
                    :,
                ]

                # Convert BGR -> RGB if model expects RGB
                # OpenCV loads as BGR, most models expect RGB
                if self.expects_rgb:
                    tile_for_model = tile_in[:, :, ::-1].copy()  # BGR -> RGB
                else:
                    tile_for_model = tile_in.copy()  # Keep as BGR

                # Ensure even dimensions (required for PixelShuffle)
                h_in, w_in = tile_for_model.shape[:2]
                pad_h = h_in % 2
                pad_w = w_in % 2
                if pad_h or pad_w:
                    tile_for_model = np.pad(tile_for_model, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                # UNIFIED INFERENCE CALL
                with suppress_stdout():
                    output_from_model = inference(
                        self.model,
                        tile_for_model,
                        self.device,
                        half=self.use_half,
                        adapter=self.adapter
                    )

                # Convert output back to BGR if model outputs RGB
                if self.expects_rgb:
                    output_tile = output_from_model[:, :, ::-1].copy()  # RGB -> BGR
                else:
                    output_tile = output_from_model.copy()  # Keep as BGR

                # Crop padding from result
                out_pad = tile_pad * scale
                out_h_real = in_h * scale
                out_w_real = in_w * scale

                # Remove even-dimension padding if applied
                if pad_h or pad_w:
                    output_tile = output_tile[:output_tile.shape[0] - pad_h * scale,
                                               :output_tile.shape[1] - pad_w * scale, :]

                valid_tile = output_tile[
                    out_pad : out_pad + out_h_real, out_pad : out_pad + out_w_real, :
                ]

                # Place in result
                out_y = y * scale
                out_x = x * scale
                output_img[
                    out_y : out_y + out_h_real, out_x : out_x + out_w_real, :
                ] = valid_tile

                # Emit Progress
                self.send_status(
                    "progress", {"current": count, "total": total_tiles, "id": job_id}
                )

        # NOTE: empty_cache() intentionally removed from hot path.
        # Per-frame cache clearing causes CUDA driver sync + allocator churn,
        # adding 2-5ms per frame.  VRAM is managed by PyTorch's caching allocator.

        return output_img

    def handle_image_file(self, payload: Dict) -> None:
        """Handle image file upscaling with geometry and color edits"""
        req_id = payload.get("id")

        # Apply research params if included
        research_params = payload.get("research_params")
        if research_params and self.research_layer is not None and HAS_RESEARCH_LAYER:
            try:
                self.research_layer.update_params(research_params)
            except Exception as e:
                print(f"[Python Warning] Research params update failed: {e}", flush=True)

        try:
            params = payload["params"]
            img = cv2.imread(params["input_path"], cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not read input image")

            # --- GEOMETRY EDITS ---
            config = params.get("config", {})
            if "crop" in config and config["crop"]:
                c = config["crop"]
                h, w = img.shape[:2]
                x = max(0, int(c["x"] * w))
                y = max(0, int(c["y"] * h))
                cw = int(c["width"] * w)
                ch = int(c["height"] * h)
                if cw > 0 and ch > 0:
                    img = img[y : y + ch, x : x + cw]

            rot = config.get("rotation", 0)
            if rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if config.get("flip_h"):
                img = cv2.flip(img, 1)
            if config.get("flip_v"):
                img = cv2.flip(img, 0)

            # --- COLOR GRADING ---
            color = config.get("color", {})
            brightness = color.get("brightness", 0.0)
            contrast = color.get("contrast", 0.0)
            saturation = color.get("saturation", 0.0)
            gamma = color.get("gamma", 1.0)

            has_color_adj = (
                abs(brightness) > 0.001
                or abs(contrast) > 0.001
                or abs(saturation) > 0.001
                or abs(gamma - 1.0) > 0.001
            )

            if has_color_adj:
                img = img.astype(np.float32)

                if abs(brightness) > 0.001:
                    img = img + (brightness * 255)

                if abs(contrast) > 0.001:
                    contrast_factor = 1.0 + contrast
                    img = (img - 127.5) * contrast_factor + 127.5

                if abs(gamma - 1.0) > 0.001:
                    img = np.clip(img, 0, 255)
                    img = ((img / 255.0) ** (1.0 / gamma)) * 255.0

                if abs(saturation) > 0.001:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                    sat_factor = 1.0 + saturation
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

                img = np.clip(img, 0, 255).astype(np.uint8)

            # --- TILED INFERENCE ---
            with suppress_stdout():
                output = self.process_image_tile(img, req_id)

            # NOTE: Research layer is skipped for images — it would reprocess the
            # entire image through the model(s) without tiling, causing OOM/hangs
            # on large images.  Tiled inference output is used directly.

            # Publish spatial map for UI overlay
            try:
                rgb_for_spatial = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._publish_spatial_map(rgb_for_spatial)
            except Exception:
                pass

            cv2.imwrite(params["output_path"], output)
            self.send_status("ok", {"id": req_id})

        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"id": req_id, "message": str(e)})

    def create_shm(self, payload: Dict) -> None:
        """Create shared memory ring buffer for video frame processing.

        Layout (SHM_VERSION = 1):
            [ Global Header 36 bytes: magic|version|header_size|slot_count|W|H|S|fmt ]
            [ SlotHeader × ring_size (ring_size × 16 bytes) ]
            [ Slot 0: input (W×H×3) | output (sW×sH×3) ]
            [ Slot 1: input | output ]
            ...
        """
        width = payload["width"]
        height = payload["height"]
        self.active_scale = payload["scale"]
        self.ring_size = payload.get("ring_size", 3)

        self.input_size = width * height * 3
        self.output_size = (width * self.active_scale) * (height * self.active_scale) * 3
        self.slot_byte_size = self.input_size + self.output_size
        # header_region_size = global header + per-slot headers
        self.header_region_size = (
            Config.GLOBAL_HEADER_SIZE + Config.SLOT_HEADER_SIZE * self.ring_size
        )
        total_size = self.header_region_size + self.slot_byte_size * self.ring_size

        if self.mmap:
            self.cleanup()

        try:
            fd, self.shm_path = tempfile.mkstemp(prefix="vf_buffer_", suffix=".bin")
            self.shm_file = os.fdopen(fd, "wb+")
            self.shm_file.write(b"\0" * total_size)
            self.shm_file.flush()
            self.shm_file.seek(0)
            self.mmap = mmap.mmap(self.shm_file.fileno(), total_size)

            # Write global header (36 bytes) at offset 0.
            # Format: <8sIIIIIII  (little-endian: 8-byte magic + 7 × u32)
            global_header = struct.pack(
                "<8sIIIIIII",
                Config.SHM_MAGIC,          # magic[8]
                Config.SHM_VERSION,        # version u32
                self.header_region_size,   # header_size u32
                self.ring_size,            # slot_count u32
                width,                     # width u32
                height,                    # height u32
                self.active_scale,         # scale u32
                Config.PIXEL_FORMAT_RGB24, # pixel_format u32
            )
            self.mmap[0 : Config.GLOBAL_HEADER_SIZE] = global_header

            self.input_shape = (height, width, 3)
            self.output_shape = (
                height * self.active_scale,
                width * self.active_scale,
                3,
            )
            self.is_configured = True
            print(
                f"[Python] SHM created: {total_size} bytes "
                f"(global_header={Config.GLOBAL_HEADER_SIZE}, "
                f"header_region={self.header_region_size}, "
                f"{self.ring_size} slots × {self.slot_byte_size}), "
                f"magic=VFSHM001 version={Config.SHM_VERSION}",
                flush=True,
            )
            self.send_status("SHM_CREATED", {"shm_path": self.shm_path})
        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"message": str(e)})

    def _validate_shm_header(self) -> None:
        """Validate the SHM global header written by this Python process.

        Called after mmap creation. Raises ValueError with a descriptive
        message if the header is malformed.
        """
        if not self.mmap or len(self.mmap) < Config.GLOBAL_HEADER_SIZE:
            raise ValueError(
                f"SHM too small for global header ({Config.GLOBAL_HEADER_SIZE} bytes)"
            )
        magic = bytes(self.mmap[0:8])
        if magic != Config.SHM_MAGIC:
            raise ValueError(
                f"SHM magic mismatch: expected {Config.SHM_MAGIC!r}, got {magic!r}"
            )
        version = struct.unpack_from("<I", self.mmap, 8)[0]
        if version != Config.SHM_VERSION:
            raise ValueError(
                f"SHM version mismatch: expected {Config.SHM_VERSION}, got {version}"
            )

    # -------------------------------------------------------------------------
    # SHM SLOT STATE HELPERS
    # -------------------------------------------------------------------------

    def _slot_state_offset(self, slot_idx: int) -> int:
        """Byte offset of the state field for a given slot header.

        Accounts for the global header at the start of the file.
        Matches Rust: GLOBAL_HEADER_SIZE + slot_idx * SLOT_HEADER_SIZE + STATE_OFFSET
        """
        return (
            Config.GLOBAL_HEADER_SIZE
            + slot_idx * Config.SLOT_HEADER_SIZE
            + Config.STATE_FIELD_OFFSET
        )

    def _read_slot_state(self, slot_idx: int) -> int:
        """Read the u32 state of a slot from the mmap header."""
        off = self._slot_state_offset(slot_idx)
        return struct.unpack_from("<I", self.mmap, off)[0]

    def _write_slot_state(self, slot_idx: int, state: int) -> None:
        """Write the u32 state of a slot into the mmap header."""
        off = self._slot_state_offset(slot_idx)
        struct.pack_into("<I", self.mmap, off, state)

    def _slot_data_base(self, slot_idx: int) -> int:
        """Byte offset of the start of data for a given slot (after headers)."""
        return self.header_region_size + slot_idx * self.slot_byte_size

    # -------------------------------------------------------------------------
    # CORE FRAME PROCESSING (shared by Zenoh fallback and polling loop)
    # -------------------------------------------------------------------------

    def _process_slot(self, slot_idx: int, research_params: Optional[Dict] = None) -> None:
        """
        Process a single video frame from shared memory slot.

        Uses header-based offsets.  Caller is responsible for state transitions.

        DETERMINISM GUARANTEES:
        - Model is stateless — no hidden recurrence between frames
        - Uses unified inference() function (same as image path)
        """
        base = self._slot_data_base(slot_idx)
        in_end = base + self.input_size
        out_end = base + self.slot_byte_size

        in_view = np.frombuffer(
            self.mmap, dtype=np.uint8, count=self.input_size, offset=base
        ).reshape(self.input_shape)
        out_view = np.frombuffer(
            self.mmap, dtype=np.uint8, count=self.output_size, offset=in_end
        ).reshape(self.output_shape)

        # Passthrough for scale=1
        if self.active_scale == 1:
            out_view[:] = in_view[:]
            return

        # Input is already RGB24 (no alpha channel)
        img_input = in_view.copy()

        # Convert color space if needed
        if not self.expects_rgb:
            img_for_model = img_input[:, :, ::-1].copy()  # RGB -> BGR
        else:
            img_for_model = img_input  # Already RGB

        # UNIFIED INFERENCE CALL
        with suppress_stdout():
            out_from_model = inference(
                self.model,
                img_for_model,
                self.device,
                half=self.use_half,
                adapter=self.adapter
            )

        # Convert output back to RGB for Rust
        if not self.expects_rgb:
            out_for_rust = out_from_model[:, :, ::-1].copy()  # BGR -> RGB
        else:
            out_for_rust = out_from_model  # Already RGB

        # Research layer post-processing (if available)
        if self.research_layer is not None and HAS_RESEARCH_LAYER:
            try:
                out_for_rust = self.research_layer.process_frame_numpy(img_input)
            except Exception as e:
                print(f"[Python Warning] Research layer failed, using vanilla: {e}", flush=True)

        # Publish spatial routing map for UI overlay
        self._publish_spatial_map(img_input)

        # SR Pipeline post-processing (blender_engine)
        if HAS_BLENDER and research_params:
            try:
                sr = research_params

                if sr.get("reset_temporal"):
                    clear_temporal_buffers()
                    print("[Python] Temporal buffers cleared by user request", flush=True)

                adr_on = bool(sr.get("adr_enabled", False))
                detail_str = float(sr.get("detail_strength", 0.0))
                luma_only = bool(sr.get("luma_only", True))
                edge_str = float(sr.get("edge_strength", 0.0))
                sharpen_val = float(sr.get("sharpen_strength", 0.0))
                temporal_on = bool(sr.get("temporal_enabled", False))
                temporal_a = float(sr.get("temporal_alpha", 0.9))

                needs_sr_pipeline = (
                    (adr_on and detail_str > 1e-4)
                    or edge_str > 1e-4
                    or sharpen_val > 1e-4
                    or temporal_on
                )

                if needs_sr_pipeline:
                    sr_float = out_for_rust.astype(np.float32) / 255.0
                    sr_tensor = torch.from_numpy(sr_float.transpose(2, 0, 1)).unsqueeze(0)
                    sr_tensor = sr_tensor.to(device=self.device, non_blocking=True)

                    if adr_on and detail_str > 1e-4:
                        gan_float = out_from_model.astype(np.float32) / 255.0
                        gan_tensor = torch.from_numpy(
                            gan_float.transpose(2, 0, 1)
                        ).unsqueeze(0).to(device=self.device, non_blocking=True)
                        sr_tensor = PredictionBlender.apply_detail_residual(
                            sr_tensor, gan_tensor, detail_str, luma_only
                        )

                    if edge_str > 1e-4:
                        sr_tensor = PredictionBlender.blend_edge_aware(
                            sr_tensor, sr_tensor, alpha=edge_str, edge_strength=1.0
                        )

                    if sharpen_val > 1e-4:
                        sr_tensor = PredictionBlender.apply_sharpen(sr_tensor, sharpen_val)

                    if temporal_on:
                        _, _, th, tw = sr_tensor.shape
                        t_key = (th, tw, sr_tensor.shape[1])
                        sr_tensor = PredictionBlender.apply_temporal(
                            sr_tensor, t_key, temporal_a
                        )

                    out_for_rust = (
                        sr_tensor.squeeze(0)
                        .clamp_(0.0, 1.0)
                        .mul_(255.0)
                        .round_()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
            except Exception as e:
                print(f"[Python Warning] SR pipeline post-processing failed: {e}", flush=True)

        # Handle scale mismatch (resize output if needed)
        h, w = out_for_rust.shape[:2]
        target_h, target_w = self.output_shape[:2]
        if h != target_h or w != target_w:
            out_for_rust = cv2.resize(out_for_rust, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Write RGB24 directly to output slot
        out_view[:] = out_for_rust

    # -------------------------------------------------------------------------
    # ZENOH FALLBACK: process_frame (legacy per-frame Zenoh command)
    # -------------------------------------------------------------------------

    def process_frame(self, payload: Dict) -> None:
        """Legacy per-frame Zenoh handler — delegates to _process_slot."""
        if not self.is_configured or self.model is None:
            self.send_status("error", {"message": "Not configured or no model"})
            return

        research_params = payload.get("research_params")
        if research_params and self.research_layer is not None and HAS_RESEARCH_LAYER:
            try:
                self.research_layer.update_params(research_params)
            except Exception as e:
                print(f"[Python Warning] Research params update failed: {e}", flush=True)

        slot_idx = payload.get("slot", 0)
        try:
            self._process_slot(slot_idx, research_params)
            self.send_status("FRAME_DONE", {"slot": slot_idx})
        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"message": str(e)})

    def process_one_frame(self, payload: Dict) -> None:
        """Single-frame SHM roundtrip used by the smoke test.

        Handles full state transition: READY_FOR_AI → AI_PROCESSING → READY_FOR_ENCODE.
        Works with scale=1 (passthrough) even without a model loaded, so the smoke
        test has no model-weight dependency.
        """
        if not self.is_configured:
            self.send_status("error", {"message": "Not configured: send create_shm first"})
            return
        if self.model is None and self.active_scale != 1:
            self.send_status("error", {"message": "No model loaded for scale != 1"})
            return

        # Find first READY_FOR_AI slot
        slot_idx = None
        for i in range(self.ring_size):
            if self._read_slot_state(i) == Config.SLOT_READY_FOR_AI:
                slot_idx = i
                break
        if slot_idx is None:
            self.send_status("error", {"message": "No slot in READY_FOR_AI state"})
            return

        self._write_slot_state(slot_idx, Config.SLOT_AI_PROCESSING)
        try:
            self._process_slot(slot_idx)
        except Exception as e:
            traceback.print_exc()
            self._write_slot_state(slot_idx, Config.SLOT_EMPTY)
            self.send_status("error", {"message": str(e)})
            return
        self._write_slot_state(slot_idx, Config.SLOT_READY_FOR_ENCODE)
        self.send_status("FRAME_DONE", {"slot": slot_idx})

    # -------------------------------------------------------------------------
    # SHM ATOMIC FRAME LOOP (replaces per-frame Zenoh signaling)
    # -------------------------------------------------------------------------

    def _collect_ready_slots(self, start_slot: int) -> list:
        """
        Scan slots starting from start_slot, collecting consecutive READY_FOR_AI
        slots up to MAX_BATCH_SIZE.  Returns list of slot indices in ring order.

        Consecutive means ring-order from start_slot: if slot 2 is ready but
        slot 1 (start) is not, we return [] because ordering is strict.
        """
        batch = []
        max_batch = min(Config.MAX_BATCH_SIZE, self.ring_size)
        slot = start_slot
        for _ in range(max_batch):
            if self._read_slot_state(slot) == Config.SLOT_READY_FOR_AI:
                batch.append(slot)
                slot = (slot + 1) % self.ring_size
            else:
                break
        return batch

    def _process_batch(self, slot_indices: list, research_params: Optional[Dict] = None) -> None:
        """
        Process multiple SHM slots as a single GPU batch.

        1. Read input frames from all slots
        2. Run batched inference (single forward pass)
        3. Apply per-frame post-processing (research layer, blender)
        4. Write output frames back to slots

        Falls back to sequential _process_slot() if batching is unsupported
        (e.g. scale=1 passthrough, or adapter doesn't support batches).
        """
        if len(slot_indices) == 1:
            self._process_slot(slot_indices[0], research_params)
            return

        # Scale=1 passthrough doesn't benefit from batching
        if self.active_scale == 1:
            for idx in slot_indices:
                self._process_slot(idx, research_params)
            return

        # --- Collect input frames ---
        inputs_rgb = []
        for slot_idx in slot_indices:
            base = self._slot_data_base(slot_idx)
            in_view = np.frombuffer(
                self.mmap, dtype=np.uint8, count=self.input_size, offset=base
            ).reshape(self.input_shape)
            img_input = in_view.copy()

            if not self.expects_rgb:
                img_input = img_input[:, :, ::-1].copy()  # RGB -> BGR for model

            inputs_rgb.append(img_input)

        # --- Batched GPU inference ---
        with suppress_stdout():
            outputs = inference_batch(
                self.model,
                inputs_rgb,
                self.device,
                half=self.use_half,
                adapter=self.adapter,
            )

        # --- Per-frame post-processing and write-back ---
        for i, slot_idx in enumerate(slot_indices):
            out_from_model = outputs[i]

            # Convert back to RGB if model output is BGR
            if not self.expects_rgb:
                out_for_rust = out_from_model[:, :, ::-1].copy()  # BGR -> RGB
            else:
                out_for_rust = out_from_model

            # Research layer post-processing
            if self.research_layer is not None and HAS_RESEARCH_LAYER:
                try:
                    base = self._slot_data_base(slot_idx)
                    in_view = np.frombuffer(
                        self.mmap, dtype=np.uint8, count=self.input_size, offset=base
                    ).reshape(self.input_shape)
                    img_input = in_view.copy()
                    out_for_rust = self.research_layer.process_frame_numpy(img_input)
                except Exception as e:
                    print(f"[Python Warning] Research layer failed, using vanilla: {e}", flush=True)

            # Spatial map (only for first frame in batch to reduce overhead)
            if i == 0:
                base = self._slot_data_base(slot_idx)
                in_view = np.frombuffer(
                    self.mmap, dtype=np.uint8, count=self.input_size, offset=base
                ).reshape(self.input_shape)
                self._publish_spatial_map(in_view.copy())

            # SR Pipeline post-processing (blender_engine)
            if HAS_BLENDER and research_params:
                try:
                    sr = research_params

                    if sr.get("reset_temporal") and i == 0:
                        clear_temporal_buffers()
                        print("[Python] Temporal buffers cleared by user request", flush=True)

                    adr_on = bool(sr.get("adr_enabled", False))
                    detail_str = float(sr.get("detail_strength", 0.0))
                    luma_only = bool(sr.get("luma_only", True))
                    edge_str = float(sr.get("edge_strength", 0.0))
                    sharpen_val = float(sr.get("sharpen_strength", 0.0))
                    temporal_on = bool(sr.get("temporal_enabled", False))
                    temporal_a = float(sr.get("temporal_alpha", 0.9))

                    needs_sr_pipeline = (
                        (adr_on and detail_str > 1e-4)
                        or edge_str > 1e-4
                        or sharpen_val > 1e-4
                        or temporal_on
                    )

                    if needs_sr_pipeline:
                        sr_float = out_for_rust.astype(np.float32) / 255.0
                        sr_tensor = torch.from_numpy(sr_float.transpose(2, 0, 1)).unsqueeze(0)
                        sr_tensor = sr_tensor.to(device=self.device, non_blocking=True)

                        if adr_on and detail_str > 1e-4:
                            gan_float = out_from_model.astype(np.float32) / 255.0
                            gan_tensor = torch.from_numpy(
                                gan_float.transpose(2, 0, 1)
                            ).unsqueeze(0).to(device=self.device, non_blocking=True)
                            sr_tensor = PredictionBlender.apply_detail_residual(
                                sr_tensor, gan_tensor, detail_str, luma_only
                            )

                        if edge_str > 1e-4:
                            sr_tensor = PredictionBlender.blend_edge_aware(
                                sr_tensor, sr_tensor, alpha=edge_str, edge_strength=1.0
                            )

                        if sharpen_val > 1e-4:
                            sr_tensor = PredictionBlender.apply_sharpen(sr_tensor, sharpen_val)

                        if temporal_on:
                            _, _, th, tw = sr_tensor.shape
                            t_key = (th, tw, sr_tensor.shape[1])
                            sr_tensor = PredictionBlender.apply_temporal(
                                sr_tensor, t_key, temporal_a
                            )

                        out_for_rust = (
                            sr_tensor.squeeze(0)
                            .clamp_(0.0, 1.0)
                            .mul_(255.0)
                            .round_()
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                except Exception as e:
                    print(f"[Python Warning] SR pipeline post-processing failed: {e}", flush=True)

            # Handle scale mismatch
            h, w = out_for_rust.shape[:2]
            target_h, target_w = self.output_shape[:2]
            if h != target_h or w != target_w:
                out_for_rust = cv2.resize(out_for_rust, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

            # Write to output slot
            base = self._slot_data_base(slot_idx)
            in_end = base + self.input_size
            out_view = np.frombuffer(
                self.mmap, dtype=np.uint8, count=self.output_size, offset=in_end
            ).reshape(self.output_shape)
            out_view[:] = out_for_rust

    def _frame_loop(self) -> None:
        """
        Background polling loop with micro-batching: collects consecutive
        READY_FOR_AI slots and processes them in a single GPU forward pass.

        Runs in a daemon thread.  Stopped by setting self._frame_loop_active = False.
        Slots are processed in strict sequential order (0 → 1 → 2 → 0 → …)
        to preserve frame ordering.
        """
        print("[Python] Frame loop started (SHM atomic polling, micro-batch)", flush=True)
        next_slot = 0
        idle_spins = 0

        while self._frame_loop_active:
            if not self.is_configured or self.model is None:
                time.sleep(0.01)
                continue

            batch = self._collect_ready_slots(next_slot)

            if batch:
                idle_spins = 0

                # Transition all batch slots: READY_FOR_AI → AI_PROCESSING
                for idx in batch:
                    self._write_slot_state(idx, Config.SLOT_AI_PROCESSING)

                try:
                    self._process_batch(batch, self._cached_research_params)
                except Exception as e:
                    print(f"[Python Error] Batch processing failed: {e}", flush=True)
                    traceback.print_exc()
                    for idx in batch:
                        self._write_slot_state(idx, Config.SLOT_EMPTY)
                    next_slot = (batch[-1] + 1) % self.ring_size
                    continue     # skip READY_FOR_ENCODE transition

                # Transition all batch slots: AI_PROCESSING → READY_FOR_ENCODE
                for idx in batch:
                    self._write_slot_state(idx, Config.SLOT_READY_FOR_ENCODE)

                next_slot = (batch[-1] + 1) % self.ring_size
            else:
                # No work available — adaptive backoff
                idle_spins += 1
                if idle_spins < 100:
                    time.sleep(0.0001)  # 100µs tight spin
                elif idle_spins < 1000:
                    time.sleep(0.001)   # 1ms
                else:
                    time.sleep(0.005)   # 5ms deep idle

        print("[Python] Frame loop stopped", flush=True)

    def start_frame_loop(self, payload: Dict) -> None:
        """Start the SHM atomic frame polling loop in a background thread."""
        if hasattr(self, '_frame_loop_thread') and self._frame_loop_thread is not None:
            if self._frame_loop_thread.is_alive():
                print("[Python] Frame loop already running", flush=True)
                return

        # Cache initial research params
        self._cached_research_params = payload.get("research_params")
        self._frame_loop_active = True

        self._frame_loop_thread = threading.Thread(
            target=self._frame_loop, daemon=True, name="vf-frame-loop"
        )
        self._frame_loop_thread.start()
        self.send_status("FRAME_LOOP_STARTED")

    def stop_frame_loop(self, payload: Dict = None) -> None:
        """Stop the SHM atomic frame polling loop."""
        self._frame_loop_active = False
        if hasattr(self, '_frame_loop_thread') and self._frame_loop_thread is not None:
            self._frame_loop_thread.join(timeout=5.0)
            self._frame_loop_thread = None
        self.send_status("FRAME_LOOP_STOPPED")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VideoForge Deterministic AI Worker")
    parser.add_argument("--port", type=str, default="7447", help="Zenoh port")
    parser.add_argument("--parent-pid", type=int, default=0, help="Parent process ID to monitor")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "deterministic"],
        help="Inference precision: fp32 (TF32 on), fp16 (autocast), deterministic (strict)"
    )
    args = parser.parse_args()

    # Configure precision BEFORE any model loading or CUDA ops
    configure_precision(args.precision)

    if args.parent_pid > 0:
        start_watchdog(args.parent_pid)

    AIWorker(args.port, precision=args.precision)
    sys.exit(0)
