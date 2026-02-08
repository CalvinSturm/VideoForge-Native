"""
VideoForge Architecture Adapters — Unified forward() interface for all SR families.

Each adapter wraps a pretrained nn.Module and handles:
  - Spatial padding to window_size multiples (Transformers)
  - Dynamic scale inference from input/output dimension comparison
  - HWC uint8 ↔ NCHW float32 conversion bookkeeping
  - Output cropping back to (original_dim × scale)

Adapter taxonomy:
  TransformerAdapter  — SwinIR, HAT (window-based, needs reflect-pad)
  DiffusionAdapter    — ResShift, SR3 (black-box forward, may change H/W)
  LightweightAdapter  — FSRCNN (plain conv, no padding quirks)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# BASE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAdapter(ABC):
    """Common bookkeeping shared by every SR adapter."""

    def __init__(self, model: nn.Module, scale: int) -> None:
        self.model: nn.Module = model
        self.scale: int = scale

    # ------------------------------------------------------------------
    @abstractmethod
    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        """NCHW float32 [0,1] → padded/prepared tensor for the model."""
        ...

    @abstractmethod
    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        """Model output → cropped NCHW float32 [0,1] at (orig_h*s, orig_w*s)."""
        ...

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end: NCHW float32 [0,1] in → NCHW float32 [0,1] out.

        Wraps pre_process → model.forward → post_process.
        """
        _, _, orig_h, orig_w = x.shape
        prepared = self.pre_process(x)
        with torch.cuda.amp.autocast():
            raw_out = self.model(prepared)
        return self.post_process(raw_out, orig_h, orig_w)

    # ------------------------------------------------------------------
    @staticmethod
    def infer_scale(model: nn.Module, device: torch.device) -> int:
        """
        Run a tiny probe tensor through the model to measure the actual
        upscale factor.  Falls back to 4 on any failure.
        """
        probe_h, probe_w = 16, 16
        probe = torch.zeros(1, 3, probe_h, probe_w, device=device, dtype=torch.float32)
        try:
            model.eval()
            with torch.inference_mode(), torch.cuda.amp.autocast():
                out = model(probe)
            out_h = out.shape[2]
            detected = max(1, round(out_h / probe_h))
            return detected
        except Exception:
            return 4


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER (SwinIR, HAT)
# ═══════════════════════════════════════════════════════════════════════════════

class TransformerAdapter(BaseAdapter):
    """
    Handles window-based vision transformers that require input dimensions to
    be exact multiples of `window_size`.

    Uses ``F.pad(mode='reflect')`` to reach the nearest multiple, then crops
    the SR output back to ``(orig_h × scale, orig_w × scale)``.
    """

    def __init__(
        self,
        model: nn.Module,
        scale: int,
        window_size: int = 8,
    ) -> None:
        super().__init__(model, scale)
        self.window_size: int = window_size

    # ------------------------------------------------------------------
    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        target_h = orig_h * self.scale
        target_w = orig_w * self.scale
        return out[:, :, :target_h, :target_w]


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFUSION (ResShift, SR3)
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionAdapter(BaseAdapter):
    """
    Black-box wrapper for diffusion SR models.

    Assumptions
    -----------
    * ``model.forward(x)`` performs the full SR pipeline internally
      (noise schedule, sampling loop, etc.) and returns the final image.
    * Output may not be an exact integer multiple of input — we resize to
      ``(orig_h × scale, orig_w × scale)`` if needed.
    """

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return x  # no spatial prep for black-box diffusion

    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        target_h = orig_h * self.scale
        target_w = orig_w * self.scale
        _, _, oh, ow = out.shape
        if oh != target_h or ow != target_w:
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
        return out.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT (FSRCNN, RCAN, EDSR, RealESRGAN)
# ═══════════════════════════════════════════════════════════════════════════════

class LightweightAdapter(BaseAdapter):
    """
    Plain-conv SR models with no special spatial constraints.

    Handles any minor output-size mismatch (e.g. rounding from PixelShuffle)
    by bilinear resize to ``(orig_h × scale, orig_w × scale)``.
    """

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        target_h = orig_h * self.scale
        target_w = orig_w * self.scale
        _, _, oh, ow = out.shape
        if oh != target_h or ow != target_w:
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return out.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical model_key → adapter class + kwargs
_ADAPTER_REGISTRY: dict[str, tuple[type[BaseAdapter], dict]] = {
    # Transformers
    "swinir":   (TransformerAdapter, {"window_size": 8}),
    "hat":      (TransformerAdapter, {"window_size": 16}),
    # Diffusion
    "resshift": (DiffusionAdapter, {}),
    "sr3":      (DiffusionAdapter, {}),
    # Lightweight / Conv
    "fsrcnn":   (LightweightAdapter, {}),
    "rcan":     (LightweightAdapter, {}),
    "edsr":     (LightweightAdapter, {}),
    "realesrgan": (LightweightAdapter, {}),
}


def create_adapter(
    model_key: str,
    model: nn.Module,
    scale: int,
    device: torch.device,
) -> BaseAdapter:
    """
    Build the correct adapter for *model_key*.

    Parameters
    ----------
    model_key : str
        Lowercase canonical name (e.g. ``"swinir"``, ``"hat"``, ``"fsrcnn"``).
    model : nn.Module
        The already-loaded, eval-mode model (on *device*).
    scale : int
        Upscale factor.  If 0 or negative, auto-detected via a probe tensor.
    device : torch.device
        CUDA device for the probe (if scale detection is needed).

    Raises
    ------
    ValueError
        If *model_key* is not in the registry.
    """
    key = model_key.lower().split("_")[0]  # "realesrgan_x4plus" → "realesrgan"
    if key not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Registered adapters: {sorted(_ADAPTER_REGISTRY)}"
        )

    adapter_cls, extra_kwargs = _ADAPTER_REGISTRY[key]

    if scale <= 0:
        scale = BaseAdapter.infer_scale(model, device)
        print(f"[ArchWrappers] Auto-detected scale: {scale}x", flush=True)

    return adapter_cls(model=model, scale=scale, **extra_kwargs)
