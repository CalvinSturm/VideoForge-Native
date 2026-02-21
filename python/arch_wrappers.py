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

    def __init__(self, model: nn.Module, scale: int, use_autocast: bool = True) -> None:
        self.model: nn.Module = model
        self.scale: int = scale
        self.use_autocast: bool = use_autocast

    # ------------------------------------------------------------------
    @abstractmethod
    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        """NCHW float32 [0,1] -> padded/prepared tensor for the model."""
        ...

    @abstractmethod
    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        """Model output -> cropped NCHW float32 [0,1] at (orig_h*s, orig_w*s)."""
        ...

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end: NCHW float32 [0,1] in -> NCHW float32 [0,1] out.

        Wraps pre_process -> model.forward -> post_process.
        """
        _, _, orig_h, orig_w = x.shape
        prepared = self.pre_process(x)
        if self.use_autocast:
            with torch.amp.autocast("cuda"):
                raw_out = self.model(prepared)
        else:
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
# EDSR / RCAN (official models trained on [0,255] with DIV2K mean subtraction)
# ═══════════════════════════════════════════════════════════════════════════════

class EDSRRCANAdapter(BaseAdapter):
    """
    Adapter for official EDSR/RCAN models trained on [0, 255] range with
    DIV2K mean subtraction.

    Our pipeline feeds [0, 1] NCHW float32.  This adapter:
      - pre_process:  x * 255 − mean
      - post_process: (y + mean) / 255, then clamp and crop
    """

    # DIV2K per-channel mean (RGB, [0, 255] scale)
    _MEAN = [0.4488 * 255.0, 0.4371 * 255.0, 0.4040 * 255.0]

    def __init__(self, model: nn.Module, scale: int) -> None:
        super().__init__(model, scale)
        self._mean_t: Optional[torch.Tensor] = None

    def _mean(self, device: torch.device) -> torch.Tensor:
        if self._mean_t is None or self._mean_t.device != device:
            self._mean_t = torch.tensor(
                self._MEAN, device=device, dtype=torch.float32
            ).view(1, 3, 1, 1)
        return self._mean_t

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return x * 255.0 - self._mean(x.device)

    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        out = (out + self._mean(out.device)) / 255.0
        target_h = orig_h * self.scale
        target_w = orig_w * self.scale
        _, _, oh, ow = out.shape
        if oh != target_h or ow != target_w:
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return out.clamp(0.0, 1.0)


class ScaledRangeAdapter(BaseAdapter):
    """
    For full model objects that already include MeanShift (sub_mean / add_mean).

    Only converts [0,1] → [0,255] before the model and [0,255] → [0,1] after.
    Does NOT apply DIV2K mean subtraction (the model handles it internally).
    """

    def pre_process(self, x: torch.Tensor) -> torch.Tensor:
        return x * 255.0

    def post_process(
        self, out: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        out = out / 255.0
        target_h = orig_h * self.scale
        target_w = orig_w * self.scale
        _, _, oh, ow = out.shape
        if oh != target_h or ow != target_w:
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
        return out.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTWEIGHT (FSRCNN, RealESRGAN, etc.)
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
# The key is matched against the lowercase model_key prefix (before '_').
_ADAPTER_REGISTRY: dict[str, tuple[type[BaseAdapter], dict]] = {
    # ── Transformers ──────────────────────────────────────────────────
    "swinir":   (TransformerAdapter, {"window_size": 8}),
    "swin2sr":  (TransformerAdapter, {"window_size": 8}),
    "hat":      (TransformerAdapter, {"window_size": 16}),
    "ipt":      (TransformerAdapter, {"window_size": 8}),
    "edt":      (TransformerAdapter, {"window_size": 8}),
    "dat":      (TransformerAdapter, {"window_size": 16}),
    # ── Diffusion ─────────────────────────────────────────────────────
    "resshift": (DiffusionAdapter, {}),
    "sr3":      (DiffusionAdapter, {}),
    "stablesr": (DiffusionAdapter, {}),
    "dit":      (DiffusionAdapter, {}),
    # ── GAN / Conv ────────────────────────────────────────────────────
    "rcan":     (EDSRRCANAdapter, {}),
    "edsr":     (EDSRRCANAdapter, {}),
    "mdsr":     (EDSRRCANAdapter, {}),
    "realesrgan": (LightweightAdapter, {}),
    "bsrgan":   (LightweightAdapter, {}),
    "spsr":     (LightweightAdapter, {}),
    "realbasicvsr": (LightweightAdapter, {}),
    # ── Lightweight ───────────────────────────────────────────────────
    "fsrcnn":   (LightweightAdapter, {}),
    "carn":     (LightweightAdapter, {}),
    "lapsrn":   (LightweightAdapter, {}),
    "omnisr":   (LightweightAdapter, {}),
    "mosr":     (LightweightAdapter, {}),
    "nomos":    (LightweightAdapter, {}),
    "compact":  (LightweightAdapter, {}),
    "span":     (LightweightAdapter, {}),
}


def _match_adapter_key(model_key: str) -> Optional[str]:
    """
    Find the best matching adapter registry key for a model_key.

    Tries, in order:
      1. Exact prefix match (split on '_')[0]
      2. Contains match — longest registered key found anywhere in model_key
    """
    lower = model_key.lower()

    # 1. Prefix match (fast path for standard names like "realesrgan_x4plus")
    prefix = lower.split("_")[0].split("-")[0]
    # Strip leading digits+x for "4xFFHQDAT" → "ffhqdat"
    import re
    stripped = re.sub(r"^\d+x[-_]?", "", prefix)
    for candidate in (prefix, stripped):
        if candidate in _ADAPTER_REGISTRY:
            return candidate

    # 2. Contains match — find longest registered key present in model_key
    #    e.g. "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN" -> "swinir"
    matches = [k for k in _ADAPTER_REGISTRY if k in lower]
    if matches:
        return max(matches, key=len)  # prefer longest match

    # 3. Normalized match — strip separators for cases like "Swin_IR" -> "swinir"
    normalized = lower.replace("-", "").replace("_", "")
    matches = [k for k in _ADAPTER_REGISTRY if k in normalized]
    if matches:
        return max(matches, key=len)

    return None


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
        Model identifier (e.g. ``"RealESRGAN_x4plus"``, ``"4xFFHQDAT"``).
    model : nn.Module
        The already-loaded, eval-mode model (on *device*).
    scale : int
        Upscale factor.  If 0 or negative, auto-detected via a probe tensor.
    device : torch.device
        CUDA device for the probe (if scale detection is needed).

    Raises
    ------
    ValueError
        If *model_key* cannot be matched to any registered adapter.
    """
    # ONNX models have all pre/post-processing baked into the graph — skip
    # every architecture-specific adapter and use a plain pass-through.
    if getattr(model, "_vf_onnx", False):
        print(f"[ArchWrappers] ONNX model, using LightweightAdapter (pass-through, FP32)", flush=True)
        return LightweightAdapter(model=model, scale=scale, use_autocast=False)

    # Spandrel-loaded models handle their own padding, mean shift, and cropping.
    # Use a pass-through adapter with autocast disabled — transformer attention
    # layers (SwinIR, HAT, DAT) overflow in FP16, producing NaN.
    if getattr(model, "_vf_spandrel", False):
        print(f"[ArchWrappers] Spandrel model, using LightweightAdapter (pass-through, FP32)", flush=True)
        if scale <= 0:
            scale = BaseAdapter.infer_scale(model, device)
        return LightweightAdapter(model=model, scale=scale, use_autocast=False)

    # If the model was loaded as-is with MeanShift layers intact,
    # use ScaledRangeAdapter to avoid double mean subtraction.
    if getattr(model, "_vf_has_mean_shift", False):
        print(f"[ArchWrappers] Model has MeanShift, using ScaledRangeAdapter", flush=True)
        if scale <= 0:
            scale = BaseAdapter.infer_scale(model, device)
        return ScaledRangeAdapter(model=model, scale=scale)

    key = _match_adapter_key(model_key)
    if key is None:
        # Fallback to LightweightAdapter (plain conv, no special padding)
        print(
            f"[ArchWrappers] No adapter match for '{model_key}', "
            f"falling back to LightweightAdapter", flush=True
        )
        if scale <= 0:
            scale = BaseAdapter.infer_scale(model, device)
        return LightweightAdapter(model=model, scale=scale)

    adapter_cls, extra_kwargs = _ADAPTER_REGISTRY[key]
    print(f"[ArchWrappers] Matched '{model_key}' -> {key} ({adapter_cls.__name__})", flush=True)

    if scale <= 0:
        scale = BaseAdapter.infer_scale(model, device)
        print(f"[ArchWrappers] Auto-detected scale: {scale}x", flush=True)

    return adapter_cls(model=model, scale=scale, **extra_kwargs)
