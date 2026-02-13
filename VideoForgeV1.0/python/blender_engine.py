"""
VideoForge Prediction Blender — GPU-resident tensor blending & stabilization.

All operations stay on CUDA to avoid host↔device copies.  The only CPU touch
point is the final ``.cpu()`` call performed by the caller after blending.

Public API
----------
PredictionBlender.blend(primary, secondary, alpha)
    → torch.lerp on GPU, clamped to [0, 1].
PredictionBlender.blend_masked(primary, secondary, mask, alpha)
    → spatially-varying blend weighted by a soft mask.
PredictionBlender.blend_luma_only(primary, secondary, alpha)
    → YCbCr luminance-only blend (preserves primary chroma).
PredictionBlender.blend_edge_aware(primary, secondary, alpha, edge_strength)
    → Sobel edge mask modulates blend strength per-pixel.
PredictionBlender.apply_detail_residual(structure, gan, detail_strength, luma_only)
    → Adaptive Detail Residual: extract GAN high-freq texture, inject into structure.
PredictionBlender.apply_sharpen(tensor, strength)
    → GPU unsharp mask via Gaussian blur subtraction.
PredictionBlender.apply_temporal(tensor, key, alpha)
    → Exponential moving average temporal stabilization.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR-SPACE CONVERSION (GPU, BT.601)
# ═══════════════════════════════════════════════════════════════════════════════

# BT.601 RGB→YCbCr matrix  (row-major, applied to [R,G,B] column)
_RGB_TO_YCBCR = torch.tensor(
    [[ 0.299,    0.587,    0.114   ],
     [-0.168736, -0.331264, 0.5     ],
     [ 0.5,     -0.418688, -0.081312]],
    dtype=torch.float32,
)

_YCBCR_TO_RGB = torch.tensor(
    [[1.0,  0.0,       1.402   ],
     [1.0, -0.344136, -0.714136],
     [1.0,  1.772,     0.0     ]],
    dtype=torch.float32,
)

# Offset added after RGB→YCbCr  (Y stays [0,1], Cb/Cr shift to [0,1])
_YCBCR_OFFSET = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)


def _ensure_matrix_on(device: torch.device) -> None:
    """Move colour-space matrices to *device* once (lazy init)."""
    global _RGB_TO_YCBCR, _YCBCR_TO_RGB, _YCBCR_OFFSET
    if _RGB_TO_YCBCR.device != device:
        _RGB_TO_YCBCR = _RGB_TO_YCBCR.to(device)
        _YCBCR_TO_RGB = _YCBCR_TO_RGB.to(device)
        _YCBCR_OFFSET = _YCBCR_OFFSET.to(device)


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) RGB [0,1] → (B,3,H,W) YCbCr [0,1]."""
    _ensure_matrix_on(rgb.device)
    B, C, H, W = rgb.shape
    flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)        # (N, 3)
    ycbcr = flat @ _RGB_TO_YCBCR.T + _YCBCR_OFFSET       # (N, 3)
    return ycbcr.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B,3,H,W)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) YCbCr [0,1] → (B,3,H,W) RGB [0,1]."""
    _ensure_matrix_on(ycbcr.device)
    B, C, H, W = ycbcr.shape
    flat = ycbcr.permute(0, 2, 3, 1).reshape(-1, 3)       # (N, 3)
    rgb = (flat - _YCBCR_OFFSET) @ _YCBCR_TO_RGB.T        # (N, 3)
    return rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2)    # (B,3,H,W)


# ═══════════════════════════════════════════════════════════════════════════════
# SOBEL EDGE DETECTION (GPU, depthwise conv2d)
# ═══════════════════════════════════════════════════════════════════════════════

_SOBEL_X: Optional[torch.Tensor] = None
_SOBEL_Y: Optional[torch.Tensor] = None


def _ensure_sobel_on(device: torch.device) -> None:
    global _SOBEL_X, _SOBEL_Y
    if _SOBEL_X is not None and _SOBEL_X.device == device:
        return
    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=device,
    ).view(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=device,
    ).view(1, 1, 3, 3)
    _SOBEL_X = sx
    _SOBEL_Y = sy


def _get_edge_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised Sobel gradient magnitude.

    Input : (B, C, H, W) float32 [0,1]  (RGB or single-channel)
    Output: (B, 1, H, W) float32 [0,1]  edge magnitude
    """
    _ensure_sobel_on(tensor.device)

    # Convert to grayscale if RGB/RGBA
    if tensor.shape[1] >= 3:
        gray = (
            0.299 * tensor[:, 0:1]
            + 0.587 * tensor[:, 1:2]
            + 0.114 * tensor[:, 2:3]
        )
    else:
        gray = tensor[:, 0:1]

    padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(padded, _SOBEL_X)
    gy = F.conv2d(padded, _SOBEL_Y)
    mag = torch.sqrt(gx * gx + gy * gy)

    # Normalise to [0, 1]
    max_val = mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
    return (mag / max_val).clamp_(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL STABILIZATION — thread-safe EMA buffers
# ═══════════════════════════════════════════════════════════════════════════════

_temporal_lock = threading.Lock()
_temporal_buffers: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _apply_temporal(
    tensor: torch.Tensor,
    key: Tuple[int, int, int],
    alpha: float = 0.15,
) -> torch.Tensor:
    """
    Exponential moving average across frames.

    ``ema = (1 - alpha) * ema_prev + alpha * tensor``

    *key* is ``(height, width, channels)`` — one buffer per resolution.
    Thread-safe via ``_temporal_lock``.
    """
    alpha = max(0.0, min(1.0, alpha))
    with _temporal_lock:
        prev = _temporal_buffers.get(key)
        if prev is None or prev.shape != tensor.shape or prev.device != tensor.device:
            # First frame or resolution change — seed with current
            _temporal_buffers[key] = tensor.clone()
            return tensor
        ema = torch.lerp(prev, tensor, alpha)
        _temporal_buffers[key] = ema.clone()
    return ema


def clear_temporal_buffers() -> None:
    """Flush all temporal EMA state (e.g. on seek or new video)."""
    with _temporal_lock:
        _temporal_buffers.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION BLENDER
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionBlender:
    """
    Stateless, GPU-only blending utilities for combining two SR predictions.

    All inputs/outputs are ``(B, C, H, W)`` float32 tensors on the same CUDA
    device.  Size mismatches are resolved via bicubic resize to the *primary*
    tensor's spatial dimensions.
    """

    # ──────────────────────────────────────────────────────────────────
    # CORE BLEND
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def blend(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        """
        Blend two SR outputs:  ``result = lerp(primary, secondary, alpha)``.

        Parameters
        ----------
        primary : Tensor (B, C, H, W)
            Dominant prediction (weight = 1 − alpha).
        secondary : Tensor (B, C, H, W)
            Secondary prediction (weight = alpha).
        alpha : float
            Blend ratio in [0, 1].  0 → pure primary, 1 → pure secondary.

        Returns
        -------
        Tensor (B, C, H, W), float32, same device, clamped [0, 1].
        """
        alpha = max(0.0, min(1.0, alpha))
        secondary = PredictionBlender._match_spatial(secondary, primary)
        return torch.lerp(primary, secondary, alpha).clamp_(0.0, 1.0)

    @staticmethod
    def blend_masked(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        """
        Spatially-varying blend using a soft mask.

        ``result = primary * (1 - mask*alpha) + secondary * mask*alpha``

        Parameters
        ----------
        mask : Tensor (B, 1, H, W) or (1, 1, H, W)
            Per-pixel blend weight in [0, 1].  Broadcast-safe.
        """
        alpha = max(0.0, min(1.0, alpha))
        secondary = PredictionBlender._match_spatial(secondary, primary)
        mask = PredictionBlender._match_spatial(mask, primary)
        weight = (mask * alpha).clamp_(0.0, 1.0)
        return (primary * (1.0 - weight) + secondary * weight).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # LUMINANCE-ONLY BLEND (YCbCr)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def blend_luma_only(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        alpha: float = 0.3,
    ) -> torch.Tensor:
        """
        Blend only the luminance (Y) channel in YCbCr space.

        Preserves the chroma (Cb, Cr) of *primary*, avoiding colour shifts
        that full-RGB blending can introduce with GAN outputs.

        Input tensors must be (B, 3, H, W) RGB float32 [0,1].
        """
        alpha = max(0.0, min(1.0, alpha))
        secondary = PredictionBlender._match_spatial(secondary, primary)

        p_ycbcr = _rgb_to_ycbcr(primary)
        s_ycbcr = _rgb_to_ycbcr(secondary)

        # Blend only Y channel
        blended_y = torch.lerp(p_ycbcr[:, 0:1], s_ycbcr[:, 0:1], alpha)
        result_ycbcr = torch.cat([blended_y, p_ycbcr[:, 1:3]], dim=1)

        return _ycbcr_to_rgb(result_ycbcr).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # EDGE-AWARE BLEND
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def blend_edge_aware(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        alpha: float = 0.3,
        edge_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Edge-aware blend: stronger blend on edges, weaker on flat regions.

        Computes a Sobel edge mask from *primary*, scales it by
        *edge_strength*, then uses it as a spatially-varying weight.

        Useful for injecting edge detail from a sharp model while keeping
        flat-region smoothness from a denoising model.
        """
        alpha = max(0.0, min(1.0, alpha))
        edge_strength = max(0.0, min(3.0, edge_strength))
        secondary = PredictionBlender._match_spatial(secondary, primary)

        edge_mask = _get_edge_mask(primary) * edge_strength
        edge_mask = edge_mask.clamp_(0.0, 1.0)

        weight = (edge_mask * alpha).clamp_(0.0, 1.0)
        return (primary * (1.0 - weight) + secondary * weight).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # FREQUENCY-AWARE BLEND
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def blend_frequency(
        primary: torch.Tensor,
        secondary: torch.Tensor,
        alpha_low: float = 0.1,
        alpha_high: float = 0.5,
        sigma: float = 4.0,
    ) -> torch.Tensor:
        """
        Blend low and high frequency bands with independent alphas.

        Uses Gaussian blur to split each prediction into low+high bands,
        then blends each band separately before recombining.

        Useful for: keep primary's structure, inject secondary's texture.
        """
        secondary = PredictionBlender._match_spatial(secondary, primary)

        low_p, high_p = PredictionBlender._freq_split(primary, sigma)
        low_s, high_s = PredictionBlender._freq_split(secondary, sigma)

        blended_low = torch.lerp(low_p, low_s, max(0.0, min(1.0, alpha_low)))
        blended_high = torch.lerp(high_p, high_s, max(0.0, min(1.0, alpha_high)))

        return (blended_low + blended_high).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # HALLUCINATION SUPPRESSION
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def hallucination_suppress(
        sr_output: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor,
        reduction: float = 0.5,
    ) -> torch.Tensor:
        """
        Suppress hallucinated detail by blending toward *reference* in
        regions where *mask* is high.
        """
        reduction = max(0.0, min(1.0, reduction))
        reference = PredictionBlender._match_spatial(reference, sr_output)
        mask = PredictionBlender._match_spatial(mask, sr_output)
        weight = (mask * reduction).clamp_(0.0, 1.0)
        return (sr_output * (1.0 - weight) + reference * weight).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # ADAPTIVE DETAIL RESIDUAL (ADR)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def apply_detail_residual(
        structure: torch.Tensor,
        gan: torch.Tensor,
        detail_strength: float = 0.3,
        luma_only: bool = False,
        sigma: float = 1.2,
    ) -> torch.Tensor:
        """
        Adaptive Detail Residual — extract high-frequency texture from the
        secondary (GAN) output and inject it into the primary (structure) output.

        Why ADR improves realism:
          Structure-focused models (RCAN, EDSR) produce clean, artifact-free
          results but often lack fine texture (skin pores, fabric weave, foliage
          grain).  GAN models (RealESRGAN, ESRGAN) hallucinate rich texture but
          can introduce colour shifts and ringing.  ADR isolates *only* the
          high-frequency detail from the GAN output (via Gaussian residual
          extraction) and adds it to the structure base, combining the best of
          both without full-image blending artefacts.

        Residual extraction:
          ``blur = GaussianBlur(gan, sigma=1.2)``
          ``residual = gan - blur``
          This isolates detail above ~1.2px wavelength — the texture band that
          GANs excel at synthesising.

        Artifact prevention:
          - The residual is *additive*, not a replacement — structure geometry
            is never overwritten, only enriched.
          - ``detail_strength`` clamps to [0, 1] so the residual cannot exceed
            the original GAN texture energy.
          - Final clamp to [0, 1] prevents out-of-range pixel values.
          - When ``luma_only=True`` the residual is restricted to the Y channel
            in YCbCr space, preventing GAN chroma artefacts from leaking in.

        Parameters
        ----------
        structure : Tensor (B, C, H, W)
            Primary (structure) SR output, float32 [0,1].
        gan : Tensor (B, C, H, W)
            Secondary (GAN / texture) SR output, float32 [0,1].
        detail_strength : float
            How much GAN texture to inject.  0 = disabled, 1 = full residual.
        luma_only : bool
            If True, inject residual into luminance channel only (YCbCr),
            preserving the structure model's chroma to avoid colour shifts.
        sigma : float
            Gaussian blur sigma for residual extraction.  1.2 targets the
            texture-detail band without touching structural edges.

        Returns
        -------
        Tensor (B, C, H, W), float32, same device, clamped [0, 1].
        """
        if detail_strength < 1e-4:
            return structure

        detail_strength = max(0.0, min(1.0, detail_strength))

        # Ensure GAN output matches structure spatial dimensions
        gan = PredictionBlender._match_spatial(gan, structure)

        if luma_only and structure.shape[1] == 3:
            # Work in YCbCr: inject residual into Y only, keep structure chroma
            gan_ycbcr = _rgb_to_ycbcr(gan)
            struct_ycbcr = _rgb_to_ycbcr(structure)

            # Extract luminance residual: high-freq detail from GAN's Y channel
            gan_y = gan_ycbcr[:, 0:1]                           # (B, 1, H, W)
            gan_y_low, gan_y_high = PredictionBlender._freq_split(gan_y, sigma)

            # Inject into structure Y channel
            enhanced_y = struct_ycbcr[:, 0:1] + detail_strength * gan_y_high

            # Recombine with structure chroma (Cb, Cr untouched)
            result_ycbcr = torch.cat([enhanced_y, struct_ycbcr[:, 1:3]], dim=1)
            return _ycbcr_to_rgb(result_ycbcr).clamp_(0.0, 1.0)

        # Full-RGB residual injection
        gan_low, gan_high = PredictionBlender._freq_split(gan, sigma)
        return (structure + detail_strength * gan_high).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # UNSHARP MASK (GPU)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def apply_sharpen(
        tensor: torch.Tensor,
        strength: float = 0.3,
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """
        GPU unsharp mask:  ``sharpened = tensor + strength * (tensor - blur)``.

        Parameters
        ----------
        strength : float
            Sharpening intensity.  0 = no effect, 1 = aggressive.
        sigma : float
            Gaussian blur sigma for the low-pass component.
        """
        if strength < 1e-4:
            return tensor

        strength = max(0.0, min(2.0, strength))
        low, high = PredictionBlender._freq_split(tensor, sigma)
        return (tensor + strength * high).clamp_(0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────
    # TEMPORAL STABILIZATION
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def apply_temporal(
        tensor: torch.Tensor,
        key: Tuple[int, int, int],
        alpha: float = 0.15,
    ) -> torch.Tensor:
        """
        Apply temporal EMA stabilization.

        Delegates to the module-level ``_apply_temporal`` which manages
        thread-safe buffers keyed by ``(height, width, channels)``.
        """
        if alpha < 1e-4:
            return tensor
        return _apply_temporal(tensor, key, alpha)

    # ──────────────────────────────────────────────────────────────────
    # RGBA SUPPORT
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def split_alpha(
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Split (B,4,H,W) RGBA into (B,3,H,W) RGB + (B,1,H,W) alpha.

        If input is (B,3,H,W) or fewer channels, returns (tensor, None).
        """
        if tensor.shape[1] >= 4:
            return tensor[:, :3], tensor[:, 3:4]
        return tensor, None

    @staticmethod
    def merge_alpha(
        rgb: torch.Tensor,
        alpha: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Recombine RGB + alpha into RGBA.  If alpha is None, returns RGB as-is.

        Alpha is bilinearly resized to match RGB spatial dims if needed.
        """
        if alpha is None:
            return rgb
        _, _, rh, rw = rgb.shape
        _, _, ah, aw = alpha.shape
        if ah != rh or aw != rw:
            alpha = F.interpolate(
                alpha, size=(rh, rw), mode="bilinear", align_corners=False
            )
        return torch.cat([rgb, alpha], dim=1)

    # ──────────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _match_spatial(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize *src* to match *ref*'s spatial dims if they differ."""
        _, _, rh, rw = ref.shape
        _, _, sh, sw = src.shape
        if sh != rh or sw != rw:
            src = F.interpolate(src, size=(rh, rw), mode="bicubic", align_corners=False)
        return src

    @staticmethod
    def _freq_split(
        x: torch.Tensor, sigma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split *x* into (low, high) frequency bands via Gaussian blur."""
        ks = int(math.ceil(sigma * 6)) | 1  # ensure odd
        B, C, H, W = x.shape

        # Build separable 1-D Gaussian kernel
        coords = torch.arange(ks, device=x.device, dtype=x.dtype) - (ks - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()

        # Depthwise separable convolution (horizontal then vertical)
        kernel_h = g.view(1, 1, 1, ks).expand(C, 1, 1, ks)
        kernel_v = g.view(1, 1, ks, 1).expand(C, 1, ks, 1)

        padded = F.pad(x, (ks // 2, ks // 2, ks // 2, ks // 2), mode="reflect")
        low = F.conv2d(padded, kernel_h, groups=C)
        low = F.conv2d(
            F.pad(low, (0, 0, ks // 2, ks // 2), mode="reflect"),
            kernel_v,
            groups=C,
        )

        # Crop to original size
        low = low[:, :, :H, :W]
        high = x - low
        return low, high
