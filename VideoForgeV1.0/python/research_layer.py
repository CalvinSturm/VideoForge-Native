"""
VideoForge Research Layer — Multi-Model Super-Resolution Blending Framework

A modular, research-grade blending system that intercepts outputs from the existing
production upscaler and blends multiple SR model outputs using frequency-aware,
hallucination-guided, and edge-routed strategies.

This is an add-on research sandbox. It does NOT replace the core engine.

Device Discipline:
  - Weight blending: CPU ONLY (WeightBlender)
  - Prediction blending: GPU (PredictionBlender)
  - Model inference: GPU + AMP (ModelSlot.infer)
  - SHM → Tensor conversion: CPU → GPU
"""

import json
import math
import mmap
import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION STRUCTURES
# =============================================================================

class HFMethod(Enum):
    LAPLACIAN = "laplacian"
    SOBEL = "sobel"
    HIGHPASS = "highpass"
    FFT = "fft"


class PerformancePreset(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    ULTRA = "ultra"


class ModelRole(Enum):
    STRUCTURE = "structure"
    TEXTURE = "texture"
    PERCEPTUAL = "perceptual"
    DIFFUSION = "diffusion"


@dataclass
class BlendParameters:
    """All UI-controllable parameters. No hardcoded constants — everything is configurable."""

    # Global model weights (sum does NOT need to equal 1; normalized at blend time)
    alpha_structure: float = 0.5
    alpha_texture: float = 0.3
    alpha_perceptual: float = 0.15
    alpha_diffusion: float = 0.05

    # Frequency band weights (applied during frequency-separated blending)
    low_freq_strength: float = 1.0
    mid_freq_strength: float = 1.0
    high_freq_strength: float = 1.0

    # Hallucination controls
    h_sensitivity: float = 1.0       # Multiplier on hallucination threshold
    h_blend_reduction: float = 0.5   # How much to reduce aggressive model contribution [0,1]

    # Spatial routing
    edge_model_bias: float = 0.7     # Bias toward structure model on edges
    texture_model_bias: float = 0.7  # Bias toward texture model in textured regions
    flat_region_suppression: float = 0.3  # Suppress detail enhancement in flat regions

    # Analysis config
    hf_method: str = "laplacian"     # 'laplacian', 'sobel', 'highpass', 'fft'

    # Performance preset (affects which models run)
    preset: str = "balanced"

    # Frequency band Gaussian sigmas (configurable for experimentation)
    freq_low_sigma: float = 4.0
    freq_mid_sigma: float = 1.5

    # Spatial routing thresholds
    edge_threshold: float = 0.5
    texture_threshold: float = 0.2

    # Pipeline stage mix ratio: spatial vs frequency blend contribution
    spatial_freq_mix: float = 0.5  # 0.0 = all spatial, 1.0 = all frequency

    def get_active_roles(self) -> List[ModelRole]:
        """Determine which model roles are active based on preset."""
        mapping = {
            "fast": [ModelRole.STRUCTURE],
            "balanced": [ModelRole.STRUCTURE, ModelRole.TEXTURE],
            "high_quality": [ModelRole.STRUCTURE, ModelRole.TEXTURE, ModelRole.PERCEPTUAL],
            "ultra": [ModelRole.STRUCTURE, ModelRole.TEXTURE, ModelRole.PERCEPTUAL, ModelRole.DIFFUSION],
        }
        return mapping.get(self.preset, [ModelRole.STRUCTURE, ModelRole.TEXTURE])

    def get_alpha(self, role: ModelRole) -> float:
        return {
            ModelRole.STRUCTURE: self.alpha_structure,
            ModelRole.TEXTURE: self.alpha_texture,
            ModelRole.PERCEPTUAL: self.alpha_perceptual,
            ModelRole.DIFFUSION: self.alpha_diffusion,
        }[role]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_structure": self.alpha_structure,
            "alpha_texture": self.alpha_texture,
            "alpha_perceptual": self.alpha_perceptual,
            "alpha_diffusion": self.alpha_diffusion,
            "low_freq_strength": self.low_freq_strength,
            "mid_freq_strength": self.mid_freq_strength,
            "high_freq_strength": self.high_freq_strength,
            "h_sensitivity": self.h_sensitivity,
            "h_blend_reduction": self.h_blend_reduction,
            "edge_model_bias": self.edge_model_bias,
            "texture_model_bias": self.texture_model_bias,
            "flat_region_suppression": self.flat_region_suppression,
            "hf_method": self.hf_method,
            "preset": self.preset,
            "freq_low_sigma": self.freq_low_sigma,
            "freq_mid_sigma": self.freq_mid_sigma,
            "edge_threshold": self.edge_threshold,
            "texture_threshold": self.texture_threshold,
            "spatial_freq_mix": self.spatial_freq_mix,
        }


# =============================================================================
# HIGH-FREQUENCY ENERGY ANALYSIS
# =============================================================================

class HFAnalyzer:
    """
    Computes high-frequency energy maps using four distinct methods.
    All outputs are normalized per-channel to [0, 1] for use as hallucination masks,
    frequency routing maps, and edge-aware blending weights.
    """

    # Cached kernels to avoid re-allocation per frame
    _kernel_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _get_cached_kernel(name: str, device: torch.device, dtype: torch.dtype,
                           factory) -> torch.Tensor:
        key = f"{name}_{device}_{dtype}"
        if key not in HFAnalyzer._kernel_cache:
            HFAnalyzer._kernel_cache[key] = factory().to(device=device, dtype=dtype)
        return HFAnalyzer._kernel_cache[key]

    @staticmethod
    def compute(image: torch.Tensor, method: str) -> torch.Tensor:
        """
        Dispatch to the selected HF analysis method.

        Args:
            image: (B, C, H, W) tensor on GPU, float32, range [0, 1]
            method: one of 'laplacian', 'sobel', 'highpass', 'fft'

        Returns:
            (B, C, H, W) normalized energy map on same device as input
        """
        dispatch = {
            "laplacian": HFAnalyzer.laplacian_energy,
            "sobel": HFAnalyzer.sobel_energy,
            "highpass": HFAnalyzer.highpass_energy,
            "fft": HFAnalyzer.fft_energy,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"Unknown HF method: {method}. Valid: {list(dispatch.keys())}")
        return fn(image)

    @staticmethod
    def _normalize_per_channel(energy: torch.Tensor) -> torch.Tensor:
        """Normalize each channel independently to [0, 1]."""
        B, C, H, W = energy.shape
        flat = energy.view(B, C, -1)
        mins = flat.min(dim=-1, keepdim=True).values
        maxs = flat.max(dim=-1, keepdim=True).values
        denom = (maxs - mins).clamp(min=1e-8)
        normalized = (flat - mins) / denom
        return normalized.view(B, C, H, W)

    @staticmethod
    def laplacian_energy(image: torch.Tensor) -> torch.Tensor:
        """Laplacian filter energy: measures second-order intensity changes."""
        def _make():
            return torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)

        kernel = HFAnalyzer._get_cached_kernel("laplacian", image.device, image.dtype, _make)
        B, C, H, W = image.shape
        kernel_expanded = kernel.expand(C, 1, 3, 3)
        laplacian = F.conv2d(image, kernel_expanded, padding=1, groups=C)
        energy = laplacian.abs()
        return HFAnalyzer._normalize_per_channel(energy)

    @staticmethod
    def sobel_energy(image: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude: measures first-order edge strength."""
        def _make_x():
            return torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        def _make_y():
            return torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)

        sobel_x = HFAnalyzer._get_cached_kernel("sobel_x", image.device, image.dtype, _make_x)
        sobel_y = HFAnalyzer._get_cached_kernel("sobel_y", image.device, image.dtype, _make_y)

        B, C, H, W = image.shape
        kx = sobel_x.expand(C, 1, 3, 3)
        ky = sobel_y.expand(C, 1, 3, 3)

        gx = F.conv2d(image, kx, padding=1, groups=C)
        gy = F.conv2d(image, ky, padding=1, groups=C)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return HFAnalyzer._normalize_per_channel(magnitude)

    @staticmethod
    def highpass_energy(image: torch.Tensor) -> torch.Tensor:
        """High-pass convolution filter: 5x5 kernel subtracting local mean."""
        def _make():
            return (torch.ones(1, 1, 5, 5) / 25.0)

        box = HFAnalyzer._get_cached_kernel("highpass_box", image.device, image.dtype, _make)
        B, C, H, W = image.shape
        box_expanded = box.expand(C, 1, 5, 5)
        low_pass = F.conv2d(image, box_expanded, padding=2, groups=C)
        high_pass = (image - low_pass).abs()
        return HFAnalyzer._normalize_per_channel(high_pass)

    @staticmethod
    def fft_energy(image: torch.Tensor) -> torch.Tensor:
        """FFT high-band power: energy in the upper frequency spectrum. Batched computation."""
        B, C, H, W = image.shape

        # Batch FFT across all B*C channels simultaneously
        flat = image.reshape(B * C, H, W)
        fft = torch.fft.fft2(flat)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # High-pass mask: suppress center (low frequencies)
        cy, cx = H // 2, W // 2
        radius = min(H, W) // 4
        yy = torch.arange(H, device=image.device).unsqueeze(1).expand(H, W)
        xx = torch.arange(W, device=image.device).unsqueeze(0).expand(H, W)
        dist = torch.sqrt(((yy - cy).float()) ** 2 + ((xx - cx).float()) ** 2)
        mask = (dist > radius).float().unsqueeze(0)  # (1, H, W) broadcasts over B*C

        fft_high = fft_shifted * mask
        spatial = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)))
        energy = spatial.abs().reshape(B, C, H, W)

        return HFAnalyzer._normalize_per_channel(energy)


# =============================================================================
# HALLUCINATION MASK GENERATION
# =============================================================================

class HallucinationDetector:
    """
    Detects hallucinated frequencies by comparing SR output HF energy against
    a bicubic-upscaled reference from the original LR input.

    Bicubic interpolation cannot hallucinate frequencies not present in the input.

    Regions where SR HF energy greatly exceeds bicubic HF energy are flagged
    as potential hallucinations.
    """

    @staticmethod
    def generate_mask(
        lr_input: torch.Tensor,
        sr_output: torch.Tensor,
        scale: int,
        hf_method: str,
        sensitivity: float,
    ) -> torch.Tensor:
        """
        Generate a hallucination mask.

        Args:
            lr_input: (B, C, H, W) low-resolution input
            sr_output: (B, C, sH, sW) super-resolved output
            scale: upscaling factor
            hf_method: HF analysis method to use
            sensitivity: multiplier on the detection threshold

        Returns:
            (B, 1, sH, sW) hallucination mask in [0, 1] where 1 = hallucinated
        """
        # Bicubic interpolation cannot hallucinate frequencies not present in the input.
        bicubic_ref = F.interpolate(
            lr_input, scale_factor=scale, mode="bicubic", align_corners=False,
        ).clamp(0, 1)

        _, _, sh, sw = sr_output.shape
        if bicubic_ref.shape[2] != sh or bicubic_ref.shape[3] != sw:
            bicubic_ref = F.interpolate(
                bicubic_ref, size=(sh, sw), mode="bicubic", align_corners=False
            ).clamp(0, 1)

        hf_sr = HFAnalyzer.compute(sr_output, hf_method)
        hf_bic = HFAnalyzer.compute(bicubic_ref, hf_method)

        B, C, H, W = hf_bic.shape
        bic_mean = hf_bic.view(B, C, -1).mean(dim=-1, keepdim=True).unsqueeze(-1)
        bic_std = hf_bic.view(B, C, -1).std(dim=-1, keepdim=True).unsqueeze(-1)

        threshold = bic_mean + sensitivity * bic_std.clamp(min=1e-6)
        excess = (hf_sr - threshold).clamp(min=0)
        denom = threshold.clamp(min=1e-6)
        hallucination_per_channel = (excess / denom).clamp(0, 1)

        # Collapse channels to single mask (max across channels)
        hallucination_mask = hallucination_per_channel.max(dim=1, keepdim=True).values
        return hallucination_mask


# =============================================================================
# FREQUENCY BAND SPLITTING
# =============================================================================

class FrequencyBandSplitter:
    """
    Splits an image into low, mid, and high frequency bands using
    Gaussian blur cascades. Fully differentiable, GPU-resident.
    """

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        return kernel_2d

    @staticmethod
    def split(
        image: torch.Tensor,
        low_sigma: float = 4.0,
        mid_sigma: float = 1.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split image into three frequency bands.

        Returns:
            (low, mid, high) frequency bands, each (B, C, H, W)
            Invariant: low + mid + high ≈ image
        """
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype

        ks_low = int(math.ceil(low_sigma * 6)) | 1
        k_low = FrequencyBandSplitter._gaussian_kernel(ks_low, low_sigma, device, dtype)
        k_low = k_low.unsqueeze(0).unsqueeze(0).expand(C, 1, ks_low, ks_low)
        pad_low = ks_low // 2
        low = F.conv2d(image, k_low, padding=pad_low, groups=C)

        ks_mid = int(math.ceil(mid_sigma * 6)) | 1
        k_mid = FrequencyBandSplitter._gaussian_kernel(ks_mid, mid_sigma, device, dtype)
        k_mid = k_mid.unsqueeze(0).unsqueeze(0).expand(C, 1, ks_mid, ks_mid)
        pad_mid = ks_mid // 2
        mid_low = F.conv2d(image, k_mid, padding=pad_mid, groups=C)

        mid = mid_low - low
        high = image - mid_low
        return low, mid, high


# =============================================================================
# SPATIAL ROUTING (EDGE / TEXTURE / FLAT CLASSIFICATION)
# =============================================================================

class SpatialRouter:
    """
    Classifies pixels into edge, texture, and flat regions using gradient
    analysis. Returns soft routing masks for model contribution weighting.
    """

    @staticmethod
    def compute_routing_masks(
        image: torch.Tensor,
        edge_threshold: float = 0.5,
        texture_threshold: float = 0.2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spatial routing masks.

        Returns:
            (edge_mask, texture_mask, flat_mask) each (B, 1, H, W) in [0, 1]
            Invariant: edge_mask + texture_mask + flat_mask ≈ 1.0
        """
        gray = image.mean(dim=1, keepdim=True)

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            device=image.device, dtype=image.dtype
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            device=image.device, dtype=image.dtype
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        gmax = gradient_mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        gradient_norm = gradient_mag / gmax

        box = torch.ones(1, 1, 5, 5, device=image.device, dtype=image.dtype) / 25.0
        local_mean = F.conv2d(gray, box, padding=2)
        local_sq_mean = F.conv2d(gray ** 2, box, padding=2)
        local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)
        var_max = local_var.amax(dim=(2, 3), keepdim=True).clamp(min=1e-8)
        var_norm = local_var / var_max

        # Soft classification using sigmoid transitions
        edge_mask = torch.sigmoid((gradient_norm - edge_threshold) * 10.0)
        texture_indicator = torch.sigmoid((var_norm - texture_threshold) * 10.0)
        texture_mask = texture_indicator * (1.0 - edge_mask)
        flat_mask = (1.0 - edge_mask - texture_mask).clamp(0, 1)

        return edge_mask, texture_mask, flat_mask


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelSlot:
    """Holds a model, its role, and lifecycle state with GPU ↔ CPU migration."""

    def __init__(self, role: ModelRole, model: Optional[nn.Module] = None, scale: int = 4):
        self.role = role
        self.model = model
        self.scale = scale
        self.enabled = model is not None
        self.device = torch.device("cpu")

    def to_gpu(self, device: torch.device) -> None:
        if self.model is not None:
            self.model = self.model.to(device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.device = device

    def free_gpu(self) -> None:
        """Release GPU memory by moving model to CPU."""
        if self.model is not None and self.device.type == "cuda":
            self.model = self.model.to("cpu")
            self.device = torch.device("cpu")
            torch.cuda.empty_cache()

    def infer(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Run inference with AMP on GPU. Model migrates to device if needed."""
        if self.model is None:
            raise RuntimeError(f"No model loaded for role {self.role.value}")

        if self.device != device:
            self.to_gpu(device)

        tensor = tensor.to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                return self.model(tensor)


# =============================================================================
# WEIGHT BLENDING (CPU-ONLY)
# =============================================================================

class WeightBlender:
    """
    Interpolate model state dicts on CPU. Used when models share architecture
    and you want to create blended weight checkpoints.
    NO GPU memory may be used during weight blending.
    """

    @staticmethod
    def blend_state_dicts(
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Linearly interpolate multiple state dicts on CPU.

        Returns:
            Blended state dict (all tensors on CPU)
        """
        if len(state_dicts) != len(weights):
            raise ValueError("state_dicts and weights must have same length")
        if len(state_dicts) == 0:
            raise ValueError("Need at least one state dict")

        w_sum = sum(weights)
        if w_sum < 1e-8:
            raise ValueError("Weight sum is effectively zero")
        norm_weights = [w / w_sum for w in weights]

        keys = set(state_dicts[0].keys())
        for sd in state_dicts[1:]:
            if set(sd.keys()) != keys:
                raise ValueError("All state dicts must have identical key sets for weight blending")

        blended = {}
        for key in state_dicts[0].keys():
            # CPU-ONLY: ensure all tensors are on CPU before blending
            tensors = [sd[key].cpu().float() for sd in state_dicts]
            result = torch.zeros_like(tensors[0])
            for t, w in zip(tensors, norm_weights):
                result += t * w
            blended[key] = result.to(dtype=state_dicts[0][key].dtype)

        return blended

    @staticmethod
    def blend_and_load(
        model: nn.Module,
        state_dicts: List[Dict[str, torch.Tensor]],
        weights: List[float],
        device: torch.device,
    ) -> nn.Module:
        """Blend on CPU, then move finished model to GPU in one shot."""
        blended = WeightBlender.blend_state_dicts(state_dicts, weights)
        model.load_state_dict(blended, strict=True)
        model.eval()
        model = model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        return model


# =============================================================================
# PREDICTION BLENDING (GPU-ONLY)
# =============================================================================

class PredictionBlender:
    """
    GPU-resident tensor blending operations for combining model predictions.
    All operations stay on GPU to avoid CPU↔GPU transfers.
    """

    @staticmethod
    def weighted_average(
        tensors: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        """Alpha-weighted combination of prediction tensors on GPU."""
        w_sum = sum(weights)
        if w_sum < 1e-8:
            raise ValueError("Weight sum is zero")

        result = torch.zeros_like(tensors[0])
        for t, w in zip(tensors, weights):
            result = result + t * (w / w_sum)
        return result

    @staticmethod
    def masked_blend(
        a: torch.Tensor,
        b: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Blend two tensors using a spatial mask. mask=1 selects b, mask=0 selects a."""
        return a * (1.0 - mask) + b * mask

    @staticmethod
    def frequency_weighted_blend(
        tensors: List[torch.Tensor],
        alphas: List[float],
        band_weights: Tuple[float, float, float],
        low_sigma: float = 4.0,
        mid_sigma: float = 1.5,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Split each tensor into frequency bands, weight bands independently, recombine.
        Returns (blended, (low_band, mid_band, high_band)) for diagnostics.
        """
        lw, mw, hw = band_weights
        accum_low = torch.zeros_like(tensors[0])
        accum_mid = torch.zeros_like(tensors[0])
        accum_high = torch.zeros_like(tensors[0])
        total_alpha = 0.0

        for tensor, alpha in zip(tensors, alphas):
            if alpha < 1e-8:
                continue
            low, mid, high = FrequencyBandSplitter.split(tensor, low_sigma, mid_sigma)
            accum_low = accum_low + low * lw * alpha
            accum_mid = accum_mid + mid * mw * alpha
            accum_high = accum_high + high * hw * alpha
            total_alpha += alpha

        if total_alpha < 1e-8:
            return tensors[0], (accum_low, accum_mid, accum_high)

        inv = 1.0 / total_alpha
        out_low = accum_low * inv
        out_mid = accum_mid * inv
        out_high = accum_high * inv
        combined = out_low + out_mid + out_high
        return combined, (out_low, out_mid, out_high)

    @staticmethod
    def hallucination_suppress(
        blend: torch.Tensor,
        safe: torch.Tensor,
        mask: torch.Tensor,
        reduction: float,
    ) -> torch.Tensor:
        """Pull blend toward safe output where hallucination mask is high."""
        return blend * (1.0 - mask * reduction) + safe * (mask * reduction)

    @staticmethod
    def detail_inject(
        base: torch.Tensor,
        detail_source: torch.Tensor,
        alpha: float,
        low_sigma: float = 4.0,
        mid_sigma: float = 1.5,
    ) -> torch.Tensor:
        """Inject only the high-frequency detail from detail_source into base."""
        _, _, src_detail = FrequencyBandSplitter.split(detail_source, low_sigma, mid_sigma)
        _, _, base_detail = FrequencyBandSplitter.split(base, low_sigma, mid_sigma)
        new_detail = src_detail - base_detail
        return (base + new_detail * alpha).clamp(0, 1)


# =============================================================================
# LOCK-FREE SHM RING BUFFER (PYTHON SIDE)
# =============================================================================

# Must match Rust #[repr(C)] RingBufferState layout exactly
RING_HEADER_SIZE = 32  # 4 × u64: write_cursor, read_cursor, buffer_size, frame_id

class ShmRingReader:
    """
    Python-side reader for the lock-free SHM ring buffer written by Rust.
    Reads the atomic cursors and frame data without locks.
    The Rust side uses Acquire/Release ordering on the cursors; Python reads
    are inherently ordered on x86 (TSO) but we use struct.unpack for correctness.
    """

    def __init__(self, shm_path: str, slot_count: int, slot_byte_size: int):
        self.slot_count = slot_count
        self.slot_byte_size = slot_byte_size

        file = open(shm_path, "r+b")
        total = RING_HEADER_SIZE + slot_count * slot_byte_size
        self.mm = mmap.mmap(file.fileno(), total)
        file.close()

    def _read_header(self) -> Tuple[int, int, int, int]:
        raw = self.mm[:RING_HEADER_SIZE]
        write_cur, read_cur, buf_size, frame_id = struct.unpack("<QQQQ", raw)
        return write_cur, read_cur, buf_size, frame_id

    def has_data(self) -> bool:
        w, r, _, _ = self._read_header()
        return w != r

    def read_slot(self, index: int) -> memoryview:
        """Return a memoryview into slot data (zero-copy)."""
        offset = RING_HEADER_SIZE + index * self.slot_byte_size
        return memoryview(self.mm)[offset: offset + self.slot_byte_size]

    def advance_read_cursor(self) -> None:
        """Advance read cursor by 1 (modulo buffer_size). Matches Rust Release semantics."""
        _, r, buf_size, _ = self._read_header()
        new_r = (r + 1) % buf_size
        self.mm[8:16] = struct.pack("<Q", new_r)

    def current_frame_id(self) -> int:
        _, _, _, fid = self._read_header()
        return fid

    def close(self) -> None:
        if self.mm:
            self.mm.close()


# =============================================================================
# ZENOH CONTROL CHANNEL LISTENER (PYTHON SIDE)
# =============================================================================

class ZenohControlListener:
    """
    Subscribes to vf/control/** and applies parameter updates to the research layer.
    Mirrors the Rust-side control.rs subscriber so either side can originate updates.
    """

    def __init__(self, session, research_layer: "VideoForgeResearchLayer",
                 prefix: str = "vf/control"):
        self.session = session
        self.layer = research_layer
        self.prefix = prefix
        self.sub = None
        self.pub = None

    def start(self) -> None:
        self.sub = self.session.declare_subscriber(
            f"{self.prefix}/**", self._on_message
        )
        self.pub = self.session.declare_publisher(f"{self.prefix}/status")
        print(f"[ResearchLayer] Zenoh control listener active on {self.prefix}/**", flush=True)

    def _on_message(self, sample) -> None:
        try:
            topic = str(sample.key_expr)
            payload = json.loads(sample.payload.to_bytes().decode("utf-8"))

            if topic.endswith("/params"):
                self.layer.update_params(payload)
                self._ack("params_updated", self.layer.get_params())

            elif topic.endswith("/model_enable"):
                role_name = payload.get("role")
                enabled = payload.get("enabled", True)
                role = ModelRole(role_name)
                if enabled:
                    self.layer.enable_model(role)
                else:
                    self.layer.disable_model(role)
                self._ack("model_toggled", {"role": role_name, "enabled": enabled})

            elif topic.endswith("/get_params"):
                self._ack("current_params", self.layer.get_params())

            elif topic.endswith("/get_diagnostics"):
                diag = self.layer.get_diagnostics()
                # Convert to serializable format (shapes only for large data)
                summary = {}
                for k, v in diag.items():
                    if v is not None:
                        summary[k] = {"shape": list(v.shape), "min": float(v.min()),
                                       "max": float(v.max()), "mean": float(v.mean())}
                    else:
                        summary[k] = None
                self._ack("diagnostics", summary)

            elif topic.endswith("/blend_control"):
                # Handle the specific JSON format from Rust control.rs:
                # {"primary": str, "secondary": str, "alpha": float, "hallucination_view": bool}
                primary = payload.get("primary")
                secondary = payload.get("secondary")
                alpha = payload.get("alpha", 0.5)
                h_view = payload.get("hallucination_view", False)

                # Map to research layer params
                updates = {}
                if primary and secondary:
                    # Both models active => balanced mode minimum
                    updates["preset"] = "balanced"
                    updates["alpha_structure"] = alpha if primary == "structure" else (1.0 - alpha)
                    updates["alpha_texture"] = alpha if primary == "texture" else (1.0 - alpha)

                # Warn if both models are diffusion (12GB VRAM risk)
                if primary == "diffusion" and secondary == "diffusion":
                    print("[ResearchLayer] WARNING: Both slots are diffusion models. "
                          "Estimated VRAM >12GB. Risk of OOM.", flush=True)
                    self._ack("vram_warning", {"message": "Both models are diffusion. >12GB VRAM risk."})

                if updates:
                    self.layer.update_params(updates)

                self._ack("blend_control_applied", payload)

        except Exception as e:
            print(f"[ResearchLayer] Control error: {e}", flush=True)
            traceback.print_exc()

    def _ack(self, status: str, data: Any) -> None:
        if self.pub:
            msg = json.dumps({"status": status, "data": data})
            self.pub.put(msg.encode("utf-8"))

    def stop(self) -> None:
        if self.sub:
            self.sub.undeclare()
        if self.pub:
            self.pub.undeclare()


# =============================================================================
# MAIN RESEARCH LAYER
# =============================================================================

class VideoForgeResearchLayer:
    """
    Research-grade multi-model blending framework for super-resolution.

    This layer:
    - Manages multiple SR models assigned to specialist roles
    - Computes HF energy maps and hallucination masks
    - Splits frequency bands for independent weighting
    - Routes model contributions spatially (edges, textures, flat regions)
    - Blends outputs using configurable strategies via PredictionBlender (GPU)
    - Exposes all parameters for live UI control

    The visual signature emerges from how models are combined, not from a single network.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = BlendParameters()
        self.models: Dict[ModelRole, ModelSlot] = {}
        self.scale: int = 4

        # Diagnostics: accessible for logging/visualization
        self.last_hf_energy: Optional[torch.Tensor] = None
        self.last_hallucination_mask: Optional[torch.Tensor] = None
        self.last_edge_mask: Optional[torch.Tensor] = None
        self.last_texture_mask: Optional[torch.Tensor] = None
        self.last_flat_mask: Optional[torch.Tensor] = None
        self.last_freq_bands: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.last_model_outputs: Dict[ModelRole, torch.Tensor] = {}

        # Optional Zenoh control listener
        self._control_listener: Optional[ZenohControlListener] = None

    # -----------------------------------------------------------------
    # MODEL MANAGEMENT
    # -----------------------------------------------------------------

    def register_model(self, role: ModelRole, model: nn.Module, scale: int = 4) -> None:
        slot = ModelSlot(role, model, scale)
        self.models[role] = slot
        self.scale = scale
        print(f"[ResearchLayer] Registered {role.value} model (scale={scale})", flush=True)

    def enable_model(self, role: ModelRole) -> None:
        if role in self.models:
            self.models[role].enabled = True
            self.models[role].to_gpu(self.device)

    def disable_model(self, role: ModelRole) -> None:
        if role in self.models:
            self.models[role].enabled = False
            self.models[role].free_gpu()

    def get_active_models(self) -> List[ModelSlot]:
        active_roles = self.params.get_active_roles()
        return [
            self.models[role]
            for role in active_roles
            if role in self.models and self.models[role].enabled
        ]

    # -----------------------------------------------------------------
    # ZENOH CONTROL INTEGRATION
    # -----------------------------------------------------------------

    def attach_zenoh_control(self, session, prefix: str = "vf/control") -> None:
        """Attach a Zenoh control listener for live UI parameter updates."""
        self._control_listener = ZenohControlListener(session, self, prefix)
        self._control_listener.start()

    def detach_zenoh_control(self) -> None:
        if self._control_listener:
            self._control_listener.stop()
            self._control_listener = None

    # -----------------------------------------------------------------
    # PARAMETER UPDATE (LIVE UI CONTROL)
    # -----------------------------------------------------------------

    def update_params(self, updates: Dict[str, Any]) -> None:
        """Update blending parameters from UI. All parameters are hot-swappable."""
        for key, value in updates.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                print(f"[ResearchLayer] Warning: unknown parameter '{key}'", flush=True)

    def get_params(self) -> Dict[str, Any]:
        return self.params.to_dict()

    # -----------------------------------------------------------------
    # CORE INFERENCE (GPU + AMP)
    # -----------------------------------------------------------------

    def _run_models(self, lr_tensor: torch.Tensor) -> Dict[ModelRole, torch.Tensor]:
        """Run all active models. GPU + AMP for each forward pass."""
        outputs = {}
        active = self.get_active_models()

        if not active:
            raise RuntimeError("No active models. Register and enable at least one model.")

        for slot in active:
            output = slot.infer(lr_tensor, self.device)
            outputs[slot.role] = output.float()

        self.last_model_outputs = outputs
        return outputs

    # -----------------------------------------------------------------
    # BLENDING STRATEGIES (delegate to PredictionBlender on GPU)
    # -----------------------------------------------------------------

    def _weighted_blend(self, outputs: Dict[ModelRole, torch.Tensor]) -> torch.Tensor:
        """Strategy 1: Alpha-weighted combination."""
        tensors = []
        weights = []
        for role, tensor in outputs.items():
            alpha = self.params.get_alpha(role)
            if alpha > 1e-8:
                tensors.append(tensor)
                weights.append(alpha)
        if not tensors:
            raise RuntimeError("All model weights are zero")
        return PredictionBlender.weighted_average(tensors, weights)

    def _frequency_blend(self, outputs: Dict[ModelRole, torch.Tensor]) -> torch.Tensor:
        """Strategy 2: Frequency-separated blending with per-band weights."""
        tensors = []
        alphas = []
        for role, tensor in outputs.items():
            alpha = self.params.get_alpha(role)
            if alpha > 1e-8:
                tensors.append(tensor)
                alphas.append(alpha)

        if not tensors:
            raise RuntimeError("No valid model outputs for frequency blending")

        band_weights = (
            self.params.low_freq_strength,
            self.params.mid_freq_strength,
            self.params.high_freq_strength,
        )
        combined, bands = PredictionBlender.frequency_weighted_blend(
            tensors, alphas, band_weights,
            self.params.freq_low_sigma, self.params.freq_mid_sigma,
        )
        self.last_freq_bands = bands
        return combined

    def _hallucination_guided_blend(
        self,
        lr_input: torch.Tensor,
        outputs: Dict[ModelRole, torch.Tensor],
    ) -> torch.Tensor:
        """Strategy 3: Reduce aggressive models where HF exceeds bicubic reference."""
        base_blend = self._weighted_blend(outputs)

        structure_output = outputs.get(ModelRole.STRUCTURE)
        if structure_output is None:
            return base_blend

        h_mask = HallucinationDetector.generate_mask(
            lr_input=lr_input,
            sr_output=base_blend,
            scale=self.scale,
            hf_method=self.params.hf_method,
            sensitivity=self.params.h_sensitivity,
        )
        self.last_hallucination_mask = h_mask

        return PredictionBlender.hallucination_suppress(
            base_blend, structure_output, h_mask, self.params.h_blend_reduction,
        )

    def _edge_aware_blend(self, outputs: Dict[ModelRole, torch.Tensor]) -> torch.Tensor:
        """Strategy 4: Edge-aware spatial routing."""
        ref = outputs.get(ModelRole.STRUCTURE) or next(iter(outputs.values()))

        edge_mask, texture_mask, flat_mask = SpatialRouter.compute_routing_masks(
            ref, self.params.edge_threshold, self.params.texture_threshold,
        )
        self.last_edge_mask = edge_mask
        self.last_texture_mask = texture_mask
        self.last_flat_mask = flat_mask

        result = torch.zeros_like(ref)
        total_weight = torch.zeros(1, device=ref.device)

        for role, tensor in outputs.items():
            alpha = self.params.get_alpha(role)
            if alpha < 1e-8:
                continue

            spatial_weight = torch.ones_like(edge_mask) * alpha

            if role == ModelRole.STRUCTURE:
                spatial_weight = spatial_weight + edge_mask * self.params.edge_model_bias
            elif role == ModelRole.TEXTURE:
                spatial_weight = spatial_weight + texture_mask * self.params.texture_model_bias
            elif role in (ModelRole.PERCEPTUAL, ModelRole.DIFFUSION):
                spatial_weight = spatial_weight * (1.0 - flat_mask * self.params.flat_region_suppression)

            result = result + tensor * spatial_weight
            total_weight = total_weight + spatial_weight

        return result / total_weight.clamp(min=1e-8)

    def _diffusion_detail_inject(
        self,
        base_result: torch.Tensor,
        outputs: Dict[ModelRole, torch.Tensor],
    ) -> torch.Tensor:
        """Strategy 5: Inject diffusion output as detail layer only."""
        diffusion_output = outputs.get(ModelRole.DIFFUSION)
        if diffusion_output is None or self.params.alpha_diffusion < 1e-8:
            return base_result

        return PredictionBlender.detail_inject(
            base_result, diffusion_output, self.params.alpha_diffusion,
            self.params.freq_low_sigma, self.params.freq_mid_sigma,
        )

    # -----------------------------------------------------------------
    # FULL PIPELINE
    # -----------------------------------------------------------------

    def process_frame(self, lr_input: torch.Tensor) -> torch.Tensor:
        """
        Full research-layer pipeline for a single frame.

        Args:
            lr_input: (B, C, H, W) tensor, float32 [0, 1], on CPU or GPU

        Returns:
            (B, C, sH, sW) blended super-resolved output, float32 [0, 1]
        """
        lr_gpu = lr_input.to(self.device)
        outputs = self._run_models(lr_gpu)

        # HF energy of LR input for diagnostics
        self.last_hf_energy = HFAnalyzer.compute(lr_gpu, self.params.hf_method)

        # Compute spatial routing masks (always, for diagnostics/visualization)
        ref = outputs.get(ModelRole.STRUCTURE) or next(iter(outputs.values()))
        edge_mask, texture_mask, flat_mask = SpatialRouter.compute_routing_masks(
            ref, self.params.edge_threshold, self.params.texture_threshold,
        )
        self.last_edge_mask = edge_mask
        self.last_texture_mask = texture_mask
        self.last_flat_mask = flat_mask

        if len(outputs) == 1:
            # Single-model path: apply frequency weighting + hallucination suppression
            result = next(iter(outputs.values()))

            # Frequency band reweighting (boost/suppress low/mid/high independently)
            band_weights = (
                self.params.low_freq_strength,
                self.params.mid_freq_strength,
                self.params.high_freq_strength,
            )
            has_freq_adj = any(abs(w - 1.0) > 1e-4 for w in band_weights)
            if has_freq_adj:
                low, mid, high = FrequencyBandSplitter.split(
                    result, self.params.freq_low_sigma, self.params.freq_mid_sigma,
                )
                self.last_freq_bands = (low, mid, high)
                result = (
                    low * band_weights[0]
                    + mid * band_weights[1]
                    + high * band_weights[2]
                )

            # Hallucination suppression (compare model output vs bicubic reference)
            if self.params.h_sensitivity > 1e-4 and self.params.h_blend_reduction > 1e-4:
                h_mask = HallucinationDetector.generate_mask(
                    lr_input=lr_gpu,
                    sr_output=result,
                    scale=self.scale,
                    hf_method=self.params.hf_method,
                    sensitivity=self.params.h_sensitivity,
                )
                self.last_hallucination_mask = h_mask
                # Pull toward bicubic reference where hallucination is detected
                bicubic_ref = F.interpolate(
                    lr_gpu, scale_factor=self.scale, mode="bicubic", align_corners=False,
                ).clamp(0, 1)
                _, _, sh, sw = result.shape
                if bicubic_ref.shape[2] != sh or bicubic_ref.shape[3] != sw:
                    bicubic_ref = F.interpolate(
                        bicubic_ref, size=(sh, sw), mode="bicubic", align_corners=False,
                    ).clamp(0, 1)
                result = PredictionBlender.hallucination_suppress(
                    result, bicubic_ref, h_mask, self.params.h_blend_reduction,
                )

            return result.clamp(0, 1)

        # Multi-model path: full blending pipeline
        # Stage A: Edge-aware spatial routing blend
        spatial_blend = self._edge_aware_blend(outputs)

        # Stage B: Frequency-separated refinement
        freq_blend = self._frequency_blend(outputs)

        # Stage C: Combine spatial and frequency blends (configurable mix)
        mix = self.params.spatial_freq_mix
        combined = (1.0 - mix) * spatial_blend + mix * freq_blend

        # Stage D: Hallucination suppression against combined baseline
        result = self._hallucination_guided_blend(lr_gpu, outputs)
        # Merge hallucination-suppressed weighted blend with spatial+freq combined
        result = PredictionBlender.weighted_average([combined, result], [0.5, 0.5])

        # Stage E: Diffusion detail injection
        result = self._diffusion_detail_inject(result, outputs)

        return result.clamp(0, 1)

    # -----------------------------------------------------------------
    # NUMPY / SHM INTERFACE
    # -----------------------------------------------------------------

    def process_frame_numpy(self, lr_rgb: np.ndarray) -> np.ndarray:
        """Process a frame from numpy (uint8 RGB) through the full pipeline."""
        # SHM → CPU → GPU tensor conversion
        img_float = lr_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        output = self.process_frame(tensor)

        result = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
        return (result * 255.0).round().astype(np.uint8)

    def process_shm_frame(
        self,
        input_buffer: np.ndarray,
        output_buffer: np.ndarray,
        has_alpha: bool = True,
    ) -> None:
        """Process a frame directly from/to SHM buffers (zero-copy read)."""
        if has_alpha:
            rgb_in = input_buffer[:, :, :3].copy()
        else:
            rgb_in = input_buffer.copy()

        rgb_out = self.process_frame_numpy(rgb_in)

        if has_alpha:
            output_buffer[:, :, :3] = rgb_out
            output_buffer[:, :, 3] = 255
        else:
            output_buffer[:] = rgb_out

    def process_shm_ring_frame(self, ring: ShmRingReader,
                                input_shape: Tuple[int, int, int],
                                output_shape: Tuple[int, int, int],
                                input_size: int) -> bool:
        """
        Read one frame from the SHM ring buffer, process it, write result back.
        Returns True if a frame was processed, False if ring was empty.
        """
        if not ring.has_data():
            return False

        w, r, buf_size, _ = ring._read_header()
        slot_idx = r % ring.slot_count
        slot_data = ring.read_slot(slot_idx)

        in_bytes = bytes(slot_data[:input_size])
        in_arr = np.frombuffer(in_bytes, dtype=np.uint8).reshape(input_shape)

        out_arr = np.frombuffer(slot_data[input_size:], dtype=np.uint8).reshape(output_shape)

        self.process_shm_frame(in_arr, out_arr, has_alpha=(input_shape[2] == 4))
        ring.advance_read_cursor()
        return True

    # -----------------------------------------------------------------
    # DIAGNOSTICS
    # -----------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Optional[np.ndarray]]:
        """Return diagnostic maps for visualization/logging."""
        def _to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
            if t is None:
                return None
            if t.dim() == 4:
                return t.squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
            return t.cpu().float().numpy()

        diag = {
            "hf_energy": _to_numpy(self.last_hf_energy),
            "hallucination_mask": _to_numpy(self.last_hallucination_mask),
            "edge_mask": _to_numpy(self.last_edge_mask),
            "texture_mask": _to_numpy(self.last_texture_mask),
            "flat_mask": _to_numpy(self.last_flat_mask),
        }

        if self.last_freq_bands is not None:
            diag["freq_low"] = _to_numpy(self.last_freq_bands[0])
            diag["freq_mid"] = _to_numpy(self.last_freq_bands[1])
            diag["freq_high"] = _to_numpy(self.last_freq_bands[2])

        return diag

    def get_model_output(self, role: ModelRole) -> Optional[np.ndarray]:
        t = self.last_model_outputs.get(role)
        if t is None:
            return None
        return t.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


# =============================================================================
# FACTORY
# =============================================================================

def create_research_layer(
    models: Optional[Dict[str, nn.Module]] = None,
    scale: int = 4,
    device: Optional[torch.device] = None,
) -> VideoForgeResearchLayer:
    """Factory function to create and configure a research layer instance."""
    layer = VideoForgeResearchLayer(device=device)
    layer.scale = scale

    if models:
        role_map = {
            "structure": ModelRole.STRUCTURE,
            "texture": ModelRole.TEXTURE,
            "perceptual": ModelRole.PERCEPTUAL,
            "diffusion": ModelRole.DIFFUSION,
        }
        for name, model in models.items():
            role = role_map.get(name)
            if role is None:
                print(f"[ResearchLayer] Warning: unknown role '{name}', skipping", flush=True)
                continue
            layer.register_model(role, model, scale)
            layer.enable_model(role)

    return layer


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("[ResearchLayer] Running self-test...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResearchLayer] Device: {device}", flush=True)

    class DummySRModel(nn.Module):
        def __init__(self, scale: int = 4):
            super().__init__()
            self.scale = scale
            self.conv = nn.Conv2d(3, 3, 3, 1, 1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
            return self.conv(x).clamp(0, 1)

    # --- HF Analysis ---
    print("\n--- HF Analysis Methods ---", flush=True)
    test_img = torch.rand(1, 3, 64, 64, device=device)
    for method in ["laplacian", "sobel", "highpass", "fft"]:
        energy = HFAnalyzer.compute(test_img, method)
        print(f"  {method}: shape={energy.shape}, range=[{energy.min():.3f}, {energy.max():.3f}]", flush=True)

    # --- Frequency Band Splitting ---
    print("\n--- Frequency Band Splitting ---", flush=True)
    low, mid, high = FrequencyBandSplitter.split(test_img)
    recon = low + mid + high
    recon_error = (recon - test_img).abs().max().item()
    print(f"  Low: {low.shape}, Mid: {mid.shape}, High: {high.shape}", flush=True)
    print(f"  Reconstruction error: {recon_error:.6f}", flush=True)

    # --- Spatial Routing ---
    print("\n--- Spatial Routing ---", flush=True)
    edge_m, tex_m, flat_m = SpatialRouter.compute_routing_masks(test_img)
    coverage = (edge_m + tex_m + flat_m).mean().item()
    print(f"  Edge: mean={edge_m.mean():.3f}, Texture: mean={tex_m.mean():.3f}, Flat: mean={flat_m.mean():.3f}", flush=True)
    print(f"  Total coverage: {coverage:.3f} (should be ~1.0)", flush=True)

    # --- Hallucination Detection ---
    print("\n--- Hallucination Detection ---", flush=True)
    lr = torch.rand(1, 3, 16, 16, device=device)
    sr = torch.rand(1, 3, 64, 64, device=device)
    h_mask = HallucinationDetector.generate_mask(lr, sr, scale=4, hf_method="laplacian", sensitivity=1.0)
    print(f"  Mask: shape={h_mask.shape}, range=[{h_mask.min():.3f}, {h_mask.max():.3f}]", flush=True)

    # --- PredictionBlender ---
    print("\n--- PredictionBlender (GPU) ---", flush=True)
    t1 = torch.rand(1, 3, 64, 64, device=device)
    t2 = torch.rand(1, 3, 64, 64, device=device)
    blended_pred = PredictionBlender.weighted_average([t1, t2], [0.6, 0.4])
    print(f"  Weighted avg: shape={blended_pred.shape}", flush=True)
    masked = PredictionBlender.masked_blend(t1, t2, torch.ones(1, 1, 64, 64, device=device) * 0.5)
    print(f"  Masked blend: shape={masked.shape}", flush=True)
    freq_out, freq_bands = PredictionBlender.frequency_weighted_blend(
        [t1, t2], [0.7, 0.3], (1.0, 1.0, 1.0))
    print(f"  Freq blend: shape={freq_out.shape}, bands={[b.shape for b in freq_bands]}", flush=True)
    suppressed = PredictionBlender.hallucination_suppress(t1, t2, h_mask, 0.5)
    print(f"  Halluc suppress: shape={suppressed.shape}", flush=True)
    injected = PredictionBlender.detail_inject(t1, t2, 0.3)
    print(f"  Detail inject: shape={injected.shape}", flush=True)

    # --- Weight Blending (CPU) ---
    print("\n--- WeightBlender (CPU) ---", flush=True)
    sd1 = {"w": torch.randn(3, 3)}
    sd2 = {"w": torch.randn(3, 3)}
    blended_sd = WeightBlender.blend_state_dicts([sd1, sd2], [0.7, 0.3])
    expected = sd1["w"] * 0.7 + sd2["w"] * 0.3
    blend_error = (blended_sd["w"] - expected).abs().max().item()
    print(f"  Blend error: {blend_error:.8f}", flush=True)

    # --- Full Pipeline ---
    print("\n--- Full Pipeline ---", flush=True)
    layer = create_research_layer(
        models={"structure": DummySRModel(4), "texture": DummySRModel(4)},
        scale=4, device=device,
    )
    layer.update_params({"preset": "balanced", "hf_method": "sobel"})

    lr_input = torch.rand(1, 3, 32, 32, device=device)
    output = layer.process_frame(lr_input)
    print(f"  Input: {lr_input.shape} -> Output: {output.shape}", flush=True)
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]", flush=True)

    # --- NumPy Interface ---
    print("\n--- NumPy Interface ---", flush=True)
    lr_np = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    out_np = layer.process_frame_numpy(lr_np)
    print(f"  Input: {lr_np.shape} -> Output: {out_np.shape}", flush=True)

    # --- Diagnostics ---
    print("\n--- Diagnostics ---", flush=True)
    diag = layer.get_diagnostics()
    for name, arr in diag.items():
        if arr is not None:
            print(f"  {name}: shape={arr.shape}", flush=True)
        else:
            print(f"  {name}: None", flush=True)

    # --- Parameter Serialization ---
    print("\n--- Parameter Serialization ---", flush=True)
    params_json = json.dumps(layer.get_params(), indent=2)
    print(f"  {params_json}", flush=True)

    print("\n[ResearchLayer] Self-test complete.", flush=True)
