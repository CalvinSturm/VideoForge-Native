"""
VideoForge Model Manager — Registry, VRAM control, SHM ingestion, process_frame().

This is the single entry point for all SR inference.  It owns:
  - A model registry mapping ``model_key`` → (nn.Module, BaseAdapter, scale)
  - VRAM discipline: only ONE heavy model (Transformer/Diffusion) resident at a time
  - The global ``threading.Lock`` that serialises ``process_frame()``
  - Zero-copy SHM → pinned-memory → CUDA pipeline
  - GPU post-processing: luma blend, edge-aware mask, sharpen, temporal EMA
"""

from __future__ import annotations

import gc
import os
import sys
import threading
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ── determinism (must precede any CUDA kernel launch) ──────────────────────
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from arch_wrappers import BaseAdapter, create_adapter  # noqa: E402
from blender_engine import PredictionBlender, clear_temporal_buffers  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Model families classified by VRAM weight class
_HEAVY_FAMILIES = frozenset({"swinir", "hat", "resshift", "sr3"})

# Global inference lock — one frame at a time
_INFERENCE_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_weight_path(model_key: str) -> str:
    """
    Resolve ``weights/{model_key}.pth``.

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    candidates = [
        os.path.join(WEIGHTS_DIR, f"{model_key}.pth"),
        os.path.join(WEIGHTS_DIR, f"{model_key}.pt"),
    ]
    # Also try nested: weights/model_key/model_key.pth
    candidates.append(
        os.path.join(WEIGHTS_DIR, model_key, f"{model_key}.pth")
    )

    for p in candidates:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"Weight file not found for '{model_key}'.  "
        f"Searched: {candidates}"
    )


def _extract_state_dict(loaded: object) -> Dict[str, torch.Tensor]:
    """
    Pull an ``nn.Module`` state-dict out of whatever ``torch.load`` returned.
    """
    if isinstance(loaded, nn.Module):
        return loaded.state_dict()

    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"torch.load returned unexpected type {type(loaded).__name__}"
        )

    # Prefer EMA weights
    for key in ("params_ema", "params", "state_dict", "model"):
        if key in loaded:
            obj = loaded[key]
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, nn.Module):
                return obj.state_dict()

    # Already a bare state-dict?
    if any(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded

    raise RuntimeError("Could not locate state_dict inside checkpoint")


def _load_module(model_key: str) -> Tuple[nn.Module, int]:
    """
    Load a model from ``weights/{model_key}.pth``, place on CPU, eval mode.

    Returns ``(model, scale)``.  Scale is auto-detected via a probe tensor
    if the checkpoint does not encode it explicitly.

    Raises ``RuntimeError`` if loading fails for any reason.
    """
    path = _resolve_weight_path(model_key)
    print(f"[ModelManager] Loading {model_key} from {path}", flush=True)

    loaded = torch.load(path, map_location="cpu", weights_only=False)

    # ── Full model objects (torch.save(model, …)) ────────────────────
    if isinstance(loaded, nn.Module):
        loaded.eval()
        for p in loaded.parameters():
            p.requires_grad_(False)
        scale = BaseAdapter.infer_scale(loaded.to(DEVICE), DEVICE)
        loaded.cpu()
        print(f"[ModelManager] Loaded full model object, detected scale={scale}x", flush=True)
        return loaded, scale

    # ── State-dict checkpoint ─────────────────────────────────────────
    state_dict = _extract_state_dict(loaded)

    # Try to build the architecture from known families
    model: Optional[nn.Module] = None
    key_lower = model_key.lower()

    if key_lower.startswith("realesrgan"):
        model = _build_realesrgan(state_dict, model_key)
    elif key_lower.startswith("rcan"):
        model = _build_rcan(state_dict, model_key)
    elif key_lower.startswith("edsr"):
        model = _build_edsr(state_dict, model_key)
    else:
        raise RuntimeError(
            f"No architecture constructor for '{model_key}'.  "
            f"Provide a full model object (.pth) or register the architecture."
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Detect scale
    model_on_device = model.to(DEVICE)
    scale = BaseAdapter.infer_scale(model_on_device, DEVICE)
    model_on_device.cpu()
    torch.cuda.empty_cache()
    print(f"[ModelManager] Built {model_key}, detected scale={scale}x", flush=True)
    return model, scale


# ── Architecture builders ─────────────────────────────────────────────────

def _build_realesrgan(
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


def _build_rcan(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """Build RCAN from shm_worker definitions."""
    sys.path.insert(0, SCRIPT_DIR)
    from shm_worker import RCAN, remap_rcan_keys

    scale = 4
    for s in (2, 3, 4, 8):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    # Check if keys need remapping (official RCAN format)
    needs_remap = any(k.startswith("head.") for k in state_dict) and any(
        k.startswith("tail.") for k in state_dict
    )
    if not needs_remap:
        pass

    model = RCAN(scale=scale)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        remapped = remap_rcan_keys(state_dict)
        model.load_state_dict(remapped, strict=True)
    return model


def _build_edsr(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """Build EDSR from shm_worker definitions."""
    sys.path.insert(0, SCRIPT_DIR)
    from shm_worker import EDSR

    scale = 4
    for s in (2, 3, 4):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    model = EDSR(scale=scale)
    model.load_state_dict(state_dict, strict=True)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class _SlotEntry:
    __slots__ = ("model", "adapter", "scale", "family")

    def __init__(
        self, model: nn.Module, adapter: BaseAdapter, scale: int, family: str
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.scale = scale
        self.family = family


# Module-level registry (survives across calls)
_registry: Dict[str, _SlotEntry] = {}
_current_heavy: Optional[str] = None  # key of the heavy model on GPU (if any)


def _family_of(model_key: str) -> str:
    return model_key.lower().split("_")[0]


def _is_heavy(model_key: str) -> bool:
    return _family_of(model_key) in _HEAVY_FAMILIES


def unload_heavy_models() -> None:
    """
    Evict the current heavy model from GPU.

    Deletes the nn.Module, runs ``gc.collect()`` and ``torch.cuda.empty_cache()``.
    """
    global _current_heavy
    if _current_heavy is not None and _current_heavy in _registry:
        entry = _registry.pop(_current_heavy)
        del entry.adapter
        del entry.model
        print(f"[ModelManager] Evicted heavy model '{_current_heavy}'", flush=True)
    _current_heavy = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_loaded(model_key: str) -> _SlotEntry:
    """
    Return the registry entry for *model_key*, loading from disk if needed.

    VRAM guard: if *model_key* is a heavy model and a different heavy model
    is already resident, the old one is evicted first.
    """
    global _current_heavy

    if model_key in _registry:
        entry = _registry[model_key]
        entry.model.to(DEVICE)
        return entry

    # VRAM guard
    if _is_heavy(model_key) and _current_heavy is not None and _current_heavy != model_key:
        unload_heavy_models()

    model, scale = _load_module(model_key)
    model.to(DEVICE)

    adapter = create_adapter(model_key, model, scale, DEVICE)
    family = _family_of(model_key)
    entry = _SlotEntry(model=model, adapter=adapter, scale=scale, family=family)
    _registry[model_key] = entry

    if _is_heavy(model_key):
        _current_heavy = model_key

    print(
        f"[ModelManager] Registered '{model_key}' "
        f"(family={family}, scale={scale}x, heavy={_is_heavy(model_key)})",
        flush=True,
    )
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# SHM INGESTION → PINNED MEMORY → CUDA
# ═══════════════════════════════════════════════════════════════════════════════

def _ingest_shm(
    shm_name: str, width: int, height: int, channels: int
) -> torch.Tensor:
    """
    Zero-copy path:  SHM → numpy → torch CPU (pinned) → CUDA NCHW float32 [0,1].

    The SharedMemory handle is closed in a ``finally`` block to prevent OS leaks.
    Supports 3-channel (RGB) and 4-channel (RGBA) inputs.
    """
    expected_size = width * height * channels
    shm: Optional[SharedMemory] = None
    try:
        shm = SharedMemory(name=shm_name, create=False)
        if shm.size < expected_size:
            raise ValueError(
                f"SHM '{shm_name}' size {shm.size} < expected {expected_size} "
                f"({width}×{height}×{channels})"
            )

        arr = np.frombuffer(shm.buf, dtype=np.uint8, count=expected_size)
        arr = arr.reshape((height, width, channels))

        cpu_tensor = torch.as_tensor(arr)  # (H, W, C) uint8
        cpu_tensor = cpu_tensor.pin_memory()

        # HWC uint8 → NCHW float32 [0, 1] on CUDA
        gpu_tensor = (
            cpu_tensor
            .to(DEVICE, non_blocking=True)
            .permute(2, 0, 1)            # (C, H, W)
            .unsqueeze(0)                 # (1, C, H, W)
            .to(torch.float32)
            .div_(255.0)
        )
        return gpu_tensor

    finally:
        if shm is not None:
            shm.close()


def _tensor_to_hwc_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    NCHW float32 [0,1] on GPU → HWC uint8 numpy on CPU.
    """
    out = (
        tensor
        .squeeze(0)          # (C, H, W)
        .clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .to(torch.uint8)
        .permute(1, 2, 0)   # (H, W, C)
        .cpu()
        .numpy()
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def process_frame(
    shm_buffer_name: str,
    width: int,
    height: int,
    channels: int,
    primary_model: str,
    secondary_model: Optional[str] = None,
    blend_alpha: float = 0.3,
    *,
    detail_strength: float = 0.0,
    luma_only: bool = False,
    edge_strength: float = 0.0,
    sharpen_strength: float = 0.0,
    temporal_enabled: bool = False,
    temporal_alpha: float = 0.15,
    return_gpu_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Synchronous, blocking SR inference on a shared-memory frame.

    Pipeline order (all GPU-resident):
      1. SHM ingest → CUDA NCHW float32
      2. RGBA split (if 4-channel)
      3. Primary model inference (structure)
      4. Optional secondary model inference (GAN / texture)
      5. Adaptive Detail Residual — extract GAN high-freq, inject into structure
      6. Luminance-only blend (if luma_only=True)
      7. Edge-aware masking (if edge_strength > 0)
      8. Optional unsharp mask (sharpen_strength > 0)
      9. Optional temporal EMA (temporal_enabled=True)
      10. RGBA merge (if split in step 2)
      11. GPU → CPU → numpy HWC uint8 (or return GPU tensor)

    Parameters
    ----------
    shm_buffer_name : str
        Name of an existing ``SharedMemory`` segment containing a raw
        HWC uint8 image of shape ``(height, width, channels)``.
    width, height, channels : int
        Frame dimensions.  ``channels`` may be 3 (RGB) or 4 (RGBA).
    primary_model : str
        Model key, e.g. ``"RealESRGAN_x4plus"``, ``"swinir_x4"``.
    secondary_model : str | None
        Optional second model for prediction blending.
    blend_alpha : float
        Blend ratio when *secondary_model* is provided.  0 → pure primary.
    detail_strength : float
        Adaptive Detail Residual intensity.  Extracts high-frequency texture
        from the secondary (GAN) output and injects it into the primary
        (structure) output.  0 = disabled, 1 = full GAN residual.
        Requires *secondary_model* to be set; skipped otherwise.
    luma_only : bool
        Blend only the Y channel in YCbCr space (preserves primary chroma).
    edge_strength : float
        If > 0, use Sobel edge mask to modulate blend strength per-pixel.
    sharpen_strength : float
        Unsharp mask intensity.  0 = disabled.
    temporal_enabled : bool
        Apply exponential moving average across frames.
    temporal_alpha : float
        EMA smoothing factor.  Lower = more smoothing.
    return_gpu_tensor : bool
        If True, return ``(1, C, H_out, W_out)`` CUDA tensor instead of numpy.

    Returns
    -------
    np.ndarray or torch.Tensor
        Upscaled frame.  numpy: HWC uint8, shape ``(H*s, W*s, C)``.
        Tensor: NCHW float32 [0,1] on CUDA.
    """
    if not shm_buffer_name or not isinstance(shm_buffer_name, str):
        raise ValueError(f"Invalid shm_buffer_name: {shm_buffer_name!r}")

    with _INFERENCE_LOCK:
        # 1. Ingest from SHM → CUDA tensor
        gpu_input = _ingest_shm(shm_buffer_name, width, height, channels)

        # 2. RGBA split — process RGB through SR, reattach alpha at the end
        alpha_channel: Optional[torch.Tensor] = None
        if channels == 4:
            gpu_input, alpha_channel = PredictionBlender.split_alpha(gpu_input)

        # 3. Primary inference (structure model)
        primary_entry = _ensure_loaded(primary_model)
        primary_out = primary_entry.adapter.forward(gpu_input)

        # 4. Optional secondary inference (GAN / texture model)
        has_secondary = (
            secondary_model is not None and secondary_model != primary_model
        )
        secondary_out: Optional[torch.Tensor] = None
        if has_secondary:
            secondary_entry = _ensure_loaded(secondary_model)
            secondary_out = secondary_entry.adapter.forward(gpu_input)

        result = primary_out

        if has_secondary and secondary_out is not None:
            # 5. Adaptive Detail Residual — extract GAN high-freq texture and
            #    inject into the structure output.  This adds realistic fine
            #    detail (pores, weave, grain) without overwriting geometry.
            #    Must run BEFORE blending so the residual is injected into the
            #    clean structure base, not into an already-blended mix.
            if detail_strength > 1e-4:
                result = PredictionBlender.apply_detail_residual(
                    result, secondary_out, detail_strength, luma_only
                )

            # 6. Luminance-only blend — merge remaining secondary contribution
            #    in YCbCr Y channel only, preserving structure chroma to prevent
            #    GAN colour shifts.
            if luma_only and result.shape[1] == 3:
                result = PredictionBlender.blend_luma_only(
                    result, secondary_out, blend_alpha
                )
            # 7. Edge-aware masking — Sobel-weighted spatially-varying blend,
            #    stronger on edges (sharp detail), weaker on flat regions.
            elif edge_strength > 1e-4:
                result = PredictionBlender.blend_edge_aware(
                    result, secondary_out, blend_alpha, edge_strength
                )
            else:
                result = PredictionBlender.blend(
                    result, secondary_out, blend_alpha
                )

        # 8. Sharpen — GPU unsharp mask for final crispness
        if sharpen_strength > 1e-4:
            result = PredictionBlender.apply_sharpen(result, sharpen_strength)

        # 9. Temporal stabilization — EMA across frames to reduce flicker
        if temporal_enabled:
            _, _, oh, ow = result.shape
            t_key = (oh, ow, result.shape[1])
            result = PredictionBlender.apply_temporal(result, t_key, temporal_alpha)

        # 10. RGBA merge — reattach alpha channel (bilinear-resized to SR dims)
        if alpha_channel is not None:
            result = PredictionBlender.merge_alpha(result, alpha_channel)

        # 11. Return
        if return_gpu_tensor:
            return result

        return _tensor_to_hwc_uint8(result)


def reset_temporal() -> None:
    """Clear all temporal EMA buffers (call on seek, new video, etc.)."""
    clear_temporal_buffers()
    print("[ModelManager] Temporal buffers cleared", flush=True)
