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

# ── Safe defaults (precision fully configured by shm_worker.configure_precision) ──
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# NOTE: TF32 and deterministic_algorithms flags are set by
# shm_worker.configure_precision() at startup — NOT here.

from arch_wrappers import BaseAdapter, create_adapter  # noqa: E402
from blender_engine import PredictionBlender, clear_temporal_buffers  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# Model families classified by VRAM weight class (see _HEAVY_KEYWORDS below)

# Global inference lock — one frame at a time
_INFERENCE_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_weight_path(model_key: str) -> str:
    """
    Resolve the weight file for *model_key*.

    Search order:
      1. ``weights/{model_key}.pth``
      2. ``weights/{model_key}.pt``
      3. ``weights/{model_key}.safetensors``
      4. ``weights/{model_key}/{model_key}.pth``
      5. Scan ``weights/`` for any file whose name (sans extension) matches *model_key*

    Raises ``FileNotFoundError`` if the file does not exist.
    """
    exts = [".pth", ".pt", ".safetensors", ".bin"]

    # Direct name match
    candidates = [os.path.join(WEIGHTS_DIR, f"{model_key}{ext}") for ext in exts]
    # Nested directory
    candidates.extend(
        os.path.join(WEIGHTS_DIR, model_key, f"{model_key}{ext}") for ext in exts
    )

    for p in candidates:
        if os.path.isfile(p):
            return p

    # Fallback: scan directory for a file whose stem matches model_key (case-insensitive)
    if os.path.isdir(WEIGHTS_DIR):
        for fname in os.listdir(WEIGHTS_DIR):
            fpath = os.path.join(WEIGHTS_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            stem = fname
            for ext in exts:
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            if stem.lower() == model_key.lower():
                return fpath

    raise FileNotFoundError(
        f"Weight file not found for '{model_key}'.  "
        f"Searched: {candidates} and scanned {WEIGHTS_DIR}"
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


def _detect_family(model_key: str, state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Detect architecture family from model key and state-dict key patterns.

    Returns a family tag used to route to the correct builder.
    """
    key_lower = model_key.lower()

    # ── Explicit family from model key name ──────────────────────────
    # IMPORTANT: Transformer checks must come first — model names like
    # "Swin_2SR_..._BSRGAN" contain GAN training method names but are
    # transformer architectures.
    normalized = key_lower.replace("-", "").replace("_", "")
    if "swinir" in normalized or "swin2sr" in normalized:
        return "swinir"
    if "hat" in key_lower and "hat" in key_lower.split("_")[0].lower():
        return "hat"
    if "dat" in key_lower:
        return "dat"
    if "realesrgan" in key_lower or "esrgan" in key_lower:
        return "realesrgan"
    if "bsrgan" in key_lower:
        return "realesrgan"  # same RRDBNet architecture
    if "spsr" in key_lower:
        return "realesrgan"  # RRDBNet-based
    if "realbasicvsr" in key_lower:
        return "realesrgan"  # RRDBNet backbone
    if key_lower.startswith("rcan") or "_rcan" in key_lower:
        return "rcan"
    if key_lower.startswith("edsr") or key_lower.startswith("mdsr"):
        return "edsr"
    if "omnisr" in key_lower or "omni_sr" in key_lower:
        return "omnisr"
    if "mosr" in key_lower:
        return "mosr"

    # ── Auto-detect from state-dict key patterns ─────────────────────
    keys_str = " ".join(list(state_dict.keys())[:50])

    # RRDBNet pattern: body.N.rdb*
    if "body." in keys_str and ".rdb" in keys_str:
        return "realesrgan"
    # RCAN pattern: head/body/tail with RCAB
    if "head." in keys_str and "tail." in keys_str:
        return "rcan"
    # SwinIR pattern: layers with RSTB/SwinTransformerBlock
    if "layers." in keys_str and ("attn." in keys_str or "rstb" in keys_str.lower()):
        return "swinir"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# OFFICIAL EDSR/RCAN MODEL STUBS (for unpickling full model objects)
# ═══════════════════════════════════════════════════════════════════════════════

_STUBS_REGISTERED = False


def _register_official_model_stubs() -> None:
    """
    Register fake ``model.*`` modules so ``torch.load`` can unpickle full model
    objects saved by the official EDSR-PyTorch / RCAN repos.

    These repos save the entire ``nn.Module`` with ``torch.save(model, path)``,
    which records class paths like ``model.rcan.RCAN``.  We provide compatible
    stub classes that pickle can instantiate.
    """
    global _STUBS_REGISTERED
    if _STUBS_REGISTERED:
        return
    _STUBS_REGISTERED = True

    import math
    import types

    # ── model.common ──────────────────────────────────────────────────
    common = types.ModuleType("model.common")

    def default_conv(in_ch, out_ch, kernel_size, bias=True):
        return nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=bias)

    common.default_conv = default_conv

    class MeanShift(nn.Conv2d):
        def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.4040),
                     rgb_std=(1.0, 1.0, 1.0), sign=-1):
            super().__init__(3, 3, kernel_size=1)
            std = torch.Tensor(rgb_std)
            self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
            self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
            for p in self.parameters():
                p.requires_grad = False

    common.MeanShift = MeanShift

    class Upsampler(nn.Sequential):
        def __init__(self, conv=default_conv, scale=4, n_feat=64,
                     bn=False, act=False, bias=True):
            m = []
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    m.append(conv(n_feat, 4 * n_feat, 3, bias))
                    m.append(nn.PixelShuffle(2))
            elif scale == 3:
                m.append(conv(n_feat, 9 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(3))
            super().__init__(*m)

    common.Upsampler = Upsampler

    class ResBlock(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     bias=True, bn=False, act=None, res_scale=1):
            super().__init__()
            m = [conv(n_feat, n_feat, kernel_size, bias=bias),
                 nn.ReLU(True),
                 conv(n_feat, n_feat, kernel_size, bias=bias)]
            self.body = nn.Sequential(*m)
            self.res_scale = res_scale

        def forward(self, x):
            return x + self.body(x) * self.res_scale

    common.ResBlock = ResBlock

    # ── model.rcan ────────────────────────────────────────────────────
    rcan_mod = types.ModuleType("model.rcan")

    class CALayer(nn.Module):
        """Channel Attention with fc1/relu1(PReLU)/fc2 structure."""
        def __init__(self, channel=64, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
            self.relu1 = nn.PReLU(channel // reduction)
            self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.fc1(y)
            y = self.relu1(y)
            y = self.fc2(y)
            y = self.sigmoid(y)
            return x * y

    class SALayer(nn.Module):
        """Spatial Attention with a single 7x7 conv."""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 1, 7, padding=3, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            y = self.conv1(avg_out)
            y = self.sigmoid(y)
            return x * y

    class CSAM(nn.Module):
        """Combined Channel + Spatial Attention Module."""
        def __init__(self, channel=64, reduction=16):
            super().__init__()
            self.ca = CALayer(channel, reduction)
            self.sa = SALayer()

        def forward(self, x):
            x = self.ca(x)
            x = self.sa(x)
            return x

    rcan_mod.CALayer = CALayer

    class RCAB_Stub(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     reduction=16, bias=True, bn=False, act=None, res_scale=1):
            super().__init__()
            modules = [conv(n_feat, n_feat, kernel_size, bias=bias),
                       act if act else nn.ReLU(True),
                       conv(n_feat, n_feat, kernel_size, bias=bias),
                       CSAM(n_feat, reduction)]
            self.body = nn.Sequential(*modules)
            self.res_scale = res_scale

        def forward(self, x):
            return x + self.body(x) * self.res_scale

    rcan_mod.RCAB = RCAB_Stub

    class ResidualGroup_Stub(nn.Module):
        def __init__(self, conv=default_conv, n_feat=64, kernel_size=3,
                     reduction=16, act=None, res_scale=1, n_resblocks=20):
            super().__init__()
            modules = [RCAB_Stub(conv, n_feat, kernel_size, reduction,
                                 act=act, res_scale=res_scale)
                       for _ in range(n_resblocks)]
            modules.append(conv(n_feat, n_feat, kernel_size))
            self.body = nn.Sequential(*modules)

        def forward(self, x):
            return x + self.body(x)

    rcan_mod.ResidualGroup = ResidualGroup_Stub

    class RCAN_Stub(nn.Module):
        def __init__(self, args=None, conv=default_conv):
            super().__init__()

        def forward(self, x):
            pass

    rcan_mod.RCAN = RCAN_Stub

    # ── model.edsr ────────────────────────────────────────────────────
    edsr_mod = types.ModuleType("model.edsr")

    class EDSR_Stub(nn.Module):
        def __init__(self, args=None, conv=default_conv):
            super().__init__()

        def forward(self, x):
            pass

    edsr_mod.EDSR = EDSR_Stub

    # ── Register in sys.modules ───────────────────────────────────────
    model_pkg = types.ModuleType("model")
    model_pkg.__path__ = []
    model_pkg.common = common
    model_pkg.rcan = rcan_mod
    model_pkg.edsr = edsr_mod

    sys.modules["model"] = model_pkg
    sys.modules["model.common"] = common
    sys.modules["model.rcan"] = rcan_mod
    sys.modules["model.edsr"] = edsr_mod

    print("[ModelManager] Registered official EDSR/RCAN model stubs", flush=True)


def _load_via_spandrel(path: str, model_key: str) -> Tuple[nn.Module, int]:
    """
    Load any SR model via spandrel (supports 30+ architectures).

    Spandrel auto-detects the architecture from state-dict key patterns
    and builds the correct nn.Module.
    """
    try:
        import spandrel
    except ImportError:
        raise RuntimeError(
            "spandrel package required for loading this model type.  "
            "Install: pip install spandrel"
        )

    print(f"[ModelManager] Loading via spandrel: {model_key}", flush=True)
    model_descriptor = spandrel.ModelLoader(device="cpu").load_from_file(path)
    model = model_descriptor.model
    scale = model_descriptor.scale

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Tag as spandrel-loaded so create_adapter uses a pass-through adapter.
    # Spandrel models handle their own padding, mean subtraction, and cropping.
    model._vf_spandrel = True  # type: ignore[attr-defined]

    print(f"[ModelManager] Spandrel loaded {model_key}: "
          f"arch={model_descriptor.architecture.name}, scale={scale}x", flush=True)
    return model, scale


class _Swin2SRWrapper(nn.Module):
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


def _load_swin2sr_hf(
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
        print(f"[ModelManager] Swin2SR HF load: {len(missing)} missing keys", flush=True)

    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)

    wrapper = _Swin2SRWrapper(hf_model, window_size=8, scale=scale)
    print(
        f"[ModelManager] Loaded Swin2SR via HuggingFace transformers: "
        f"embed={embed_dim}, depths={depths}, scale={scale}x, upsampler={upsampler}",
        flush=True,
    )
    return wrapper, scale


def _load_module(model_key: str) -> Tuple[nn.Module, int]:
    """
    Load a model from ``weights/{model_key}.pth``, place on CPU, eval mode.

    Returns ``(model, scale)``.  Scale is auto-detected via a probe tensor
    if the checkpoint does not encode it explicitly.

    Loading strategy (in order):
      1. Full model objects (``torch.save(model, path)``)
      2. Known architecture builders (RCAN, EDSR, RealESRGAN) for legacy compat
      3. **Spandrel** — auto-detects 30+ SR architectures from state-dict keys
         (SwinIR, HAT, DAT, OmniSR, MOSR, SPAN, ESRGAN, etc.)

    Raises ``RuntimeError`` if loading fails for any reason.
    """
    path = _resolve_weight_path(model_key)
    print(f"[ModelManager] Loading {model_key} from {path}", flush=True)

    # ── Safetensors files need a different loader ─────────────────────
    if path.lower().endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise RuntimeError(
                "safetensors package required for .safetensors files. "
                "Install: pip install safetensors"
            )
        print(f"[ModelManager] Loading safetensors: {path}", flush=True)
        loaded = load_file(path, device="cpu")
    else:
        # Try loading — if it fails with an import error the file is likely a full
        # model pickle from the official EDSR/RCAN repo.  Register stubs and retry.
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"[ModelManager] Pickle import failed ({e}), registering model stubs and retrying", flush=True)
            _register_official_model_stubs()
            loaded = torch.load(path, map_location="cpu", weights_only=False)

    # ── Full model objects (torch.save(model, …)) ────────────────────
    # Extract state dict and rebuild with OUR architecture so the adapter
    # and pipeline can work correctly (official models include MeanShift
    # layers that we handle in the adapter instead).
    if isinstance(loaded, nn.Module):
        print(f"[ModelManager] Loaded full model object ({type(loaded).__name__}), extracting state dict", flush=True)
        loaded.eval()
        state_dict = loaded.state_dict()
        family = _detect_family(model_key, state_dict)
        print(f"[ModelManager] Full model family='{family}', rebuilding with our architecture", flush=True)

        # Strip MeanShift keys (sub_mean / add_mean) — handled by adapter
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("sub_mean.") and not k.startswith("add_mean.")}

        model: Optional[nn.Module] = None
        if family == "rcan":
            try:
                model = _build_rcan(state_dict, model_key)
            except Exception as e:
                print(f"[ModelManager] RCAN rebuild failed: {e}", flush=True)
        elif family == "edsr":
            try:
                model = _build_edsr(state_dict, model_key)
            except Exception as e:
                print(f"[ModelManager] EDSR rebuild failed: {e}", flush=True)

        if model is not None:
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            model_on_device = model.to(DEVICE)
            scale = BaseAdapter.infer_scale(model_on_device, DEVICE)
            model_on_device.cpu()
            torch.cuda.empty_cache()
            print(f"[ModelManager] Rebuilt from full model, scale={scale}x", flush=True)
            return model, scale

        # Fallback: use the loaded model as-is (includes MeanShift).
        # Tag it so create_adapter can pick ScaledRangeAdapter instead of
        # EDSRRCANAdapter (which would double-apply mean shift).
        for p in loaded.parameters():
            p.requires_grad_(False)
        has_sub_mean = hasattr(loaded, "sub_mean")
        if has_sub_mean:
            loaded._vf_has_mean_shift = True  # type: ignore[attr-defined]
        scale = BaseAdapter.infer_scale(loaded.to(DEVICE), DEVICE)
        loaded.cpu()
        print(f"[ModelManager] Using full model as-is (has_mean_shift={has_sub_mean}), "
              f"detected scale={scale}x", flush=True)
        return loaded, scale

    # ── State-dict checkpoint ─────────────────────────────────────────
    state_dict = _extract_state_dict(loaded)
    family = _detect_family(model_key, state_dict)
    print(f"[ModelManager] Detected family='{family}' for {model_key}", flush=True)

    # Strip MeanShift keys for RCAN/EDSR — we handle range/mean in adapter
    if family in ("rcan", "edsr"):
        stripped = {k: v for k, v in state_dict.items()
                    if not k.startswith("sub_mean.") and not k.startswith("add_mean.")}
        if len(stripped) < len(state_dict):
            print(f"[ModelManager] Stripped {len(state_dict) - len(stripped)} MeanShift keys", flush=True)
            state_dict = stripped

    model: Optional[nn.Module] = None

    # Try architecture builders first for RCAN/EDSR
    if family == "rcan":
        try:
            model = _build_rcan(state_dict, model_key)
        except Exception as e:
            print(f"[ModelManager] RCAN builder failed: {e}, trying spandrel", flush=True)
    elif family == "edsr":
        try:
            model = _build_edsr(state_dict, model_key)
        except Exception as e:
            print(f"[ModelManager] EDSR builder failed: {e}, trying spandrel", flush=True)

    # For everything else (including realesrgan), use spandrel — it handles
    # RRDBNet, SwinIR, HAT, DAT, OmniSR, MOSR, SPAN, ESRGAN, and 20+ more
    if model is None:
        try:
            return _load_via_spandrel(path, model_key)
        except Exception as spandrel_err:
            print(f"[ModelManager] Spandrel failed: {spandrel_err}", flush=True)

            # Try HuggingFace Swin2SR loader for HF-format state dicts
            is_hf_swin2sr = any(
                k.startswith("swin2sr.") for k in list(state_dict.keys())[:10]
            )
            if is_hf_swin2sr:
                try:
                    return _load_swin2sr_hf(state_dict, model_key)
                except Exception as hf_err:
                    raise RuntimeError(
                        f"Could not load '{model_key}' (family='{family}'). "
                        f"Spandrel: {spandrel_err}. HF Swin2SR: {hf_err}."
                    )

            # Legacy RRDBNet builder for realesrgan-family
            if family == "realesrgan":
                try:
                    model = _build_realesrgan(state_dict, model_key)
                except Exception as rrdb_err:
                    raise RuntimeError(
                        f"Could not load '{model_key}' (family='{family}'). "
                        f"Spandrel: {spandrel_err}. RRDBNet: {rrdb_err}."
                    )
            else:
                raise RuntimeError(
                    f"Could not load '{model_key}' (family='{family}'). "
                    f"Spandrel failed: {spandrel_err}. "
                    f"Ensure the weight file is a valid SR model checkpoint."
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
    """
    Build RCAN from shm_worker definitions.

    Detects architecture parameters (num_feat, num_group, num_rcab) from the
    state dict keys and tensor shapes so we match the actual checkpoint.
    """
    sys.path.insert(0, SCRIPT_DIR)
    from shm_worker import RCAN

    # ── Detect scale from model key ─────────────────────────────────
    scale = 4
    for s in (2, 3, 4, 8):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    # ── Detect num_feat from head conv weight shape ──────────────────
    num_feat = 64
    if "head.0.weight" in state_dict:
        num_feat = state_dict["head.0.weight"].shape[0]

    # ── Detect num_group (residual groups) ───────────────────────────
    # body.N.body.* = ResidualGroup N; the last body.M is a plain conv
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

    # num_group = count of indices that have sub-body (ResidualGroup)
    num_group = len(group_indices) if group_indices else 10

    # ── Detect num_rcab (RCABs per group) ────────────────────────────
    # Within body.0.body.*, sequential entries are RCABs, last is a conv
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
            # Last index in the group Sequential is a plain conv, not RCAB
            num_rcab = max(rcab_indices)  # 0..num_rcab-1 are RCABs, num_rcab is conv

    print(f"[ModelManager] RCAN params: num_feat={num_feat}, num_group={num_group}, "
          f"num_rcab={num_rcab}, scale={scale}", flush=True)

    model = RCAN(num_feat=num_feat, num_group=num_group,
                 num_rcab=num_rcab, scale=scale)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[ModelManager] RCAN strict load failed: {e}", flush=True)
        # Log key differences for debugging
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        missing = model_keys - weight_keys
        unexpected = weight_keys - model_keys
        if missing:
            print(f"[ModelManager] Missing keys ({len(missing)}): {sorted(missing)[:5]}...", flush=True)
        if unexpected:
            print(f"[ModelManager] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...", flush=True)
        # Try non-strict as last resort
        model.load_state_dict(state_dict, strict=False)
        print("[ModelManager] RCAN loaded with strict=False", flush=True)
    return model


def _build_edsr(
    state_dict: Dict[str, torch.Tensor], model_key: str
) -> nn.Module:
    """
    Build EDSR from shm_worker definitions.

    Handles two state-dict formats:
      - Official EDSR-PyTorch (head/body/tail) — loads directly
      - BasicSR (conv_first/conv_after_body/upsample/conv_last) — remapped first

    Detects architecture parameters (num_feat, num_block, res_scale) from the
    state dict.  The official EDSR .pt files are the *large* model
    (num_feat=256, num_block=32, res_scale=0.1).
    """
    sys.path.insert(0, SCRIPT_DIR)
    from shm_worker import EDSR, remap_edsr_keys

    # ── Remap BasicSR format if needed ───────────────────────────────
    is_basicsr = any(k.startswith("conv_first.") for k in state_dict)
    if is_basicsr:
        print("[ModelManager] Detected BasicSR EDSR format, remapping to official format", flush=True)
        state_dict = remap_edsr_keys(state_dict)

    # ── Detect scale from model key ──────────────────────────────────
    scale = 4
    for s in (2, 3, 4):
        if f"x{s}" in model_key.lower() or f"_{s}x" in model_key.lower():
            scale = s
            break

    # ── Detect num_feat from head.0.weight shape ─────────────────────
    num_feat = 256  # default to large model
    if "head.0.weight" in state_dict:
        num_feat = state_dict["head.0.weight"].shape[0]

    # ── Detect num_block from body keys ──────────────────────────────
    # body.N.body.* are ResBlocks; body.M (without .body.) is conv_after_body
    block_indices = set()
    for k in state_dict:
        if k.startswith("body."):
            parts = k.split(".")
            if len(parts) > 2 and parts[1].isdigit() and parts[2] == "body":
                block_indices.add(int(parts[1]))
    num_block = len(block_indices) if block_indices else 32

    # ── res_scale: large=0.1, baseline=1.0 ───────────────────────────
    res_scale = 0.1 if num_feat >= 128 else 1.0

    print(f"[ModelManager] EDSR params: num_feat={num_feat}, num_block={num_block}, "
          f"res_scale={res_scale}, scale={scale}", flush=True)

    model = EDSR(num_feat=num_feat, num_block=num_block,
                 res_scale=res_scale, scale=scale)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[ModelManager] EDSR strict load failed: {e}", flush=True)
        model_keys = set(model.state_dict().keys())
        weight_keys = set(state_dict.keys())
        missing = model_keys - weight_keys
        unexpected = weight_keys - model_keys
        if missing:
            print(f"[ModelManager] Missing keys ({len(missing)}): {sorted(missing)[:5]}...", flush=True)
        if unexpected:
            print(f"[ModelManager] Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...", flush=True)
        # Try non-strict as last resort
        model.load_state_dict(state_dict, strict=False)
        print("[ModelManager] EDSR loaded with strict=False", flush=True)
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


_HEAVY_KEYWORDS = frozenset({"swinir", "swin2sr", "hat", "dat", "resshift", "sr3", "stablesr", "dit", "ipt", "edt"})


def _family_of(model_key: str) -> str:
    return model_key.lower().split("_")[0]


def _is_heavy(model_key: str) -> bool:
    lower = model_key.lower()
    return any(kw in lower for kw in _HEAVY_KEYWORDS)


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
