"""
ONNX Model Loader — OnnxModelWrapper and ONNX Runtime inference.

Extracted from model_manager.py. Lazy-imports onnxruntime at call time.
"""

import logging
import os
import threading
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("videoforge")


class OnnxModelWrapper(nn.Module):
    """
    Wraps an ONNX Runtime InferenceSession as an nn.Module so ONNX models
    plug into the same inference pipeline as PyTorch models.

    forward() accepts an (N, 3, H, W) float32 tensor on any device and
    returns an (N, 3, H*scale, W*scale) float32 tensor on the same device.
    """

    def __init__(self, session, scale: int, preferred_tile_size: int = 512) -> None:
        super().__init__()
        self.session = session
        self.scale = scale
        self.preferred_tile_size = preferred_tile_size
        self._infer_device = torch.device("cpu")
        self._vf_onnx = True  # sentinel for create_adapter bypass

    def to(self, device=None, dtype=None, **kwargs):
        if device is not None:
            self._infer_device = torch.device(device) if isinstance(device, str) else device
        return self

    def half(self):
        return self  # ORT manages its own precision

    def eval(self):
        return self

    def parameters(self, recurse=True):
        return iter([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_device = x.device
        inp_info = self.session.get_inputs()[0]
        _ORT_TO_NP = {
            "tensor(float)":   np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)":  np.float64,
        }
        np_dtype = _ORT_TO_NP.get(inp_info.type, np.float32)
        np_input = x.cpu().numpy().astype(np_dtype)
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {inp_info.name: np_input})[0]
        out = torch.from_numpy(result.astype(np.float32))  # always return fp32
        if input_device.type != "cpu":
            out = out.to(input_device, non_blocking=True)
        return out


def _extract_scale_from_stem(stem: str) -> int:
    """Mirror models.rs extract_scale() — extract upscale factor from filename stem."""
    lower = stem.lower()
    for scale in [2, 3, 4, 8]:
        for pattern in [f"_x{scale}", f"x{scale}plus", f"_{scale}x", f"{scale}x_", f"{scale}x-"]:
            if pattern in lower:
                return scale
    if len(lower) >= 2 and lower[1] == "x" and lower[0] in "2348":
        return int(lower[0])
    return 4


def _probe_onnx_session(session, inp_info, timeout: float = 20.0) -> bool:
    """
    Run a tiny dummy inference in a background thread.
    Returns True if it completes within *timeout* seconds, False if it hangs.
    """
    _ORT_TO_NP = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)":  np.float64,
    }
    np_dtype = _ORT_TO_NP.get(inp_info.type, np.float32)
    dummy = np.zeros((1, 3, 64, 64), dtype=np_dtype)

    success = [False]

    def _run():
        try:
            session.run(None, {inp_info.name: dummy})
            success[0] = True
        except Exception:
            success[0] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return success[0] or not t.is_alive()


def load_onnx_model(path: str) -> Tuple[nn.Module, int]:
    """Load an ONNX file and return (OnnxModelWrapper, scale)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError(
            "onnxruntime package required for .onnx files. "
            "Install: pip install onnxruntime-gpu"
        )

    # ORT's CUDA provider needs cudnn64_9.dll which lives inside PyTorch's lib
    try:
        import torch as _torch
        _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_torch_lib)
    except Exception:
        pass

    stem = os.path.splitext(os.path.basename(path))[0]
    scale = _extract_scale_from_stem(stem)

    _TRANSFORMER_KEYS = {"dat", "swin", "hat", "realweb", "omnisr", "lmlt"}
    preferred_tile_size = 256 if any(k in stem.lower() for k in _TRANSFORMER_KEYS) else 512

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        log.info(f"Loading ONNX (CUDA EP probe): {os.path.basename(path)}")
        cuda_session = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        inp_info = cuda_session.get_inputs()[0]
        if _probe_onnx_session(cuda_session, inp_info):
            log.info(f"ONNX CUDA EP OK — scale={scale}x tile={preferred_tile_size}px")
            return OnnxModelWrapper(cuda_session, scale, preferred_tile_size), scale
        else:
            log.warning(
                f"CUDA EP probe timed out for {os.path.basename(path)}, "
                f"falling back to CPU EP (inference will be slower)"
            )

    log.info(f"Loading ONNX (CPU EP): {os.path.basename(path)}")
    cpu_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    log.info(f"ONNX CPU EP loaded — scale={scale}x tile={preferred_tile_size}px")
    return OnnxModelWrapper(cpu_session, scale, preferred_tile_size), scale
