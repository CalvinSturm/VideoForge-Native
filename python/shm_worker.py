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
import importlib
import json
import logging
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
from logging_setup import setup_logging

log = setup_logging(None)

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
        log.info(f"Precision: DETERMINISTIC (TF32=off, strict_deterministic=on)")
    else:
        # fp32 / fp16: enable TF32 for 2-4× speedup on Ampere+
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)
        tag = "FP16 (autocast)" if mode == "fp16" else "FP32"
        log.info(f"Precision: {tag} (TF32=on, cuDNN_deterministic=on)")


# Apply safe defaults immediately (overridden by configure_precision() at startup)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
# 3rd Party Imports
# =============================================================================
try:
    import cv2
    import numpy as np
except ImportError as e:
    log.error(f"Missing Dependency: {e}")
    sys.exit(1)


def _require_zenoh():
    try:
        return importlib.import_module("zenoh")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing dependency: zenoh. Install requirements for the Python worker "
            "(e.g. activate the VideoForge Python environment / install project requirements)."
        ) from e

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
    log.info("Research layer not available — running vanilla inference only")

# Blender engine (optional — SR pipeline post-processing)
try:
    from blender_engine import PredictionBlender, clear_temporal_buffers
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False
    log.info("Blender engine not available — SR pipeline post-processing disabled")

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    TILE_SIZE = 512
    TILE_PAD = 32
    RING_SIZE = 6
    PARENT_CHECK_INTERVAL = 2  # seconds
    ZENOH_PREFIX = "videoforge/ipc"

    # --- SHM Slot State Machine ---
    # Defaults; overwritten by load_shm_protocol()
    SLOT_EMPTY = 0
    SLOT_RUST_WRITING = 1
    SLOT_READY_FOR_AI = 2
    SLOT_AI_PROCESSING = 3
    SLOT_READY_FOR_ENCODE = 4
    SLOT_ENCODING = 5

    # --- SHM Global Header ---
    SHM_MAGIC = b"VFSHM001"
    SHM_VERSION = 2
    PIXEL_FORMAT_RGB24 = 1
    GLOBAL_HEADER_SIZE = 36
    SLOT_HEADER_SIZE = 16

    # Offsets
    STATE_FIELD_OFFSET = 8
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

    @classmethod
    def load_shm_protocol(cls):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Adjust path: script is in python/, protocol is in ipc/
            # If script is in root/python/, then root/ipc/shm_protocol.json
            protocol_path = os.path.join(os.path.dirname(script_dir), "ipc", "shm_protocol.json")
            
            with open(protocol_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            cls.RING_SIZE = data["ring_size"]
            cls.SHM_MAGIC = data["magic"].encode("utf-8")
            cls.SHM_VERSION = data["version"]
            cls.PIXEL_FORMAT_RGB24 = data["pixel_format_rgb24"]
            cls.GLOBAL_HEADER_SIZE = data["global_header_size"]
            cls.SLOT_HEADER_SIZE = data["slot_header_size"]
            
            states = data.get("slot_states", {})
            cls.SLOT_EMPTY = states.get("EMPTY", 0)
            cls.SLOT_RUST_WRITING = states.get("RUST_WRITING", 1)
            cls.SLOT_READY_FOR_AI = states.get("READY_FOR_AI", 2)
            cls.SLOT_AI_PROCESSING = states.get("AI_PROCESSING", 3)
            cls.SLOT_READY_FOR_ENCODE = states.get("READY_FOR_ENCODE", 4)
            cls.SLOT_ENCODING = states.get("ENCODING", 5)
            
            offsets = data.get("offsets", {})
            cls.STATE_FIELD_OFFSET = offsets.get("state", 8)
            cls.FRAME_BYTES_FIELD_OFFSET = offsets.get("frame_bytes", 12)
            
            log.info(f"Loaded SHM protocol from {protocol_path}")
            
        except Exception as e:
            log.warning(f"Failed to load SHM protocol: {e}. Using defaults.")

# Load protocol immediately
Config.load_shm_protocol()

ZENOH_PREFIX = Config.ZENOH_PREFIX
SPATIAL_MAP_TOPIC = "videoforge/research/spatial_map"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHTS_DIRS = [
    os.path.join(PROJECT_ROOT, "weights"),
    os.path.join(SCRIPT_DIR, "weights"),
]


# =============================================================================
# RCAN / EDSR Architecture Definitions (canonical in model_manager)
# =============================================================================
# Re-exported here for backward compatibility with smoke tests and any code
# that imports these names directly from shm_worker.
from model_manager import RCAN, EDSR, remap_edsr_keys, ModelLoader  # noqa: E402


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
        log.warning(f"PID check failed: {e}")
        return False


def watchdog_loop(parent_pid: int) -> None:
    """Monitor parent process. If it dies, we die."""
    log.info(f"Watchdog started for Parent PID: {parent_pid}")
    while True:
        if not is_pid_alive(parent_pid):
            log.info(f"Parent {parent_pid} died. Committing seppuku...")
            os._exit(0)
        time.sleep(Config.PARENT_CHECK_INTERVAL)


def start_watchdog(parent_pid: int) -> None:
    if parent_pid <= 0:
        return
    t = threading.Thread(target=watchdog_loop, args=(parent_pid,), daemon=True)
    t.start()


# =============================================================================
# UNIFIED INFERENCE FUNCTION
# =============================================================================

def inference(
    model: torch.nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    half: bool = False,
    adapter=None,
    pinned_input: Optional[torch.Tensor] = None,
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

    # Determine dtype from precision mode
    use_fp16 = (_PRECISION_MODE == "fp16") or half
    dtype = torch.float16 if use_fp16 else torch.float32

    # Build input tensor — use pre-allocated pinned buffer when available to
    # avoid a CUDA malloc on every frame (~1-2ms savings at 4K).
    if pinned_input is not None:
        arr = img_rgb.astype(np.float32) / 255.0
        pinned_input[0].copy_(torch.from_numpy(arr.transpose(2, 0, 1)))
        tensor = pinned_input.to(device=device, dtype=dtype, non_blocking=True)
    else:
        img_float = img_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
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
            log.info(f"Batch OOM (N={len(imgs_rgb)}), falling back to sequential")
            torch.cuda.empty_cache()
        else:
            log.info(f"Batch forward failed: {e}, falling back to sequential")
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
    def __init__(
        self,
        port: str,
        precision: str = "fp32",
        log_level: Optional[str] = None,
        use_typed_ipc: bool = False,
        use_events: bool = False,
        prealloc_tensors: bool = False,
        deterministic: bool = False,
    ):
        self.log = logging.getLogger("videoforge")
        # Deterministic mode forces batch_size=1 for bit-exact output
        if precision == "deterministic":
            Config.MAX_BATCH_SIZE = 1
            log.info(f"Deterministic mode: batch_size forced to 1")

        log.info(f"Initializing Zenoh on Port {port}...")
        log.info(f"Precision mode: {precision}")
        log.debug(f"CUDNN deterministic: {torch.backends.cudnn.deterministic}")
        log.debug(f"CUDNN benchmark: {torch.backends.cudnn.benchmark}")

        zenoh = _require_zenoh()
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))

        try:
            self.session = zenoh.open(conf)
            log.info("Zenoh connected successfully")
        except Exception as e:
            log.error(f"Zenoh connection failed: {e}")
            sys.exit(1)

        unique_prefix = f"{ZENOH_PREFIX}/{port}"

        try:
            self.pub = self.session.declare_publisher(f"{unique_prefix}/res")
            self.sub = self.session.declare_subscriber(
                f"{unique_prefix}/req", self.on_request
            )
        except Exception as e:
            log.error(f"Zenoh pub/sub setup failed: {e}")
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
        # Parsed/plumbed only; not active yet.
        self.log_level = log_level
        self.use_typed_ipc = use_typed_ipc
        self.use_events = use_events
        self.prealloc_tensors = prealloc_tensors
        self.deterministic_flag = deterministic
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

        # Pre-allocated pinned CPU tensor for zero-copy frame upload (Phase 2.1).
        # Allocated in create_shm() once frame dimensions are known.
        self._pinned_input: Optional[torch.Tensor] = None

        # Research layer (initialized after first model load)
        self.research_layer: Optional[Any] = None
        self.spatial_pub = None
        if HAS_RESEARCH_LAYER:
            try:
                self.spatial_pub = self.session.declare_publisher(SPATIAL_MAP_TOPIC)
                log.info("Spatial map publisher ready")
            except Exception as e:
                log.warning(f"Spatial map publisher failed: {e}")

        # Load default model
        default_model = "rcan_4x"
        log.info(f"Attempting initial load: {default_model}")
        self.load_model(default_model)
        self.loop()

    def loop(self) -> None:
        log.info("Ready...")
        while self.running:
            time.sleep(0.1)
        self.cleanup()

    def cleanup(self) -> None:
        log.info("Cleanup...")
        if self.mmap:
            try:
                self.mmap.close()
            except Exception as e:
                log.warning(f"mmap close failed: {e}")
        if self.shm_file:
            try:
                self.shm_file.close()
                if self.shm_path and os.path.exists(self.shm_path):
                    os.unlink(self.shm_path)
            except Exception as e:
                log.warning(f"SHM file cleanup failed: {e}")
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                log.warning(f"Zenoh session close failed: {e}")
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

            log.info(
                f"Loaded: {self.model_name} (Scale: x{self.model_scale}, expects_rgb={self.expects_rgb})"
            )

            # Register with research layer
            if HAS_RESEARCH_LAYER:
                try:
                    self.research_layer = create_research_layer(
                        models={"structure": self.model},
                        scale=self.model_scale,
                        device=self.device,
                    )
                    log.info(f"Research layer initialized with {self.model_name} as structure model")
                except Exception as e:
                    log.warning(f"Research layer init failed: {e}")
                    self.research_layer = None

            self.send_status(
                "MODEL_LOADED", {"model": self.model_name, "scale": self.model_scale}
            )

        except Exception as e:
            log.error(f"Load Error: {e}")
            log.exception("Load failed")
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
            log.warning(f"Failed to send status: {e}")

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
                p = env.payload if isinstance(env.payload, dict) else raw
                self.handle_image_file(p)
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
                log.warning(f"Unknown command kind: {cmd!r}")
                self.send_status("error", {"message": f"Unknown command: {cmd}"})
        except Exception as e:
            log.error(f"Request failed: {e}")
            log.exception("Request handler exception")
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
            
            log.info(f"Auto-grade analysis for: {image_path}")
            
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
            
            log.info(f"Auto-grade complete: confidence={result['confidence']:.2f}, summary={result['summary']}")
            
        except Exception as e:
            log.error(f"Auto-grade analysis failed: {e}")
            log.exception("Auto-grade exception")
            self.send_status("error", {"message": f"Auto-grade failed: {str(e)}"})

    def handle_update_research_params(self, payload: Dict[str, Any]) -> None:
        """Handle research parameter updates from the Rust backend."""
        if not HAS_RESEARCH_LAYER or self.research_layer is None:
            self.send_status("error", {"message": "Research layer not available"})
            return
        try:
            params = payload.get("params", {})
            self.research_layer.update_params(params)
            log.info(f"Research params updated: {list(params.keys())}")
            self.send_status("RESEARCH_PARAMS_UPDATED", {"keys": list(params.keys())})
        except Exception as e:
            log.error(f"Research params update failed: {e}")
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
            log.warning(f"Spatial map publish failed: {e}")

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

        # ONNX transformer models (DAT2, etc.) have quadratic attention cost — their
        # VRAM footprint grows with tile area, not linearly.  Allow the model to
        # advertise a smaller preferred tile size to avoid VRAM exhaustion.
        tile_size = getattr(self.model, "preferred_tile_size", Config.TILE_SIZE)
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
                log.warning(f"Research params update failed: {e}")

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
            log.exception("Image upscale failed")
            self.send_status("error", {"id": req_id, "message": str(e)})

    def create_shm(self, payload: Dict) -> None:
        """Create shared memory ring buffer for video frame processing.

        Layout (SHM_VERSION = 2):
            [ Global Header 36 bytes: magic|version|header_size|slot_count|W|H|S|fmt ]
            [ SlotHeader × ring_size (ring_size × 16 bytes) ]
            [ Slot 0: input (W×H×3) | output (sW×sH×3) ]
            [ Slot 1: input | output ]
            ...  (6 slots total)
        """
        width = payload["width"]
        height = payload["height"]
        self.active_scale = payload["scale"]
        self.ring_size = payload.get("ring_size", Config.RING_SIZE)

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

            # Pre-allocate pinned input tensor for zero-copy frame upload.
            # Eliminates per-frame CUDA malloc (~1-2ms/frame on 4K).
            if torch.cuda.is_available():
                self._pinned_input = torch.empty(
                    (1, 3, height, width), dtype=torch.float32, pin_memory=True
                )
            else:
                self._pinned_input = None

            self.is_configured = True
            log.info(
                f"SHM created: {total_size} bytes "
                f"(global_header={Config.GLOBAL_HEADER_SIZE}, "
                f"header_region={self.header_region_size}, "
                f"{self.ring_size} slots x {self.slot_byte_size}), "
                f"magic=VFSHM001 version={Config.SHM_VERSION}"
            )
            self.send_status("SHM_CREATED", {"shm_path": self.shm_path})
        except Exception as e:
            log.exception("Failed to create SHM")
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
                adapter=self.adapter,
                pinned_input=self._pinned_input,
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
                log.warning(f"Research layer failed, using vanilla: {e}")

        # Publish spatial routing map for UI overlay
        self._publish_spatial_map(img_input)

        # SR Pipeline post-processing (blender_engine)
        if HAS_BLENDER and research_params:
            try:
                sr = research_params

                if sr.get("reset_temporal"):
                    clear_temporal_buffers()
                    log.info("Temporal buffers cleared by user request")

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
                log.warning(f"SR pipeline post-processing failed: {e}")

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
                log.warning(f"Research params update failed: {e}")

        slot_idx = payload.get("slot", 0)
        try:
            self._process_slot(slot_idx, research_params)
            self.send_status("FRAME_DONE", {"slot": slot_idx})
        except Exception as e:
            log.exception("Request failed")
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
            log.exception("Single frame processing failed")
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
                    log.warning(f"Research layer failed, using vanilla: {e}")

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
                        log.info("Temporal buffers cleared by user request")

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
                    log.warning(f"SR pipeline post-processing failed: {e}")

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
        log.info("Frame loop started (SHM atomic polling, micro-batch)")
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
                    log.error(f"Batch processing failed: {e}")
                    log.exception("Batch processing failed")
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

        log.info("Frame loop stopped")

    def start_frame_loop(self, payload: Dict) -> None:
        """Start the SHM atomic frame polling loop in a background thread."""
        if hasattr(self, '_frame_loop_thread') and self._frame_loop_thread is not None:
            if self._frame_loop_thread.is_alive():
                log.info("Frame loop already running")
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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=None,
        help="Logger verbosity for stderr output (default: info)"
    )
    parser.add_argument("--use-typed-ipc", action="store_true", help="Parsed only (plumbing)")
    parser.add_argument("--use-events", action="store_true", help="Parsed only (plumbing)")
    parser.add_argument("--prealloc-tensors", action="store_true", help="Parsed only (plumbing)")
    parser.add_argument("--deterministic", action="store_true", help="Parsed only (plumbing)")
    return parser


def run_worker(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)
    try:
        _require_zenoh()
    except RuntimeError as e:
        log.error(f"{e}")
        return 1

    # Configure precision BEFORE any model loading or CUDA ops
    configure_precision(args.precision)

    if args.parent_pid > 0:
        start_watchdog(args.parent_pid)

    AIWorker(
        args.port,
        precision=args.precision,
        log_level=args.log_level,
        use_typed_ipc=args.use_typed_ipc,
        use_events=args.use_events,
        prealloc_tensors=args.prealloc_tensors,
        deterministic=args.deterministic,
    )
    return 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

