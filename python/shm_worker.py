import argparse
import json
import mmap
import os
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager

# 3rd Party Imports
try:
    import cv2
    import numpy as np
    import torch
    import zenoh
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError as e:
    print(f"[Python Critical] Missing Dependency: {e}", flush=True)

# -----------------------------------------------------------------------------
# CONFIGURATION (Extracted Magic Numbers)
# -----------------------------------------------------------------------------
class Config:
    TILE_SIZE = 512
    TILE_PAD = 32
    RING_SIZE = 3
    PARENT_CHECK_INTERVAL = 2  # seconds
    ZENOH_PREFIX = "videoforge/ipc"


# -----------------------------------------------------------------------------
# CONSTANTS & PATHS
# -----------------------------------------------------------------------------
ZENOH_PREFIX = Config.ZENOH_PREFIX

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHTS_DIRS = [
    os.path.join(PROJECT_ROOT, "weights"),
    os.path.join(SCRIPT_DIR, "weights"),
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


@contextmanager
def suppress_stdout():
    """Suppress stdout from noisy C++ libraries like PyTorch/Basicsr"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def find_weight_file(filename):
    if not filename.endswith(".pth"):
        filename += ".pth"
    for d in WEIGHTS_DIRS:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def find_weights_recurse(state_dict, target_key="conv_first.weight"):
    """Recursive search for model weights to handle nested dicts"""
    if target_key in state_dict:
        return state_dict
    for key, value in state_dict.items():
        if isinstance(value, dict):
            found = find_weights_recurse(value, target_key)
            if found is not None:
                return found
    return None


# -----------------------------------------------------------------------------
# SAFE WRAPPER CLASS
# -----------------------------------------------------------------------------


class SafeRealESRGANer(RealESRGANer):
    """
    Overrides __init__ to bypass buggy weight loading logic.
    We load the model manually and inject it ready-to-go.
    """

    def __init__(
        self,
        scale,
        model_path,
        model=None,
        tile=0,
        tile_pad=10,
        pre_pad=10,
        half=False,
        device=None,
        gpu_id=None,
    ):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model

        if self.device == torch.device("cpu"):
            self.model.cpu()
            self.model.eval()
        else:
            self.model.to(self.device)
            self.model.eval()
            if self.half:
                self.model = self.model.half()


# -----------------------------------------------------------------------------
# WATCHDOG (Suicide Pact)
# -----------------------------------------------------------------------------
import threading
import ctypes

def is_pid_alive(pid):
    """Check if PID is alive on Windows using ctypes kernel32"""
    try:
        # PROCESS_QUERY_INFORMATION (0x0400) or PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
        # SYNCHRONIZE (0x00100000)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False

        exit_code = ctypes.c_ulong()
        if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            ctypes.windll.kernel32.CloseHandle(handle)
            # STAY_ALIVE (259) means still running
            return exit_code.value == 259

        ctypes.windll.kernel32.CloseHandle(handle)
        return False
    except Exception as e:
        print(f"[Python Warning] PID check failed: {e}", flush=True)
        return False

def watchdog_loop(parent_pid):
    """Monitor parent. If it dies, we die."""
    print(f"[Python] Watchdog started for Parent PID: {parent_pid}", flush=True)
    while True:
        if not is_pid_alive(parent_pid):
            print(f"[Python] Parent {parent_pid} died. Committing seppuku...", flush=True)
            # Hard exit, no cleanup needed (OS handles it)
            os._exit(0)
        time.sleep(Config.PARENT_CHECK_INTERVAL)

def start_watchdog(parent_pid):
    if parent_pid <= 0: return
    t = threading.Thread(target=watchdog_loop, args=(parent_pid,), daemon=True)
    t.start()

# -----------------------------------------------------------------------------
# WORKER CLASS
# -----------------------------------------------------------------------------


class AIWorker:
    def __init__(self, port):
        print(f"[Python] Initializing Zenoh on Port {port}...", flush=True)
        conf = zenoh.Config()
        conf.insert_json5("connect/endpoints", json.dumps([f"tcp/127.0.0.1:{port}"]))

        # Zenoh connection with validation
        try:
            self.session = zenoh.open(conf)
            print("[Python] Zenoh connected successfully", flush=True)
        except Exception as e:
            print(f"[Python CRITICAL] Zenoh connection failed: {e}", flush=True)
            sys.exit(1)

        # FIX: Match the unique key prefix from Rust
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
        self.upsampler = None
        self.model_scale = 4
        self.active_scale = 4
        self.is_configured = False
        self.running = True

        # FIX: Initialize the flag here to prevent AttributeError
        self.is_rgb_model = False

        # Try to load standard model first
        default_model = "RealESRGAN_x4plus"
        print(f"[Python] Attempting initial load: {default_model}", flush=True)
        self.load_model(default_model)
        self.loop()

    def loop(self):
        print("[Python] Ready...", flush=True)
        while self.running:
            time.sleep(0.1)
        self.cleanup()

    def cleanup(self):
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
        # Free GPU memory on cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_filename):
        weight_path = find_weight_file(model_filename)
        if not weight_path:
            self.send_status("error", {"message": f"Model missing: {model_filename}"})
            return

        print(f"[Python] Loading: {model_filename}", flush=True)

        # 3. Standard RealESRGAN models are always BGR output
        self.is_rgb_model = False

        try:
            # Load Weights FIRST to auto-detect scale
            loadnet = torch.load(weight_path, map_location=torch.device("cpu"))
            
            # Smart Search for weights inside the file structure
            state_dict = find_weights_recurse(loadnet, "conv_first.weight")
            if state_dict is None:
                state_dict = loadnet

            # Determine Architecture (Standard vs Anime 6B)
            # Default to 23 blocks (Standard)
            num_blocks = 23
            name_lower = model_filename.lower()
            if "anime_6b" in name_lower:
                num_blocks = 6

            # --- ARCHITECTURE MATCHING ---
            # We try likely scales to find one that matches the weight shapes.
            # This handles cases where:
            # 1. Weights are Standard (3ch in) but RRDBNet(4) uses PixelUnshuffle (48ch in). -> We fall back to RRDBNet(1) or compatible.
            # 2. Weights are Anime (48ch in) and RRDBNet(4) matches.
            # 3. Weights are x4 but labeled x2 (User issue).
            
            target_in_ch = 3
            if "conv_first.weight" in state_dict:
                 target_in_ch = state_dict["conv_first.weight"].shape[1]

            match_found = False
            best_model = None
            final_scale = 4

            # Preference order: Try inferred scale first (from filename), then 4, 2, 1
            search_scales = [4, 2, 1]
            if "x2" in name_lower and 2 in search_scales:
                search_scales.remove(2)
                search_scales.insert(0, 2)
            
            # If we detected a scale from conv_last previously, prioritize that
            if "conv_last.weight" in state_dict:
                out_shape = state_dict["conv_last.weight"].shape
                out_filters = out_shape[0]
                detected_out_scale = int((out_filters / 3) ** 0.5)
                if detected_out_scale in [1, 2, 4, 8] and detected_out_scale not in search_scales:
                    search_scales.insert(0, detected_out_scale)

            print(f"[Python] Auto-matching architecture. Target In-Channels: {target_in_ch}. Trying scales: {search_scales}", flush=True)

            for try_scale in search_scales:
                try:
                    test_model = RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=num_blocks,
                        num_grow_ch=32,
                        scale=try_scale,
                    )
                    
                    # Check if input shape matches
                    # We create a dummy variable to check expected input channels
                    # Or simpler: check conv_first.weight.shape[1] of the init model
                    model_in_ch = test_model.conv_first.weight.shape[1]
                    
                    if model_in_ch == target_in_ch:
                        # Found a structural match!
                        best_model = test_model
                        final_scale = try_scale
                        match_found = True
                        print(f"[Python] Match found: Scale {try_scale} (Model expects {model_in_ch} ch)", flush=True)
                        break
                    else:
                        print(f"[Python] Scale {try_scale} mismatch: Model expects {model_in_ch} ch, Weights have {target_in_ch} ch", flush=True)
                except Exception as e:
                    print(f"[Python] Architecture init failed for scale {try_scale}: {e}", flush=True)

            if not match_found:
                # Fallback to default scale 4 if nothing matched exactly (hope for loose load)
                print("[Python] No exact match found, falling back to scale 4", flush=True)
                final_scale = 4
                best_model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=num_blocks,
                    num_grow_ch=32,
                    scale=4,
                )

            self.model_scale = final_scale
            model = best_model

            # Setup Device
            if torch.cuda.is_available():
                device = "cuda"
                torch.cuda.empty_cache()
            else:
                device = "cpu"

            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"[Python] Strict load failed: {e}", flush=True)
                print("[Python] Retrying loose load...", flush=True)
                model.load_state_dict(state_dict, strict=False)

            model.eval()
            model = model.to(device)

            # Init Upsampler
            self.upsampler = SafeRealESRGANer(
                scale=self.model_scale,
                model_path=None,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=(device == "cuda"),
                gpu_id=0 if device == "cuda" else None,
            )

            self.model_name = model_filename
            print(
                f"[Python] Loaded: {model_filename} (Active Scale: x{self.model_scale})",
                flush=True,
            )
            self.send_status(
                "MODEL_LOADED", {"model": model_filename, "scale": self.model_scale}
            )

        except Exception as e:
            print(f"[Python CRITICAL] Load Error:", flush=True)
            traceback.print_exc()
            self.send_status("error", {"message": f"Load Failed: {str(e)}"})

    def send_status(self, status, extra=None):
        payload = {"status": status}
        if extra:
            payload.update(extra)
        try:
            self.pub.put(json.dumps(payload).encode("utf-8"))
        except Exception as e:
            print(f"[Python Warning] Failed to send status: {e}", flush=True)

    def on_request(self, sample):
        try:
            payload = json.loads(sample.payload.to_bytes().decode("utf-8"))
            cmd = payload.get("command")
            if cmd == "create_shm":
                self.create_shm(payload)
            elif cmd == "process_frame":
                self.process_frame(payload)
            elif cmd == "load_model":
                new_model = payload.get("params", {}).get("model_name")
                if new_model:
                    self.load_model(new_model)
            elif cmd == "upscale_image_file":
                self.handle_image_file(payload)
            elif cmd == "shutdown":
                self.running = False
        except Exception as e:
            print(f"[Python Error] Request failed: {e}", flush=True)
            self.send_status("error", {"message": str(e)})

    # -------------------------------------------------------------------------
    # TILING LOGIC (Progress-Aware)
    # -------------------------------------------------------------------------
    def process_image_tile(self, img, job_id):
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

        # Mirror Pad the image
        img_padded = np.pad(
            img, ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0)), mode="reflect"
        )

        # Determine dtype for tensor transfers (optimized)
        target_dtype = torch.float16 if self.upsampler.half else torch.float32

        for y in y_steps:
            for x in x_steps:
                count += 1

                # Extract tile from padded image
                pad_y = y
                pad_x = x

                # Dimensions of the valid content in this tile (clip at edges)
                in_h = min(tile_size, h - y)
                in_w = min(tile_size, w - x)

                # Slice: include padding
                tile_in = img_padded[
                    pad_y : pad_y + in_h + 2 * tile_pad,
                    pad_x : pad_x + in_w + 2 * tile_pad,
                    :,
                ]

                # Normalize
                tile_in = tile_in.astype(np.float32) / 255.0

                # BGR -> RGB + Negative Stride Fix (FIX: Added .copy())
                if not self.is_rgb_model:
                    tile_in = tile_in[:, :, ::-1].copy()

                # FIX: Ensure even dimensions for Anime models (PixelUnshuffle requirement)
                # Pad right/bottom if odd
                h_in, w_in = tile_in.shape[:2]
                pad_h = h_in % 2
                pad_w = w_in % 2
                if pad_h or pad_w:
                    tile_in = np.pad(tile_in, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

                # Optimized Tensor Transfer: directly convert to target dtype on GPU
                tile_tensor = torch.from_numpy(tile_in.transpose(2, 0, 1)).to(
                    device=self.upsampler.device, dtype=target_dtype, non_blocking=True
                ).unsqueeze(0)

                # Inference
                with torch.no_grad():
                    if hasattr(self.upsampler.model, "module"):
                        output_tensor = self.upsampler.model.module(tile_tensor)
                    else:
                        output_tensor = self.upsampler.model(tile_tensor)

                # Post Process
                output_tile = (
                    output_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                )
                if output_tile.ndim == 3:
                    output_tile = output_tile.transpose(1, 2, 0)

                # RGB -> BGR + Negative Stride Fix (FIX: Added .copy())
                if not self.is_rgb_model:
                    output_tile = output_tile[:, :, ::-1].copy()

                output_tile = (output_tile * 255.0).round().astype(np.uint8)

                # Crop Padding from Result
                out_pad = tile_pad * scale
                out_h_real = in_h * scale
                out_w_real = in_w * scale

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

        # Free GPU memory after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_img

    def handle_image_file(self, payload):
        req_id = payload.get("id")
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

                # Apply brightness: shift values
                if abs(brightness) > 0.001:
                    img = img + (brightness * 255)

                # Apply contrast: scale around mean
                if abs(contrast) > 0.001:
                    contrast_factor = 1.0 + contrast
                    img = (img - 127.5) * contrast_factor + 127.5

                # Apply gamma correction
                if abs(gamma - 1.0) > 0.001:
                    img = np.clip(img, 0, 255)
                    img = ((img / 255.0) ** (1.0 / gamma)) * 255.0

                # Apply saturation in HSV space
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

            cv2.imwrite(params["output_path"], output)
            self.send_status("ok", {"id": req_id})

        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"id": req_id, "message": str(e)})

    def create_shm(self, payload):
        width = payload["width"]
        height = payload["height"]
        self.active_scale = payload["scale"]
        self.ring_size = payload.get("ring_size", 3)

        self.input_size = width * height * 4
        self.slot_byte_size = self.input_size + (
            (width * self.active_scale) * (height * self.active_scale) * 4
        )
        total_size = self.slot_byte_size * self.ring_size

        if self.mmap:
            self.cleanup()

        try:
            fd, self.shm_path = tempfile.mkstemp(prefix="vf_buffer_", suffix=".bin")
            self.shm_file = os.fdopen(fd, "wb+")
            self.shm_file.write(b"\0" * total_size)
            self.shm_file.flush()
            self.shm_file.seek(0)
            self.mmap = mmap.mmap(self.shm_file.fileno(), total_size)

            self.input_shape = (height, width, 4)
            self.output_shape = (
                height * self.active_scale,
                width * self.active_scale,
                4,
            )
            self.is_configured = True
            self.send_status("SHM_CREATED", {"shm_path": self.shm_path})
        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"message": str(e)})

    def process_frame(self, payload):
        if not self.is_configured:
            return
        slot_idx = payload.get("slot", 0)
        try:
            base = slot_idx * self.slot_byte_size
            in_end = base + self.input_size
            out_end = base + self.slot_byte_size

            in_view = np.frombuffer(
                self.mmap, dtype=np.uint8, count=self.input_size, offset=base
            ).reshape(self.input_shape)
            out_view = np.frombuffer(
                self.mmap, dtype=np.uint8, count=(out_end - in_end), offset=in_end
            ).reshape(self.output_shape)

            if self.active_scale == 1:
                out_view[:] = in_view[:]
                self.send_status("FRAME_DONE", {"slot": slot_idx})
                return

            # Rust(RGBA) -> BGR (Model Input) + Fix negative strides
            img_bgr = in_view[:, :, :3][:, :, ::-1].copy()

            with suppress_stdout():
                out_raw, _ = self.upsampler.enhance(img_bgr, outscale=self.model_scale)

            h, w = out_raw.shape[:2]
            target_h, target_w = self.output_shape[:2]
            if h != target_h or w != target_w:
                out_raw = cv2.resize(out_raw, (target_w, target_h))

            # BGR (Model Output) -> RGBA (Rust Input) + Fix negative strides
            if self.is_rgb_model:
                out_view[:, :, :3] = out_raw
            else:
                out_view[:, :, :3] = out_raw[:, :, ::-1].copy()

            out_view[:, :, 3] = 255
            self.send_status("FRAME_DONE", {"slot": slot_idx})
        except Exception as e:
            traceback.print_exc()
            self.send_status("error", {"message": str(e)})
        finally:
            # Free GPU memory after each frame to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="7447")
    parser.add_argument("--parent-pid", type=int, default=0, help="Parent Process ID to monitor")
    args = parser.parse_args()
    
    if args.parent_pid > 0:
        start_watchdog(args.parent_pid)
        
    AIWorker(args.port)
    sys.exit(0)
