import base64
import io
import json
import math
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler

from load_model import load_model  # Your improved load_model function

# ─────────────────────────────────────────────
# CONFIGURATION & GLOBALS
# ─────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_SCALE = 4
DEFAULT_TILE_SIZE = 128
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "upscale_engine.log")

# Set up logging
logger = logging.getLogger("upscale_engine")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Rotating file handler (max 5MB, keep 3 backups)
file_handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Starting upscale engine")

# Model state
model_lock = threading.Lock()
model: Optional[torch.nn.Module] = None
model_scale: int = DEFAULT_MODEL_SCALE
tile_size: int = DEFAULT_TILE_SIZE
model_name: str = "RealESRGAN_x4plus"
weights_path: Optional[str] = None

# Cancellation management
_cancel_flags: Dict[str, bool] = {}
_cancel_lock = threading.Lock()

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def send(obj: Dict[str, Any]) -> None:
    """Send a JSON object to stdout."""
    try:
        json_str = json.dumps(obj)
        logger.debug(f"Sending: {json_str}")
        sys.stdout.write(json_str + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        logger.warning("BrokenPipeError on send; exiting")
        sys.exit(0)
    except Exception:
        logger.exception("Exception during send")

def send_event(event: str, id: Optional[str] = None, **payload) -> None:
    """Send an event message."""
    send({
        "event": event,
        "id": id,
        **payload,
    })

def send_ok(id: str, result: Optional[Dict[str, Any]] = None) -> None:
    send({
        "status": "ok",
        "id": id,
        "result": result or {},
    })

def send_error(id: str, code: str, message: str) -> None:
    logger.error(f"Error [{code}] on job {id}: {message}")
    send({
        "status": "error",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        },
    })

def is_cancelled(job_id: str) -> bool:
    with _cancel_lock:
        return _cancel_flags.get(job_id, False)

def set_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_flags[job_id] = True
    logger.info(f"Cancellation set for job {job_id}")

def clear_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_flags.pop(job_id, None)
    logger.info(f"Cancellation cleared for job {job_id}")

def decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")

def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def list_available_models() -> List[str]:
    """Recursively scan WEIGHTS_DIR for all .pth files, returning paths relative to WEIGHTS_DIR."""
    models = []
    for root, dirs, files in os.walk(WEIGHTS_DIR):
        for file in files:
            if file.endswith(".pth"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, WEIGHTS_DIR)
                models.append(rel_path.replace("\\", "/"))  # Normalize for Windows paths
    logger.debug(f"Available models: {models}")
    return models

def load_weights(model_name: str) -> str:
    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights")
    )

    candidate = os.path.join(
        base,
        model_name,
        f"{model_name}.pth"
    )

    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"Model weights not found for '{model_name}': {candidate}"
        )

    return candidate

def reload_model(model_name_param: str) -> None:
    global model, model_scale, weights_path, model_name
    with model_lock:
        logger.info(f"Reloading model '{model_name_param}'")
        weights_file = load_weights(model_name_param)
        logger.info(f"Loading model weights from: {weights_file}")
        model = load_model(weights_file, DEVICE)
        model_name = model_name_param
        # Infer model scale from name, default fallback
        if "x4" in model_name_param:
            model_scale = 4
        elif "x2" in model_name_param:
            model_scale = 2
        else:
            model_scale = DEFAULT_MODEL_SCALE
        weights_path = weights_file
        logger.info(f"Model '{model_name}' loaded with scale {model_scale} from {weights_path}")

# ─────────────────────────────────────────────
# UPSCALING FUNCTIONS
# ─────────────────────────────────────────────

def upscale_stream(
    img: Image.Image,
    job_id: str,
) -> Image.Image:
    """Upscale an image in tiles with progress events and cancellation."""

    w, h = img.size
    logger.info(f"Upscaling job {job_id}: input size {w}x{h}")
    out = Image.new("RGB", (w * model_scale, h * model_scale))

    x_tiles = math.ceil(w / tile_size)
    y_tiles = math.ceil(h / tile_size)
    total_tiles = x_tiles * y_tiles
    tile_index = 0

    start_time = time.time()

    for ty in range(y_tiles):
        for tx in range(x_tiles):
            if is_cancelled(job_id):
                send_event("cancelled", id=job_id)
                logger.info(f"Job {job_id} cancelled during upscaling")
                raise RuntimeError("CANCELLED")

            tile_index += 1

            left = tx * tile_size
            top = ty * tile_size
            right = min(left + tile_size, w)
            bottom = min(top + tile_size, h)

            crop = img.crop((left, top, right, bottom))
            arr = np.array(crop).astype("float32") / 255.0
            tensor = (
                torch.from_numpy(arr)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            with torch.no_grad():
                out_tensor = model(tensor)

            out_arr = (
                out_tensor.squeeze(0)
                .clamp(0, 1)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )

            out_crop = Image.fromarray((out_arr * 255).astype("uint8"))
            out.paste(out_crop, (left * model_scale, top * model_scale))

            # Estimate ETA
            elapsed = time.time() - start_time
            tiles_left = total_tiles - tile_index
            eta = elapsed / tile_index * tiles_left if tile_index else 0

            send_event(
                "progress",
                id=job_id,
                value=tile_index / total_tiles,
                tile=tile_index,
                total=total_tiles,
                coords={"x": tx, "y": ty},
                eta_seconds=eta,
            )

    logger.info(f"Job {job_id} completed successfully")
    return out

def upscale_batch(
    images_base64: List[str],
    job_id: str,
) -> List[str]:
    """Upscale a batch of images, returning list of upscaled base64 images."""

    logger.info(f"Starting batch upscale job {job_id} with {len(images_base64)} images")
    results = []

    for idx, b64 in enumerate(images_base64):
        if is_cancelled(job_id):
            send_event("cancelled", id=job_id)
            logger.info(f"Job {job_id} cancelled during batch upscale at index {idx}")
            raise RuntimeError("CANCELLED")

        img = decode_image(b64)
        out = upscale_stream(img, job_id)
        out_b64 = encode_image(out)
        results.append(out_b64)
        send_event("progress", id=job_id, value=(idx+1)/len(images_base64), tile=idx+1, total=len(images_base64))

    logger.info(f"Batch upscale job {job_id} completed")
    return results

def upscale_video(
    frames_base64: List[str],
    job_id: str,
) -> None:
    """
    Upscale video frames one by one and send 'frame' events for each.
    The frames_base64 list must be ordered by frame index.
    """

    logger.info(f"Starting video upscale job {job_id} with {len(frames_base64)} frames")

    for idx, b64 in enumerate(frames_base64):
        if is_cancelled(job_id):
            send_event("cancelled", id=job_id)
            logger.info(f"Job {job_id} cancelled during video upscale at frame {idx}")
            raise RuntimeError("CANCELLED")

        img = decode_image(b64)
        out = upscale_stream(img, job_id)
        out_b64 = encode_image(out)

        send_event(
            "frame",
            id=job_id,
            frame=idx,
            total=len(frames_base64),
            image_base64=out_b64,
        )
        send_event("progress", id=job_id, value=(idx + 1) / len(frames_base64), tile=idx + 1, total=len(frames_base64))

    logger.info(f"Video upscale job {job_id} completed")

# ─────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────

def handle_upscale_image(req: Dict[str, Any]) -> None:
    job_id = req["id"]
    img = decode_image(req["params"]["image_base64"])

    try:
        out = upscale_stream(img, job_id)
        send_ok(job_id, {"image_base64": encode_image(out)})
    except RuntimeError as e:
        if str(e) == "CANCELLED":
            send_error(job_id, "CANCELLED", "Job cancelled by user")
        else:
            logger.exception(f"Unexpected error in upscale_image job {job_id}")
            send_error(job_id, "FAILED", "Unexpected error during upscale")
    finally:
        clear_cancel(job_id)

def handle_upscale_frame(req: Dict[str, Any]) -> None:
    job_id = req["id"]
    frame_index = req["params"]["frame_index"]
    img = decode_image(req["params"]["image_base64"])

    try:
        out = upscale_stream(img, job_id)
        out_b64 = encode_image(out)

        send_event(
            "frame",
            id=job_id,
            frame=frame_index,
            image_base64=out_b64,
        )
        send_ok(job_id, {"frame": frame_index})
    except RuntimeError as e:
        if str(e) == "CANCELLED":
            send_error(job_id, "CANCELLED", "Frame cancelled")
        else:
            logger.exception(f"Unexpected error in upscale_frame job {job_id}")
            send_error(job_id, "FAILED", "Unexpected error during frame upscale")
    finally:
        clear_cancel(job_id)

def handle_upscale_batch(req: Dict[str, Any]) -> None:
    job_id = req["id"]
    images_b64: List[str] = req["params"]["images_base64"]

    try:
        results = upscale_batch(images_b64, job_id)
        send_ok(job_id, {"images_base64": results})
    except RuntimeError as e:
        if str(e) == "CANCELLED":
            send_error(job_id, "CANCELLED", "Batch cancelled")
        else:
            logger.exception(f"Unexpected error in upscale_batch job {job_id}")
            send_error(job_id, "FAILED", "Unexpected error during batch upscale")
    finally:
        clear_cancel(job_id)

def handle_upscale_video(req: Dict[str, Any]) -> None:
    job_id = req["id"]
    frames_b64: List[str] = req["params"]["frames_base64"]

    try:
        upscale_video(frames_b64, job_id)
        send_ok(job_id, {"frames": len(frames_b64)})
    except RuntimeError as e:
        if str(e) == "CANCELLED":
            send_error(job_id, "CANCELLED", "Video cancelled")
        else:
            logger.exception(f"Unexpected error in upscale_video job {job_id}")
            send_error(job_id, "FAILED", "Unexpected error during video upscale")
    finally:
        clear_cancel(job_id)

def handle_load_model(req: Dict[str, Any]) -> None:
    job_id = req["id"]
    new_model_name = req["params"].get("model_name")

    if not new_model_name:
        send_error(job_id, "BAD_REQUEST", "Missing 'model_name' param")
        return

    try:
        reload_model(new_model_name)
        send_ok(job_id, {"model": new_model_name, "scale": model_scale})
    except Exception as e:
        logger.exception(f"Failed to load model '{new_model_name}': {str(e)}")
        send_error(job_id, "LOAD_FAILED", str(e))  # Now it will send the exact error message

def handle_cancel(req: Dict[str, Any]) -> None:
    target = req["params"].get("id")
    if not target:
        send_error(req.get("id", ""), "BAD_REQUEST", "Missing job id to cancel")
        return

    set_cancel(target)
    send_ok(req.get("id", ""), {"cancelled": target})

def handle_list_models(req: Dict[str, Any]) -> None:
    job_id = req.get("id", "")
    try:
        models = list_available_models()
        send_ok(job_id, {"models": models})
    except Exception as e:
        logger.exception("Failed to list models")
        send_error(job_id, "LIST_FAILED", str(e))

# ─────────────────────────────────────────────
# REQUEST DISPATCH
# ─────────────────────────────────────────────

COMMANDS = {
    "upscale_image": handle_upscale_image,
    "upscale_frame": handle_upscale_frame,
    "upscale_batch": handle_upscale_batch,
    "upscale_video": handle_upscale_video,
    "load_model": handle_load_model,
    "cancel": handle_cancel,
    "list_models": handle_list_models,
}

def process_request(req: Dict[str, Any]) -> None:
    cmd = req.get("command")
    handler = COMMANDS.get(cmd)

    if not handler:
        send_error(
            req.get("id", ""),
            "UNKNOWN_COMMAND",
            f"Unknown command: {cmd}",
        )
        return

    logger.info(f"Processing command '{cmd}' id={req.get('id')}")
    handler(req)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main() -> None:
    try:
        reload_model(model_name)
    except Exception as e:
        logger.exception("Failed to load initial model")
        send_error("", "INIT_FAILED", str(e))
        sys.exit(1)

    send_event(
        "ready",
        backend="python",
        device=str(DEVICE),
        model=model_name,
        scale=model_scale,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            process_request(req)
        except Exception:
            send({
                "status": "error",
                "id": "",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Unhandled exception",
                },
                "traceback": traceback.format_exc(),
            })

if __name__ == "__main__":
    main()