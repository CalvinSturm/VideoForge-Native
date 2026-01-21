import sys
import json
import base64
import threading
import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np
import cv2
from realesrgan import RealESRGANer

# ───────────── CONFIG ─────────────
MODEL_PATH = r"videoForge2\weights\RealESRGAN_x4plus\RealESRGAN_x4plus.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 4
TILE = 0
TILE_PAD = 10
PRE_PAD = 0
HALF = DEVICE == "cuda"

# ───────────── GLOBALS ─────────────
upsampler_lock = threading.Lock()
upsampler: Optional[RealESRGANer] = None
_cancel_flags: Dict[str, bool] = {}
_cancel_lock = threading.Lock()


# ───────────── UTILITIES ─────────────
def send(obj: Dict[str, Any]) -> None:
    """Send JSON to stdout."""
    try:
        json_str = json.dumps(obj)
        sys.stdout.write(json_str + "\n")
        sys.stdout.flush()
    except Exception:
        sys.exit(1)


def send_ok(req_id: str, result: Optional[Dict[str, Any]] = None) -> None:
    send({"status": "ok", "id": req_id, "result": result or {}, "error": None})


def send_error(req_id: str, error_msg: str) -> None:
    send({"status": "error", "id": req_id, "result": {}, "error": error_msg})


def send_progress(req_id: str, value: float, tile: int = 0, total: int = 0) -> None:
    send({"status": "progress", "id": req_id, "result": {"value": value, "tile": tile, "total": total}, "error": None})


def is_cancelled(job_id: str) -> bool:
    with _cancel_lock:
        return _cancel_flags.get(job_id, False)


def set_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_flags[job_id] = True


def clear_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_flags.pop(job_id, None)


def decode_image(b64: str):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


# ───────────── MODEL ─────────────
def load_model() -> None:
    global upsampler
    with upsampler_lock:
        upsampler = RealESRGANer(
            scale=SCALE,
            model_path=MODEL_PATH,
            dni_weight=None,
            model=None,
            tile=TILE,
            tile_pad=TILE_PAD,
            pre_pad=PRE_PAD,
            half=HALF,
            device=DEVICE,
        )


# ───────────── UPSCALING ─────────────
def upscale_image(job_id: str, img_b64: str) -> str:
    if is_cancelled(job_id):
        raise RuntimeError("CANCELLED")

    img = decode_image(img_b64)

    h, w = img.shape[:2]
    out_img = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.uint8)

    # For simplicity, RealESRGANer will handle full image if TILE=0
    out, _ = upsampler.enhance(img, outscale=SCALE)

    if is_cancelled(job_id):
        raise RuntimeError("CANCELLED")

    return encode_image(out)


def upscale_batch(job_id: str, images_b64: List[str]) -> List[str]:
    results = []
    total = len(images_b64)
    for idx, img_b64 in enumerate(images_b64):
        if is_cancelled(job_id):
            raise RuntimeError("CANCELLED")

        out_b64 = upscale_image(job_id, img_b64)
        results.append(out_b64)
        send_progress(job_id, value=(idx + 1) / total, tile=idx + 1, total=total)
    return results


def upscale_video(job_id: str, frames_b64: List[str]) -> None:
    total = len(frames_b64)
    for idx, frame_b64 in enumerate(frames_b64):
        if is_cancelled(job_id):
            raise RuntimeError("CANCELLED")

        out_b64 = upscale_image(job_id, frame_b64)
        send({"status": "frame", "id": job_id, "result": {"frame": idx, "total": total, "image_base64": out_b64}, "error": None})
        send_progress(job_id, value=(idx + 1) / total, tile=idx + 1, total=total)


# ───────────── HANDLERS ─────────────
def handle_request(req: Dict[str, Any]) -> None:
    cmd = req.get("command")
    job_id = req.get("id", "unknown")

    try:
        if cmd == "upscale_image_base64":
            out_b64 = upscale_image(job_id, req["params"]["image"])
            send_ok(job_id, {"image": out_b64})

        elif cmd == "upscale_batch":
            images = req["params"]["images_base64"]
            results = upscale_batch(job_id, images)
            send_ok(job_id, {"images_base64": results})

        elif cmd == "upscale_video":
            frames = req["params"]["frames_base64"]
            upscale_video(job_id, frames)
            send_ok(job_id, {"frames": len(frames)})

        elif cmd == "cancel":
            target = req["params"].get("id")
            if target:
                set_cancel(target)
                send_ok(job_id, {"cancelled": target})
            else:
                send_error(job_id, "Missing job id to cancel")

        else:
            send_error(job_id, f"Unknown command: {cmd}")

    except RuntimeError as e:
        if str(e) == "CANCELLED":
            send_error(job_id, "Job cancelled")
        else:
            send_error(job_id, str(e))
    finally:
        clear_cancel(job_id)


# ───────────── MAIN LOOP ─────────────
def main():
    try:
        load_model()
    except Exception as e:
        send_error("", f"Failed to load model: {str(e)}")
        sys.exit(1)

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            handle_request(req)
        except Exception as e:
            send_error(req.get("id", "unknown"), f"Unhandled exception: {str(e)}")


if __name__ == "__main__":
    main()