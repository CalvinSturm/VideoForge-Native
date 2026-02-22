import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(SCRIPT_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from logging_setup import setup_logging
from shm_worker import PreallocBuffers, enforce_deterministic_mode


def _utc_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _default_precision(device_choice: str) -> str:
    if device_choice == "cuda":
        return "fp16"
    if device_choice == "cpu":
        return "fp32"
    return "fp16" if torch.cuda.is_available() else "fp32"


def _mock_infer(x: torch.Tensor, scale: int) -> torch.Tensor:
    y = x * 0.9 + 0.1
    if scale > 1:
        y = F.interpolate(
            y,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )
    return y


def _run_iteration(
    *,
    iteration: int,
    width: int,
    height: int,
    scale: int,
    dtype: torch.dtype,
    device: torch.device,
    prealloc_tensors: bool,
    pool: PreallocBuffers | None,
) -> dict:
    input_u8 = np.full((height, width, 3), fill_value=iteration % 251, dtype=np.uint8)

    t0 = time.perf_counter()
    cpu = torch.from_numpy(input_u8.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0) / 255.0
    t1 = time.perf_counter()

    _sync_if_cuda(device)
    if prealloc_tensors and pool is not None:
        pool.ensure(height, width, scale, dtype, device)
        td0 = time.perf_counter()
        x = pool.copy_in_from_cpu(cpu)
        _sync_if_cuda(device)
        td1 = time.perf_counter()
    else:
        td0 = time.perf_counter()
        x = cpu.to(device=device, dtype=dtype, non_blocking=True)
        _sync_if_cuda(device)
        td1 = time.perf_counter()

    ti0 = time.perf_counter()
    with torch.no_grad():
        y = _mock_infer(x, scale)
        if prealloc_tensors and pool is not None:
            out_buf = pool.get_output()
            if out_buf.shape != y.shape:
                pool.output_gpu = torch.empty_like(y)
                out_buf = pool.output_gpu
            out_buf.copy_(y)
            y = out_buf
    _sync_if_cuda(device)
    ti1 = time.perf_counter()

    tp0 = time.perf_counter()
    _ = y.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    _sync_if_cuda(device)
    tp1 = time.perf_counter()

    total = tp1 - t0
    return {
        "cpu_to_tensor": (t1 - t0) * 1000.0,
        "to_device": (td1 - td0) * 1000.0 if device.type == "cuda" else 0.0,
        "mock_infer": (ti1 - ti0) * 1000.0,
        "postprocess": (tp1 - tp0) * 1000.0,
        "total": total * 1000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoForge frame loop micro-benchmark")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prealloc-tensors", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default="-",
        help="Output JSONL path, or '-' for stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = setup_logging("info")

    try:
        device = _resolve_device(args.device)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    precision = args.precision or _default_precision(args.device)
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if device.type == "cpu" and dtype == torch.float16:
        log.warning("fp16 requested on CPU; using fp32 for compatibility")
        dtype = torch.float32
        precision = "fp32"

    enforce_deterministic_mode(log, args.deterministic)

    pool = PreallocBuffers(log) if args.prealloc_tensors else None
    cfg = {
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "precision": precision,
        "device": str(device),
        "prealloc_tensors": args.prealloc_tensors,
        "deterministic": args.deterministic,
    }

    writer = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        total_iters = args.warmup + args.iterations
        for i in range(total_iters):
            durations = _run_iteration(
                iteration=i,
                width=args.width,
                height=args.height,
                scale=args.scale,
                dtype=dtype,
                device=device,
                prealloc_tensors=args.prealloc_tensors,
                pool=pool,
            )
            if i < args.warmup:
                continue

            sample = {
                "schema_version": "videoforge.perf_sample.v1",
                "ts_utc": _utc_rfc3339(),
                "iteration": i - args.warmup,
                "config": cfg,
                "durations_ms": durations,
            }
            writer.write(json.dumps(sample) + "\n")
    finally:
        if pool is not None:
            pool.clear()
        if writer is not sys.stdout:
            writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

