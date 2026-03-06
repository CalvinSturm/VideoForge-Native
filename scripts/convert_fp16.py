#!/usr/bin/env python3
"""
Convert an ONNX model's internal ops from FP32 to FP16.

Usage:
    python scripts/convert_fp16.py path/to/model.onnx
    python scripts/convert_fp16.py path/to/model.onnx --output path/to/model.fp16.onnx

The converted model keeps FP32 inputs/outputs (keep_io_types=True) so the
inference pipeline needs no changes — only internal compute uses FP16,
which runs ~2× faster on Tensor Core GPUs (Ampere+).

Requirements:
    pip install onnx onnxconverter-common
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def convert_fp16(input_path: str, output_path: str | None = None) -> str:
    """Convert an ONNX model to FP16 internals, keeping IO as FP32."""
    try:
        import onnx
    except ImportError:
        print("ERROR: 'onnx' package not installed. Run: pip install onnx", file=sys.stderr)
        sys.exit(1)

    try:
        from onnxconverter_common import float16
    except ImportError:
        print(
            "ERROR: 'onnxconverter-common' package not installed. "
            "Run: pip install onnxconverter-common",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Model not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        # model.onnx → model.fp16.onnx
        output_path = input_path.with_suffix(".fp16.onnx")
    else:
        output_path = Path(output_path)

    input_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Loading  {input_path} ({input_mb:.1f} MB)...")

    model = onnx.load(str(input_path))

    print("Converting internal ops to FP16 (keep_io_types=True)...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,      # Inputs & outputs stay FP32
        min_positive_val=1e-7,    # Clamp small constants to avoid FP16 underflow
        max_finite_val=65504.0,   # FP16 max representable value
    )

    onnx.save(model_fp16, str(output_path))
    output_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"Saved    {output_path} ({output_mb:.1f} MB)")
    print(f"Size reduction: {input_mb:.1f} MB → {output_mb:.1f} MB ({output_mb/input_mb*100:.0f}%)")
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ONNX model internal ops from FP32 to FP16.",
        epilog="The native engine auto-detects .fp16.onnx variants when --precision fp16 is set.",
    )
    parser.add_argument("model", help="Path to the FP32 ONNX model")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: <model>.fp16.onnx)",
    )
    args = parser.parse_args()
    convert_fp16(args.model, args.output)


if __name__ == "__main__":
    main()
