"""
Export a VideoForge .pth model to ONNX.

Usage:
  python tools/export_onnx.py weights/RealESRGAN_x4plus.pth
  python tools/export_onnx.py weights/RealESRGAN_x4plus.pth --output weights/RealESRGAN_x4plus.onnx --half
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import torch


def export(src: str, dst: str, opset: int, half: bool) -> None:
    import spandrel
    name = os.path.splitext(os.path.basename(src))[0]
    print(f"Loading {name} via spandrel...")
    desc = spandrel.ModelLoader(device="cpu").load_from_file(src)
    model = desc.model
    scale = desc.scale
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dtype = torch.float16 if half else torch.float32
    if half:
        model = model.half()
    dummy = torch.zeros(1, 3, 64, 64, dtype=dtype)
    print(f"Exporting → {dst}  (scale={scale}x, opset={opset}, {'fp16' if half else 'fp32'})")
    torch.onnx.export(
        model, dummy, dst,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input":  {0: "batch_size", 2: "height", 3: "width"},
                      "output": {0: "batch_size", 2: "height", 3: "width"}},
        opset_version=opset, do_constant_folding=True,
    )
    print(f"Done. {os.path.getsize(dst)/1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("--output", "-o")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--half", action="store_true")
    args = p.parse_args()
    export(args.src, args.output or os.path.splitext(args.src)[0] + ".onnx", args.opset, args.half)
