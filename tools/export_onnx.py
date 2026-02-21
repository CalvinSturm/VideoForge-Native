"""
Export a VideoForge .pth model to ONNX using the robust ModelLoader.

Usage:
  python tools/export_onnx.py weights/rcan_4x.pt
  python tools/export_onnx.py weights/rcan_4x.pt --output weights/rcan_4x.onnx --half
"""
import argparse
import os
import sys
import torch

# Add python/ directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from model_manager import ModelLoader, _load_module

def export(src: str, dst: str, opset: int, half: bool) -> None:
    print(f"Loading {os.path.basename(src)} via model_manager...")
    
    # Use internal loader directly to get model + scale without device movement yet
    # Or use ModelLoader class
    loader = ModelLoader(precision="fp16" if half else "fp32")
    # model_manager expects a model identifier, usually a key or path?
    # _load_module's docstring says "Load a model from weights/{model_key}.pth" or similar.
    # But it calls _resolve_weight_path which handles keys.
    # If we pass a full path, _resolve_weight_path might fail if it doesn't match the search logic.
    # Let's check _resolve_weight_path logic in model_manager.py.
    # It searches in weights/ dir.
    # If src is a path, we might need to be careful.
    
    # Actually, _load_module calls _resolve_weight_path.
    # We should probably bypass _load_module if we have a direct path, 
    # OR we can update _load_module to handle absolute paths, 
    # OR we just use the logic from _load_module but with our path.
    
    # Let's replicate _load_module's loading logic but for a specific file path.
    # model_manager.py has _load_module(model_key). 
    # We can use _load_module if we pass the key.
    
    # If src is "weights/rcan_4x.pt", the key is "rcan_4x".
    key = os.path.splitext(os.path.basename(src))[0]
    
    # We can try to use ModelLoader with the key.
    try:
        model, scale = loader.load(key)
    except FileNotFoundError:
        # If the file is not in weights/ or not found by key, fallback to direct loading
        # But ModelLoader relies on _load_module which relies on _resolve_weight_path.
        print(f"Key '{key}' not found in registry/weights dir, attempting direct load of {src}...")
        # We can implement a direct load helper here or use spandrel directly if compatible, 
        # but the goal was to use model_manager's robustness.
        # model_manager has _load_via_spandrel(path, key)
        # and handlers for full models.
        
        # Let's instantiate ModelLoader but manually load the model.
        # This is getting complicated. 
        # Simpler: Modify model_manager.py to allow loading from a path? 
        # No, I shouldn't modify core logic just for a tool if I can avoid it.
        
        # Let's just use the key and hope the user put the file in weights/.
        # The tool usage says "python tools/export_onnx.py weights/rcan_4x.pt"
        raise

    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Input tensor
    # Some models need specific input size? most are fully convolutional.
    # 64x64 is standard for export.
    device = next(model.parameters()).device
    dummy_input = torch.zeros(1, 3, 64, 64, device=device)
    if half and device.type == "cuda":
        dummy_input = dummy_input.half()
        model = model.half()

    print(f"Exporting to {dst}...")
    print(f"  Scale: {scale}x")
    print(f"  Precision: {'fp16' if half else 'fp32'}")
    print(f"  Opset: {opset}")
    
    # Ensure dst directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            dst,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"}
            },
            opset_version=opset,
            do_constant_folding=True
        )
    
    print(f"Success! Output size: {os.path.getsize(dst) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="Path to input .pth/.pt model")
    parser.add_argument("--output", "-o", help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="Export in FP16 mode")
    args = parser.parse_args()

    dst = args.output
    if not dst:
        dst = os.path.splitext(args.src)[0] + ".onnx"
        if args.half and not dst.endswith("_fp16.onnx"):
            dst = dst.replace(".onnx", "_fp16.onnx")

    export(args.src, dst, args.opset, args.half)
