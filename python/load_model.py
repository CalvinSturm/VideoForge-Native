import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
from typing import Optional

def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # Initialize the model architecture
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    # Load weights, mapping to correct device (CPU/GPU)
    weights = torch.load(weights_path, map_location=device)

    # Extract actual state dict if wrapped
    if isinstance(weights, dict):
        if 'params_ema' in weights:
            weights = weights['params_ema']
        elif 'state_dict' in weights:
            weights = weights['state_dict']

    # Load weights strictly (ensures exact matching keys)
    model.load_state_dict(weights, strict=True)

    # Move to device first, then set to eval mode
    model.to(device)
    model.eval()

    return model
