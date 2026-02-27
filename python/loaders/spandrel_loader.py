"""
Spandrel Model Loader — Auto-detects 30+ SR architectures.

Extracted from model_manager.py. Lazy-imports spandrel at call time.
"""

import logging
from typing import Tuple

import torch.nn as nn

log = logging.getLogger("videoforge")


def load_via_spandrel(path: str, model_key: str) -> Tuple[nn.Module, int]:
    """
    Load any SR model via spandrel (supports 30+ architectures).

    Spandrel auto-detects the architecture from state-dict key patterns
    and builds the correct nn.Module.
    """
    try:
        import spandrel
    except ImportError:
        raise RuntimeError(
            "spandrel package required for loading this model type.  "
            "Install: pip install spandrel"
        )

    log.info(f"Loading via spandrel: {model_key}")
    model_descriptor = spandrel.ModelLoader(device="cpu").load_from_file(path)
    model = model_descriptor.model
    scale = model_descriptor.scale

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Tag as spandrel-loaded so create_adapter uses a pass-through adapter.
    model._vf_spandrel = True  # type: ignore[attr-defined]

    log.info(f"Spandrel loaded {model_key}: "
          f"arch={model_descriptor.architecture.name}, scale={scale}x")
    return model, scale
