"""VideoForge Model Loaders — Third-party and legacy model loading."""

from loaders.spandrel_loader import load_via_spandrel  # noqa: F401
from loaders.swin2sr_loader import Swin2SRWrapper, load_swin2sr_hf  # noqa: F401
from loaders.onnx_loader import OnnxModelWrapper, load_onnx_model  # noqa: F401
from loaders.official_stubs import register_official_model_stubs  # noqa: F401
