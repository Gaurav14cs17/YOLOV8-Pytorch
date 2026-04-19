# model/__init__.py
"""YOLOv8 Model Package.

Submodules:
- blocks: Building blocks (ConvBlock, CSPBlock, SPPBlock, etc.)
- backbone: Feature extraction backbone
- neck: Feature Pyramid Network (FPN/PAN)
- head: Detection head with DFL
- factory: Model variant factories
- fusion: Layer fusion for inference
- yolo_core: Main YOLO class
- variants: Alternative architectures (RepViT, Ghost, Light)
"""

# Building blocks
from .blocks import (
    ConvBlock,
    ResidualBlock,
    CSPBlock,
    SPPBlock,
    RepViTBlock,
    RepViTCSPBlock
)

# Architecture components
from .backbone import Backbone
from .neck import Neck
from .head import DFL, Head
from .fusion import fuse_conv_bn, FuseLayer

# Main YOLO class
from .yolo_core import YOLO

# Model factory functions
from .factory import (
    yolo_v8_n,
    yolo_v8_s,
    yolo_v8_m,
    yolo_v8_l,
    yolo_v8_x
)

__all__ = [
    # Blocks
    'ConvBlock',
    'ResidualBlock',
    'CSPBlock',
    'SPPBlock',
    'RepViTBlock',
    'RepViTCSPBlock',
    # Architecture
    'Backbone',
    'Neck',
    'DFL',
    'Head',
    # Fusion
    'fuse_conv_bn',
    'FuseLayer',
    # Main class
    'YOLO',
    # Factory
    'yolo_v8_n',
    'yolo_v8_s',
    'yolo_v8_m',
    'yolo_v8_l',
    'yolo_v8_x',
]

# Optional: import variants (may have additional dependencies)
try:
    from .variants import (
        MobileRepViTBackbone,
        GhostNeck,
        YOLOv8Hybrid,
    )
    __all__.extend(['MobileRepViTBackbone', 'GhostNeck', 'YOLOv8Hybrid'])
except ImportError:
    pass
