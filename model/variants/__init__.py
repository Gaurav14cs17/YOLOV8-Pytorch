# variants/__init__.py
"""Alternative model architectures and lightweight variants.

Modules:
- backboneRepVit: Mobile-optimized RepViT backbone
- ghostneck: GhostNet-style efficient neck
- lightmodel: Lightweight hybrid model
"""

from .backboneRepVit import (
    MobileRepViTBackbone,
    MobileRepViTBlock,
    MobileRepViTBackbonePresets,
    MobileSPPBlock,
)

from .ghostneck import (
    GhostBottleneck,
    GhostModule,
    SELayer,
    GhostFusionBlock,
    GhostNeck,
    LightGhostNeck,
    HybridGhostNeck,
    GhostNeckPresets,
)

from .lightmodel import (
    RepViTBackbone,
    YOLOv8Hybrid,
    GhostHead,
)

# Backward compatibility aliases
BackboneRepViT = MobileRepViTBackbone
LightYOLO = YOLOv8Hybrid
GhostConv = GhostModule

__all__ = [
    # RepViT Backbone
    'MobileRepViTBackbone',
    'MobileRepViTBlock',
    'MobileRepViTBackbonePresets',
    'MobileSPPBlock',
    'BackboneRepViT',  # alias
    # Ghost modules
    'GhostBottleneck',
    'GhostModule',
    'GhostConv',  # alias
    'SELayer',
    'GhostFusionBlock',
    'GhostNeck',
    'LightGhostNeck',
    'HybridGhostNeck',
    'GhostNeckPresets',
    # Light model
    'RepViTBackbone',
    'YOLOv8Hybrid',
    'GhostHead',
    'LightYOLO',  # alias
]
