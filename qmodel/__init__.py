"""Quantized RepNeXt YOLO model package.

Provides quantization-aware RepNeXt YOLO models with:
- Structural reparameterization for efficient inference
- Quantization support for deployment
- Optional pruning capabilities

Modules:
- repnext: Quantized RepNeXt building blocks
- backbone: Quantized RepNeXt backbone
- neck: Quantized RepNeXt FPN/PANet/BiFPN necks
- head: Quantized detection head with DFL
- yolo: Complete quantized RepNeXt YOLO model
- pruning: Pruning utilities and configuration
"""

from .repnext import (
    LayerNorm,
    QRepNeXtBlock,
    QRepNeXtDownsample,
    QRepNeXtStem,
    QRepNeXtStage,
    QRepNeXtCSPBlock,
    QRepNeXtSPP,
)

from .backbone import QRepNeXtBackbone, QRepNeXtBackboneCSP
from .neck import QRepNeXtNeck, QRepNeXtNeckLite, QRepNeXtBiFPN
from .head import QRepNeXtHead, DFL

from .pruning import (
    PrunableMixin,
    PruningConfig,
)

from .yolo import (
    QYOLORepNeXt,
    QuantizedYOLO,
    VARIANTS,
    create_qyolo_repnext,
    qyolo_repnext_n,
    qyolo_repnext_t,
    qyolo_repnext_s,
    qyolo_repnext_m,
    qyolo_repnext_l,
    qyolo_repnext_x,
    qyolo_repnext_n_lite,
    qyolo_repnext_s_lite,
    qyolo_repnext_s_bifpn,
)


__all__ = [
    # RepNeXt blocks
    'LayerNorm',
    'QRepNeXtBlock',
    'QRepNeXtDownsample',
    'QRepNeXtStem',
    'QRepNeXtStage',
    'QRepNeXtCSPBlock',
    'QRepNeXtSPP',
    # Network components
    'QRepNeXtBackbone',
    'QRepNeXtBackboneCSP',
    'QRepNeXtNeck',
    'QRepNeXtNeckLite',
    'QRepNeXtBiFPN',
    'QRepNeXtHead',
    'DFL',
    # Pruning
    'PrunableMixin',
    'PruningConfig',
    # Models
    'QYOLORepNeXt',
    'QuantizedYOLO',
    'VARIANTS',
    'create_qyolo_repnext',
    # Factory functions
    'qyolo_repnext_n',
    'qyolo_repnext_t',
    'qyolo_repnext_s',
    'qyolo_repnext_m',
    'qyolo_repnext_l',
    'qyolo_repnext_x',
    'qyolo_repnext_n_lite',
    'qyolo_repnext_s_lite',
    'qyolo_repnext_s_bifpn',
]
