"""YOLOv8-RepNeXt Model Package.

Based on RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization
Paper: https://arxiv.org/abs/2406.16004
"""

from .yolo import (
    YOLO,
    yolo_v8_n,
    yolo_v8_s,
    yolo_v8_m,
    yolo_v8_l,
    yolo_v8_x,
    yolo_v8_n_lite,
    yolo_v8_s_lite,
    yolo_v8_s_bifpn,
    yolo_v8_m_bifpn,
)

from .backbone import Backbone, BackbonePlain
from .neck import Neck, NeckLite, NeckBiFPN
from .head import Head

__all__ = [
    'YOLO',
    'yolo_v8_n',
    'yolo_v8_s',
    'yolo_v8_m',
    'yolo_v8_l',
    'yolo_v8_x',
    'yolo_v8_n_lite',
    'yolo_v8_s_lite',
    'yolo_v8_s_bifpn',
    'yolo_v8_m_bifpn',
    'Backbone',
    'BackbonePlain',
    'Neck',
    'NeckLite',
    'NeckBiFPN',
    'Head',
]
