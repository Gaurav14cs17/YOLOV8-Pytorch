"""YOLOv8 with RepNeXt backbone and neck.

Based on RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization
Paper: https://arxiv.org/abs/2406.16004

Features:
- RepNeXt backbone with multi-branch reparameterization
- RepNeXt neck (FPN/PANet) for multi-scale feature fusion
- Efficient inference through structural reparameterization
"""

import torch
import torch.nn as nn
from typing import List

from .backbone import Backbone, BackbonePlain
from .neck import Neck, NeckLite, NeckBiFPN
from .head import Head
from .fuse_layer import FuseLayer


class YOLO(nn.Module):
    """YOLOv8 with RepNeXt backbone and neck.

    Combines RepNeXt's structural reparameterization with YOLOv8's
    detection head for efficient object detection.

    Args:
        width: List of channel dimensions [in_ch, stem, s2, s3, s4, s5]
        depth: List of block depths [d1, d2, d3]
        num_classes: Number of detection classes
        backbone_type: 'csp' or 'plain' backbone variant
        neck_type: 'standard', 'lite', or 'bifpn' neck variant
        mlp_ratio: MLP expansion ratio in RepNeXt blocks
        deploy: If True, use fused single-branch architecture
    """

    def __init__(self, width: List[int], depth: List[int], num_classes: int,
                 backbone_type: str = 'csp', neck_type: str = 'standard',
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()

        # Select backbone
        if backbone_type == 'plain':
            self.backbone = BackbonePlain(width, depth, mlp_ratio, deploy)
        else:
            self.backbone = Backbone(width, depth, mlp_ratio, deploy)

        # Select neck
        if neck_type == 'lite':
            self.neck = NeckLite(width, depth, mlp_ratio, deploy)
        elif neck_type == 'bifpn':
            self.neck = NeckBiFPN(width, depth, mlp_ratio, deploy)
        else:
            self.neck = Neck(width, depth, mlp_ratio, deploy)

        # Detection head
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        # Model attributes
        self.nc = num_classes
        self.no = num_classes + self.head.dfl_channels * 4
        self.deploy = deploy

        # Initialize stride
        self._init_stride()

    def _init_stride(self):
        """Initialize detection head strides."""
        img_dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(img_dummy)
            feats = self.neck(feats)
        self.head.stride = torch.tensor([256 / f.shape[-2] for f in feats])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(list(feats))

    def fuse(self):
        """Fuse Conv+BN layers for inference."""
        FuseLayer.fuse_module(self)
        return self

    def reparameterize(self):
        """Convert RepNeXt blocks to inference mode.

        Fuses multi-branch structure into single large-kernel convolutions.
        Call this before inference for better speed.
        """
        self.backbone.reparameterize()
        self.neck.reparameterize()
        self.deploy = True
        return self


# ========================================
# Factory functions for different variants
# ========================================

def yolo_v8_n(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Nano variant (~2.5M params)."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes, deploy=deploy)


def yolo_v8_s(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Small variant (~8M params)."""
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes, deploy=deploy)


def yolo_v8_m(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Medium variant (~17M params)."""
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes, deploy=deploy)


def yolo_v8_l(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Large variant (~40M params)."""
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes, deploy=deploy)


def yolo_v8_x(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt XLarge variant (~70M params)."""
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes, deploy=deploy)


# Lite variants with lighter neck
def yolo_v8_n_lite(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Nano Lite - for edge devices."""
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes, neck_type='lite', deploy=deploy)


def yolo_v8_s_lite(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Small Lite - fast model for mobile."""
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes, neck_type='lite', deploy=deploy)


# BiFPN variants for better feature fusion
def yolo_v8_s_bifpn(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Small with BiFPN neck."""
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes, neck_type='bifpn', deploy=deploy)


def yolo_v8_m_bifpn(num_classes: int = 80, deploy: bool = False):
    """YOLOv8-RepNeXt Medium with BiFPN neck."""
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes, neck_type='bifpn', deploy=deploy)

