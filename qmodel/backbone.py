"""Quantized RepNeXt backbone.

Multi-scale feature extraction using RepNeXt blocks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .repnext import (
    QRepNeXtStem, QRepNeXtStage, QRepNeXtCSPBlock, QRepNeXtSPP
)


class QRepNeXtBackbone(nn.Module):
    """Quantized RepNeXt Backbone for YOLOv8.

    Extracts P3, P4, P5 features at different scales.
    """

    def __init__(self, dims: List[int] = None, depths: List[int] = None,
                 mlp_ratio: float = 2.0, deploy: bool = False):
        """Initialize backbone.

        Args:
            dims: Channel dimensions for each stage [stem, s1, s2, s3, s4]
            depths: Number of blocks per stage [s1, s2, s3, s4]
            mlp_ratio: MLP expansion ratio in RepNeXt blocks
            deploy: Whether to use inference mode (reparameterized)
        """
        super().__init__()

        if dims is None:
            dims = [48, 96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 6, 2]

        # Stem: 4x downsample
        self.stem = QRepNeXtStem(3, dims[0])

        # Stage 1: dims[0] -> dims[1], 2x downsample (P2, stride 8)
        self.stage1 = QRepNeXtStage(dims[0], dims[1], depths[0],
                                    downsample=True, mlp_ratio=mlp_ratio, deploy=deploy)

        # Stage 2: dims[1] -> dims[2], 2x downsample (P3, stride 16)
        self.stage2 = QRepNeXtStage(dims[1], dims[2], depths[1],
                                    downsample=True, mlp_ratio=mlp_ratio, deploy=deploy)

        # Stage 3: dims[2] -> dims[3], 2x downsample (P4, stride 32)
        self.stage3 = QRepNeXtStage(dims[2], dims[3], depths[2],
                                    downsample=True, mlp_ratio=mlp_ratio, deploy=deploy)

        # Stage 4: dims[3] -> dims[4], 2x downsample (P5, stride 64)
        self.stage4 = QRepNeXtStage(dims[3], dims[4], depths[3],
                                    downsample=True, mlp_ratio=mlp_ratio, deploy=deploy)

        # SPP at the end
        self.spp = QRepNeXtSPP(dims[4], dims[4])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Tuple of (P3, P4, P5) feature maps
        """
        x = self.stem(x)       # stride 4
        x = self.stage1(x)     # stride 8, P2
        p3 = self.stage2(x)    # stride 16, P3
        p4 = self.stage3(p3)   # stride 32, P4
        p5 = self.stage4(p4)   # stride 64, P5
        p5 = self.spp(p5)

        return p3, p4, p5

    def reparameterize(self) -> None:
        """Reparameterize all stages for inference."""
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            if hasattr(stage, 'reparameterize'):
                stage.reparameterize()


class QRepNeXtBackboneCSP(nn.Module):
    """Quantized RepNeXt Backbone with CSP blocks.

    YOLOv8-style backbone using RepNeXt CSP blocks.
    """

    def __init__(self, dims: List[int] = None, depths: List[int] = None,
                 mlp_ratio: float = 2.0, deploy: bool = False):
        """Initialize CSP backbone.

        Args:
            dims: Channel dimensions [input, p1, p2, p3, p4, p5]
            depths: Number of blocks per stage [p2, p3, p4]
            mlp_ratio: MLP expansion ratio
            deploy: Inference mode flag
        """
        super().__init__()

        if dims is None:
            dims = [3, 32, 64, 128, 256, 512]
        if depths is None:
            depths = [1, 2, 2]

        # P1: stride 2
        self.p1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )

        # P2: stride 4
        self.p2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[2]),
            nn.GELU(),
            QRepNeXtCSPBlock(dims[2], dims[2], depths[0], mlp_ratio, deploy)
        )

        # P3: stride 8
        self.p3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.GELU(),
            QRepNeXtCSPBlock(dims[3], dims[3], depths[1], mlp_ratio, deploy)
        )

        # P4: stride 16
        self.p4 = nn.Sequential(
            nn.Conv2d(dims[3], dims[4], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[4]),
            nn.GELU(),
            QRepNeXtCSPBlock(dims[4], dims[4], depths[2], mlp_ratio, deploy)
        )

        # P5: stride 32
        self.p5 = nn.Sequential(
            nn.Conv2d(dims[4], dims[5], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[5]),
            nn.GELU(),
            QRepNeXtCSPBlock(dims[5], dims[5], depths[0], mlp_ratio, deploy),
            QRepNeXtSPP(dims[5], dims[5])
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale features."""
        x = self.p1(x)
        x = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

    def reparameterize(self) -> None:
        """Reparameterize all blocks for inference."""
        for stage in [self.p2, self.p3, self.p4, self.p5]:
            for m in stage.modules():
                if hasattr(m, 'reparameterize'):
                    m.reparameterize()
