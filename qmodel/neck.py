"""Quantized RepNeXt Feature Pyramid Network.

Multi-scale feature fusion with RepNeXt blocks and quantization support.
"""

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from typing import List, Tuple

from .repnext import QRepNeXtCSPBlock, QRepNeXtBlock


class QRepNeXtNeck(nn.Module):
    """Quantized RepNeXt FPN + PANet Neck.

    Top-down and bottom-up feature fusion using RepNeXt blocks.
    """

    def __init__(self, dims: List[int] = None, depth: int = 1,
                 mlp_ratio: float = 2.0, deploy: bool = False):
        """Initialize neck.

        Args:
            dims: Channel dimensions [p3, p4, p5]
            depth: Number of RepNeXt blocks in CSP
            mlp_ratio: MLP expansion ratio
            deploy: Inference mode flag
        """
        super().__init__()

        if dims is None:
            dims = [128, 256, 512]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.quant_cat1 = nnq.FloatFunctional()
        self.quant_cat2 = nnq.FloatFunctional()
        self.quant_cat3 = nnq.FloatFunctional()
        self.quant_cat4 = nnq.FloatFunctional()

        # Top-down pathway
        self.reduce_p5 = nn.Sequential(
            nn.Conv2d(dims[2], dims[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )
        self.td_block1 = QRepNeXtCSPBlock(dims[1] * 2, dims[1], depth, mlp_ratio, deploy)

        self.reduce_p4 = nn.Sequential(
            nn.Conv2d(dims[1], dims[0], 1, 1, 0, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )
        self.td_block2 = QRepNeXtCSPBlock(dims[0] * 2, dims[0], depth, mlp_ratio, deploy)

        # Bottom-up pathway
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )
        self.bu_block1 = QRepNeXtCSPBlock(dims[0] + dims[1], dims[1], depth, mlp_ratio, deploy)

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.GELU()
        )
        self.bu_block2 = QRepNeXtCSPBlock(dims[1] + dims[2], dims[2], depth, mlp_ratio, deploy)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse multi-scale features.

        Args:
            p3: P3 features (stride 8)
            p4: P4 features (stride 16)
            p5: P5 features (stride 32)

        Returns:
            Fused (p3, p4, p5) features
        """
        # Top-down
        p5_up = self.reduce_p5(p5)
        p4 = self.td_block1(self.quant_cat1.cat([self.upsample(p5_up), p4], dim=1))

        p4_up = self.reduce_p4(p4)
        p3 = self.td_block2(self.quant_cat2.cat([self.upsample(p4_up), p3], dim=1))

        # Bottom-up
        p4 = self.bu_block1(self.quant_cat3.cat([self.down_conv1(p3), p4], dim=1))
        p5 = self.bu_block2(self.quant_cat4.cat([self.down_conv2(p4), p5], dim=1))

        return p3, p4, p5

    def reparameterize(self) -> None:
        """Reparameterize all blocks for inference."""
        for block in [self.td_block1, self.td_block2, self.bu_block1, self.bu_block2]:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class QRepNeXtNeckLite(nn.Module):
    """Lightweight quantized RepNeXt neck using single blocks."""

    def __init__(self, dims: List[int] = None, mlp_ratio: float = 2.0,
                 deploy: bool = False):
        """Initialize lightweight neck."""
        super().__init__()

        if dims is None:
            dims = [128, 256, 512]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.quant_cat1 = nnq.FloatFunctional()
        self.quant_cat2 = nnq.FloatFunctional()
        self.quant_cat3 = nnq.FloatFunctional()
        self.quant_cat4 = nnq.FloatFunctional()

        # Top-down
        self.reduce_p5 = nn.Conv2d(dims[2], dims[1], 1)
        self.td_block1 = QRepNeXtBlock(dims[1], mlp_ratio=mlp_ratio, deploy=deploy)

        self.reduce_p4 = nn.Conv2d(dims[1], dims[0], 1)
        self.td_block2 = QRepNeXtBlock(dims[0], mlp_ratio=mlp_ratio, deploy=deploy)

        # Bottom-up
        self.down_conv1 = nn.Conv2d(dims[0], dims[0], 3, 2, 1)
        self.bu_block1 = QRepNeXtBlock(dims[1], mlp_ratio=mlp_ratio, deploy=deploy)

        self.down_conv2 = nn.Conv2d(dims[1], dims[1], 3, 2, 1)
        self.bu_block2 = QRepNeXtBlock(dims[2], mlp_ratio=mlp_ratio, deploy=deploy)

        # Channel adjustment after concat
        self.adjust_p4 = nn.Conv2d(dims[1] * 2, dims[1], 1)
        self.adjust_p3 = nn.Conv2d(dims[0] * 2, dims[0], 1)
        self.adjust_bu_p4 = nn.Conv2d(dims[0] + dims[1], dims[1], 1)
        self.adjust_bu_p5 = nn.Conv2d(dims[1] + dims[2], dims[2], 1)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Top-down
        p5_up = self.reduce_p5(p5)
        p4 = self.td_block1(self.adjust_p4(self.quant_cat1.cat([self.upsample(p5_up), p4], dim=1)))

        p4_up = self.reduce_p4(p4)
        p3 = self.td_block2(self.adjust_p3(self.quant_cat2.cat([self.upsample(p4_up), p3], dim=1)))

        # Bottom-up
        p4 = self.bu_block1(self.adjust_bu_p4(self.quant_cat3.cat([self.down_conv1(p3), p4], dim=1)))
        p5 = self.bu_block2(self.adjust_bu_p5(self.quant_cat4.cat([self.down_conv2(p4), p5], dim=1)))

        return p3, p4, p5

    def reparameterize(self) -> None:
        for block in [self.td_block1, self.td_block2, self.bu_block1, self.bu_block2]:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class QRepNeXtBiFPN(nn.Module):
    """Quantized RepNeXt Bi-directional FPN with weighted fusion."""

    def __init__(self, dims: List[int] = None, depth: int = 1,
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()

        if dims is None:
            dims = [128, 256, 512]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.quant_ops = nnq.FloatFunctional()

        # Learnable weights for fusion
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_td = nn.Parameter(torch.ones(2))
        self.w_p4_bu = nn.Parameter(torch.ones(3))
        self.w_p5_bu = nn.Parameter(torch.ones(2))

        # Lateral connections
        self.lateral_p5 = nn.Conv2d(dims[2], dims[1], 1)
        self.lateral_p4 = nn.Conv2d(dims[1], dims[0], 1)

        # Top-down blocks
        self.td_block1 = QRepNeXtCSPBlock(dims[1], dims[1], depth, mlp_ratio, deploy)
        self.td_block2 = QRepNeXtCSPBlock(dims[0], dims[0], depth, mlp_ratio, deploy)

        # Bottom-up
        self.down_conv1 = nn.Conv2d(dims[0], dims[1], 3, 2, 1)
        self.down_conv2 = nn.Conv2d(dims[1], dims[2], 3, 2, 1)

        self.bu_block1 = QRepNeXtCSPBlock(dims[1], dims[1], depth, mlp_ratio, deploy)
        self.bu_block2 = QRepNeXtCSPBlock(dims[2], dims[2], depth, mlp_ratio, deploy)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor,
                p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Weighted top-down
        w_p4 = torch.softmax(self.w_p4_td, dim=0)
        p4_td = self.td_block1(
            w_p4[0] * self.upsample(self.lateral_p5(p5)) + w_p4[1] * p4
        )

        w_p3 = torch.softmax(self.w_p3_td, dim=0)
        p3_out = self.td_block2(
            w_p3[0] * self.upsample(self.lateral_p4(p4_td)) + w_p3[1] * p3
        )

        # Weighted bottom-up
        w_p4_bu = torch.softmax(self.w_p4_bu, dim=0)
        p4_out = self.bu_block1(
            w_p4_bu[0] * p4 + w_p4_bu[1] * p4_td + w_p4_bu[2] * self.down_conv1(p3_out)
        )

        w_p5_bu = torch.softmax(self.w_p5_bu, dim=0)
        p5_out = self.bu_block2(
            w_p5_bu[0] * p5 + w_p5_bu[1] * self.down_conv2(p4_out)
        )

        return p3_out, p4_out, p5_out

    def reparameterize(self) -> None:
        for block in [self.td_block1, self.td_block2, self.bu_block1, self.bu_block2]:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()
