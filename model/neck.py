"""RepNeXt Neck for YOLO (FPN/PANet with RepNeXt blocks).

Based on RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization
Paper: https://arxiv.org/abs/2406.16004

Features:
- Feature Pyramid Network (FPN) for top-down pathway
- Path Aggregation Network (PANet) for bottom-up pathway
- RepNeXt blocks for efficient multi-scale feature fusion
"""

import torch
import torch.nn as nn
from typing import Tuple, List

from .repnext import (
    RepNeXtBlock,
    RepNeXtCSPBlock,
    reparameterize_model
)


class Neck(nn.Module):
    """RepNeXt-based Feature Pyramid Network with Path Aggregation.

    Implements FPN + PANet using RepNeXt blocks for efficient
    multi-scale feature fusion.

    Args:
        width: List of channel dimensions [in_ch, stem, s2, s3, s4, s5]
        depth: List of block depths [d1, d2, d3]
        mlp_ratio: MLP expansion ratio in RepNeXt blocks
        deploy: If True, use fused single-branch architecture
    """

    def __init__(self, width: List[int], depth: List[int],
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.deploy = deploy

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down pathway (FPN)
        self.top_to_mid = RepNeXtCSPBlock(
            width[4] + width[5], width[4], depth[0], mlp_ratio, deploy
        )
        self.mid_to_small = RepNeXtCSPBlock(
            width[3] + width[4], width[3], depth[0], mlp_ratio, deploy
        )

        # Bottom-up pathway (PANet)
        self.downsample_mid = nn.Sequential(
            nn.Conv2d(width[3], width[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[3]),
            nn.GELU()
        )
        self.mid_fuse = RepNeXtCSPBlock(
            width[3] + width[4], width[4], depth[0], mlp_ratio, deploy
        )

        self.downsample_top = nn.Sequential(
            nn.Conv2d(width[4], width[4], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[4]),
            nn.GELU()
        )
        self.top_fuse = RepNeXtCSPBlock(
            width[4] + width[5], width[5], depth[0], mlp_ratio, deploy
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: Tuple[torch.Tensor, ...]
                ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through FPN + PANet."""
        p3, p4, p5 = feats

        # Top-down pathway (FPN)
        p4_td = self.top_to_mid(
            torch.cat([self.upsample(p5), p4], dim=1))
        p3_td = self.mid_to_small(
            torch.cat([self.upsample(p4_td), p3], dim=1))

        # Bottom-up pathway (PANet)
        p4_out = self.mid_fuse(
            torch.cat([self.downsample_mid(p3_td), p4_td], dim=1))
        p5_out = self.top_fuse(
            torch.cat([self.downsample_top(p4_out), p5], dim=1))

        return p3_td, p4_out, p5_out

    def reparameterize(self):
        reparameterize_model(self)
        self.deploy = True
        return self


class NeckLite(nn.Module):
    """Lightweight RepNeXt neck for mobile/edge deployment."""

    def __init__(self, width: List[int], depth: List[int],
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down
        self.top_to_mid = self._fusion_block(
            width[4] + width[5], width[4], depth[0], mlp_ratio, deploy)
        self.mid_to_small = self._fusion_block(
            width[3] + width[4], width[3], depth[0], mlp_ratio, deploy)

        # Bottom-up
        self.downsample_mid = nn.Sequential(
            nn.Conv2d(width[3], width[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[3]),
            nn.GELU()
        )
        self.mid_fuse = self._fusion_block(
            width[3] + width[4], width[4], depth[0], mlp_ratio, deploy)

        self.downsample_top = nn.Sequential(
            nn.Conv2d(width[4], width[4], 3, 2, 1, bias=False),
            nn.BatchNorm2d(width[4]),
            nn.GELU()
        )
        self.top_fuse = self._fusion_block(
            width[4] + width[5], width[5], depth[0], mlp_ratio, deploy)

    def _fusion_block(self, in_ch, out_ch, n, mlp_ratio, deploy):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            *[RepNeXtBlock(out_ch, mlp_ratio=mlp_ratio, deploy=deploy)
              for _ in range(n)]
        )

    def forward(self, feats):
        p3, p4, p5 = feats
        p4_td = self.top_to_mid(torch.cat([self.upsample(p5), p4], dim=1))
        p3_td = self.mid_to_small(torch.cat([self.upsample(p4_td), p3], dim=1))
        p4_out = self.mid_fuse(
            torch.cat([self.downsample_mid(p3_td), p4_td], dim=1))
        p5_out = self.top_fuse(
            torch.cat([self.downsample_top(p4_out), p5], dim=1))
        return p3_td, p4_out, p5_out

    def reparameterize(self):
        reparameterize_model(self)
        self.deploy = True
        return self


class NeckBiFPN(nn.Module):
    """RepNeXt-based Bi-directional Feature Pyramid Network."""

    def __init__(self, width: List[int], depth: List[int],
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.epsilon = 1e-4

        # Learnable weights
        self.w_p4_td = nn.Parameter(torch.ones(2))
        self.w_p3_td = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(3))

        # Processing blocks
        self.p5_td_block = RepNeXtBlock(width[5], mlp_ratio=mlp_ratio,
                                        deploy=deploy)
        self.p4_td_block = RepNeXtBlock(width[4], mlp_ratio=mlp_ratio,
                                        deploy=deploy)
        self.p3_td_block = RepNeXtBlock(width[3], mlp_ratio=mlp_ratio,
                                        deploy=deploy)
        self.p4_out_block = RepNeXtBlock(width[4], mlp_ratio=mlp_ratio,
                                         deploy=deploy)
        self.p5_out_block = RepNeXtBlock(width[5], mlp_ratio=mlp_ratio,
                                         deploy=deploy)

        # Channel adjustment
        self.p5_to_p4 = nn.Conv2d(width[5], width[4], 1, bias=False)
        self.p4_to_p3 = nn.Conv2d(width[4], width[3], 1, bias=False)
        self.p3_to_p4 = nn.Conv2d(width[3], width[4], 1, bias=False)
        self.p4_to_p5 = nn.Conv2d(width[4], width[5], 1, bias=False)

        self.downsample_p3 = nn.MaxPool2d(3, 2, 1)
        self.downsample_p4 = nn.MaxPool2d(3, 2, 1)

    def forward(self, feats):
        p3, p4, p5 = feats

        w_p4_td = self.w_p4_td.relu() / (self.w_p4_td.relu().sum() +
                                          self.epsilon)
        w_p3_td = self.w_p3_td.relu() / (self.w_p3_td.relu().sum() +
                                          self.epsilon)
        w_p4_out = self.w_p4_out.relu() / (self.w_p4_out.relu().sum() +
                                            self.epsilon)
        w_p5_out = self.w_p5_out.relu() / (self.w_p5_out.relu().sum() +
                                            self.epsilon)

        # Top-down
        p4_td = self.p4_td_block(
            w_p4_td[0] * p4 + w_p4_td[1] * self.p5_to_p4(self.upsample(p5)))
        p3_td = self.p3_td_block(
            w_p3_td[0] * p3 + w_p3_td[1] * self.p4_to_p3(self.upsample(p4_td)))

        # Bottom-up
        p4_out = self.p4_out_block(
            w_p4_out[0] * p4 + w_p4_out[1] * p4_td +
            w_p4_out[2] * self.p3_to_p4(self.downsample_p3(p3_td)))
        p5_out = self.p5_out_block(
            w_p5_out[0] * p5 + w_p5_out[1] * self.p5_td_block(p5) +
            w_p5_out[2] * self.p4_to_p5(self.downsample_p4(p4_out)))

        return p3_td, p4_out, p5_out

    def reparameterize(self):
        reparameterize_model(self)
        self.deploy = True
        return self

