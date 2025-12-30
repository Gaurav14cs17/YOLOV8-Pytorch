"""Quantized RepNeXt blocks with structural reparameterization.

Based on RepNeXt paper with quantization-aware training support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as nnq
from typing import Tuple


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first format for quantization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:  # channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class QRepNeXtBlock(nn.Module):
    """Quantized RepNeXt Block with structural reparameterization.

    Multi-branch design during training, fuses to single 7x7 depthwise conv.
    """

    def __init__(self, dim: int, kernel_size: int = 7, mlp_ratio: float = 2.0,
                 deploy: bool = False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio
        self.deploy = deploy
        self.padding = kernel_size // 2

        # Quantization ops
        self.quant_add1 = nnq.FloatFunctional()
        self.quant_add2 = nnq.FloatFunctional()

        if deploy:
            self.token_mixer = nn.Conv2d(dim, dim, kernel_size, 1, self.padding,
                                         groups=dim, bias=True)
        else:
            # Multi-branch for training
            self.dwconv_7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
            self.bn_7x7 = nn.BatchNorm2d(dim)

            self.dwconv_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
            self.bn_5x5 = nn.BatchNorm2d(dim)

            self.dwconv_3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(dim)

            self.dwconv_1x1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(dim)

            self.bn_identity = nn.BatchNorm2d(dim)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # Channel mixer
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.deploy:
            x = self.token_mixer(x)
        else:
            x = (self.bn_7x7(self.dwconv_7x7(x)) +
                 self.bn_5x5(self.dwconv_5x5(x)) +
                 self.bn_3x3(self.dwconv_3x3(x)) +
                 self.bn_1x1(self.dwconv_1x1(x)) +
                 self.bn_identity(x))

        x = self.norm(x)
        input_channel_mixer = x

        # Channel mixer
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        # Residual connections with quantized add
        x = self.quant_add1.add(input_channel_mixer, x)
        x = self.quant_add2.add(identity, x)
        return x

    def reparameterize(self) -> None:
        """Fuse multi-branch structure into single convolution."""
        if self.deploy:
            return

        kernel_size = 7
        fused_kernel = torch.zeros(self.dim, 1, kernel_size, kernel_size,
                                   device=self.dwconv_7x7.weight.device)
        fused_bias = torch.zeros(self.dim, device=self.dwconv_7x7.weight.device)

        k7, b7 = self._fuse_conv_bn(self.dwconv_7x7, self.bn_7x7)
        fused_kernel += k7
        fused_bias += b7

        k5, b5 = self._fuse_conv_bn(self.dwconv_5x5, self.bn_5x5)
        fused_kernel += F.pad(k5, [1, 1, 1, 1])
        fused_bias += b5

        k3, b3 = self._fuse_conv_bn(self.dwconv_3x3, self.bn_3x3)
        fused_kernel += F.pad(k3, [2, 2, 2, 2])
        fused_bias += b3

        k1, b1 = self._fuse_conv_bn(self.dwconv_1x1, self.bn_1x1)
        fused_kernel += F.pad(k1, [3, 3, 3, 3])
        fused_bias += b1

        identity_kernel = torch.zeros(self.dim, 1, kernel_size, kernel_size,
                                      device=self.dwconv_7x7.weight.device)
        identity_kernel[:, 0, kernel_size // 2, kernel_size // 2] = 1.0
        ki, bi = self._fuse_identity_bn(self.bn_identity, identity_kernel)
        fused_kernel += ki
        fused_bias += bi

        self.token_mixer = nn.Conv2d(self.dim, self.dim, kernel_size, 1,
                                     kernel_size // 2, groups=self.dim, bias=True)
        self.token_mixer.weight.data = fused_kernel
        self.token_mixer.bias.data = fused_bias

        for attr in ['dwconv_7x7', 'bn_7x7', 'dwconv_5x5', 'bn_5x5',
                     'dwconv_3x3', 'bn_3x3', 'dwconv_1x1', 'bn_1x1', 'bn_identity']:
            if hasattr(self, attr):
                delattr(self, attr)

        self.deploy = True

    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d,
                          identity_kernel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return identity_kernel * t, bn.bias - bn.running_mean * bn.weight / std


class QRepNeXtDownsample(nn.Module):
    """Quantized RepNeXt Downsample layer."""

    def __init__(self, dim: int, mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.dim = dim
        self.out_dim = dim * 2
        self.deploy = deploy

        self.quant_add = nnq.FloatFunctional()

        if deploy:
            self.dwconv = nn.Conv2d(dim, self.out_dim, 7, 2, 3, groups=dim, bias=True)
        else:
            self.dwconv_7x7 = nn.Conv2d(dim, self.out_dim, 7, 2, 3, groups=dim, bias=False)
            self.bn_7x7 = nn.BatchNorm2d(self.out_dim)

            self.dwconv_5x5 = nn.Conv2d(dim, self.out_dim, 5, 2, 2, groups=dim, bias=False)
            self.bn_5x5 = nn.BatchNorm2d(self.out_dim)

            self.dwconv_3x3 = nn.Conv2d(dim, self.out_dim, 3, 2, 1, groups=dim, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(self.out_dim)

        self.norm = LayerNorm(self.out_dim, eps=1e-6, data_format="channels_first")

        hidden_dim = int(self.out_dim * mlp_ratio)
        self.pwconv1 = nn.Linear(self.out_dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            x = self.dwconv(x)
        else:
            x = (self.bn_7x7(self.dwconv_7x7(x)) +
                 self.bn_5x5(self.dwconv_5x5(x)) +
                 self.bn_3x3(self.dwconv_3x3(x)))

        x = self.norm(x)
        identity = x

        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return self.quant_add.add(identity, x)

    def reparameterize(self) -> None:
        if self.deploy:
            return

        kernel_size = 7
        fused_kernel = torch.zeros(self.out_dim, 1, kernel_size, kernel_size,
                                   device=self.dwconv_7x7.weight.device)
        fused_bias = torch.zeros(self.out_dim, device=self.dwconv_7x7.weight.device)

        k7, b7 = self._fuse_conv_bn(self.dwconv_7x7, self.bn_7x7)
        fused_kernel += k7
        fused_bias += b7

        k5, b5 = self._fuse_conv_bn(self.dwconv_5x5, self.bn_5x5)
        fused_kernel += F.pad(k5, [1, 1, 1, 1])
        fused_bias += b5

        k3, b3 = self._fuse_conv_bn(self.dwconv_3x3, self.bn_3x3)
        fused_kernel += F.pad(k3, [2, 2, 2, 2])
        fused_bias += b3

        self.dwconv = nn.Conv2d(self.dim, self.out_dim, kernel_size, 2,
                                kernel_size // 2, groups=self.dim, bias=True)
        self.dwconv.weight.data = fused_kernel
        self.dwconv.bias.data = fused_bias

        for attr in ['dwconv_7x7', 'bn_7x7', 'dwconv_5x5', 'bn_5x5', 'dwconv_3x3', 'bn_3x3']:
            if hasattr(self, attr):
                delattr(self, attr)

        self.deploy = True

    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std


class QRepNeXtStem(nn.Module):
    """Quantized RepNeXt Stem."""

    def __init__(self, in_channels: int = 3, out_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class QRepNeXtStage(nn.Module):
    """Quantized RepNeXt Stage."""

    def __init__(self, in_dim: int, out_dim: int, depth: int,
                 downsample: bool = True, mlp_ratio: float = 2.0,
                 deploy: bool = False):
        super().__init__()

        layers = []
        if downsample and in_dim != out_dim:
            layers.append(QRepNeXtDownsample(in_dim, mlp_ratio, deploy))
            in_dim = out_dim

        for _ in range(depth):
            layers.append(QRepNeXtBlock(in_dim, kernel_size=7, mlp_ratio=mlp_ratio,
                                        deploy=deploy))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def reparameterize(self) -> None:
        for block in self.blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class QRepNeXtCSPBlock(nn.Module):
    """Quantized CSP block using RepNeXt blocks."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1,
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        mid_ch = out_ch // 2

        self.split_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )
        self.path_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )

        self.repnext_blocks = nn.ModuleList([
            QRepNeXtBlock(mid_ch, kernel_size=7, mlp_ratio=mlp_ratio, deploy=deploy)
            for _ in range(n)
        ])

        self.final_conv = nn.Sequential(
            nn.Conv2d((2 + n) * mid_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

        self.quant_cat = nnq.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.split_conv(x)
        right = self.path_conv(x)
        out_list = [left, right]

        for block in self.repnext_blocks:
            out_list.append(block(out_list[-1]))

        return self.final_conv(self.quant_cat.cat(out_list, dim=1))

    def reparameterize(self) -> None:
        for block in self.repnext_blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class QRepNeXtSPP(nn.Module):
    """Quantized Spatial Pyramid Pooling with RepNeXt style."""

    def __init__(self, in_ch: int, out_ch: int, kernel_sizes: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        mid_ch = in_ch // 2

        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU()
        )

        self.pools = nn.ModuleList([
            nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * (1 + len(kernel_sizes)), out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

        self.quant_cat = nnq.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        pooled = [x] + [pool(x) for pool in self.pools]
        return self.fuse(self.quant_cat.cat(pooled, dim=1))

