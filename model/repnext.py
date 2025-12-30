"""RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization.

Based on paper: https://arxiv.org/abs/2406.16004
GitHub: https://github.com/suous/RepNeXt

Key features:
- Multi-branch reparameterization for faster convergence
- Large kernel depthwise convolutions (7x7)
- LayerNorm with channels_first format
- GELU activation in channel mixer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_first (NCHW) or channels_last (NHWC) format."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, 
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class RepNeXtBlock(nn.Module):
    """RepNeXt Block with structural reparameterization.
    
    During training: Uses multi-branch design with identity, 1x1, 3x3, 5x5, and 7x7 branches
    During inference: Fuses into single 7x7 depthwise convolution
    
    Architecture:
        Token Mixer: Multi-branch depthwise conv (reparameterizable to 7x7)
        Channel Mixer: MLP with GELU activation
        Residual: identity + block output
    """
    
    def __init__(self, dim: int, kernel_size: int = 7, mlp_ratio: float = 2.0,
                 deploy: bool = False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio
        self.deploy = deploy
        
        # Padding for each kernel size to maintain spatial dimensions
        self.padding = kernel_size // 2
        
        if deploy:
            # Single fused branch for inference
            self.token_mixer = nn.Conv2d(dim, dim, kernel_size, 1, self.padding, 
                                         groups=dim, bias=True)
        else:
            # Multi-branch design for training
            # Main branch: 7x7 depthwise conv
            self.dwconv_7x7 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False)
            self.bn_7x7 = nn.BatchNorm2d(dim)
            
            # 5x5 branch
            self.dwconv_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
            self.bn_5x5 = nn.BatchNorm2d(dim)
            
            # 3x3 branch
            self.dwconv_3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(dim)
            
            # 1x1 branch (equivalent to channel-wise scaling)
            self.dwconv_1x1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(dim)
            
            # Identity branch (BN only)
            self.bn_identity = nn.BatchNorm2d(dim)
        
        # Normalization after token mixer
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # Channel mixer (MLP)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Token mixer
        if self.deploy:
            x = self.token_mixer(x)
        else:
            # Multi-branch during training
            x = (self.bn_7x7(self.dwconv_7x7(x)) + 
                 self.bn_5x5(self.dwconv_5x5(x)) + 
                 self.bn_3x3(self.dwconv_3x3(x)) + 
                 self.bn_1x1(self.dwconv_1x1(x)) + 
                 self.bn_identity(x))
        
        x = self.norm(x)
        
        # Channel mixer with residual on channel mixer only
        input_channel_mixer = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        return identity + input_channel_mixer + x
    
    def reparameterize(self) -> None:
        """Fuse multi-branch structure into single convolution for inference."""
        if self.deploy:
            return
            
        # Initialize fused kernel and bias
        kernel_size = 7  # Largest kernel
        fused_kernel = torch.zeros(self.dim, 1, kernel_size, kernel_size, 
                                   device=self.dwconv_7x7.weight.device)
        fused_bias = torch.zeros(self.dim, device=self.dwconv_7x7.weight.device)
        
        # Fuse 7x7 branch
        k7, b7 = self._fuse_conv_bn(self.dwconv_7x7, self.bn_7x7)
        fused_kernel += k7
        fused_bias += b7
        
        # Fuse 5x5 branch (pad to 7x7)
        k5, b5 = self._fuse_conv_bn(self.dwconv_5x5, self.bn_5x5)
        fused_kernel += F.pad(k5, [1, 1, 1, 1])
        fused_bias += b5
        
        # Fuse 3x3 branch (pad to 7x7)
        k3, b3 = self._fuse_conv_bn(self.dwconv_3x3, self.bn_3x3)
        fused_kernel += F.pad(k3, [2, 2, 2, 2])
        fused_bias += b3
        
        # Fuse 1x1 branch (pad to 7x7)
        k1, b1 = self._fuse_conv_bn(self.dwconv_1x1, self.bn_1x1)
        fused_kernel += F.pad(k1, [3, 3, 3, 3])
        fused_bias += b1
        
        # Fuse identity branch (create identity kernel and fuse with BN)
        identity_kernel = torch.zeros(self.dim, 1, kernel_size, kernel_size,
                                      device=self.dwconv_7x7.weight.device)
        identity_kernel[:, 0, kernel_size // 2, kernel_size // 2] = 1.0
        ki, bi = self._fuse_identity_bn(self.bn_identity, identity_kernel)
        fused_kernel += ki
        fused_bias += bi
        
        # Create fused conv layer
        self.token_mixer = nn.Conv2d(self.dim, self.dim, kernel_size, 1, 
                                     kernel_size // 2, groups=self.dim, bias=True)
        self.token_mixer.weight.data = fused_kernel
        self.token_mixer.bias.data = fused_bias
        
        # Remove training branches
        for attr in ['dwconv_7x7', 'bn_7x7', 'dwconv_5x5', 'bn_5x5', 
                     'dwconv_3x3', 'bn_3x3', 'dwconv_1x1', 'bn_1x1', 'bn_identity']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.deploy = True
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv and BatchNorm layers."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias
    
    def _fuse_identity_bn(self, bn: nn.BatchNorm2d, 
                          identity_kernel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse identity with BatchNorm."""
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = identity_kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias


class RepNeXtDownsample(nn.Module):
    """RepNeXt Downsample layer with stride 2.
    
    Doubles channels and halves spatial dimensions.
    Shortcut bypasses only the channel mixer for BN fusion capability.
    """
    
    def __init__(self, dim: int, mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        self.dim = dim
        out_dim = dim * 2
        self.out_dim = out_dim
        self.deploy = deploy
        
        if deploy:
            # Single fused branch for inference
            self.dwconv = nn.Conv2d(dim, out_dim, 7, 2, 3, groups=dim, bias=True)
        else:
            # Multi-branch design for training
            # 7x7 stride-2 depthwise conv (main branch)
            self.dwconv_7x7 = nn.Conv2d(dim, out_dim, 7, 2, 3, groups=dim, bias=False)
            self.bn_7x7 = nn.BatchNorm2d(out_dim)
            
            # 5x5 branch
            self.dwconv_5x5 = nn.Conv2d(dim, out_dim, 5, 2, 2, groups=dim, bias=False)
            self.bn_5x5 = nn.BatchNorm2d(out_dim)
            
            # 3x3 branch  
            self.dwconv_3x3 = nn.Conv2d(dim, out_dim, 3, 2, 1, groups=dim, bias=False)
            self.bn_3x3 = nn.BatchNorm2d(out_dim)
        
        self.norm = LayerNorm(out_dim, eps=1e-6, data_format="channels_first")
        
        # Channel mixer
        hidden_dim = int(out_dim * mlp_ratio)
        self.pwconv1 = nn.Linear(out_dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixer with downsampling
        if self.deploy:
            x = self.dwconv(x)
        else:
            x = (self.bn_7x7(self.dwconv_7x7(x)) + 
                 self.bn_5x5(self.dwconv_5x5(x)) + 
                 self.bn_3x3(self.dwconv_3x3(x)))
        
        x = self.norm(x)
        identity = x  # Shortcut bypasses channel mixer only
        
        # Channel mixer
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        return identity + x
    
    def reparameterize(self) -> None:
        """Fuse multi-branch structure into single convolution."""
        if self.deploy:
            return
            
        kernel_size = 7
        fused_kernel = torch.zeros(self.out_dim, 1, kernel_size, kernel_size,
                                   device=self.dwconv_7x7.weight.device)
        fused_bias = torch.zeros(self.out_dim, device=self.dwconv_7x7.weight.device)
        
        # Fuse 7x7 branch
        k7, b7 = self._fuse_conv_bn(self.dwconv_7x7, self.bn_7x7)
        fused_kernel += k7
        fused_bias += b7
        
        # Fuse 5x5 branch (pad to 7x7)
        k5, b5 = self._fuse_conv_bn(self.dwconv_5x5, self.bn_5x5)
        fused_kernel += F.pad(k5, [1, 1, 1, 1])
        fused_bias += b5
        
        # Fuse 3x3 branch (pad to 7x7)
        k3, b3 = self._fuse_conv_bn(self.dwconv_3x3, self.bn_3x3)
        fused_kernel += F.pad(k3, [2, 2, 2, 2])
        fused_bias += b3
        
        # Create fused conv
        self.dwconv = nn.Conv2d(self.dim, self.out_dim, kernel_size, 2,
                                kernel_size // 2, groups=self.dim, bias=True)
        self.dwconv.weight.data = fused_kernel
        self.dwconv.bias.data = fused_bias
        
        # Remove training branches
        for attr in ['dwconv_7x7', 'bn_7x7', 'dwconv_5x5', 'bn_5x5', 
                     'dwconv_3x3', 'bn_3x3']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.deploy = True
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv and BatchNorm layers."""
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias


class RepNeXtStem(nn.Module):
    """RepNeXt Stem: Initial feature extraction with overlapping patch embedding."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 48):
        super().__init__()
        # Overlapping patch embedding (similar to ConvNeXt)
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


class RepNeXtStage(nn.Module):
    """RepNeXt Stage: Sequence of RepNeXt blocks with optional downsampling."""
    
    def __init__(self, in_dim: int, out_dim: int, depth: int, 
                 downsample: bool = True, mlp_ratio: float = 2.0,
                 deploy: bool = False):
        super().__init__()
        
        layers = []
        
        # Downsample at the beginning of each stage (except first)
        if downsample and in_dim != out_dim:
            layers.append(RepNeXtDownsample(in_dim, mlp_ratio, deploy))
            in_dim = out_dim
        
        # Stack RepNeXt blocks
        for _ in range(depth):
            layers.append(RepNeXtBlock(in_dim, kernel_size=7, mlp_ratio=mlp_ratio, 
                                       deploy=deploy))
        
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
    
    def reparameterize(self) -> None:
        """Reparameterize all blocks in the stage."""
        for block in self.blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class RepNeXtCSPBlock(nn.Module):
    """CSP-style block using RepNeXt blocks.
    
    Combines CSP (Cross Stage Partial) design with RepNeXt for 
    better gradient flow and feature reuse.
    """
    
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, 
                 mlp_ratio: float = 2.0, deploy: bool = False):
        super().__init__()
        mid_ch = out_ch // 2
        
        # Split convolutions
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
        
        # RepNeXt blocks
        self.repnext_blocks = nn.ModuleList([
            RepNeXtBlock(mid_ch, kernel_size=7, mlp_ratio=mlp_ratio, deploy=deploy)
            for _ in range(n)
        ])
        
        # Final fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d((2 + n) * mid_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.split_conv(x)
        right = self.path_conv(x)
        out_list = [left, right]
        
        for block in self.repnext_blocks:
            out_list.append(block(out_list[-1]))
        
        return self.final_conv(torch.cat(out_list, dim=1))
    
    def reparameterize(self) -> None:
        """Reparameterize all RepNeXt blocks."""
        for block in self.repnext_blocks:
            if hasattr(block, 'reparameterize'):
                block.reparameterize()


class RepNeXtSPP(nn.Module):
    """Spatial Pyramid Pooling with RepNeXt-style design."""
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        pooled = [x] + [pool(x) for pool in self.pools]
        return self.fuse(torch.cat(pooled, dim=1))


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Recursively reparameterize all RepNeXt blocks in the model."""
    for module in model.modules():
        # Skip the model itself to avoid infinite recursion
        if module is model:
            continue
        if hasattr(module, 'reparameterize') and callable(module.reparameterize):
            # Only call reparameterize on leaf-level RepNeXt blocks
            if isinstance(module, (RepNeXtBlock, RepNeXtDownsample)):
                module.reparameterize()
    return model

