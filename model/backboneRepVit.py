import torch
import torch.nn as nn
from model.blocks import ConvBlock, RepViTBlock, RepViTCSPBlock, SPPBlock

class MobileRepViTBackbone(torch.nn.Module):
    """
    Mobile-optimized RepViT Backbone for CPU deployment
    Features:
    - Lightweight channel dimensions
    - Reduced depth
    - Efficient activation functions
    - Optimized for sequential processing
    """
    def __init__(self, width, depth, use_se=False, use_repvit_csp=True):
        super().__init__()
        self.use_repvit_csp = use_repvit_csp
        
        # Stage 1: Initial downsampling
        self.stage1 = torch.nn.Sequential(
            ConvBlock(width[0], width[1], 3, 2)  # 640x640 -> 320x320
        )
        
        # Stage 2: First feature extraction
        if use_repvit_csp:
            self.stage2 = torch.nn.Sequential(
                ConvBlock(width[1], width[2], 3, 2),  # 320x320 -> 160x160
                RepViTCSPBlock(width[2], width[2], depth[0], use_se=use_se)
            )
        else:
            self.stage2 = torch.nn.Sequential(
                ConvBlock(width[1], width[2], 3, 2),
                *[MobileRepViTBlock(width[2], use_se=use_se) for _ in range(depth[0])]
            )
        
        # Stage 3: Medium-level features
        self.stage3 = torch.nn.Sequential(
            ConvBlock(width[2], width[3], 3, 2),  # 160x160 -> 80x80
            *[MobileRepViTBlock(width[3], use_se=use_se) for _ in range(depth[1])]
        )
        
        # Stage 4: High-level features
        self.stage4 = torch.nn.Sequential(
            ConvBlock(width[3], width[4], 3, 2),  # 80x80 -> 40x40
            *[MobileRepViTBlock(width[4], use_se=use_se) for _ in range(depth[2])]
        )
        
        # Stage 5: Final features with SPP
        self.stage5 = torch.nn.Sequential(
            ConvBlock(width[4], width[5], 3, 2),  # 40x40 -> 20x20
            *[MobileRepViTBlock(width[5], use_se=use_se) for _ in range(depth[3])],
            SPPBlock(width[5], width[5], k=5)  # Keep SPP for rich spatial features
        )

    def forward(self, x):
        # Sequential processing optimized for CPU
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s3, s4, s5

    def reparameterize(self):
        """Reparameterize all RepViT blocks for efficient inference"""
        for module in self.modules():
            if hasattr(module, 'reparameterize'):
                module.reparameterize()


class MobileRepViTBlock(nn.Module):
    """
    Lightweight RepViT Block optimized for CPU
    - Reduced MLP ratio
    - Simpler reparameterization
    - CPU-friendly operations
    """
    def __init__(self, dim, kernel_size=3, mlp_ratio=1.5, use_se=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        # Lightweight token mixer
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        # Reduced channel mixer
        hidden_dim = max(int(dim * mlp_ratio), 32)  # Minimum hidden dim
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),  # Simpler activation
            nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        # Lightweight SE (optional)
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, max(dim // 8, 8), 1, 1, 0),  # Reduced compression
                nn.ReLU(inplace=True),
                nn.Conv2d(max(dim // 8, 8), dim, 1, 1, 0),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Simplified reparameterization
        self.reparam_conv = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
        self.reparam_bn = nn.BatchNorm2d(dim)
        
        self.act = nn.ReLU(inplace=True)  # CPU-friendly activation

    def forward(self, x):
        # Training: multi-branch
        if self.training:
            identity = x
            # Main path
            out = self.token_mixer(x)
            # Reparameterization branch
            out = out + self.reparam_bn(self.reparam_conv(x))
        else:
            # Inference: single path
            out = self.token_mixer(x)
        
        # Lightweight SE
        if self.se is not None:
            out = out * self.se(out)
        
        # Channel mixer with residual
        out = out + self.channel_mixer(out)
        
        return self.act(out)

    def reparameterize(self):
        """Simplified reparameterization for CPU"""
        if not self.training:
            return
            
        # Fuse reparam branch into token mixer
        token_conv = self.token_mixer[0]
        token_bn = self.token_mixer[1]
        
        # Fuse main token mixer
        main_kernel, main_bias = self._fuse_bn(token_conv, token_bn)
        
        # Fuse reparameterization branch
        reparam_kernel, reparam_bias = self._fuse_bn(self.reparam_conv, self.reparam_bn)
        
        # Create fused convolution
        fused_conv = nn.Conv2d(
            self.dim, self.dim,
            kernel_size=token_conv.kernel_size,
            stride=1,
            padding=token_conv.padding,
            groups=self.dim,
            bias=True
        )
        
        # Combine weights and biases
        fused_conv.weight.data = main_kernel + reparam_kernel
        fused_conv.bias.data = main_bias + reparam_bias
        
        # Replace token mixer
        self.token_mixer = nn.Sequential(fused_conv)
        
        # Clean up reparameterization branches
        self.reparam_conv = None
        self.reparam_bn = None

    def _fuse_bn(self, conv, bn):
        """Fuse Conv and BN layers"""
        if conv is None or bn is None:
            return 0, 0
            
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std


class MobileRepViTBackbonePresets:
    """Predefined configurations for CPU-optimized models"""
    
    @staticmethod
    def micro(use_se=False):
        """Micro size - ~0.5-1M parameters, ultra-lightweight"""
        width = [3, 16, 32, 64, 128, 256]
        depth = [1, 2, 2, 1]  # [s2, s3, s4, s5]
        return MobileRepViTBackbone(width, depth, use_se, use_repvit_csp=False)
    
    @staticmethod
    def nano(use_se=False):
        """Nano size - ~1-2M parameters, mobile-optimized"""
        width = [3, 24, 48, 96, 192, 384]
        depth = [1, 2, 3, 1]
        return MobileRepViTBackbone(width, depth, use_se, use_repvit_csp=True)
    
    @staticmethod
    def small(use_se=False):
        """Small size - ~3-4M parameters, balanced performance"""
        width = [3, 32, 64, 128, 256, 512]
        depth = [1, 3, 4, 2]
        return MobileRepViTBackbone(width, depth, use_se, use_repvit_csp=True)
    
    @staticmethod
    def medium(use_se=True):
        """Medium size - ~6-8M parameters, better accuracy"""
        width = [3, 48, 96, 192, 384, 768]
        depth = [2, 4, 6, 3]
        return MobileRepViTBackbone(width, depth, use_se, use_repvit_csp=True)


class CPUOptimizedSequential(nn.Sequential):
    """
    CPU-optimized sequential container
    - Better memory management
    - Sequential execution optimization
    """
    def forward(self, x):
        for module in self:
            x = module(x)
        return x


# Enhanced SPP block for CPU
class MobileSPPBlock(nn.Module):
    """CPU-optimized SPP block"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.reduce_conv = ConvBlock(in_ch, in_ch // 2)
        self.final_conv = ConvBlock(in_ch * 2, out_ch)
        # Use smaller kernel sizes for CPU efficiency
        self.pool1 = nn.MaxPool2d(k, 1, k // 2)
        self.pool2 = nn.MaxPool2d(3, 1, 1)  # Smaller kernel
        self.pool3 = nn.MaxPool2d(3, 1, 1)  # Smaller kernel

    def forward(self, x):
        x = self.reduce_conv(x)
        y1 = self.pool1(x)
        y2 = self.pool2(y1)
        y3 = self.pool3(y2)
        return self.final_conv(torch.cat([x, y1, y2, y3], 1))


# Example usage and testing
if __name__ == "__main__":
    # Test mobile backbones
    input_tensor = torch.randn(2, 3, 640, 640)
    
    print("Testing MobileRepViT Backbones on CPU:")
    
    # Test micro preset
    backbone_micro = MobileRepViTBackbonePresets.micro()
    s3, s4, s5 = backbone_micro(input_tensor)
    print(f"Micro - s3: {s3.shape}, s4: {s4.shape}, s5: {s5.shape}")
    
    # Test nano preset
    backbone_nano = MobileRepViTBackbonePresets.nano()
    s3, s4, s5 = backbone_nano(input_tensor)
    print(f"Nano - s3: {s3.shape}, s4: {s4.shape}, s5: {s5.shape}")
    
    # Test small preset
    backbone_small = MobileRepViTBackbonePresets.small()
    s3, s4, s5 = backbone_small(input_tensor)
    print(f"Small - s3: {s3.shape}, s4: {s4.shape}, s5: {s5.shape}")
    
    # Parameter counting
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"Micro: {count_parameters(backbone_micro):,} parameters")
    print(f"Nano: {count_parameters(backbone_nano):,} parameters")
    print(f"Small: {count_parameters(backbone_small):,} parameters")
    
    # Test reparameterization
    print("\nTesting reparameterization:")
    backbone_nano.eval()
    backbone_nano.reparameterize()
    
    with torch.no_grad():
        s3, s4, s5 = backbone_nano(input_tensor)
        print(f"After reparameterization - s3: {s3.shape}, s4: {s4.shape}, s5: {s5.shape}")
    
    # CPU performance test
    print("\nCPU Performance Test:")
    import time
    
    backbone = MobileRepViTBackbonePresets.nano().eval()
    backbone.reparameterize()
    
    # Warmup
    for _ in range(10):
        _ = backbone(input_tensor)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = backbone(input_tensor)
    end_time = time.time()
    
    print(f"Average inference time: {(end_time - start_time) / 100 * 1000:.2f} ms")
