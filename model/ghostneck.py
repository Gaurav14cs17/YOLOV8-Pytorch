import torch
import torch.nn as nn
from model.blocks import ConvBlock

class GhostBottleneck(nn.Module):
    """GhostNet Bottleneck for efficient feature fusion"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        
        # Expansion conv
        self.ghost1 = GhostModule(in_ch, out_ch, relu=True)
        
        # Depthwise conv for spatial processing
        if stride > 1:
            self.dw_conv = nn.Conv2d(out_ch, out_ch, 3, stride, 1, groups=out_ch, bias=False)
            self.dw_bn = nn.BatchNorm2d(out_ch)
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SELayer(out_ch)
        
        # Pointwise conv
        self.ghost2 = GhostModule(out_ch, out_ch, relu=False)
        
        # Shortcut connection
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = x
        
        # Expansion
        out = self.ghost1(x)
        
        # Depthwise
        if self.stride > 1:
            out = self.dw_conv(out)
            out = self.dw_bn(out)
        
        # SE attention
        if self.use_se:
            out = self.se(out)
        
        # Pointwise
        out = self.ghost2(out)
        
        # Shortcut
        if self.stride == 1:
            out += self.shortcut(residual)
        else:
            out = self.shortcut(residual) + out if self.use_se else out
            
        return out


class GhostModule(nn.Module):
    """Ghost module for efficient feature extraction"""
    def __init__(self, in_ch, out_ch, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_ch = out_ch
        init_ch = out_ch // ratio
        new_ch = init_ch * (ratio - 1)
        
        # Primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        
        # Cheap operation (depthwise)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, dw_size, 1, dw_size//2, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer"""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostFusionBlock(nn.Module):
    """GhostNet-based feature fusion block"""
    def __init__(self, in_ch, out_ch, depth=1, use_se=False):
        super().__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(GhostBottleneck(in_ch, out_ch, use_se=use_se))
            else:
                layers.append(GhostBottleneck(out_ch, out_ch, use_se=use_se))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class GhostNeck(torch.nn.Module):
    """GhostNet-based neck for efficient feature pyramid fusion"""
    def __init__(self, width, depth, use_se=True):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # Top to middle fusion
        self.top_to_mid = GhostFusionBlock(width[4] + width[5], width[4], depth[0], use_se=use_se)
        
        # Middle to small fusion  
        self.mid_to_small = GhostFusionBlock(width[3] + width[4], width[3], depth[0], use_se=use_se)

        # Downsampling layers
        self.downsample_mid = ConvBlock(width[3], width[3], 3, 2)
        self.mid_fuse = GhostFusionBlock(width[3] + width[4], width[4], depth[0], use_se=use_se)

        self.downsample_top = ConvBlock(width[4], width[4], 3, 2)
        self.top_fuse = GhostFusionBlock(width[4] + width[5], width[5], depth[0], use_se=use_se)

    def forward(self, feats):
        feat_small, feat_mid, feat_large = feats
        
        # Top-down path (large -> mid -> small)
        merged_mid = self.top_to_mid(torch.cat([self.upsample(feat_large), feat_mid], dim=1))
        merged_small = self.mid_to_small(torch.cat([self.upsample(merged_mid), feat_small], dim=1))
        
        # Bottom-up path (small -> mid -> large)
        mid_down = self.mid_fuse(torch.cat([self.downsample_mid(merged_small), merged_mid], dim=1))
        top_down = self.top_fuse(torch.cat([self.downsample_top(mid_down), feat_large], dim=1))
        
        return merged_small, mid_down, top_down


class LightGhostNeck(torch.nn.Module):
    """Lighter version of GhostNet neck for maximum efficiency"""
    def __init__(self, width, depth, use_se=False):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # Simplified fusion blocks
        self.top_to_mid = self._make_ghost_block(width[4] + width[5], width[4], depth[0])
        self.mid_to_small = self._make_ghost_block(width[3] + width[4], width[3], depth[0])
        
        self.downsample_mid = ConvBlock(width[3], width[3], 3, 2)
        self.mid_fuse = self._make_ghost_block(width[3] + width[4], width[4], depth[0])
        
        self.downsample_top = ConvBlock(width[4], width[4], 3, 2)
        self.top_fuse = self._make_ghost_block(width[4] + width[5], width[5], depth[0])

    def _make_ghost_block(self, in_ch, out_ch, depth):
        layers = []
        for i in range(depth):
            layers.append(GhostModule(in_ch if i == 0 else out_ch, out_ch, relu=(i < depth-1)))
        return nn.Sequential(*layers)

    def forward(self, feats):
        feat_small, feat_mid, feat_large = feats
        
        merged_mid = self.top_to_mid(torch.cat([self.upsample(feat_large), feat_mid], dim=1))
        merged_small = self.mid_to_small(torch.cat([self.upsample(merged_mid), feat_small], dim=1))
        
        mid_down = self.mid_fuse(torch.cat([self.downsample_mid(merged_small), merged_mid], dim=1))
        top_down = self.top_fuse(torch.cat([self.downsample_top(mid_down), feat_large], dim=1))
        
        return merged_small, mid_down, top_down


class HybridGhostNeck(torch.nn.Module):
    """Hybrid neck combining Ghost modules with traditional convolutions"""
    def __init__(self, width, depth, use_se=True):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # Use Ghost bottlenecks for complex fusions
        self.top_to_mid = GhostBottleneck(width[4] + width[5], width[4], use_se=use_se)
        self.mid_to_small = GhostBottleneck(width[3] + width[4], width[3], use_se=use_se)
        
        # Use regular convs for simple downsampling
        self.downsample_mid = ConvBlock(width[3], width[3], 3, 2)
        self.mid_fuse = GhostBottleneck(width[3] + width[4], width[4], use_se=use_se)
        
        self.downsample_top = ConvBlock(width[4], width[4], 3, 2)
        self.top_fuse = GhostBottleneck(width[4] + width[5], width[5], use_se=use_se)

    def forward(self, feats):
        feat_small, feat_mid, feat_large = feats
        
        merged_mid = self.top_to_mid(torch.cat([self.upsample(feat_large), feat_mid], dim=1))
        merged_small = self.mid_to_small(torch.cat([self.upsample(merged_mid), feat_small], dim=1))
        
        mid_down = self.mid_fuse(torch.cat([self.downsample_mid(merged_small), merged_mid], dim=1))
        top_down = self.top_fuse(torch.cat([self.downsample_top(mid_down), feat_large], dim=1))
        
        return merged_small, mid_down, top_down


class GhostNeckPresets:
    """Predefined configurations for GhostNet necks"""
    
    @staticmethod
    def nano(use_se=False):
        """Nano size - ultra lightweight"""
        width = [256, 512, 1024]  # [small, mid, large] input channels
        depth = [1]  # fusion depth
        return LightGhostNeck(width, depth, use_se)
    
    @staticmethod
    def small(use_se=True):
        """Small size - balanced performance"""
        width = [512, 1024, 2048]
        depth = [1]
        return GhostNeck(width, depth, use_se)
    
    @staticmethod
    def medium(use_se=True):
        """Medium size - better fusion"""
        width = [512, 1024, 2048]
        depth = [2]
        return GhostNeck(width, depth, use_se)
    
    @staticmethod
    def large(use_se=True):
        """Large size - maximum performance"""
        width = [512, 1024, 2048]
        depth = [3]
        return GhostNeck(width, depth, use_se)


# Example usage and testing
if __name__ == "__main__":
    # Test different neck configurations
    feat_small = torch.randn(2, 256, 80, 80)
    feat_mid = torch.randn(2, 512, 40, 40)
    feat_large = torch.randn(2, 1024, 20, 20)
    feats = (feat_small, feat_mid, feat_large)
    
    print("Testing GhostNet-based Necks:")
    
    # Test nano preset
    neck_nano = GhostNeckPresets.nano()
    out_small, out_mid, out_large = neck_nano(feats)
    print(f"Nano - small: {out_small.shape}, mid: {out_mid.shape}, large: {out_large.shape}")
    
    # Test small preset
    neck_small = GhostNeckPresets.small()
    out_small, out_mid, out_large = neck_small(feats)
    print(f"Small - small: {out_small.shape}, mid: {out_mid.shape}, large: {out_large.shape}")
    
    # Test custom neck
    custom_width = [128, 256, 512]
    custom_depth = [1]
    custom_neck = GhostNeck(custom_width, custom_depth, use_se=True)
    out_small, out_mid, out_large = custom_neck(feats)
    print(f"Custom - small: {out_small.shape}, mid: {out_mid.shape}, large: {out_large.shape}")
    
    # Parameter counting
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"Nano neck: {count_parameters(neck_nano):,} parameters")
    print(f"Small neck: {count_parameters(neck_small):,} parameters")
    
    # Performance comparison
    import time
    
    print("\nPerformance Test:")
    neck_nano.eval()
    
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            _ = neck_nano(feats)
        end_time = time.time()
        print(f"Average inference time: {(end_time - start_time) / 100 * 1000:.2f} ms")
