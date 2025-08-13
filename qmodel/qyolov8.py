import torch
import torch.nn as nn
import torch.nn.quantized as nnq

class Swish(nn.Module):
    """Swish activation function implemented using quantized operations."""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.quant_ops = nnq.FloatFunctional()

    def forward(self, x):
        return self.quant_ops.mul(self.sigmoid(x), x)


class BlockBuilder:
    """Helper class to construct common building blocks."""
    
    @staticmethod
    def conv_bn_act(in_channels, out_channels, kernel_size=1, stride=1):
        """Conv2d + BatchNorm + Swish activation block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            Swish()
        )
    
    @staticmethod
    def residual_block(channels, use_add=True):
        """Basic residual block with optional skip connection."""
        layers = [
            BlockBuilder.conv_bn_act(channels, channels, 3),
            BlockBuilder.conv_bn_act(channels, channels, 3)
        ]
        return ResidualBlock(layers, use_add)


class ResidualBlock(nn.Module):
    """Residual block implementation with quantized add operation."""
    
    def __init__(self, layers, use_add=True):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.use_add = use_add
        self.quant_add = nnq.FloatFunctional()
        
    def forward(self, x):
        residual = self.layers(x)
        return self.quant_add.add(x, residual) if self.use_add else residual


class CSPBlock(nn.Module):
    """Cross Stage Partial block with multiple residual connections."""
    
    def __init__(self, in_channels, out_channels, num_blocks=1, use_add=True):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.initial_conv = BlockBuilder.conv_bn_act(in_channels, out_channels)
        self.residual_blocks = nn.ModuleList([
            BlockBuilder.residual_block(mid_channels, use_add) 
            for _ in range(num_blocks)
        ])
        self.final_conv = BlockBuilder.conv_bn_act(
            (2 + num_blocks) * mid_channels, out_channels
        )
        self.quant_cat = nnq.FloatFunctional()
        
    def forward(self, x):
        # Split input channels
        features = list(self.initial_conv(x).chunk(2, dim=1))
        
        # Process through residual blocks
        for block in self.residual_blocks:
            features.append(block(features[-1]))
            
        # Concatenate and process
        return self.final_conv(self.quant_cat.cat(features, dim=1))


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer with multiple max pools."""
    
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.input_conv = BlockBuilder.conv_bn_act(in_channels, mid_channels)
        self.output_conv = BlockBuilder.conv_bn_act(mid_channels * 4, out_channels)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)
        self.quant_cat = nnq.FloatFunctional()
        
    def forward(self, x):
        x = self.input_conv(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.output_conv(self.quant_cat.cat([x, p1, p2, p3], dim=1))


class Backbone(nn.Module):
    """DarkNet backbone with multiple stages."""
    
    def __init__(self, width_channels, depth_blocks):
        super().__init__()
        
        # Stage definitions
        self.stages = nn.ModuleList([
            # Stage 1 (p1/2)
            nn.Sequential(
                BlockBuilder.conv_bn_act(width_channels[0], width_channels[1], 3, 2)
            ),
            # Stage 2 (p2/4)
            nn.Sequential(
                BlockBuilder.conv_bn_act(width_channels[1], width_channels[2], 3, 2),
                CSPBlock(width_channels[2], width_channels[2], depth_blocks[0])
            ),
            # Stage 3 (p3/8)
            nn.Sequential(
                BlockBuilder.conv_bn_act(width_channels[2], width_channels[3], 3, 2),
                CSPBlock(width_channels[3], width_channels[3], depth_blocks[1])
            ),
            # Stage 4 (p4/16)
            nn.Sequential(
                BlockBuilder.conv_bn_act(width_channels[3], width_channels[4], 3, 2),
                CSPBlock(width_channels[4], width_channels[4], depth_blocks[2])
            ),
            # Stage 5 (p5/32)
            nn.Sequential(
                BlockBuilder.conv_bn_act(width_channels[4], width_channels[5], 3, 2),
                CSPBlock(width_channels[5], width_channels[5], depth_blocks[0]),
                SpatialPyramidPooling(width_channels[5], width_channels[5])
            )
        ])
        
    def forward(self, x):
        features = []
        current = x
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        return features[2], features[3], features[4]  # p3, p4, p5


class FeaturePyramid(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""
    
    def __init__(self, width_channels, depth_blocks):
        super().__init__()
        self.quant_ops = nnq.FloatFunctional()
        self.upsample = nn.Upsample(scale_factor=2)
        
        # Fusion blocks
        self.fusion_blocks = nn.ModuleDict({
            'top_down_1': CSPBlock(width_channels[4] + width_channels[5], width_channels[4], depth_blocks[0], False),
            'top_down_2': CSPBlock(width_channels[3] + width_channels[4], width_channels[3], depth_blocks[0], False),
            'bottom_up_1': BlockBuilder.conv_bn_act(width_channels[3], width_channels[3], 3, 2),
            'bottom_up_2': CSPBlock(width_channels[3] + width_channels[4], width_channels[4], depth_blocks[0], False),
            'bottom_up_3': BlockBuilder.conv_bn_act(width_channels[4], width_channels[4], 3, 2),
            'bottom_up_4': CSPBlock(width_channels[4] + width_channels[5], width_channels[5], depth_blocks[0], False)
        })
        
    def forward(self, p3, p4, p5):
        # Top-down pathway
        p4 = self.fusion_blocks['top_down_1'](self.quant_ops.cat([self.upsample(p5), p4], dim=1))
        p3 = self.fusion_blocks['top_down_2'](self.quant_ops.cat([self.upsample(p4), p3], dim=1))
        
        # Bottom-up pathway
        p4 = self.fusion_blocks['bottom_up_2'](
            self.quant_ops.cat([self.fusion_blocks['bottom_up_1'](p3), p4], dim=1))
        p5 = self.fusion_blocks['bottom_up_4'](
            self.quant_ops.cat([self.fusion_blocks['bottom_up_3'](p4), p5], dim=1))
        
        return p3, p4, p5


class DetectionHead(nn.Module):
    """YOLO detection head with separate classification and box regression branches."""
    
    def __init__(self, num_classes=80, in_channels=()):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = num_classes + 4
        self.stride = torch.zeros(len(in_channels))
        
        # Create branches for each input scale
        self.box_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        self.quant_ops = nnq.FloatFunctional()
        
        for ch in in_channels:
            # Box prediction branch
            box_channels = max(64, ch // 4)
            box_branch = nn.Sequential(
                BlockBuilder.conv_bn_act(ch, box_channels, 3),
                BlockBuilder.conv_bn_act(box_channels, box_channels, 3),
                nn.Conv2d(box_channels, 4, 1)
            )
            self.box_branches.append(box_branch)
            
            # Class prediction branch
            cls_channels = max(80, ch, num_classes)
            cls_branch = nn.Sequential(
                BlockBuilder.conv_bn_act(ch, cls_channels, 3),
                BlockBuilder.conv_bn_act(cls_channels, cls_channels, 3),
                nn.Conv2d(cls_channels, num_classes, 1)
            )
            self.cls_branches.append(cls_branch)
            
    def forward(self, *inputs):
        outputs = []
        for i, x in enumerate(inputs):
            box_out = self.box_branches[i](x)
            cls_out = self.cls_branches[i](x)
            outputs.append(self.quant_ops.cat((box_out, cls_out), dim=1))
        return outputs


class YOLOv8(nn.Module):
    """Complete YOLOv8 model with backbone, neck and head."""
    
    def __init__(self, width_channels, depth_blocks, num_classes):
        super().__init__()
        self.backbone = Backbone(width_channels, depth_blocks)
        self.neck = FeaturePyramid(width_channels, depth_blocks)
        self.head = DetectionHead(num_classes, (width_channels[3], width_channels[4], width_channels[5]))
        
        # Initialize strides
        with torch.no_grad():
            dummy_input = torch.zeros(1, width_channels[0], 256, 256)
            outputs = self.forward(dummy_input)
            self.head.stride = torch.tensor([256 / x.shape[-2] for x in outputs])
            self.stride = self.head.stride
            
    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.neck(*features)
        return self.head(*fused_features)


class QuantizedYOLO(nn.Module):
    """Wrapper for quantization-aware training."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Copy important attributes
        self.num_classes = model.head.num_classes
        self.stride = model.stride
        
    def forward(self, x):
        x = self.quant(x)
        outputs = self.model(x)
        return [self.dequant(out) for out in outputs]


def create_yolo_variant(variant_name, num_classes=80):
    """Factory function to create different YOLOv8 variants."""
    variants = {
        'nano': {'depth': [1, 2, 2], 'width': [3, 16, 32, 64, 128, 256]},
        'tiny': {'depth': [1, 2, 2], 'width': [3, 24, 48, 96, 192, 384]},
        'small': {'depth': [1, 2, 2], 'width': [3, 32, 64, 128, 256, 512]},
        'medium': {'depth': [2, 4, 4], 'width': [3, 48, 96, 192, 384, 576]},
        'large': {'depth': [3, 6, 6], 'width': [3, 64, 128, 256, 512, 512]},
        'xlarge': {'depth': [3, 6, 6], 'width': [3, 80, 160, 320, 640, 640]}
    }
    
    config = variants.get(variant_name.lower())
    if not config:
        raise ValueError(f"Unknown variant: {variant_name}. Choose from {list(variants.keys())}")
    
    return YOLOv8(config['width'], config['depth'], num_classes)
