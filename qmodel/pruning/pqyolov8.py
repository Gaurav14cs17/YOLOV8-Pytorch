import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.utils.prune as prune
from typing import List, Dict

class Swish(nn.Module):
    """Swish activation function with quantized operations."""
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.quant_ops = nnq.FloatFunctional()

    def forward(self, x):
        return self.quant_ops.mul(self.sigmoid(x), x)


class PrunableBlock:
    """Mixin class for prunable blocks with common pruning methods."""
    
    @staticmethod
    def apply_conv_pruning(module: nn.Module, amount: float = 0.2):
        """Apply L1 unstructured pruning to Conv2d layers."""
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
            
    @staticmethod
    def apply_global_pruning(model: nn.Module, amount: float = 0.2):
        """Apply global pruning across all Conv2d layers."""
        parameters_to_prune = [
            (module, 'weight') for module in model.modules() 
            if isinstance(module, nn.Conv2d)
        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')


class BlockBuilder(PrunableBlock):
    """Helper class to construct common building blocks with pruning support."""
    
    @staticmethod
    def conv_bn_act(in_channels: int, 
                   out_channels: int, 
                   kernel_size: int = 1, 
                   stride: int = 1,
                   pruning_cfg: Dict = None):
        """Conv2d + BatchNorm + Swish with optional pruning."""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                        stride, (kernel_size - 1) // 2, bias=False)
        
        # Apply pruning if configured
        if pruning_cfg and pruning_cfg.get('prune_conv', False):
            BlockBuilder.apply_conv_pruning(conv, pruning_cfg.get('amount', 0.2))
            
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            Swish()
        )
    
    @staticmethod
    def residual_block(channels: int, 
                      use_add: bool = True,
                      pruning_cfg: Dict = None):
        """Residual block with optional pruning."""
        layers = [
            BlockBuilder.conv_bn_act(channels, channels, 3, pruning_cfg=pruning_cfg),
            BlockBuilder.conv_bn_act(channels, channels, 3, pruning_cfg=pruning_cfg)
        ]
        return ResidualBlock(layers, use_add)


class ResidualBlock(nn.Module):
    """Residual block with quantized add operation."""
    
    def __init__(self, layers: List[nn.Module], use_add: bool = True):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.use_add = use_add
        self.quant_add = nnq.FloatFunctional()
        
    def forward(self, x):
        residual = self.layers(x)
        return self.quant_add.add(x, residual) if self.use_add else residual


class CSPBlock(nn.Module, PrunableBlock):
    """Cross Stage Partial block with pruning support."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_blocks: int = 1, 
                 use_add: bool = True,
                 pruning_cfg: Dict = None):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.initial_conv = BlockBuilder.conv_bn_act(
            in_channels, out_channels, pruning_cfg=pruning_cfg)
        
        self.residual_blocks = nn.ModuleList([
            BlockBuilder.residual_block(mid_channels, use_add, pruning_cfg) 
            for _ in range(num_blocks)
        ])
        
        self.final_conv = BlockBuilder.conv_bn_act(
            (2 + num_blocks) * mid_channels, out_channels, pruning_cfg=pruning_cfg
        )
        self.quant_cat = nnq.FloatFunctional()
        
    def forward(self, x):
        features = list(self.initial_conv(x).chunk(2, dim=1))
        for block in self.residual_blocks:
            features.append(block(features[-1]))
        return self.final_conv(self.quant_cat.cat(features, dim=1))


class SpatialPyramidPooling(nn.Module, PrunableBlock):
    """SPP layer with pruning support."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 pool_size: int = 5,
                 pruning_cfg: Dict = None):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.input_conv = BlockBuilder.conv_bn_act(
            in_channels, mid_channels, pruning_cfg=pruning_cfg)
        self.output_conv = BlockBuilder.conv_bn_act(
            mid_channels * 4, out_channels, pruning_cfg=pruning_cfg)
        self.pool = nn.MaxPool2d(pool_size, 1, pool_size // 2)
        self.quant_cat = nnq.FloatFunctional()
        
    def forward(self, x):
        x = self.input_conv(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.output_conv(self.quant_cat.cat([x, p1, p2, p3], dim=1))


class Backbone(nn.Module, PrunableBlock):
    """DarkNet backbone with pruning support."""
    
    def __init__(self, 
                 width_channels: List[int], 
                 depth_blocks: List[int],
                 pruning_cfg: Dict = None):
        super().__init__()
        self.pruning_cfg = pruning_cfg
        
        self.stages = nn.ModuleList([
            # Stage 1 (p1/2)
            nn.Sequential(
                BlockBuilder.conv_bn_act(
                    width_channels[0], width_channels[1], 3, 2, pruning_cfg)
            ),
            # Stage 2 (p2/4)
            nn.Sequential(
                BlockBuilder.conv_bn_act(
                    width_channels[1], width_channels[2], 3, 2, pruning_cfg),
                CSPBlock(width_channels[2], width_channels[2], depth_blocks[0], 
                        True, pruning_cfg)
            ),
            # Stage 3 (p3/8)
            nn.Sequential(
                BlockBuilder.conv_bn_act(
                    width_channels[2], width_channels[3], 3, 2, pruning_cfg),
                CSPBlock(width_channels[3], width_channels[3], depth_blocks[1], 
                        True, pruning_cfg)
            ),
            # Stage 4 (p4/16)
            nn.Sequential(
                BlockBuilder.conv_bn_act(
                    width_channels[3], width_channels[4], 3, 2, pruning_cfg),
                CSPBlock(width_channels[4], width_channels[4], depth_blocks[2], 
                        True, pruning_cfg)
            ),
            # Stage 5 (p5/32)
            nn.Sequential(
                BlockBuilder.conv_bn_act(
                    width_channels[4], width_channels[5], 3, 2, pruning_cfg),
                CSPBlock(width_channels[5], width_channels[5], depth_blocks[0], 
                        True, pruning_cfg),
                SpatialPyramidPooling(width_channels[5], width_channels[5], 
                                     pruning_cfg=pruning_cfg)
            )
        ])
        
    def forward(self, x):
        features = []
        current = x
        for stage in self.stages:
            current = stage(current)
            features.append(current)
        return features[2], features[3], features[4]


class FeaturePyramid(nn.Module, PrunableBlock):
    """FPN with pruning support."""
    
    def __init__(self, 
                 width_channels: List[int], 
                 depth_blocks: List[int],
                 pruning_cfg: Dict = None):
        super().__init__()
        self.quant_ops = nnq.FloatFunctional()
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.fusion_blocks = nn.ModuleDict({
            'top_down_1': CSPBlock(
                width_channels[4] + width_channels[5], width_channels[4], 
                depth_blocks[0], False, pruning_cfg),
            'top_down_2': CSPBlock(
                width_channels[3] + width_channels[4], width_channels[3], 
                depth_blocks[0], False, pruning_cfg),
            'bottom_up_1': BlockBuilder.conv_bn_act(
                width_channels[3], width_channels[3], 3, 2, pruning_cfg),
            'bottom_up_2': CSPBlock(
                width_channels[3] + width_channels[4], width_channels[4], 
                depth_blocks[0], False, pruning_cfg),
            'bottom_up_3': BlockBuilder.conv_bn_act(
                width_channels[4], width_channels[4], 3, 2, pruning_cfg),
            'bottom_up_4': CSPBlock(
                width_channels[4] + width_channels[5], width_channels[5], 
                depth_blocks[0], False, pruning_cfg)
        })
        
    def forward(self, p3, p4, p5):
        p4 = self.fusion_blocks['top_down_1'](self.quant_ops.cat([self.upsample(p5), p4], dim=1))
        p3 = self.fusion_blocks['top_down_2'](self.quant_ops.cat([self.upsample(p4), p3], dim=1))
        
        p4 = self.fusion_blocks['bottom_up_2'](
            self.quant_ops.cat([self.fusion_blocks['bottom_up_1'](p3), p4], dim=1))
        p5 = self.fusion_blocks['bottom_up_4'](
            self.quant_ops.cat([self.fusion_blocks['bottom_up_3'](p4), p5], dim=1))
        
        return p3, p4, p5


class DetectionHead(nn.Module, PrunableBlock):
    """Detection head with pruning support."""
    
    def __init__(self, 
                 num_classes: int = 80, 
                 in_channels: tuple = (),
                 pruning_cfg: Dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = num_classes + 4
        self.stride = torch.zeros(len(in_channels))
        self.quant_ops = nnq.FloatFunctional()
        
        self.box_branches = nn.ModuleList()
        self.cls_branches = nn.ModuleList()
        
        for ch in in_channels:
            box_channels = max(64, ch // 4)
            self.box_branches.append(nn.Sequential(
                BlockBuilder.conv_bn_act(ch, box_channels, 3, pruning_cfg=pruning_cfg),
                BlockBuilder.conv_bn_act(box_channels, box_channels, 3, pruning_cfg=pruning_cfg),
                nn.Conv2d(box_channels, 4, 1)
            ))
            
            cls_channels = max(80, ch, num_classes)
            self.cls_branches.append(nn.Sequential(
                BlockBuilder.conv_bn_act(ch, cls_channels, 3, pruning_cfg=pruning_cfg),
                BlockBuilder.conv_bn_act(cls_channels, cls_channels, 3, pruning_cfg=pruning_cfg),
                nn.Conv2d(cls_channels, num_classes, 1)
            ))
            
    def forward(self, *inputs):
        outputs = []
        for i, x in enumerate(inputs):
            box_out = self.box_branches[i](x)
            cls_out = self.cls_branches[i](x)
            outputs.append(self.quant_ops.cat((box_out, cls_out), dim=1))
        return outputs


class YOLOv8(nn.Module, PrunableBlock):
    """Complete YOLOv8 model with pruning support."""
    
    def __init__(self, 
                 width_channels: List[int], 
                 depth_blocks: List[int], 
                 num_classes: int = 80,
                 pruning_cfg: Dict = None):
        super().__init__()
        self.pruning_cfg = pruning_cfg
        
        self.backbone = Backbone(width_channels, depth_blocks, pruning_cfg)
        self.neck = FeaturePyramid(width_channels, depth_blocks, pruning_cfg)
        self.head = DetectionHead(num_classes, 
                                (width_channels[3], width_channels[4], width_channels[5]), 
                                pruning_cfg)
        
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
    
    def prune_model(self, amount: float = 0.2):
        """Apply global pruning to the model."""
        if self.pruning_cfg and self.pruning_cfg.get('global_prune', False):
            self.apply_global_pruning(self, amount)


class QuantizedYOLO(nn.Module):
    """Wrapper for quantization-aware training with pruning support."""
    
    def __init__(self, model: nn.Module):
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


def create_yolo_variant(variant_name: str, 
                       num_classes: int = 80,
                       pruning_cfg: Dict = None):
    """Factory function to create different YOLOv8 variants with pruning."""
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
    
    model = YOLOv8(config['width'], config['depth'], num_classes, pruning_cfg)
    
    # Apply initial pruning if configured
    if pruning_cfg and pruning_cfg.get('prune_on_init', False):
        model.prune_model(pruning_cfg.get('amount', 0.2))
    
    return model
