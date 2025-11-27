import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 🔸 RepViT Block with Automatic Reparameterization
# =========================================================
class RepViTBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.fused = False

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride, 0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.pwconv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(in_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, int(in_channels*se_ratio), 1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels*se_ratio), in_channels, 1),
            nn.Sigmoid()
        )
        self.shortcut = stride == 1

    def forward(self, x):
        if not self.training and not self.fused:
            if hasattr(self, 'rbr_dense') and hasattr(self, 'rbr_1x1'):
                self._fuse_reparam()

        if hasattr(self, 'rbr_reparam'):
            y = self.rbr_reparam(x)
        else:
            y = self.rbr_dense(x) + self.rbr_1x1(x)

        y = self.pwconv(y)
        y = self.bn_pw(y)
        y = F.relu(y)
        y = y * self.se(y)
        if self.shortcut:
            return y + x
        else:
            return y

    def _fuse_reparam(self):
        k3, b3 = self._fuse_conv_bn(self.rbr_dense)
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1)
        k1_pad = F.pad(k1, [1,1,1,1])
        fused_k = k3 + k1_pad
        fused_b = b3 + b1

        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.in_channels, 3, self.stride, 1, groups=self.in_channels, bias=True
        )
        self.rbr_reparam.weight.data = fused_k
        self.rbr_reparam.bias.data = fused_b

        for attr in ['rbr_dense','rbr_1x1']:
            if hasattr(self, attr): delattr(self, attr)
        self.fused = True

    @staticmethod
    def _fuse_conv_bn(branch):
        conv = branch[0]; bn = branch[1]
        w = conv.weight
        bias = torch.zeros(w.size(0), device=w.device) if conv.bias is None else conv.bias
        bn_var_rsqrt = 1.0 / torch.sqrt(bn.running_var + bn.eps)
        w_fused = w * (bn.weight * bn_var_rsqrt).reshape(-1,1,1,1)
        b_fused = bn.bias + (bias - bn.running_mean) * bn_var_rsqrt * bn.weight
        return w_fused, b_fused

# =========================================================
# RepViT Backbone
# =========================================================
class RepViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,32,3,2,1,bias=False), nn.BatchNorm2d(32), nn.ReLU(), RepViTBlock(32)
        )
        self.stage2 = nn.Sequential(
            RepViTBlock(32,stride=2), RepViTBlock(32)
        )
        self.stage3 = nn.Sequential(
            RepViTBlock(32,stride=2), RepViTBlock(32), RepViTBlock(32)
        )

    def forward(self,x):
        c2 = self.stage1(x)  # stride 2
        c3 = self.stage2(c2) # stride 4
        c4 = self.stage3(c3) # stride 8
        return [c2,c3,c4]

# =========================================================
# Ghost Module
# =========================================================
class GhostModule(nn.Module):
    def __init__(self,in_ch,out_ch,ratio=2,kernel_size=1,dw_size=3,stride=1,relu=True):
        super().__init__()
        init_ch = out_ch // ratio
        new_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride, kernel_size//2,bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch,new_ch,dw_size,1,dw_size//2,groups=init_ch,bias=False),
            nn.BatchNorm2d(new_ch),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self,x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y,z],dim=1)

# =========================================================
# GhostNeck (returns multi-scale features for multi-head)
# =========================================================
class GhostNeck(nn.Module):
    def __init__(self,chs=[32,32,32]):
        super().__init__()
        c2,c3,c4 = chs
        self.reduce_c4 = GhostModule(c4,64)
        self.reduce_c3 = GhostModule(c3+64,48)
        self.reduce_c2 = GhostModule(c2+48,32)
        self.up = nn.Upsample(scale_factor=2,mode='nearest')

    def forward(self,feats):
        c2,c3,c4 = feats
        p5 = self.reduce_c4(c4)       # low-res
        p4 = self.up(p5)
        p4 = torch.cat([p4,c3],1)
        p4 = self.reduce_c3(p4)
        p3 = self.up(p4)
        p3 = torch.cat([p3,c2],1)
        p3 = self.reduce_c2(p3)
        return p3,p4,p5   # return all scales

# =========================================================
# Ghost Head (for detection)
# =========================================================
class GhostHead(nn.Module):
    def __init__(self,in_ch,num_classes):
        super().__init__()
        self.conv = GhostModule(in_ch,32)
        self.pred = nn.Conv2d(32,num_classes+4,1)

    def forward(self,x):
        x = self.conv(x)
        x = self.pred(x)
        return x

# =========================================================
# YOLOv8Hybrid supporting single or multi-head
# =========================================================
class YOLOv8Hybrid(nn.Module):
    def __init__(self,num_classes=80,multi_head=False):
        super().__init__()
        self.multi_head = multi_head
        self.backbone = RepViTBackbone()
        self.neck = GhostNeck([32,32,32])

        if multi_head:
            # One head per scale
            self.head_p3 = GhostHead(32,num_classes)
            self.head_p4 = GhostHead(48,num_classes)
            self.head_p5 = GhostHead(64,num_classes)
        else:
            # Single fused head
            self.head = GhostHead(32,num_classes)

    def forward(self,x):
        p3,p4,p5 = self.neck(self.backbone(x))
        if self.multi_head:
            out_p3 = self.head_p3(p3)
            out_p4 = self.head_p4(p4)
            out_p5 = self.head_p5(p5)
            return [out_p3,out_p4,out_p5]
        else:
            # Fuse all scales (concatenate) for single head
            # Simple approach: upsample p4,p5 to p3 resolution and add
            p4_up = F.interpolate(p4,size=p3.shape[2:],mode='nearest')
            p5_up = F.interpolate(p5,size=p3.shape[2:],mode='nearest')
            fused = p3 + p4_up + p5_up
            out = self.head(fused)
            return out

# =========================================================
# Test Run
# =========================================================
if __name__=="__main__":
    device = torch.device('cpu')
    x = torch.randn(1,3,640,640).to(device)

    # Single-head
    model_single = YOLOv8Hybrid(num_classes=20,multi_head=False).to(device)
    model_single.train()
    y_train = model_single(x)
    print("Single-head train output shape:", y_train.shape)
    model_single.eval()
    y_eval = model_single(x)
    print("Single-head inference output shape:", y_eval.shape)

    # Multi-head
    model_multi = YOLOv8Hybrid(num_classes=20,multi_head=True).to(device)
    model_multi.train()
    y_train_multi = model_multi(x)
    print("Multi-head train output shapes:", [o.shape for o in y_train_multi])
    model_multi.eval()
    y_eval_multi = model_multi(x)
    print("Multi-head inference output shapes:", [o.shape for o in y_eval_multi])
