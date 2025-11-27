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

        # Depthwise 3x3
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Normal pointwise 1x1 (groups = 1)
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride, 0, groups=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Pointwise conv
        self.pwconv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(in_channels)

        # SE
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, int(in_channels * se_ratio), 1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels * se_ratio), in_channels, 1),
            nn.Sigmoid()
        )

        self.shortcut = stride == 1

    def forward(self, x):
        # Fuse only on inference
        if not self.training and not self.fused:
            self._fuse_reparam()

        if hasattr(self, 'rbr_reparam'):
            y = self.rbr_reparam(x)
        else:
            y = self.rbr_dense(x) + self.rbr_1x1(x)

        y = self.pwconv(y)
        y = self.bn_pw(y)
        y = y * self.se(y)
        y = F.relu(y)

        return y + x if self.shortcut else y

    # ------------------ FUSION ------------------
    def _fuse_reparam(self):
        k3, b3 = self._fuse_conv_bn(self.rbr_dense)
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1)

        # Pad 1x1 → 3x3
        k1_pad = F.pad(k1, [1, 1, 1, 1])

        fused_k = k3 + k1_pad
        fused_b = b3 + b1

        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.in_channels, 3,
            self.stride, 1, groups=self.in_channels, bias=True
        )

        self.rbr_reparam.weight.data = fused_k
        self.rbr_reparam.bias.data = fused_b

        del self.rbr_dense
        del self.rbr_1x1

        self.fused = True

    @staticmethod
    def _fuse_conv_bn(branch):
        conv = branch[0]
        bn = branch[1]

        w = conv.weight
        b = torch.zeros(w.size(0), device=w.device)

        bn_var_rsqrt = 1.0 / torch.sqrt(bn.running_var + bn.eps)

        w_fused = w * (bn.weight * bn_var_rsqrt).reshape(-1, 1, 1, 1)
        b_fused = bn.bias + (b - bn.running_mean) * bn_var_rsqrt * bn.weight

        return w_fused, b_fused


# =========================================================
# RepViT Backbone
# =========================================================
class RepViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            RepViTBlock(32)
        )
        self.stage2 = nn.Sequential(
            RepViTBlock(32, stride=2),
            RepViTBlock(32)
        )
        self.stage3 = nn.Sequential(
            RepViTBlock(32, stride=2),
            RepViTBlock(32),
            RepViTBlock(32)
        )

    def forward(self, x):
        c2 = self.stage1(x)   # 320x320 → 160x160
        c3 = self.stage2(c2)  # → 80x80
        c4 = self.stage3(c3)  # → 40x40
        return [c2, c3, c4]


# =========================================================
# Ghost Module
# =========================================================
class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2, kernel_size=1, dw_size=3, stride=1, relu=True):
        super().__init__()
        init_ch = out_ch // ratio
        new_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, dw_size, 1, dw_size // 2, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)


# =========================================================
# GhostNeck
# =========================================================
class GhostNeck(nn.Module):
    def __init__(self, chs=[32, 32, 32]):
        super().__init__()
        c2, c3, c4 = chs
        self.reduce_c4 = GhostModule(c4, 64)
        self.reduce_c3 = GhostModule(c3 + 64, 48)
        self.reduce_c2 = GhostModule(c2 + 48, 32)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, feats):
        c2, c3, c4 = feats
        p5 = self.reduce_c4(c4)
        p4 = self.reduce_c3(torch.cat([self.up(p5), c3], 1))
        p3 = self.reduce_c2(torch.cat([self.up(p4), c2], 1))
        return p3, p4, p5


# =========================================================
# Head
# =========================================================
class GhostHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = GhostModule(in_ch, 32)
        self.pred = nn.Conv2d(32, num_classes + 4, 1)

    def forward(self, x):
        return self.pred(self.conv(x))


# =========================================================
# YOLOv8 Hybrid (Single or Multi-head)
# =========================================================
class YOLOv8Hybrid(nn.Module):
    def __init__(self, num_classes=80, multi_head=False):
        super().__init__()
        self.multi_head = multi_head

        self.backbone = RepViTBackbone()
        self.neck = GhostNeck([32, 32, 32])

        if multi_head:
            self.head_p3 = GhostHead(32, num_classes)
            self.head_p4 = GhostHead(48, num_classes)
            self.head_p5 = GhostHead(64, num_classes)
        else:
            # Fix: project channels before adding
            self.proj_p4 = nn.Conv2d(48, 32, 1, bias=False)
            self.bn_p4 = nn.BatchNorm2d(32)

            self.proj_p5 = nn.Conv2d(64, 32, 1, bias=False)
            self.bn_p5 = nn.BatchNorm2d(32)

            self.head = GhostHead(32, num_classes)

    def forward(self, x):
        p3, p4, p5 = self.neck(self.backbone(x))

        if self.multi_head:
            return [
                self.head_p3(p3),
                self.head_p4(p4),
                self.head_p5(p5)
            ]
        else:
            p4_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
            p5_up = F.interpolate(p5, size=p3.shape[2:], mode='nearest')

            p4_proj = self.bn_p4(self.proj_p4(p4_up))
            p5_proj = self.bn_p5(self.proj_p5(p5_up))

            fused = p3 + p4_proj + p5_proj
            return self.head(fused)


# =========================================================
# Test Run
# =========================================================
if __name__ == "__main__":
    device = torch.device("cpu")
    x = torch.randn(1, 3, 640, 640).to(device)

    # Single-head
    model_single = YOLOv8Hybrid(num_classes=20, multi_head=False).to(device)
    model_single.train()
    print("Single-head train:", model_single(x).shape)
    model_single.eval()
    print("Single-head infer:", model_single(x).shape)

    # Multi-head
    model_multi = YOLOv8Hybrid(num_classes=20, multi_head=True).to(device)
    model_multi.train()
    print("Multi-head train:", [o.shape for o in model_multi(x)])
    model_multi.eval()
    print("Multi-head infer:", [o.shape for o in model_multi(x)])
