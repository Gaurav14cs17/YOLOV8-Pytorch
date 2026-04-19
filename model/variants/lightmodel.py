import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 🔸 RepViT Block with Automatic Reparameterization (fixed)
# =========================================================
class RepViTBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.fused = False

        # depthwise 3x3
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # depthwise 1x1 — MUST be depthwise to allow fusion into depthwise 3x3
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride, 0, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # pointwise conv (to mix channels after the depthwise)
        self.pwconv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(in_channels)

        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, int(in_channels * se_ratio)), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, int(in_channels * se_ratio)), in_channels, 1),
            nn.Sigmoid()
        )

        self.shortcut = (stride == 1)

    def forward(self, x):
        # fuse on first inference run automatically (if not fused)
        if not self.training and not self.fused:
            self._fuse_reparam()

        if hasattr(self, 'rbr_reparam'):
            y = self.rbr_reparam(x)
        else:
            y = self.rbr_dense(x) + self.rbr_1x1(x)

        y = self.pwconv(y)
        y = self.bn_pw(y)
        y = y * self.se(y)   # apply SE before activation
        y = F.relu(y)
        return x + y if self.shortcut else y

    def _fuse_reparam(self):
        # get fused kernels and biases for each branch
        k3, b3 = self._fuse_conv_bn(self.rbr_dense)   # depthwise 3x3 -> [C,1,3,3]
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1)     # depthwise 1x1 -> [C,1,1,1]

        # pad 1x1 to 3x3
        k1_pad = F.pad(k1, [1, 1, 1, 1])  # -> [C,1,3,3]

        fused_k = k3 + k1_pad
        fused_b = b3 + b1

        # create depthwise 3x3 conv with bias
        self.rbr_reparam = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=3,
            stride=self.stride, padding=1, groups=self.in_channels, bias=True
        )

        # assign weights/bias
        with torch.no_grad():
            self.rbr_reparam.weight.copy_(fused_k)
            self.rbr_reparam.bias.copy_(fused_b)

        # remove old branches to save memory / avoid accidental use
        del self.rbr_dense
        del self.rbr_1x1

        self.fused = True

    @staticmethod
    def _fuse_conv_bn(branch):
        """
        branch: nn.Sequential([Conv, BN])
        returns: (w_fused, b_fused)
        where w_fused has shape [C, 1, k, k] for depthwise convs
        """
        conv = branch[0]
        bn = branch[1]

        w = conv.weight  # e.g., [C, 1, k, k] for depthwise
        if conv.bias is None:
            bias = torch.zeros(w.size(0), device=w.device)
        else:
            bias = conv.bias

        bn_var_rsqrt = 1.0 / torch.sqrt(bn.running_var + bn.eps)  # shape [C]

        # scale the conv weights by BN gamma * inv_sqrt(var)
        # shape broadcasting: (C,1,1,1)
        w_fused = w * (bn.weight * bn_var_rsqrt).reshape(-1, 1, 1, 1)
        b_fused = bn.bias + (bias - bn.running_mean) * bn_var_rsqrt * bn.weight

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
            nn.ReLU(inplace=True),
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
        # c2: 640 -> 320
        # c3: 320 -> 160
        # c4: 160 -> 80
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
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
# GhostNeck (multi-scale features)
# =========================================================
class GhostNeck(nn.Module):
    def __init__(self, chs=[32, 32, 32]):
        super().__init__()
        c2, c3, c4 = chs
        self.reduce_c4 = GhostModule(c4, 64)            # p5 channels = 64
        self.reduce_c3 = GhostModule(c3 + 64, 48)       # p4 channels = 48
        self.reduce_c2 = GhostModule(c2 + 48, 32)       # p3 channels = 32
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, feats):
        c2, c3, c4 = feats
        p5 = self.reduce_c4(c4)
        p4 = self.reduce_c3(torch.cat([self.up(p5), c3], dim=1))
        p3 = self.reduce_c2(torch.cat([self.up(p4), c2], dim=1))
        return p3, p4, p5


# =========================================================
# GhostHead (detection head)
# =========================================================
class GhostHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = GhostModule(in_ch, 32)
        self.pred = nn.Conv2d(32, num_classes + 4, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pred(x)
        return x


# =========================================================
# YOLOv8Hybrid (single or multi-head)
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
            # project p4/p5 -> p3 channels before summation
            self.proj_p4 = nn.Conv2d(48, 32, 1, bias=False)
            self.bn_p4 = nn.BatchNorm2d(32)
            self.proj_p5 = nn.Conv2d(64, 32, 1, bias=False)
            self.bn_p5 = nn.BatchNorm2d(32)
            self.head = GhostHead(32, num_classes)

    def forward(self, x):
        p3, p4, p5 = self.neck(self.backbone(x))

        if self.multi_head:
            out_p3 = self.head_p3(p3)
            out_p4 = self.head_p4(p4)
            out_p5 = self.head_p5(p5)
            return [out_p3, out_p4, out_p5]
        else:
            p4_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
            p5_up = F.interpolate(p5, size=p3.shape[2:], mode='nearest')

            p4_proj = self.bn_p4(self.proj_p4(p4_up))
            p5_proj = self.bn_p5(self.proj_p5(p5_up))

            fused = p3 + p4_proj + p5_proj
            return self.head(fused)


# =========================================================
# Quick test run
# =========================================================
if __name__ == "__main__":
    device = torch.device("cpu")
    x = torch.randn(1, 3, 640, 640).to(device)

    # Single-head test
    model_single = YOLOv8Hybrid(num_classes=20, multi_head=False).to(device)
    model_single.train()
    y_train = model_single(x)
    print("Single-head train:", y_train.shape)

    # Important: to trigger fusion, set to eval (fusion occurs once lazily in forward)
    model_single.eval()
    y_eval = model_single(x)
    print("Single-head infer:", y_eval.shape)

    # Multi-head test
    model_multi = YOLOv8Hybrid(num_classes=20, multi_head=True).to(device)
    model_multi.train()
    y_train_multi = model_multi(x)
    print("Multi-head train:", [o.shape for o in y_train_multi])
    model_multi.eval()
    y_eval_multi = model_multi(x)
    print("Multi-head infer:", [o.shape for o in y_eval_multi])
