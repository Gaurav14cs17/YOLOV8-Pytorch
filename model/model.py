import math
import torch
from utils.util import make_anchors


def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class ConvBlock(torch.nn.Module):
    """ Convolution -> BatchNorm -> SiLU """
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p, d), d, g, False)
        self.bn = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        self.act = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class ResidualBlock(torch.nn.Module):
    """Simple residual: two ConvBlocks with optional add"""
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_residual = add
        self.blocks = torch.nn.Sequential(
            ConvBlock(ch, ch, 3),
            ConvBlock(ch, ch, 3)
        )

    def forward(self, x):
        return self.blocks(x) + x if self.add_residual else self.blocks(x)


class CSPBlock(torch.nn.Module):
    """Cross Stage Partial block"""
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        # split convs
        self.split_conv = ConvBlock(in_ch, out_ch // 2)
        self.path_conv = ConvBlock(in_ch, out_ch // 2)
        self.final_conv = ConvBlock((2 + n) * out_ch // 2, out_ch)
        # residual blocks list
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(out_ch // 2, add) for _ in range(n)])

    def forward(self, x):
        left = self.split_conv(x)
        right = self.path_conv(x)
        # run residuals sequentially on the 'right' path
        out_list = [left, right]
        out_list.extend(block(out_list[-1]) for block in self.residual_blocks)
        return self.final_conv(torch.cat(out_list, dim=1))


class SPPBlock(torch.nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.reduce_conv = ConvBlock(in_ch, in_ch // 2)
        self.final_conv = ConvBlock(in_ch * 2, out_ch)
        self.maxpool = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.reduce_conv(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.final_conv(torch.cat([x, y1, y2, y3], 1))


class Backbone(torch.nn.Module):
    """DarkNet-like backbone composed of 5 stages"""
    def __init__(self, width, depth):
        super().__init__()
        self.stage1 = torch.nn.Sequential(ConvBlock(width[0], width[1], 3, 2))
        self.stage2 = torch.nn.Sequential(
            ConvBlock(width[1], width[2], 3, 2),
            CSPBlock(width[2], width[2], depth[0])
        )
        self.stage3 = torch.nn.Sequential(
            ConvBlock(width[2], width[3], 3, 2),
            CSPBlock(width[3], width[3], depth[1])
        )
        self.stage4 = torch.nn.Sequential(
            ConvBlock(width[3], width[4], 3, 2),
            CSPBlock(width[4], width[4], depth[2])
        )
        self.stage5 = torch.nn.Sequential(
            ConvBlock(width[4], width[5], 3, 2),
            CSPBlock(width[5], width[5], depth[0]),
            SPPBlock(width[5], width[5])
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        # return three features (small, medium, large) to be used by neck
        return s3, s4, s5


class Neck(torch.nn.Module):
    """Feature pyramid (DarkFPN-like)"""
    def __init__(self, width, depth):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # merge top (large) into mid
        self.top_to_mid = CSPBlock(width[4] + width[5], width[4], depth[0], add=False)
        # merge mid (after receive top) into small
        self.mid_to_small = CSPBlock(width[3] + width[4], width[3], depth[0], add=False)

        # downsample operations and fuses to reconstruct top path
        self.downsample_mid = ConvBlock(width[3], width[3], 3, 2)
        self.mid_fuse = CSPBlock(width[3] + width[4], width[4], depth[0], add=False)

        self.downsample_top = ConvBlock(width[4], width[4], 3, 2)
        self.top_fuse = CSPBlock(width[4] + width[5], width[5], depth[0], add=False)

    def forward(self, feats):
        feat_small, feat_mid, feat_large = feats  # s3, s4, s5
        # propagate top (large) -> mid
        merged_mid = self.top_to_mid(torch.cat([self.upsample(feat_large), feat_mid], dim=1))
        # propagate mid (merged) -> small
        merged_small = self.mid_to_small(torch.cat([self.upsample(merged_mid), feat_small], dim=1))
        # fuse paths back upwards
        mid_down = self.mid_fuse(torch.cat([self.downsample_mid(merged_small), merged_mid], dim=1))
        top_down = self.top_fuse(torch.cat([self.downsample_top(mid_down), feat_large], dim=1))
        # return features ordered from small->mid->top (same ordering as Head expects)
        return merged_small, mid_down, top_down


class DFL(torch.nn.Module):
    """Distribution Focal Loss integral module for decoding discrete bounding outputs"""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        # conv maps discrete distribution to a scalar (using fixed weights 0..ch-1)
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = x

    def forward(self, x):
        # x expected shape: (batch, ch*4, n_locations)
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)  # -> (b, ch, 4, a) then -> (b, 4, ch, a) transposed
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, num_classes=80, filters=()):
        super().__init__()
        self.dfl_channels = 16
        self.num_classes = num_classes
        self.num_layers = len(filters)  # number of feature layers
        self.outputs_per_anchor = num_classes + self.dfl_channels * 4
        self.stride = torch.zeros(self.num_layers)

        # channel choices
        c1 = max(filters[0], self.num_classes)
        c2 = max(filters[0] // 4, self.dfl_channels * 4)

        self.dfl = DFL(self.dfl_channels)

        # classification and bbox branches per feature map
        self.cls_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvBlock(f, c1, 3),
                ConvBlock(c1, c1, 3),
                torch.nn.Conv2d(c1, self.num_classes, 1)
            ) for f in filters
        ])

        self.bbox_branch = torch.nn.ModuleList([
            torch.nn.Sequential(
                ConvBlock(f, c2, 3),
                ConvBlock(c2, c2, 3),
                torch.nn.Conv2d(c2, 4 * self.dfl_channels, 1)
            ) for f in filters
        ])

    def forward(self, feats):
        # feats is a list/tuple of feature tensors [f_small, f_mid, f_top]
        for i in range(self.num_layers):
            feats[i] = torch.cat((self.bbox_branch[i](feats[i]), self.cls_branch[i](feats[i])), dim=1)

        if self.training:
            return feats

        # build anchors and strides from the feature maps
        self.anchors, self.strides = [t.transpose(0, 1) for t in make_anchors(feats, self.stride, 0.5)]

        # flatten and concatenate all layer outputs
        batch = feats[0].shape[0]
        concat = torch.cat([f.view(batch, self.outputs_per_anchor, -1) for f in feats], dim=2)

        box_preds, class_preds = concat.split((self.dfl_channels * 4, self.num_classes), dim=1)
        lt, rb = torch.split(self.dfl(box_preds), 2, dim=1)  # left-top and right-bottom offsets

        lt = self.anchors.unsqueeze(0) - lt
        rb = self.anchors.unsqueeze(0) + rb
        boxes = torch.cat(((lt + rb) / 2, rb - lt), dim=1)

        return torch.cat((boxes * self.strides, class_preds.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases for stability (requires stride to be set)
        for bbox_module, cls_module, s in zip(self.bbox_branch, self.cls_branch, self.stride):
            # bbox branch final conv bias (set to 1 for stability)
            bbox_module[-1].bias.data[:] = 1.0
            # classification bias: priors tuned based on image scale and class count
            cls_module[-1].bias.data[:self.num_classes] = math.log(5 / self.num_classes / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    """Full model wrapper: backbone -> neck -> head"""
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth)
        self.neck = Neck(width, depth)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        # compute strides using dummy input, without calling head/forward recursively
        img_dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(img_dummy)
            feats = self.neck(feats)
        # strides are 256 / spatial_size (height) per feature level
        self.head.stride = torch.tensor([256 / f.shape[-2] for f in feats])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        # head expects a list
        return self.head(list(feats))

    def fuse(self):
        # fuse ConvBlock conv + bn into a single conv for inference speedup
        for m in self.modules():
            if type(m) is ConvBlock and hasattr(m, 'bn'):
                m.conv = fuse_conv(m.conv, m.bn)
                m.forward = m.fuse_forward
                delattr(m, 'bn')
        return self


# Model factory functions (variants)
def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)

def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)

def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)

def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)

def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)
