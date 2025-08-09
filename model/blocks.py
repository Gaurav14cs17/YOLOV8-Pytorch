import torch
import torch.nn as nn

# Conv+BN fusion function
def fuse_conv(conv, norm):
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.running_var + norm.eps)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)
    return fused_conv


class Conv(nn.Module):
    """Standard Conv + BN + Activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuse(self):
        self.conv = fuse_conv(self.conv, self.bn)
        self.bn = nn.Identity()
        return self


class FuseBlock(nn.Module):
    """Feature map fusion block (adds and convs two feature maps)"""
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1, c2, k=1)

    def forward(self, x1, x2):
        return self.conv(x1 + x2)

    def fuse(self):
        self.conv.fuse()
        return self
