import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p if p is not None else k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = Conv(ch, ch, 1)
        self.conv2 = Conv(ch, ch, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSP(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks):
        super().__init__()
        self.split_conv = Conv(in_ch, out_ch // 2, 1)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(out_ch // 2) for _ in range(n_blocks)])
        self.final_conv = Conv(out_ch, out_ch, 1)

    def forward(self, x):
        y1 = self.residual_blocks(self.split_conv(x))
        y2 = self.split_conv(x)
        return self.final_conv(torch.cat((y1, y2), dim=1))

class SPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        ])
        self.conv = Conv(in_ch * 4, out_ch, 1)

    def forward(self, x):
        return self.conv(torch.cat([x] + [m(x) for m in self.m], dim=1))
