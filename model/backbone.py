import torch
import torch.nn as nn
from .blocks import Conv, CSP, SPP

class DarkNet(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.stage1 = Conv(width[0], width[1], 3, 2)
        self.stage2 = nn.Sequential(
            Conv(width[1], width[2], 3, 2),
            CSP(width[2], width[2], depth[0])
        )
        self.stage3 = nn.Sequential(
            Conv(width[2], width[3], 3, 2),
            CSP(width[3], width[3], depth[1])
        )
        self.stage4 = nn.Sequential(
            Conv(width[3], width[4], 3, 2),
            CSP(width[4], width[4], depth[2])
        )
        self.stage5 = nn.Sequential(
            Conv(width[4], width[5], 3, 2),
            CSP(width[5], width[5], depth[0]),
            SPP(width[5], width[5])
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s3, s4, s5
