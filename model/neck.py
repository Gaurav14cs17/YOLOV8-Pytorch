import torch
import torch.nn as nn
from .blocks import Conv, CSP

class DarkFPN(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.reduce_conv1 = Conv(width[5], width[4], 1)
        self.csp1 = CSP(width[4] * 2, width[4], depth[0])

        self.reduce_conv2 = Conv(width[4], width[3], 1)
        self.csp2 = CSP(width[3] * 2, width[3], depth[0])

        self.down_conv1 = Conv(width[3], width[3], 3, 2)
        self.csp3 = CSP(width[3] * 2, width[4], depth[0])

        self.down_conv2 = Conv(width[4], width[4], 3, 2)
        self.csp4 = CSP(width[4] * 2, width[5], depth[0])

    def forward(self, s3, s4, s5):
        p5_up = nn.functional.interpolate(self.reduce_conv1(s5), scale_factor=2)
        p4 = self.csp1(torch.cat([p5_up, s4], dim=1))

        p4_up = nn.functional.interpolate(self.reduce_conv2(p4), scale_factor=2)
        p3 = self.csp2(torch.cat([p4_up, s3], dim=1))

        p3_down = self.down_conv1(p3)
        p4_out = self.csp3(torch.cat([p3_down, p4], dim=1))

        p4_down = self.down_conv2(p4_out)
        p5_out = self.csp4(torch.cat([p4_down, s5], dim=1))

        return p3, p4_out, p5_out
