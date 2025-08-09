import torch
import torch.nn as nn
from .backbone import DarkNet
from .neck import DarkFPN
from .head import YOLOHead

class YOLO(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        width = [3, 64, 128, 256, 512, 1024]
        depth = [1, 3, 3, 1]

        self.backbone = DarkNet(width, depth)
        self.neck = DarkFPN(width, depth)
        self.head = YOLOHead(num_classes, [256, 512, 1024])

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck(c3, c4, c5)
        outputs = self.head([p3, p4, p5])
        return outputs
