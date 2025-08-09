import torch
import torch.nn as nn
from .blocks import Conv

class YOLOHead(nn.Module):
    def __init__(self, num_classes, width):
        super().__init__()
        self.bbox_branch = nn.ModuleList([nn.Conv2d(w, 4, 1) for w in width])
        self.cls_branch = nn.ModuleList([nn.Conv2d(w, num_classes, 1) for w in width])

    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            bbox = self.bbox_branch[i](feat)
            cls = self.cls_branch[i](feat)
            outputs.append(torch.cat([bbox, cls], dim=1))
        return outputs
