# yolo.py
import torch
from backbone import Backbone
from neck import Neck
from head import Head
from fuse_layer import FuseLayer

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.backbone = Backbone(width, depth)
        self.neck = Neck(width, depth)
        self.head = Head(num_classes, (width[3], width[4], width[5]))

        img_dummy = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(img_dummy)
            feats = self.neck(feats)
        self.head.stride = torch.tensor([256 / f.shape[-2] for f in feats])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(list(feats))

    def fuse(self):
        FuseLayer.fuse_module(self)
        return self

# # Factory functions for different YOLO versions (optional)
# def yolo_v8_n(num_classes=80):
#     depth = [1, 2, 2]
#     width = [3, 16, 32, 64, 128, 256]
#     return YOLO(width, depth, num_classes)

# def yolo_v8_s(num_classes=80):
#     depth = [1, 2, 2]
#     width = [3, 32, 64, 128, 256, 512]
#     return YOLO(width, depth, num_classes)
