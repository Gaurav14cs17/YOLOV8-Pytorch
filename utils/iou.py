"""IoU (Intersection over Union) computations for bounding boxes.

Supports multiple IoU variants:
- IoU: Standard Intersection over Union
- GIoU: Generalized IoU
- DIoU: Distance IoU
- CIoU: Complete IoU (default for YOLO)
"""

import math
import torch
from typing import Tuple


class IoUCalculator:
    """Calculator for various IoU metrics between bounding boxes.

    All methods expect boxes in xyxy format: (x1, y1, x2, y2)
    """

    @staticmethod
    def compute_iou(box1: torch.Tensor, box2: torch.Tensor,
                    eps: float = 1e-7) -> torch.Tensor:
        """Compute standard IoU.

        Args:
            box1: First set of boxes (..., 4)
            box2: Second set of boxes (..., 4)
            eps: Small value to avoid division by zero

        Returns:
            IoU values
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        intersection = inter_w * inter_h

        # Areas
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        # Union
        union = area1 + area2 - intersection + eps

        return intersection / union

    @staticmethod
    def compute_giou(box1: torch.Tensor, box2: torch.Tensor,
                     eps: float = 1e-7) -> torch.Tensor:
        """Compute Generalized IoU (GIoU).

        Reference: https://arxiv.org/abs/1902.09630

        Args:
            box1: First set of boxes (..., 4)
            box2: Second set of boxes (..., 4)
            eps: Small value to avoid division by zero

        Returns:
            GIoU values (can be negative)
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        intersection = inter_w * inter_h

        # Areas
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - intersection + eps

        # IoU
        iou = intersection / union

        # Enclosing box
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + eps

        # GIoU
        return iou - (enclose_area - union) / enclose_area

    @staticmethod
    def compute_diou(box1: torch.Tensor, box2: torch.Tensor,
                     eps: float = 1e-7) -> torch.Tensor:
        """Compute Distance IoU (DIoU).

        Reference: https://arxiv.org/abs/1911.08287

        Args:
            box1: First set of boxes (..., 4)
            box2: Second set of boxes (..., 4)
            eps: Small value to avoid division by zero

        Returns:
            DIoU values
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        intersection = inter_w * inter_h

        # Areas
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - intersection + eps

        # IoU
        iou = intersection / union

        # Center distance
        center1_x = (b1_x1 + b1_x2) / 2
        center1_y = (b1_y1 + b1_y2) / 2
        center2_x = (b2_x1 + b2_x2) / 2
        center2_y = (b2_y1 + b2_y2) / 2
        center_dist_sq = (center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2

        # Enclosing box diagonal
        enclose_x1 = torch.min(b1_x1, b2_x1)
        enclose_y1 = torch.min(b1_y1, b2_y1)
        enclose_x2 = torch.max(b1_x2, b2_x2)
        enclose_y2 = torch.max(b1_y2, b2_y2)
        diagonal_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

        # DIoU
        return iou - center_dist_sq / diagonal_sq

    @staticmethod
    def compute_ciou(box1: torch.Tensor, box2: torch.Tensor,
                     eps: float = 1e-7) -> torch.Tensor:
        """Compute Complete IoU (CIoU).

        Reference: https://arxiv.org/abs/1911.08287v1

        Args:
            box1: First set of boxes (..., 4)
            box2: Second set of boxes (..., 4)
            eps: Small value to avoid division by zero

        Returns:
            CIoU values
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        # Dimensions
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection
        inter_x1 = b1_x1.maximum(b2_x1)
        inter_y1 = b1_y1.maximum(b2_y1)
        inter_x2 = b1_x2.minimum(b2_x2)
        inter_y2 = b1_y2.minimum(b2_y2)
        intersection = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Union and IoU
        union = w1 * h1 + w2 * h2 - intersection + eps
        iou = intersection / union

        # Enclosing box
        enclose_w = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        enclose_h = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        diagonal_sq = enclose_w ** 2 + enclose_h ** 2 + eps

        # Center distance
        center_dist_sq = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                          (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

        # Aspect ratio penalty
        v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)

        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        # CIoU = IoU - distance_penalty - aspect_ratio_penalty
        return iou - (center_dist_sq / diagonal_sq + v * alpha)


def compute_iou(box1: torch.Tensor, box2: torch.Tensor,
                iou_type: str = 'ciou', eps: float = 1e-7) -> torch.Tensor:
    """Compute IoU with specified type.

    Args:
        box1: First set of boxes (..., 4) in xyxy format
        box2: Second set of boxes (..., 4) in xyxy format
        iou_type: Type of IoU ('iou', 'giou', 'diou', 'ciou')
        eps: Small value to avoid division by zero

    Returns:
        IoU values
    """
    calculator = IoUCalculator()

    if iou_type == 'iou':
        return calculator.compute_iou(box1, box2, eps)
    elif iou_type == 'giou':
        return calculator.compute_giou(box1, box2, eps)
    elif iou_type == 'diou':
        return calculator.compute_diou(box1, box2, eps)
    elif iou_type == 'ciou':
        return calculator.compute_ciou(box1, box2, eps)
    else:
        raise ValueError(f"Unknown IoU type: {iou_type}")

