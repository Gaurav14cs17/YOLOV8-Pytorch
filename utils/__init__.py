"""Utils package for YOLOv8-RepNeXt training and inference.

Organized modules:
- setup: Environment setup utilities
- boxes: Bounding box operations
- iou: IoU computation (IoU, GIoU, DIoU, CIoU)
- assigner: Task-Aligned label assignment
- losses: Individual loss functions
- loss: Main ComputeLoss class
- metrics: Detection metrics
- training: Training utilities (EMA, etc.)
"""

# Setup utilities
from .setup import setup_seed, setup_multi_processes

# Box utilities
from .boxes import (
    scale,
    make_anchors,
    box_iou,
    wh2xy,
    non_max_suppression,
    BoxUtils,
)

# IoU computation
from .iou import (
    IoUCalculator,
    compute_iou,
)

# Label assignment
from .assigner import TaskAlignedAssigner

# Individual loss functions
from .losses import (
    ClassificationLoss,
    BoxLoss,
    DFLoss,
    QualityFocalLoss,
    VarifocalLoss,
)

# Main loss computation
from .loss import ComputeLoss

# Metrics
from .metrics import (
    smooth,
    compute_ap,
    MetricsCalculator,
)

# Training utilities
from .training import (
    strip_optimizer,
    clip_gradients,
    EMA,
    AverageMeter,
    LRScheduler,
)


__all__ = [
    # Setup
    'setup_seed',
    'setup_multi_processes',
    # Boxes
    'scale',
    'make_anchors',
    'box_iou',
    'wh2xy',
    'non_max_suppression',
    'BoxUtils',
    # IoU
    'IoUCalculator',
    'compute_iou',
    # Assigner
    'TaskAlignedAssigner',
    # Losses
    'ClassificationLoss',
    'BoxLoss',
    'DFLoss',
    'QualityFocalLoss',
    'VarifocalLoss',
    'ComputeLoss',
    # Metrics
    'smooth',
    'compute_ap',
    'MetricsCalculator',
    # Training
    'strip_optimizer',
    'clip_gradients',
    'EMA',
    'AverageMeter',
    'LRScheduler',
]
