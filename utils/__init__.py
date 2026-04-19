# __init__.py
"""YOLO utilities package."""

# Environment utilities (seed, multi-processing)
from .env import setup_seed, setup_multi_processes

# Bounding box operations and NMS
from .bbox import scale, make_anchors, box_iou, wh2xy, non_max_suppression

# Model utilities (optimizer)
from .model_utils import strip_optimizer, clip_gradients

# Metrics
from .metrics import smooth, compute_ap

# Exponential Moving Average
from .ema import EMA

# Training meters
from .meters import AverageMeter

# Loss computation
from .loss import ComputeLoss

# Task-aligned assigner
from .assigner import TaskAlignedAssigner


__all__ = [
    # Environment
    'setup_seed',
    'setup_multi_processes',
    # Box operations
    'scale',
    'make_anchors',
    'box_iou',
    'wh2xy',
    # NMS
    'non_max_suppression',
    # Model utils
    'strip_optimizer',
    'clip_gradients',
    # Metrics
    'smooth',
    'compute_ap',
    # EMA
    'EMA',
    # Meters
    'AverageMeter',
    # Loss
    'ComputeLoss',
    # Assigner
    'TaskAlignedAssigner',
]
