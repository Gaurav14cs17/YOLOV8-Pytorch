# qmodel/__init__.py
"""Quantized and Pruned YOLOv8 Models.

This package provides model compression techniques for YOLOv8:
- quantization: Quantization-aware training (QAT)
- pruning: Weight pruning + quantization
"""

# Quantization-only model
from .quantization import (
    YOLOv8 as QYOLOv8,
    QuantizedYOLO,
    create_yolo_variant as create_quantized_variant
)

# Pruning + Quantization model
from .pruning import (
    YOLOv8 as PQYOLOv8,
    PrunableBlock,
    create_yolo_variant as create_pruned_variant
)

__all__ = [
    # Quantization
    'QYOLOv8',
    'QuantizedYOLO',
    'create_quantized_variant',
    # Pruning + Quantization
    'PQYOLOv8',
    'PrunableBlock',
    'create_pruned_variant',
]

