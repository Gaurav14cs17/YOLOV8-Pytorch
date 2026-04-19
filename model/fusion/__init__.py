# fusion/__init__.py
"""Layer fusion utilities for inference optimization."""

from .fuse_layer import fuse_conv_bn, FuseLayer

__all__ = ['fuse_conv_bn', 'FuseLayer']

