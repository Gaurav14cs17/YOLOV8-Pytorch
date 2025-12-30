"""Pruning utilities for model compression.

Supports L1 unstructured and global pruning.
"""

import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Optional


class PrunableMixin:
    """Mixin class providing pruning methods for modules."""

    @staticmethod
    def apply_conv_pruning(module: nn.Module, amount: float = 0.2) -> None:
        """Apply L1 unstructured pruning to Conv2d layers.

        Args:
            module: Module to prune
            amount: Fraction of weights to prune
        """
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make permanent

    @staticmethod
    def apply_global_pruning(model: nn.Module, amount: float = 0.2) -> None:
        """Apply global L1 pruning across all Conv2d layers.

        Args:
            model: Model to prune
            amount: Fraction of weights to prune globally
        """
        parameters_to_prune = [
            (module, 'weight') for module in model.modules()
            if isinstance(module, nn.Conv2d)
        ]

        if not parameters_to_prune:
            return

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )

        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')

    @staticmethod
    def get_sparsity(model: nn.Module) -> float:
        """Calculate model weight sparsity.

        Args:
            model: Model to analyze

        Returns:
            Fraction of zero weights
        """
        total_params = 0
        zero_params = 0

        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class PruningConfig:
    """Configuration for pruning operations."""

    def __init__(self,
                 prune_conv: bool = False,
                 global_prune: bool = False,
                 prune_on_init: bool = False,
                 amount: float = 0.2):
        """Initialize pruning config.

        Args:
            prune_conv: Whether to prune individual conv layers
            global_prune: Whether to use global pruning
            prune_on_init: Whether to prune on model creation
            amount: Pruning fraction
        """
        self.prune_conv = prune_conv
        self.global_prune = global_prune
        self.prune_on_init = prune_on_init
        self.amount = amount

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'prune_conv': self.prune_conv,
            'global_prune': self.global_prune,
            'prune_on_init': self.prune_on_init,
            'amount': self.amount
        }

    @classmethod
    def from_dict(cls, config: Optional[Dict]) -> Optional['PruningConfig']:
        """Create from dictionary."""
        if config is None:
            return None
        return cls(**config)

