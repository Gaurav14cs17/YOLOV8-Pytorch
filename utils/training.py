"""Training utilities for model optimization."""

import copy
import math
import torch
import torch.nn as nn


def strip_optimizer(filename: str):
    """Strip optimizer from saved checkpoint for deployment.

    Args:
        filename: Path to checkpoint file
    """
    x = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, filename)


def clip_gradients(model: nn.Module, max_norm: float = 10.0):
    """Clip model gradients to prevent explosion.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


class EMA:
    """Exponential Moving Average for model weights.

    Keeps a moving average of model parameters for better generalization.
    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 tau: int = 2000, updates: int = 0):
        """Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate
            tau: Decay time constant
            updates: Initial update count
        """
        self.ema = copy.deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        """Update EMA weights.

        Args:
            model: Current model to update from
        """
        if hasattr(model, 'module'):
            model = model.module

        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    """Computes and stores running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, value: float, n: int = 1):
        """Update running average.

        Args:
            value: New value to add
            n: Number of samples (for weighted average)
        """
        if not math.isnan(float(value)):
            self.num += n
            self.sum += value * n
            self.avg = self.sum / self.num


class LRScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, epochs: int,
                 lrf: float = 0.01, warmup_epochs: int = 3,
                 warmup_bias_lr: float = 0.1, warmup_momentum: float = 0.8):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            epochs: Total training epochs
            lrf: Final learning rate factor
            warmup_epochs: Number of warmup epochs
            warmup_bias_lr: Initial bias learning rate
            warmup_momentum: Initial momentum during warmup
        """
        self.optimizer = optimizer
        self.epochs = epochs
        self.lrf = lrf
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for epoch.

        Args:
            epoch: Current epoch

        Returns:
            Learning rate multiplier
        """
        return (1 - epoch / self.epochs) * (1.0 - self.lrf) + self.lrf

    def step(self, epoch: int):
        """Update learning rate.

        Args:
            epoch: Current epoch
        """
        lr_mult = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_mult

