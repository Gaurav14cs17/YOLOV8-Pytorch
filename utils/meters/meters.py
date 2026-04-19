# meters.py
"""Training meters and trackers."""

import math


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        """Initialize the meter."""
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        """
        Update the meter with a new value.
        
        Args:
            v: Value to add
            n: Number of samples
        """
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
