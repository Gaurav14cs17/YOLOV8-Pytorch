"""Setup utilities for training environment."""

import random
import numpy
import torch


def setup_seed(seed: int = 0):
    """Setup random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """Setup multi-processing environment variables.

    Configures:
    - Multiprocess start method (fork on non-Windows)
    - OpenCV threading
    - OMP and MKL thread counts
    """
    import cv2
    from os import environ
    from platform import system

    # Set multiprocess start method as `fork` to speed up training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # Disable opencv multithreading to avoid system overload
    cv2.setNumThreads(0)

    # Setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # Setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'

