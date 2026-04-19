# transforms/transforms.py
"""Coordinate and image transformation utilities."""

import cv2
import numpy
import random


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] absolute.
    
    Args:
        x: numpy array of shape (N, 4) with normalized [cx, cy, w, h] boxes
        w: target image width
        h: target image height
        pad_w: horizontal padding offset
        pad_h: vertical padding offset
    
    Returns:
        numpy array of shape (N, 4) with absolute [x1, y1, x2, y2] boxes
    """
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w=640, h=640):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized.
    
    Warning: performs inplace clip.
    
    Args:
        x: numpy array of shape (N, 4) with absolute [x1, y1, x2, y2] boxes
        w: image width for normalization
        h: image height for normalization
    
    Returns:
        numpy array of shape (N, 4) with normalized [cx, cy, w, h] boxes
    """
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    """Get random resampling interpolation method.
    
    Returns:
        OpenCV interpolation flag
    """
    choices = (
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4
    )
    return random.choice(seq=choices)


def resize(image, input_size, augment):
    """Resize and pad image while meeting stride-multiple constraints.
    
    Args:
        image: input image (H, W, C)
        input_size: target size (single int for square)
        augment: whether to use random resampling
    
    Returns:
        Tuple of (padded_image, (ratio_w, ratio_h), (pad_w, pad_h))
    """
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(
            image,
            dsize=pad,
            interpolation=resample() if augment else cv2.INTER_LINEAR
        )
    
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    
    return image, (r, r), (w, h)


def candidates(box1, box2):
    """Filter candidate boxes based on size and aspect ratio.
    
    Args:
        box1: original boxes (4, N) - x1, y1, x2, y2
        box2: transformed boxes (4, N)
    
    Returns:
        Boolean mask of valid candidates
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)

