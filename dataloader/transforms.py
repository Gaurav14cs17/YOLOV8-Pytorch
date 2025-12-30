"""Image transformation utilities for data augmentation.

Contains coordinate conversions and geometric transforms.
"""

import math
import random
import cv2
import numpy as np


def wh2xy(x: np.ndarray, w: int = 640, h: int = 640,
          pad_w: int = 0, pad_h: int = 0) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2].

    Args:
        x: Boxes in center format (N, 4)
        w: Image width
        h: Image height
        pad_w: X padding offset
        pad_h: Y padding offset

    Returns:
        Boxes in corner format (N, 4)
    """
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # x1
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # y1
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # x2
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # y2
    return y


def xy2wh(x: np.ndarray, w: int = 640, h: int = 640) -> np.ndarray:
    """Convert pixel [x1, y1, x2, y2] to normalized [cx, cy, w, h].

    Args:
        x: Boxes in corner format (N, 4)
        w: Image width
        h: Image height

    Returns:
        Boxes in center format (N, 4)
    """
    # Clip to image bounds
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1e-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1e-3)

    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # cx
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # cy
    y[:, 2] = (x[:, 2] - x[:, 0]) / w        # w
    y[:, 3] = (x[:, 3] - x[:, 1]) / h        # h
    return y


def resample() -> int:
    """Get random interpolation method for image resizing."""
    choices = (
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LINEAR,
        cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4
    )
    return random.choice(choices)


class Resizer:
    """Image resizing with letterbox padding."""

    @staticmethod
    def resize(image: np.ndarray, input_size: int,
               augment: bool = False) -> tuple:
        """Resize and pad image while meeting stride-multiple constraints.

        Args:
            image: Input image (H, W, C)
            input_size: Target size
            augment: Whether to use random interpolation

        Returns:
            Tuple of (resized_image, (ratio_x, ratio_y), (pad_w, pad_h))
        """
        shape = image.shape[:2]  # [height, width]

        # Scale ratio (new / old)
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:  # Only scale down for validation
            r = min(r, 1.0)

        # Compute padding
        new_size = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w = (input_size - new_size[0]) / 2
        pad_h = (input_size - new_size[1]) / 2

        # Resize if needed
        if shape[::-1] != new_size:
            interp = resample() if augment else cv2.INTER_LINEAR
            image = cv2.resize(image, dsize=new_size, interpolation=interp)

        # Add border padding
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return image, (r, r), (pad_w, pad_h)


class GeometricTransform:
    """Geometric transformations for data augmentation."""

    @staticmethod
    def candidates(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
        """Filter valid transformed boxes.

        Args:
            box1: Original boxes (4, N)
            box2: Transformed boxes (4, N)

        Returns:
            Boolean mask of valid boxes
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)

    @staticmethod
    def random_perspective(image: np.ndarray, targets: np.ndarray,
                          params: dict, border: tuple = (0, 0)) -> tuple:
        """Apply random perspective transformation.

        Args:
            image: Input image
            targets: Labels (N, 5) [cls, x1, y1, x2, y2]
            params: Augmentation parameters
            border: Border offset

        Returns:
            Tuple of (transformed_image, transformed_targets)
        """
        h = image.shape[0] + border[0] * 2
        w = image.shape[1] + border[1] * 2

        # Center
        center = np.eye(3)
        center[0, 2] = -image.shape[1] / 2
        center[1, 2] = -image.shape[0] / 2

        # Perspective (identity)
        perspective = np.eye(3)

        # Rotation and Scale
        rotate = np.eye(3)
        a = random.uniform(-params['degrees'], params['degrees'])
        s = random.uniform(1 - params['scale'], 1 + params['scale'])
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

        # Translation
        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
        translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

        # Combined transformation matrix
        matrix = translate @ shear @ rotate @ perspective @ center

        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ matrix.T
            xy = xy[:, :2].reshape(n, 8)

            # Create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # Clip to image bounds
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

            # Filter valid candidates
            indices = GeometricTransform.candidates(box1=targets[:, 1:5].T * s, box2=new.T)
            targets = targets[indices]
            targets[:, 1:5] = new[indices]

        return image, targets

