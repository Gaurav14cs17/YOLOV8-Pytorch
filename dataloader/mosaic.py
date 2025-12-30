"""Mosaic data augmentation.

Creates training images by combining 4 images into one.
"""

import random
import numpy as np

from .transforms import wh2xy, GeometricTransform


class MosaicLoader:
    """Mosaic augmentation for object detection.

    Combines 4 images into a single training sample.
    """

    def __init__(self, input_size: int):
        """Initialize mosaic loader.

        Args:
            input_size: Target image size
        """
        self.input_size = input_size

    def create_mosaic(self, indices: list, load_image_fn: callable,
                      labels_list: list, params: dict) -> tuple:
        """Create mosaic from 4 images.

        Args:
            indices: List of 4 image indices
            load_image_fn: Function to load image by index
            labels_list: List of all labels
            params: Augmentation parameters

        Returns:
            Tuple of (mosaic_image, mosaic_labels)
        """
        # Initialize 4x size canvas
        mosaic_image = np.full(
            (self.input_size * 2, self.input_size * 2, 3),
            114, dtype=np.uint8
        )
        mosaic_labels = []

        # Random mosaic center
        border = [-self.input_size // 2, -self.input_size // 2]
        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        # Shuffle indices
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            # Load image
            image, _ = load_image_fn(idx)
            h, w = image.shape[:2]

            # Calculate placement coordinates
            coords = self._get_placement_coords(i, xc, yc, w, h)
            x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = coords

            # Place image in mosaic
            mosaic_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # Adjust labels
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            label = labels_list[idx].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], w, h, pad_w, pad_h)
            mosaic_labels.append(label)

        # Concatenate and clip labels
        mosaic_labels = np.concatenate(mosaic_labels, 0)
        for x in mosaic_labels[:, 1:]:
            np.clip(x, 0, 2 * self.input_size, out=x)

        # Apply random perspective
        mosaic_image, mosaic_labels = GeometricTransform.random_perspective(
            mosaic_image, mosaic_labels, params, border
        )

        return mosaic_image, mosaic_labels

    def _get_placement_coords(self, quadrant: int, xc: int, yc: int,
                              w: int, h: int) -> tuple:
        """Calculate placement coordinates for each quadrant.

        Args:
            quadrant: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
            xc: Mosaic center x
            yc: Mosaic center y
            w: Image width
            h: Image height

        Returns:
            Tuple of (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)
        """
        if quadrant == 0:  # Top left
            x1a = max(xc - w, 0)
            y1a = max(yc - h, 0)
            x2a = xc
            y2a = yc
            x1b = w - (x2a - x1a)
            y1b = h - (y2a - y1a)
            x2b = w
            y2b = h

        elif quadrant == 1:  # Top right
            x1a = xc
            y1a = max(yc - h, 0)
            x2a = min(xc + w, self.input_size * 2)
            y2a = yc
            x1b = 0
            y1b = h - (y2a - y1a)
            x2b = min(w, x2a - x1a)
            y2b = h

        elif quadrant == 2:  # Bottom left
            x1a = max(xc - w, 0)
            y1a = yc
            x2a = xc
            y2a = min(self.input_size * 2, yc + h)
            x1b = w - (x2a - x1a)
            y1b = 0
            x2b = w
            y2b = min(y2a - y1a, h)

        else:  # Bottom right
            x1a = xc
            y1a = yc
            x2a = min(xc + w, self.input_size * 2)
            y2a = min(self.input_size * 2, yc + h)
            x1b = 0
            y1b = 0
            x2b = min(w, x2a - x1a)
            y2b = min(y2a - y1a, h)

        return x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b

