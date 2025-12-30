"""Image augmentation utilities.

Contains color space augmentations and optional Albumentations wrapper.
"""

import random
import cv2
import numpy as np


class HSVAugmentation:
    """HSV color space augmentation."""

    @staticmethod
    def augment(image: np.ndarray, params: dict) -> None:
        """Apply HSV color-space augmentation in-place.

        Args:
            image: Input image (BGR format)
            params: Dict with 'hsv_h', 'hsv_s', 'hsv_v' keys
        """
        h_gain = params['hsv_h']
        s_gain = params['hsv_s']
        v_gain = params['hsv_v']

        # Random gains
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1

        # Split HSV channels
        h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

        # Create lookup tables
        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype('uint8')
        lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
        lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

        # Apply LUTs and merge
        im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)


class FlipAugmentation:
    """Flip augmentations (horizontal and vertical)."""

    @staticmethod
    def flip_ud(image: np.ndarray, labels: np.ndarray) -> tuple:
        """Flip image and labels vertically.

        Args:
            image: Input image
            labels: Labels (N, 5) [cls, cx, cy, w, h]

        Returns:
            Tuple of (flipped_image, flipped_labels)
        """
        image = np.flipud(image)
        if len(labels):
            labels[:, 2] = 1 - labels[:, 2]  # Flip y center
        return image, labels

    @staticmethod
    def flip_lr(image: np.ndarray, labels: np.ndarray) -> tuple:
        """Flip image and labels horizontally.

        Args:
            image: Input image
            labels: Labels (N, 5) [cls, cx, cy, w, h]

        Returns:
            Tuple of (flipped_image, flipped_labels)
        """
        image = np.fliplr(image)
        if len(labels):
            labels[:, 1] = 1 - labels[:, 1]  # Flip x center
        return image, labels


class MixUpAugmentation:
    """MixUp data augmentation.

    Reference: https://arxiv.org/pdf/1710.09412.pdf
    """

    @staticmethod
    def mix_up(image1: np.ndarray, label1: np.ndarray,
               image2: np.ndarray, label2: np.ndarray) -> tuple:
        """Apply MixUp augmentation.

        Args:
            image1: First image
            label1: First labels
            image2: Second image
            label2: Second labels

        Returns:
            Tuple of (mixed_image, concatenated_labels)
        """
        alpha = np.random.beta(32.0, 32.0)
        image = (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
        label = np.concatenate((label1, label2), 0)
        return image, label


class Albumentations:
    """Optional Albumentations wrapper for advanced augmentations.

    Requires albumentations package to be installed.
    """

    def __init__(self):
        """Initialize Albumentations transforms."""
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose([
                A.Blur(p=0.01),
                A.CLAHE(p=0.01),
                A.ToGray(p=0.01),
                A.MedianBlur(p=0.01),
            ], A.BboxParams('yolo', ['class_labels']))

        except ImportError:
            pass  # Albumentations not installed

    def __call__(self, image: np.ndarray, label: np.ndarray) -> tuple:
        """Apply Albumentations transforms.

        Args:
            image: Input image
            label: Labels (N, 5) [cls, cx, cy, w, h]

        Returns:
            Tuple of (transformed_image, transformed_labels)
        """
        if self.transform and len(label):
            try:
                result = self.transform(
                    image=image,
                    bboxes=label[:, 1:],
                    class_labels=label[:, 0]
                )
                image = result['image']
                if len(result['bboxes']):
                    label = np.array([
                        [c, *b] for c, b in zip(result['class_labels'], result['bboxes'])
                    ])
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
            except Exception:
                pass  # Return original on error

        return image, label


class AugmentationPipeline:
    """Complete augmentation pipeline combining all augmentations."""

    def __init__(self, params: dict):
        """Initialize pipeline.

        Args:
            params: Augmentation parameters dict
        """
        self.params = params
        self.hsv = HSVAugmentation()
        self.flip = FlipAugmentation()
        self.mixup = MixUpAugmentation()
        self.albumentations = Albumentations()

    def __call__(self, image: np.ndarray, labels: np.ndarray) -> tuple:
        """Apply full augmentation pipeline.

        Args:
            image: Input image
            labels: Labels

        Returns:
            Tuple of (augmented_image, augmented_labels)
        """
        # Albumentations
        image, labels = self.albumentations(image, labels)

        # HSV augmentation
        self.hsv.augment(image, self.params)

        # Flip up-down
        if random.random() < self.params.get('flip_ud', 0):
            image, labels = self.flip.flip_ud(image, labels)

        # Flip left-right
        if random.random() < self.params.get('flip_lr', 0):
            image, labels = self.flip.flip_lr(image, labels)

        return image, labels

