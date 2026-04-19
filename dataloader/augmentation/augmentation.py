# augmentation/augmentation.py
"""Data augmentation functions for object detection training."""

import math
import random
import cv2
import numpy

from ..transforms import candidates


def augment_hsv(image, params):
    """Apply HSV color-space augmentation.
    
    Args:
        image: BGR image (H, W, 3)
        params: dict with 'hsv_h', 'hsv_s', 'hsv_v' augmentation ranges
    """
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def random_perspective(samples, targets, params, border=(0, 0)):
    """Apply random perspective transformation.
    
    Applies a combination of:
    - Rotation
    - Scale
    - Shear
    - Translation
    
    Args:
        samples: input image
        targets: labels (N, 5) with [class, x1, y1, x2, y2]
        params: augmentation parameters dict
        border: border offset for mosaic
    
    Returns:
        Tuple of (transformed_image, transformed_targets)
    """
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():
        samples = cv2.warpAffine(samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

        # Filter candidates
        indices = candidates(box1=targets[:, 1:5].T * s, box2=new.T)
        targets = targets[indices]
        targets[:, 1:5] = new[indices]

    return samples, targets


def mix_up(image1, label1, image2, label2):
    """Apply MixUp augmentation.
    
    Reference: https://arxiv.org/pdf/1710.09412.pdf
    
    Args:
        image1, label1: first image and labels
        image2, label2: second image and labels
    
    Returns:
        Tuple of (mixed_image, concatenated_labels)
    """
    alpha = numpy.random.beta(32.0, 32.0)  # mix-up ratio
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    """Albumentations wrapper for additional augmentations."""
    
    def __init__(self):
        """Initialize Albumentations transforms if available."""
        self.transform = None
        try:
            import albumentations as album

            transforms = [
                album.Blur(p=0.01),
                album.CLAHE(p=0.01),
                album.ToGray(p=0.01),
                album.MedianBlur(p=0.01)
            ]
            self.transform = album.Compose(
                transforms,
                album.BboxParams('yolo', ['class_labels'])
            )

        except ImportError:
            pass

    def __call__(self, image, label):
        """Apply albumentations transforms.
        
        Args:
            image: input image
            label: labels (N, 5) with [class, cx, cy, w, h] normalized
        
        Returns:
            Tuple of (augmented_image, augmented_labels)
        """
        if self.transform:
            x = self.transform(
                image=image,
                bboxes=label[:, 1:],
                class_labels=label[:, 0]
            )
            image = x['image']
            label = numpy.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image, label

