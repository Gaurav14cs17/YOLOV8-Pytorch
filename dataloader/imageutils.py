import math
import random
import cv2
import numpy as np


FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp')


class AlbumentationsWrapper:
    """Wrapper for optional Albumentations augmentation."""
    def __init__(self):
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
                bbox_params=album.BboxParams(format='yolo', label_fields=['class_labels'])
            )
        except ImportError:
            pass

    def __call__(self, image, label):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=label[:, 1:].tolist(),  # Albumentations expects list of lists
                               class_labels=label[:, 0].tolist())
            image = x['image']
            # convert back to np.array with correct dtype
            label = np.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])], dtype=np.float32)
        return image, label


class BoxUtils:
    """Static methods for bounding box conversions and filters."""

    @staticmethod
    def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
        y = x.copy()
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # x1
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # y1
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # x2
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # y2
        return y

    @staticmethod
    def xy2wh(x, w=640, h=640):
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
        y = x.copy()
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # cx normalized
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # cy normalized
        y[:, 2] = (x[:, 2] - x[:, 0]) / w        # width normalized
        y[:, 3] = (x[:, 3] - x[:, 1]) / h        # height normalized
        return y

    @staticmethod
    def candidates(box1, box2):
        """
        Filters candidate boxes.

        box1, box2: shape (4,) each representing (x1, y1, x2, y2)
        Returns True if box2 is a valid candidate relative to box1.
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        aspect_ratio = max(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > 2) and (h2 > 2) and (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) and (aspect_ratio < 100)


class ImageAugmenter:
    """Handles HSV augmentation and geometric transformations."""
    def __init__(self, params):
        self.params = params

    def augment_hsv(self, image):
        h, s, v = self.params['hsv_h'], self.params['hsv_s'], self.params['hsv_v']
        r = np.random.uniform(-1, 1, 3) * np.array([h, s, v]) + 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv)
        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype('uint8')
        lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
        lut_v = np.clip(x * r[2], 0, 255).astype('uint8')
        hsv = cv2.merge((cv2.LUT(channels[0], lut_h),
                         cv2.LUT(channels[1], lut_s),
                         cv2.LUT(channels[2], lut_v)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=image)

    def random_perspective(self, image, targets, border=(0, 0)):
        h = image.shape[0] + border[0] * 2
        w = image.shape[1] + border[1] * 2

        center = np.eye(3)
        center[0, 2] = -image.shape[1] / 2  # shift center x
        center[1, 2] = -image.shape[0] / 2  # shift center y

        perspective = np.eye(3)

        rotate = np.eye(3)
        a = random.uniform(-self.params['degrees'], self.params['degrees'])  # rotation angle
        s = random.uniform(1 - self.params['scale'], 1 + self.params['scale'])  # scale factor
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)

        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * w
        translate[1, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * h

        matrix = translate @ shear @ rotate @ perspective @ center

        # Apply warpAffine only if transformation matrix is not identity or border is nonzero
        if (border[0] != 0) or (border[1] != 0) or (not np.allclose(matrix, np.eye(3))):
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

        n = len(targets)
        if n:
            # Convert boxes from [class, x_center, y_center, w, h] to corner points
            # targets shape: (n, 5)
            # Extract box corners for all targets
            xy = np.ones((n * 4, 3))
            # points: top-left, bottom-left, bottom-right, top-right
            # Arrange as (x1,y1), (x1,y2), (x2,y2), (x2,y1)
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ matrix.T
            xy = xy[:, :2].reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

            # Filter valid boxes: targets are in normalized format, multiply by s
            valid_mask = np.array([BoxUtils.candidates(targets[i, 1:5] * s, new[i]) for i in range(n)])
            targets = targets[valid_mask]
            targets[:, 1:5] = new[valid_mask]

        return image, targets
