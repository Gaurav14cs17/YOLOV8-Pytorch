import math
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

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
            self.transform = album.Compose(transforms,
                                           album.BboxParams('yolo', ['class_labels']))
        except ImportError:
            pass

    def __call__(self, image, label):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=label[:, 1:],
                               class_labels=label[:, 0])
            image = x['image']
            label = np.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image, label


class BoxUtils:
    """Static methods for bounding box conversions and filters."""

    @staticmethod
    def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
        y = x.copy()
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h
        return y

    @staticmethod
    def xy2wh(x, w=640, h=640):
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)
        y = x.copy()
        y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w
        y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h
        y[:, 2] = (x[:, 2] - x[:, 0]) / w
        y[:, 3] = (x[:, 3] - x[:, 1]) / h
        return y

    @staticmethod
    def candidates(box1, box2):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


class ImageAugmenter:
    """Handles HSV augmentation and geometric transformations."""
    def __init__(self, params):
        self.params = params

    def augment_hsv(self, image):
        h, s, v = self.params['hsv_h'], self.params['hsv_s'], self.params['hsv_v']
        r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
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
        center[0, 2] = -image.shape[1] / 2
        center[1, 2] = -image.shape[0] / 2

        perspective = np.eye(3)

        rotate = np.eye(3)
        a = random.uniform(-self.params['degrees'], self.params['degrees'])
        s = random.uniform(1 - self.params['scale'], 1 + self.params['scale'])
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        shear = np.eye(3)
        shear[0, 1] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-self.params['shear'], self.params['shear']) * math.pi / 180)

        translate = np.eye(3)
        translate[0, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * w
        translate[1, 2] = random.uniform(0.5 - self.params['translate'], 0.5 + self.params['translate']) * h

        matrix = translate @ shear @ rotate @ perspective @ center

        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

        n = len(targets)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ matrix.T
            xy = xy[:, :2].reshape(n, 8)

            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

            indices = BoxUtils.candidates(targets[:, 1:5].T * s, new.T)
            targets = targets[indices]
            targets[:, 1:5] = new[indices]

        return image, targets


class ImageLoader:
    """Handles loading and resizing images."""

    def __init__(self, input_size, augment):
        self.input_size = input_size
        self.augment = augment

    def resample_method(self):
        choices = (cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR,
                   cv2.INTER_NEAREST, cv2.INTER_LANCZOS4)
        return random.choice(choices)

    def load_image(self, filename):
        image = cv2.imread(filename)
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            interpolation = self.resample_method() if self.augment else cv2.INTER_LINEAR
            image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=interpolation)
        return image, (h, w)

    def resize_with_padding(self, image):
        shape = image.shape[:2]
        r = min(self.input_size / shape[0], self.input_size / shape[1])
        if not self.augment:
            r = min(r, 1.0)

        pad_w, pad_h = int(round(shape[1] * r)), int(round(shape[0] * r))
        w = (self.input_size - pad_w) / 2
        h = (self.input_size - pad_h) / 2

        if shape[::-1] != (pad_w, pad_h):
            interpolation = self.resample_method() if self.augment else cv2.INTER_LINEAR
            image = cv2.resize(image, dsize=(pad_w, pad_h), interpolation=interpolation)

        top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
        left, right = int(round(w - 0.1)), int(round(w + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image, (r, r), (w, h)


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment=True):
        self.filenames = filenames
        self.input_size = input_size
        self.params = params
        self.augment = augment
        self.mosaic_prob = params.get('mosaic', 0.0)
        self.mixup_prob = params.get('mix_up', 0.0)
        self.albumentations = AlbumentationsWrapper()
        self.loader = ImageLoader(input_size, augment)
        self.augmenter = ImageAugmenter(params)

        # Load labels cache or from files
        cache = self.load_label_cache(filenames)
        self.labels, self.shapes = zip(*cache.values())
        self.labels = list(self.labels)
        self.shapes = np.array(self.shapes, dtype=np.float64)
        self.n = len(self.labels)
        self.indices = range(self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mosaic = self.augment and (random.random() < self.mosaic_prob)

        if mosaic:
            image, label = self.load_mosaic(index)
            if random.random() < self.mixup_prob:
                mix_index = random.choice(self.indices)
                mix_image2, mix_label2 = self.load_mosaic(mix_index)
                image, label = self.mix_up(image, label, mix_image2, mix_label2)
        else:
            image, shape = self.loader.load_image(self.filenames[index])
            h, w = image.shape[:2]

            image, ratio, pad = self.loader.resize_with_padding(image)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = BoxUtils.wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])

            if self.augment:
                image, label = self.augmenter.random_perspective(image, label, border=(0, 0))

            nl = len(label)
            if nl:
                label[:, 1:5] = BoxUtils.xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        # Albumentations
        if self.augment:
            image, label = self.albumentations(image, label)

            # HSV augmentation
            self.augmenter.augment_hsv(image)

            # Flip UD
            if random.random() < self.params.get('flip_ud', 0.0):
                image = np.flipud(image)
                if len(label):
                    label[:, 2] = 1 - label[:, 2]

            # Flip LR
            if random.random() < self.params.get('flip_lr', 0.0):
                image = np.fliplr(image)
                if len(label):
                    label[:, 1] = 1 - label[:, 1]

        nl = len(label)
        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        sample = image.transpose((2, 0, 1))[::-1]  # HWC to CHW and BGR to RGB
        sample = np.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, (shape, (ratio, pad))

    def load_mosaic(self, index):
        input_size = self.input_size
        labels4 = []
        images4 = np.full((input_size * 2, input_size * 2, 3), 0, dtype=np.uint8)

        border = [-input_size // 2, -input_size // 2]

        xc = int(random.uniform(-border[0], 2 * input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            image, _ = self.loader.load_image(self.filenames[idx])
            shape = image.shape

            if i == 0:  # top left
                x1a = max(xc - shape[1], 0)
                y1a = max(yc - shape[0], 0)
                x2a = xc
                y2a = yc
                x1b = shape[1] - (x2a - x1a)
                y1b = shape[0] - (y2a - y1a)
                x2b = shape[1]
                y2b = shape[0]
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - shape[0], 0)
                x2a = min(xc + shape[1], 2 * input_size)
                y2a = yc
                x1b = 0
                y1b = shape[0] - (y2a - y1a)
                x2b = min(shape[1], x2a - x1a)
                y2b = shape[0]
            elif i == 2:  # bottom left
                x1a = max(xc - shape[1], 0)
                y1a = yc
                x2a = xc
                y2a = min(2 * input_size, yc + shape[0])
                x1b = shape[1] - (x2a - x1a)
                y1b = 0
                x2b = shape[1]
                y2b = min(y2a - y1a, shape[0])
            else:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + shape[1], 2 * input_size)
                y2a = min(2 * input_size, yc + shape[0])
                x1b = 0
                y1b = 0
                x2b = min(shape[1], x2a - x1a)
                y2b = min(y2a - y1a, shape[0])

            images4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            pad_w = x1a - x1b
            pad_h = y1a - y1b

            label = self.labels[idx].copy()
            if len(label):
                label[:, 1:] = BoxUtils.wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)

            labels4.append(label)

        label4 = np.concatenate(labels4, 0)
        for x in label4[:, 1:]:
            np.clip(x, 0, 2 * input_size, out=x)

        images4, label4 = self.augmenter.random_perspective(images4, label4, border=border)
        return images4, label4

    def mix_up(self, image1, label1, image2, label2):
        alpha = np.random.beta(32.0, 32.0)
        image = (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
        label = np.concatenate((label1, label2), 0)
        return image, label

    def load_label_cache(self, filenames):
        cache_path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        cache = {}
        for filename in filenames:
            try:
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                assert shape[0] > 9 and shape[1] > 9
                assert image.format.lower() in FORMATS

                label_path = filename.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = np.array(label, dtype=np.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5
                        assert (label >= 0).all()
                        assert (label[:, 1:] <= 1).all()
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:
                            label = label[i]
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)

                cache[filename] = (label, shape)
            except FileNotFound
