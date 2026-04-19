# dataset/dataset.py
"""YOLO Dataset implementation for object detection."""

import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

from ..transforms import wh2xy, xy2wh, resize, resample
from ..augmentation import augment_hsv, random_perspective, mix_up, Albumentations


FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    """YOLO Dataset for object detection training and validation.
    
    Supports:
    - Mosaic augmentation
    - MixUp augmentation
    - HSV color augmentation
    - Random perspective transformation
    - Albumentations integration
    - Horizontal/vertical flipping
    
    Args:
        filenames: list of image file paths
        input_size: target input size (int)
        params: augmentation parameters dict
        augment: whether to apply augmentation
    """
    
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size

        # Read labels
        cache = self.load_label(filenames)
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.filenames = list(cache.keys())
        self.n = len(shapes)
        self.indices = range(self.n)
        
        # Albumentations (optional)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        """Get a training sample.
        
        Args:
            index: sample index
        
        Returns:
            Tuple of (image_tensor, target_tensor, shapes_info)
        """
        index = self.indices[index]
        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            shapes = None
            image, label = self._load_mosaic(index, params)
            
            # MixUp augmentation
            if random.random() < params['mix_up']:
                mix_index = random.choice(self.indices)
                mix_image2, mix_label2 = self._load_mosaic(mix_index, params)
                image, label = mix_up(image, label, mix_image2, mix_label2)
        else:
            image, shape = self._load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)
            shapes = shape, ((h / shape[0], w / shape[1]), pad)

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            
            if self.augment:
                image, label = random_perspective(image, label, params)

        nl = len(label)
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        if self.augment:
            # Albumentations
            image, label = self.albumentations(image, label)
            nl = len(label)
            
            # HSV augmentation
            augment_hsv(image, params)
            
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    label[:, 2] = 1 - label[:, 2]
            
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, shapes

    def __len__(self):
        """Return dataset length."""
        return len(self.filenames)

    def _load_image(self, i):
        """Load and optionally resize an image.
        
        Args:
            i: image index
        
        Returns:
            Tuple of (image, original_shape)
        """
        image = cv2.imread(self.filenames[i])
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            image = cv2.resize(
                image,
                dsize=(int(w * r), int(h * r)),
                interpolation=resample() if self.augment else cv2.INTER_LINEAR
            )
        return image, (h, w)

    def _load_mosaic(self, index, params):
        """Load 4 images and create mosaic.
        
        Args:
            index: center image index
            params: augmentation parameters
        
        Returns:
            Tuple of (mosaic_image, mosaic_labels)
        """
        label4 = []
        image4 = numpy.full(
            (self.input_size * 2, self.input_size * 2, 3),
            0,
            dtype=numpy.uint8
        )
        
        border = [-self.input_size // 2, -self.input_size // 2]
        xc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))
        yc = int(random.uniform(-border[0], 2 * self.input_size + border[1]))

        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            image, _ = self._load_image(idx)
            shape = image.shape

            # Calculate placement coordinates
            x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = self._get_mosaic_coords(
                i, xc, yc, shape, self.input_size
            )

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            pad_w = x1a - x1b
            pad_h = y1a - y1b

            # Labels
            label = self.labels[idx].copy()
            if len(label):
                label[:, 1:] = wh2xy(label[:, 1:], shape[1], shape[0], pad_w, pad_h)
            label4.append(label)

        # Concat and clip labels
        label4 = numpy.concatenate(label4, 0)
        for x in label4[:, 1:]:
            numpy.clip(x, 0, 2 * self.input_size, out=x)

        # Apply random perspective
        image4, label4 = random_perspective(image4, label4, params, border)

        return image4, label4

    def _get_mosaic_coords(self, i, xc, yc, shape, input_size):
        """Calculate mosaic placement coordinates.
        
        Args:
            i: quadrant index (0-3)
            xc, yc: mosaic center coordinates
            shape: image shape (H, W, C)
            input_size: target input size
        
        Returns:
            Tuple of (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) coordinates
        """
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
            x2a = min(xc + shape[1], input_size * 2)
            y2a = yc
            x1b = 0
            y1b = shape[0] - (y2a - y1a)
            x2b = min(shape[1], x2a - x1a)
            y2b = shape[0]
        elif i == 2:  # bottom left
            x1a = max(xc - shape[1], 0)
            y1a = yc
            x2a = xc
            y2a = min(input_size * 2, yc + shape[0])
            x1b = shape[1] - (x2a - x1a)
            y1b = 0
            x2b = shape[1]
            y2b = min(y2a - y1a, shape[0])
        else:  # bottom right (i == 3)
            x1a = xc
            y1a = yc
            x2a = min(xc + shape[1], input_size * 2)
            y2a = min(input_size * 2, yc + shape[0])
            x1b = 0
            y1b = 0
            x2b = min(shape[1], x2a - x1a)
            y2b = min(y2a - y1a, shape[0])
        
        return x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader.
        
        Args:
            batch: list of (sample, target, shapes) tuples
        
        Returns:
            Tuple of (stacked_samples, concatenated_targets, shapes_list)
        """
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    @staticmethod
    def load_label(filenames):
        """Load and cache labels for all images.
        
        Args:
            filenames: list of image file paths
        
        Returns:
            Dict mapping filename to (labels, shape) tuples
        """
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        
        x = {}
        for filename in filenames:
            try:
                # Verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()
                shape = image.size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # Verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                label_path = b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'
                
                if os.path.isfile(label_path):
                    with open(label_path) as f:
                        label = [line.split() for line in f.read().strip().splitlines() if len(line)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, 'labels require 5 columns'
                        assert (label >= 0).all(), 'negative label values'
                        assert (label[:, 1:] <= 1).all(), 'non-normalized coordinates'
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
                
                if filename:
                    x[filename] = [label, shape]
            except FileNotFoundError:
                pass
        
        torch.save(x, path)
        return x

