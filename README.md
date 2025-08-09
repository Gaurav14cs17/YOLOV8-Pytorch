# YOLOV8-Pytorch

![YOLO-V8](https://github.com/Gaurav14cs17/YOLOV8-Pytorch/blob/main/images/v8_structure.jpg)


* Project description
* Directory structure (backbone, neck, head, yolo, blocks, dataset, augmentations)
* Installation instructions
* Usage example for training/inference
* Details on dataset handling & augmentations
* FuseConv2d + model fusion usage

---

## 📄 **README.md**

```markdown
# YOLO Object Detection Pipeline (Custom Modular Design)

This repository contains a modular implementation of YOLO-style object detection with:
- **Backbone** (Darknet-style)
- **Neck** (FPN/PANet fusion)
- **Detection Head**
- **Custom Dataset Loader** for YOLO format
- **Albumentations-based Augmentations**
- **BatchNorm fusion** for inference optimization

The structure is **clean and modular** so you can easily extend, modify, or replace components.

---

## 📂 Directory Structure
```

model/
│── backbone.py       # DarkNet backbone
│── neck.py           # DarkFPN / PANet neck
│── head.py           # Detection head
│── yolo.py           # Main YOLO model
│── blocks.py         # Common CNN building blocks (Conv, Bottleneck, CSP, etc.)
dataset/
│── dataset\_yolo.py   # YOLO format dataset loader
│── augmentations.py  # Albumentations wrapper for image + bbox augmentation
utils/
│── fuse.py           # Conv+BN fusion function
train.py              # Training script
infer.py              # Inference script
requirements.txt
README.md

````

---

## 📦 Installation
```bash
# Clone this repository
git clone https://github.com/Gaurav14cs17/YOLOV8-Pytorch.git
cd YOLOV8-Pytorch

# Install dependencies
pip install -r requirements.txt
````

---

## 📊 Dataset Format

We use **YOLO format**:

```
images/
    train/
    val/
labels/
    train/
    val/
```

Each label file contains:

```
class x_center y_center width height
```

Values are **normalized** (0–1 range).

---

## 🎨 Augmentations

We use **Albumentations** for powerful augmentations:

* Blur / Median Blur
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Grayscale
* Random Brightness / Contrast
* Hue, Saturation, Value shifts
* Shift / Scale / Rotate

You can customize them in:

```
dataset/augmentations.py
```

Example:

```python
from dataset.augmentations import AlbumentationsWrapper
augmenter = AlbumentationsWrapper()
image, labels = augmenter(image, labels)
```

---

## 🖼 YOLO Dataset Loader

```python
from dataset.dataset_yolo import DatasetYOLO

dataset = DatasetYOLO(
    images_path='path/to/images',
    labels_path='path/to/labels',
    img_size=640,
    augment=True
)
```

---

## 🚀 Model Example

```python
from model.yolo import YOLO
from utils.fuse import fuse_conv_and_bn

model = YOLO(num_classes=80)

# Fuse Conv+BN for faster inference
model = fuse_conv_and_bn(model)

# Forward
output = model(images)  # shape: [batch, anchors, grid_h, grid_w, classes+5]
```

---

## 🏋️ Training

```bash
python train.py --data data.yaml --epochs 100 --batch-size 16 --img 640
```

---

## 🧪 Inference

```bash
python infer.py --weights best.pt --source images/
```

---

## ⚡ Fuse Conv + BN

We include a utility to **fuse Conv2d + BatchNorm2d** for faster inference:

```python
from utils.fuse import fuse_conv_and_bn
model = fuse_conv_and_bn(model)
```

---

## 📌 TODO
* [ ] Mixed Precision Training
* [ ] Multi-scale Training
* [ ] Mosaic Augmentation
* [ ] Export to ONNX/TensorRT

---


