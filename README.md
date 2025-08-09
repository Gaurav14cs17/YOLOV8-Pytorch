---
# YOLOV8-PyTorch

![YOLO-V8](https://github.com/Gaurav14cs17/YOLOV8-Pytorch/blob/main/images/v8_structure.jpg)

A modular, PyTorch-based implementation of YOLO-style object detection with:

* **Backbone** (Darknet-style)
* **Neck** (FPN / PANet fusion)
* **Detection Head**
* **Custom YOLO Dataset Loader**
* **Albumentations-based Augmentations**
* **Conv + BatchNorm Fusion** for faster inference

---

## 📂 Directory Structure

```
model/
│── backbone.py       # DarkNet backbone
│── neck.py           # FPN / PANet neck
│── head.py           # Detection head
│── yolo.py           # YOLO model (connect backbone + neck + head)
│── blocks.py         # Common CNN blocks (Conv, Bottleneck, CSP, etc.)
dataset/
│── dataset_yolo.py   # YOLO dataset loader
│── augmentations.py  # Albumentations wrapper for image + bbox aug
utils/
│── fuse.py           # Conv + BN fusion
train.py              # Training script
infer.py              # Inference script
requirements.txt
README.md
```

---

## 📦 Installation

```bash
# Clone this repository
git clone https://github.com/Gaurav14cs17/YOLOV8-Pytorch.git
cd YOLOV8-Pytorch

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Format (YOLO)

Directory structure:

```
dataset/
    images/
        train/
        val/
    labels/
        train/
        val/
```

Label format:

```
class x_center y_center width height
```

All values are **normalized** (0–1 range).

---

## 🎨 Augmentations

We use [Albumentations](https://albumentations.ai/) for data augmentation.
Default augmentations include:

* Blur / Median Blur
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Grayscale
* Random Brightness / Contrast
* Hue / Saturation / Value shifts
* Shift / Scale / Rotate

**Example**:

```python
from dataset.augmentations import AlbumentationsWrapper

augmenter = AlbumentationsWrapper()
image, labels = augmenter(image, labels)
```

---

## 🖼 Dataset Loader

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

## 🚀 Model Usage

```python
from model.yolo import YOLO
from utils.fuse import fuse_conv_and_bn

# Create model
model = YOLO(num_classes=80)

# Fuse Conv + BN for faster inference
model = fuse_conv_and_bn(model)

# Forward pass
output = model(images)  # [batch, anchors, grid_h, grid_w, classes+5]
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

BatchNorm can be merged into Conv2d for faster inference:

```python
from utils.fuse import fuse_conv_and_bn
model = fuse_conv_and_bn(model)
```

---

## 📌 TODO

* [ ] Mixed Precision Training
* [ ] Multi-scale Training
* [ ] Mosaic Augmentation
* [ ] ONNX / TensorRT Export

---
