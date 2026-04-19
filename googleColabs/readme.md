# YOLOv8 Google Colab Notebooks

This folder contains Google Colab notebooks for training and using our custom YOLOv8 implementation.

## 📓 Available Notebooks

### `Yolov8_Train.ipynb` (Recommended)
A clean training notebook that imports from our package structure.

**Features:**
- ✅ Imports from modular packages (`model`, `utils`, `dataloader`)
- ✅ Unified output folder structure
- ✅ Epoch-wise sample visualization
- ✅ Automatic cleanup of old epoch folders (keeps last 5)
- ✅ Metrics tracking and visualization

---

## 🚀 Quick Start

### Using `Yolov8_Train.ipynb`:

1. Open in Google Colab
2. Run Setup cell to clone repo and install dependencies
3. Edit training parameters (epochs, batch size, variant)
4. Run training

```python
# Training parameters - EDIT THESE
EPOCHS = 50
BATCH_SIZE = 8
INPUT_SIZE = 640
VARIANT = 'nano'  # Options: nano, small, medium, large, xlarge
```

---

## 📁 Output Structure

All outputs are organized in a single `output/` folder:

```
output/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Latest model
├── epoch_samples/       # Detection samples (keeps last 5 epochs)
│   ├── epoch_0046/
│   │   └── sample_epoch_0046.jpg
│   ├── epoch_0047/
│   ├── epoch_0048/
│   ├── epoch_0049/
│   └── epoch_0050/
└── metrics/
    └── step.csv         # Training metrics (epoch, mAP@50, mAP)
```

---

## 📦 Package Imports

The notebook imports from our modular packages:

```python
# Model architecture
from model import YOLO, yolo_v8_n, yolo_v8_s, yolo_v8_m
from model.fusion import fuse

# Utilities
from utils import (
    setup_seed, setup_multi_processes,
    EMA, AverageMeter,
    ComputeLoss,
    compute_ap, smooth,
    non_max_suppression, scale, box_iou,
    strip_optimizer, clip_gradients
)

# Dataset
from dataloader import Dataset
```

---

## 🛠️ Classes Defined in Notebook

### `OutputManager`
Manages all training outputs in a single folder structure.

```python
output = OutputManager('output')
output.get_weight_path('best.pt')  # -> output/weights/best.pt
output.get_metrics_path('step.csv')  # -> output/metrics/step.csv
```

### `EpochVisualizer`
Saves sample detection images at each epoch and manages folder cleanup.

```python
visualizer = EpochVisualizer(output_dir=Path('output/epoch_samples'), max_folders=5)
visualizer.save_sample(model, dataloader, epoch, device, class_names)
```

### `Config`
Loads and provides access to training configuration.

```python
config = Config('config/config.yml')
config.train_path  # Training images path
config.val_path    # Validation images path
config.class_names # Class name dictionary
config.num_classes # Number of classes
config.hyp         # Hyperparameters
```

### `ColabTrainer`
Main training engine with all components integrated.

```python
trainer = ColabTrainer(
    config=config,
    epochs=50,
    batch_size=8,
    input_size=640,
    variant='nano'
)
trainer.train()
```

---

## 📊 Results Visualization

The notebook includes cells to:

1. **View output folder structure**
   ```bash
   tree output
   ```

2. **Display epoch samples**
   Shows detection samples from the last 5 epochs

3. **Plot training metrics**
   mAP@0.5 and mAP@0.5:0.95 over epochs

4. **Download trained models**
   Download `best.pt` and `last.pt` to local machine

---

## 💡 Tips

### Memory Issues
```python
# Reduce batch size
BATCH_SIZE = 4  # or 2
```

### Faster Training
```python
# Use smaller model variant
VARIANT = 'nano'  # fastest

# Use less epochs
EPOCHS = 20
```

### Better Performance
```python
# Use larger model variant
VARIANT = 'medium'  # or 'large', 'xlarge'

# Train longer
EPOCHS = 100
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE`
- Reduce `INPUT_SIZE` to 416 or 320
- Use smaller `VARIANT`

### Slow Training
- Enable GPU in Colab (Runtime → Change runtime type → GPU)
- Use T4 or A100 if available

### Import Errors
- Make sure you cloned the repository
- Check that `sys.path.insert(0, '.')` was executed
- Verify dependencies are installed

---

## 📝 Custom Dataset

To use your own dataset:

1. Upload to Google Drive or use a URL
2. Update `config/config.yml`:
   ```yaml
   data:
     train: /content/your_dataset/images/train
     val: /content/your_dataset/images/val
   names:
     0: class1
     1: class2
   ```
3. Ensure YOLO format labels exist in `/content/your_dataset/labels/train` and `val`

---

### 🚀 Open in Google Colab
<br/>
<p align="center">
  <a href="https://colab.research.google.com/github/Gaurav14cs17/YOLOV8-Pytorch/blob/main/googleColabs/Yolov8_Train.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg"
         alt="Open In Colab"
         height="60"/>
  </a>
</p>
<br/>

<p align="center">
  <strong>✨ Click the badge above to open this notebook directly in Google Colab!</strong>
</p>


## 📚 Navigation

| Previous | Up | Next |
|:---------|:--:|-----:|
| [← QModel Package](../qmodel/README.md) | [🏠 Home](../README.md) | - |

**Related Documentation:**
- [Model Documentation](../model/README.md)
- [Dataloader Documentation](../dataloader/README.md)
- [Utils Documentation](../utils/README.md)
- [QModel Documentation](../qmodel/README.md)
