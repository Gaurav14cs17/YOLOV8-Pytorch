# YOLOv8 Custom Car Detection

Hey there! 👋 Welcome to our custom YOLOv8 implementation for car detection. This isn't your standard YOLO - we've cooked up something special by combining the latest research in efficient architectures with practical object detection needs.

## What's This All About?

I built this project because I wanted a car detection system that's both accurate and efficient enough to run on modest hardware. Most existing solutions were either too heavy or not accurate enough for real-world use cases.

The magic sauce here is the combination of:
- **RepViT backbone** - gives us the power of transformers without the computational burden
- **Ghost modules** - makes the model lightweight without sacrificing performance
- **Smart training tricks** - mixed precision, EMA, and advanced augmentation

## Quick Start - Let's Get You Running

### First, Get the Code
```bash
git clone https://github.com/your-username/yolov8-car-detection.git
cd yolov8-car-detection
```

### Install What You Need
```bash
# The essentials
pip install torch torchvision
# For image augmentations
pip install albumentations opencv-python
# To automatically get the car dataset
pip install kagglehub
```

### Train Your First Model
```python
from train import start_training

# This will train for 50 epochs with sensible defaults
start_training(epochs=50, batch_size=16)
```

That's it! The code will automatically download the car dataset from Kaggle and start training.

## What Makes This Different?

### The Architecture - Technical Stuff Made Simple

Think of our model like this:

```
Your Image → Feature Extractor → Multi-scale Analysis → Detection Heads
```

The **RepViT backbone** is like having a really smart feature extractor that can "reparameterize" itself - meaning it trains one way but runs more efficiently during inference.

The **Ghost modules** are our secret weapon for efficiency. Instead of doing expensive computations everywhere, they create "ghost" features that are cheaper to compute but just as useful.

### Training Smart, Not Hard

We use several techniques that make training more stable and effective:

- **Mixed Precision**: Uses FP16 where possible to speed things up and save memory
- **EMA (Exponential Moving Average)**: Keeps a smoothed version of your model that's more stable
- **Gradient Accumulation**: Lets you effectively use larger batch sizes even with limited GPU memory
- **Smart Data Augmentation**: We don't just flip images - we use mosaic (combining 4 images), mix-up (blending images), and color adjustments to make your model robust

## Real Results You Can Expect

After training for 50 epochs (which takes about 45 minutes per epoch on a T4 GPU), you should see:

- **mAP@0.5**: Around 0.50 (that means it correctly detects cars 50% of the time at moderate overlap thresholds)
- **mAP@0.5:0.95**: Around 0.30 (more strict measurement across different overlap levels)
- **Model Size**: Only about 5MB - tiny compared to many detection models!
- **Training Speed**: Roughly 1.3 iterations per second on a T4 GPU

The loss typically goes from around 450,000 in the first epoch down to about 50,000 by epoch 50.

## When Things Go Wrong (They Will)

### Common Issues and Fixes

**"I'm running out of GPU memory!"**
```python
# Just use a smaller batch size
start_training(epochs=50, batch_size=8)  # Instead of 16
```

**"Training is unstable - loss is jumping around"**
```python
# Try a lower learning rate
config = {'lr0': 0.001}  # Instead of 0.01
```

**"The model isn't learning anything"**
- Check that your dataset downloaded correctly
- Make sure the labels are being loaded (look for "Found X images" messages)
- Try increasing the number of epochs

**"I'm getting weird errors about devices or tensors"**
```python
# This usually fixes device issues
import torch
torch.cuda.empty_cache()
```

### Debugging Tips

If you're having trouble, add this to see what's happening:

```python
import warnings
warnings.filterwarnings("default")  # Show all warnings

# Check your setup
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
```

## Making It Your Own

### Want to Detect More Than Just Cars?

```python
# Say you want to detect cars, trucks, and motorcycles
config = {
    'nc': 3,  # Now detecting 3 classes
    'cls_loss': 0.7,  # Maybe emphasize classification more
}
```

### Need Better Performance on Small Cars?

```python
# Enable multi-head detection for better small object detection
config['multi_head'] = True
```

### Customizing the Training

```python
start_training(
    epochs=100,  # Train longer
    batch_size=8,  # Smaller batches if memory is tight
    resume_checkpoint='weights/latest_model.pth'  # Continue from where you left off
)
```

## Project Structure - Where Everything Lives

```
yolov8-car-detection/
├── Yolov8Custoom.ipynb     # Main training notebook
├── weights/                # Your trained models go here
│   ├── best_model.pth     # Best model based on validation
│   ├── latest_model.pth   # Most recent model
│   └── training_log.csv   # Training history
└── dataset_cache/         # Cached dataset (auto-created)
```


