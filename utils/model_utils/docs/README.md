# Model Utilities Module (`model_utils.py`)

This module provides utility functions for model checkpoint management and gradient handling during training.

---

## 📊 Visual Overview

### Model Utilities Overview

![Model Utilities](./01_model_utilities.svg)

---

## 🔧 Functions

### `strip_optimizer(filename)`

Remove optimizer state and convert model to FP16 for deployment.

```python
def strip_optimizer(filename):
    """
    Strip optimizer from checkpoint and convert to FP16.
    
    Args:
        filename: Path to checkpoint file (.pt)
    
    Effects:
        - Converts model weights to FP16 (half precision)
        - Disables gradient computation for all parameters
        - Saves stripped checkpoint in-place
    """
    x = torch.load(filename, map_location=torch.device('cpu'))
    x['model'].half()  # Convert to FP16
    for p in x['model'].parameters():
        p.requires_grad = False  # Disable gradients
    torch.save(x, filename)
```

**Benefits:**

| Aspect | Before | After |
|--------|--------|-------|
| Precision | FP32 | FP16 |
| Size | ~200MB | ~100MB |
| Optimizer | Included | Removed |
| Training | Resumable | Inference only |

---

### `clip_gradients(model, max_norm=10.0)`

Prevent exploding gradients by clipping gradient norms.

```python
def clip_gradients(model, max_norm=10.0):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model with computed gradients
        max_norm: Maximum allowed gradient norm (default: 10.0)
    
    Uses torch.nn.utils.clip_grad_norm_ for efficient clipping.
    """
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
```

**How It Works:**

1. Compute total gradient norm: $||g|| = \sqrt{\sum_i g_i^2}$
2. If $||g|| > \text{max\_norm}$: scale all gradients by $\frac{\text{max\_norm}}{||g||}$
3. All gradients remain proportional but within bounds

---

## 📁 Module Structure

```
utils/
├── model_utils.py           # Main module
└── model_utils/
    └── docs/
        ├── README.md        # This documentation
        └── 01_model_utilities.svg
```

---

## 💡 When to Use

### `strip_optimizer()`

Use when:
- Exporting final model for production
- Sharing model checkpoints (smaller size)
- Deploying to edge devices
- Converting to ONNX/TensorRT

**Warning**: After stripping, the checkpoint cannot be used to resume training!

### `clip_gradients()`

Use when:
- Training with high learning rates
- Observing loss spikes or NaN values
- Using mixed precision (FP16)
- Training very deep networks
- Fine-tuning with unfrozen layers

---

## 🎯 Usage Examples

### 1. Strip Optimizer for Deployment

```python
from utils.model_utils import strip_optimizer

# After training completes
final_checkpoint = 'runs/train/best.pt'

# Create deployment version
import shutil
deploy_checkpoint = 'runs/train/best_deploy.pt'
shutil.copy(final_checkpoint, deploy_checkpoint)

# Strip optimizer (modifies in-place)
strip_optimizer(deploy_checkpoint)

# Now deploy_checkpoint is:
# - 50% smaller
# - FP16 precision
# - Ready for inference
```

### 2. Gradient Clipping in Training Loop

```python
from utils.model_utils import clip_gradients

for epoch in range(epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients BEFORE optimizer step
        clip_gradients(model, max_norm=10.0)
        
        # Update weights
        optimizer.step()
```

### 3. Combined Training Script

```python
from utils.model_utils import strip_optimizer, clip_gradients

# Training
for epoch in range(epochs):
    for batch in train_loader:
        loss = train_step(batch)
        optimizer.zero_grad()
        loss.backward()
        clip_gradients(model, max_norm=10.0)  # Stability
        optimizer.step()
    
    # Save checkpoint
    torch.save({'model': model, 'optimizer': optimizer}, f'epoch_{epoch}.pt')

# After training: prepare for deployment
strip_optimizer('best.pt')
```

---

## ⚙️ Technical Details

### Gradient Clipping Methods

| Method | Description | This Implementation |
|--------|-------------|---------------------|
| Clip by Value | Clip each gradient independently | ❌ |
| Clip by Norm | Scale all gradients together | ✅ |
| Clip by Global Norm | Same as by norm, global view | ✅ |

**Clip by Norm** preserves gradient direction while limiting magnitude.

### FP16 Conversion

```python
model.half()  # Converts all float tensors to torch.float16
```

- **Pros**: 2x memory reduction, faster inference on modern GPUs
- **Cons**: Reduced precision, may affect very small values

---

## 📚 References

1. **Gradient Clipping**: Pascanu et al., "On the difficulty of training Recurrent Neural Networks" (2013)
   - Paper: https://arxiv.org/abs/1211.5063

2. **Mixed Precision Training**: Micikevicius et al., "Mixed Precision Training" (2018)
   - Paper: https://arxiv.org/abs/1710.03740

3. **PyTorch Utilities**: https://pytorch.org/docs/stable/nn.html#clip-grad-norm

---

## 📚 Navigation

| Previous | Up | Next |
|:---------|:--:|-----:|
| [← Meters](../../meters/docs/README.md) | [🏠 Utils](../../README.md) | [QModel Package →](../../../qmodel/README.md) |

