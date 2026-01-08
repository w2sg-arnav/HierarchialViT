# 🏋️ Training Guide

This document provides comprehensive instructions for training HierarchicalViT models.

## Table of Contents

- [Overview](#overview)
- [Phase 1: Self-Supervised Pre-training](#phase-1-self-supervised-pre-training)
- [Phase 2: Supervised Fine-tuning](#phase-2-supervised-fine-tuning)
- [Advanced Training Features](#advanced-training-features)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Multi-GPU Training](#multi-gpu-training)
- [Troubleshooting](#troubleshooting)

---

## Overview

HViT uses a two-phase training pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Self-Supervised Pre-training (SimCLR)             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Unlabeled Images → Contrastive Learning → Encoder  │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  Phase 2: Supervised Fine-tuning                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Labeled Images → Classification → Predictions      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Self-Supervised Pre-training

### Overview

SimCLR (Simple Contrastive Learning of Representations) learns visual representations by maximizing agreement between differently augmented views of the same image.

### Quick Start

```bash
python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --data-dir /path/to/images \
    --output-dir outputs/pretrain
```

### Configuration

```yaml
# configs/pretrain.yaml

data:
  root_dir: "/path/to/sarcld2024"
  img_size: [256, 256]
  num_classes: 7
  num_workers: 4

model:
  patch_size: 16
  embed_dim_rgb: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]
  use_gradient_checkpointing: true

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0003
  weight_decay: 0.0001
  warmup_epochs: 10

# SimCLR parameters
temperature: 0.07
projection_dim: 128
```

### SimCLR Augmentations

The following augmentations are applied to create positive pairs:

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Random Crop | scale=[0.2, 1.0] | Spatial invariance |
| Horizontal Flip | p=0.5 | Orientation invariance |
| Color Jitter | brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2 | Color invariance |
| Grayscale | p=0.2 | Color robustness |
| Gaussian Blur | kernel=[0.1, 2.0], p=0.5 | Texture robustness |

### InfoNCE Loss

The contrastive loss maximizes similarity between positive pairs while minimizing similarity with negatives:

```python
def info_nce_loss(features, temperature=0.07):
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(features, features.T) / temperature
    
    # Create labels (positive pairs are diagonal elements)
    labels = torch.arange(features.size(0), device=features.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(similarity, labels)
    return loss
```

### Monitoring Pre-training

Track these metrics during pre-training:

| Metric | Description | Target |
|--------|-------------|--------|
| `loss` | InfoNCE contrastive loss | Decreasing |
| `probe_accuracy` | Linear probe on frozen features | Increasing |
| `learning_rate` | Current LR (with warmup) | Follows schedule |

---

## Phase 2: Supervised Fine-tuning

### Overview

Fine-tuning adapts the pre-trained encoder to the target classification task using labeled data.

### Quick Start

```bash
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --ssl-checkpoint outputs/pretrain/best.pth \
    --data-dir /path/to/labeled_data
```

### Configuration

```yaml
# configs/finetune.yaml

data:
  root_dir: "/path/to/sarcld2024"
  img_size: [256, 256]
  num_classes: 7

model:
  ssl_pretrained_path: null  # Set via --ssl-checkpoint
  use_dfca: true
  drop_path_rate: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_epochs: 5
  
  # Advanced features
  use_ema: true
  ema_decay: 0.9999
  use_tta: true
  use_mixup: true
  mixup_alpha: 0.8
  use_cutmix: true
  cutmix_alpha: 1.0

  # Loss
  loss_type: "combined"
  focal_gamma: 2.0
  label_smoothing: 0.1
```

### Data Augmentation

Training augmentations are severity-configurable:

```yaml
augmentations:
  severity: "moderate"  # Options: light, moderate, aggressive
```

| Severity | Augmentations |
|----------|---------------|
| `light` | Resize, RandomCrop, HorizontalFlip |
| `moderate` | + ColorJitter, GaussianBlur, RandomRotation |
| `aggressive` | + RandAugment, Cutout, GridDistortion |

### Loss Functions

#### Focal Loss

Addresses class imbalance by down-weighting easy examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma
        self.alpha = alpha  # Optional per-class weights
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

#### Combined Loss

Combines Focal Loss with Cross-Entropy and label smoothing:

```python
loss = 0.5 * focal_loss + 0.5 * ce_loss_with_smoothing
```

---

## Advanced Training Features

### Exponential Moving Average (EMA)

EMA maintains a shadow copy of model weights for smoother predictions:

```python
from hvit.utils import EMA

# Initialize EMA
ema = EMA(model, decay=0.9999)

# During training
for batch in dataloader:
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    ema.update()  # Update shadow weights

# For evaluation
ema.apply_shadow()  # Use EMA weights
evaluate(model)
ema.restore()  # Restore original weights
```

### MixUp Augmentation

Blends pairs of training examples:

```python
def mixup(x, y, alpha=0.8):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    
    return mixed_x, y_a, y_b, lam
```

### CutMix Augmentation

Cuts and pastes patches between images:

```python
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    
    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # Apply cutmix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    
    return x, y, y[idx], lam
```

### Test-Time Augmentation (TTA)

Multiple augmented views are averaged at inference:

```python
def predict_with_tta(model, image, transforms):
    predictions = []
    for transform in transforms:
        augmented = transform(image)
        pred = model(augmented)
        predictions.append(pred)
    
    return torch.stack(predictions).mean(dim=0)
```

---

## Hyperparameter Tuning

### Recommended Starting Points

| Parameter | Small Dataset (<5K) | Medium (5K-50K) | Large (>50K) |
|-----------|---------------------|-----------------|--------------|
| `batch_size` | 16 | 32 | 64-128 |
| `learning_rate` | 5e-5 | 1e-4 | 3e-4 |
| `warmup_epochs` | 10 | 5 | 3 |
| `weight_decay` | 0.05 | 0.01 | 0.001 |
| `drop_path_rate` | 0.3 | 0.2 | 0.1 |

### Learning Rate Schedule

Cosine decay with warmup is recommended:

```
LR
 │
 │    ╭──────────────────╮
 │   ╱                    ╲
 │  ╱                      ╲
 │ ╱                        ╲
 │╱                          ╲
 └────────────────────────────── Epoch
   ↑         ↑
 warmup    peak
```

### Gradient Accumulation

For limited GPU memory, use gradient accumulation:

```yaml
training:
  batch_size: 8  # Actual batch per step
  gradient_accumulation_steps: 4  # Effective batch = 8 × 4 = 32
```

---

## Multi-GPU Training

### Data Parallel (Single Node)

```bash
# Automatic DataParallel
python scripts/finetune.py --config configs/finetune.yaml
```

### Distributed Data Parallel

```bash
# Multi-GPU with torchrun
torchrun --nproc_per_node=4 scripts/finetune.py \
    --config configs/finetune.yaml \
    --distributed
```

---

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**: `training.batch_size: 16`
2. **Enable gradient checkpointing**:
   ```yaml
   model:
     use_gradient_checkpointing: true
   ```
3. **Use gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 4
   ```
4. **Reduce image size**: `data.img_size: [224, 224]`

### Training Not Converging

1. **Lower learning rate**: Start with `1e-5`
2. **Increase warmup epochs**: `warmup_epochs: 15`
3. **Check data augmentation**: Try `severity: light`
4. **Verify class balance**: Use `get_class_weights()`

### NaN Loss

1. **Reduce learning rate drastically**: `1e-6`
2. **Add gradient clipping**:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
3. **Check for corrupted data samples**

### Poor Generalization

1. **Increase regularization**:
   - `drop_path_rate: 0.3`
   - `weight_decay: 0.05`
2. **Enable all augmentations**:
   - `use_mixup: true`
   - `use_cutmix: true`
3. **Use EMA**:
   - `use_ema: true`
   - `ema_decay: 0.9999`

---

## Logging and Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/finetune/logs

# View at http://localhost:6006
```

### Logged Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `train/lr` | Current learning rate |
| `val/accuracy` | Validation accuracy |
| `val/f1_macro` | Macro-averaged F1 score |
| `val/loss` | Validation loss |

---

For more details, see:
- [Architecture Documentation](architecture.md)
- [Benchmark Results](benchmarks.md)
- [API Reference](api/core.md)
