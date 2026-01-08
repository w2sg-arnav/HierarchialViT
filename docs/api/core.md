# 📚 API Reference

This document provides the API reference for the HierarchicalViT (HViT) package.

## Table of Contents

- [Models](#models)
- [Data](#data)
- [Training](#training)
- [Utilities](#utilities)

---

## Models

### `hvit.models`

#### `create_disease_aware_hvt`

Factory function to create a DiseaseAwareHVT model.

```python
from hvit.models import create_disease_aware_hvt

model = create_disease_aware_hvt(
    current_img_size: Tuple[int, int],
    num_classes: int,
    model_params_dict: Dict[str, Any]
) -> DiseaseAwareHVT
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `current_img_size` | `Tuple[int, int]` | Input image size as (height, width) |
| `num_classes` | `int` | Number of output classes |
| `model_params_dict` | `Dict[str, Any]` | Model configuration dictionary |

**Model Parameters Dictionary:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `patch_size` | `int` | 16 | Patch size for tokenization |
| `embed_dim_rgb` | `int` | 96 | RGB embedding dimension |
| `embed_dim_spectral` | `int` | 96 | Spectral embedding dimension |
| `spectral_channels` | `int` | 0 | Number of spectral channels (0=RGB only) |
| `depths` | `List[int]` | [2, 2, 6, 2] | Number of blocks per stage |
| `num_heads` | `List[int]` | [3, 6, 12, 24] | Attention heads per stage |
| `mlp_ratio` | `float` | 4.0 | MLP expansion ratio |
| `qkv_bias` | `bool` | True | Use bias in QKV projections |
| `model_drop_rate` | `float` | 0.0 | Dropout rate |
| `attn_drop_rate` | `float` | 0.0 | Attention dropout rate |
| `drop_path_rate` | `float` | 0.1 | Drop path (stochastic depth) rate |
| `norm_layer_name` | `str` | "LayerNorm" | Normalization layer type |
| `use_dfca` | `bool` | False | Enable Disease-Focused Cross-Attention |
| `use_gradient_checkpointing` | `bool` | False | Enable gradient checkpointing |

**Example:**

```python
from hvit.models import create_disease_aware_hvt

model = create_disease_aware_hvt(
    current_img_size=(256, 256),
    num_classes=7,
    model_params_dict={
        "embed_dim_rgb": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "use_dfca": True,
        "drop_path_rate": 0.2,
    }
)
```

---

#### `DiseaseAwareHVT`

Main model class for hierarchical vision transformer.

```python
class DiseaseAwareHVT(nn.Module):
    def forward(
        self,
        rgb_img: torch.Tensor,
        spectral_img: Optional[torch.Tensor] = None,
        mode: str = 'classify'
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

**Forward Modes:**

| Mode | Return Type | Description |
|------|-------------|-------------|
| `'classify'` | `Tensor (B, C)` | Classification logits |
| `'get_embeddings'` | `Dict[str, Tensor]` | Feature embeddings |
| `'get_all_features'` | `Dict[str, Tensor]` | All intermediate features |

**Example:**

```python
import torch
from hvit.models import DiseaseAwareHVT

# Classification
logits = model(images, mode='classify')
predictions = logits.argmax(dim=1)

# Get embeddings
embeddings = model(images, mode='get_embeddings')
features = embeddings['fused_pooled']  # Shape: (B, 768)
```

---

#### `DiseaseFocusedCrossAttention`

Cross-attention module for multi-modal fusion.

```python
from hvit.models import DiseaseFocusedCrossAttention

dfca = DiseaseFocusedCrossAttention(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embed_dim` | `int` | - | Embedding dimension |
| `num_heads` | `int` | - | Number of attention heads |
| `dropout` | `float` | 0.1 | Dropout probability |

---

## Data

### `hvit.data`

#### `SARCLD2024Dataset`

Dataset class for SAR-CLD-2024 cotton leaf disease images.

```python
from hvit.data import SARCLD2024Dataset

dataset = SARCLD2024Dataset(
    root_dir: str,
    img_size: Tuple[int, int] = (256, 256),
    split: str = 'train',
    transform: Optional[Callable] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | `str` | - | Path to dataset root |
| `img_size` | `Tuple[int, int]` | (256, 256) | Target image size |
| `split` | `str` | 'train' | Dataset split ('train', 'val', 'test') |
| `transform` | `Callable` | None | Optional transform pipeline |

**Methods:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `__len__()` | `int` | Number of samples |
| `__getitem__(idx)` | `Tuple[Tensor, int]` | (image, label) tuple |
| `get_class_names()` | `List[str]` | List of class names |
| `get_class_weights()` | `Tensor` | Inverse frequency weights |

**Example:**

```python
from hvit.data import SARCLD2024Dataset, get_train_transforms

transform = get_train_transforms(img_size=(256, 256), severity='moderate')
dataset = SARCLD2024Dataset(
    root_dir='/path/to/data',
    split='train',
    transform=transform
)

image, label = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

---

#### `get_train_transforms`

Create training augmentation pipeline.

```python
from hvit.data import get_train_transforms

transform = get_train_transforms(
    img_size: Tuple[int, int],
    severity: str = 'moderate'
) -> Compose
```

**Parameters:**

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `img_size` | `Tuple[int, int]` | - | Target image size |
| `severity` | `str` | 'light', 'moderate', 'aggressive' | Augmentation strength |

---

#### `get_val_transforms`

Create validation/test transform pipeline.

```python
from hvit.data import get_val_transforms

transform = get_val_transforms(
    img_size: Tuple[int, int]
) -> Compose
```

---

#### `SimCLRAugmentation`

Augmentation pipeline for SimCLR contrastive learning.

```python
from hvit.data import SimCLRAugmentation

aug = SimCLRAugmentation(
    img_size: Tuple[int, int],
    s: float = 1.0,
    p_grayscale: float = 0.2,
    p_gaussian_blur: float = 0.5
)

# Returns two augmented views
view1, view2 = aug(batch_images)
```

---

#### `SARCLD_CLASSES`

List of class names in the SAR-CLD-2024 dataset.

```python
from hvit.data import SARCLD_CLASSES

print(SARCLD_CLASSES)
# ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 
#  'Herbicide Growth Damage', 'Leaf Hopper Jassids', 
#  'Leaf Redding', 'Leaf Variegation']
```

---

## Training

### `hvit.training`

#### `Pretrainer`

Self-supervised pre-training trainer using SimCLR.

```python
from hvit.training import Pretrainer

trainer = Pretrainer(
    model: nn.Module,
    augmentations: SimCLRAugmentation,
    loss_fn: InfoNCELoss,
    config: Dict[str, Any],
    device: str = 'cuda'
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `train_one_epoch(dataloader, epoch)` | Train for one epoch |
| `evaluate_linear_probe(epoch)` | Evaluate with linear probe |
| `save_checkpoint(epoch, path, ...)` | Save checkpoint |

---

#### `EnhancedFinetuner`

Enhanced fine-tuning trainer with EMA, MixUp, CutMix, and TTA.

```python
from hvit.training import EnhancedFinetuner

trainer = EnhancedFinetuner(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    output_dir: str
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run()` | Run full training loop |
| `train_one_epoch(epoch)` | Train for one epoch |
| `validate(epoch)` | Run validation |
| `save_checkpoint(epoch, is_best)` | Save checkpoint |

---

#### `InfoNCELoss`

Contrastive loss for self-supervised learning.

```python
from hvit.training import InfoNCELoss

loss_fn = InfoNCELoss(temperature: float = 0.07)

# features: (2*B, D) - concatenated positive pairs
loss = loss_fn(features)
```

---

#### `FocalLoss`

Focal loss for handling class imbalance.

```python
from hvit.training import FocalLoss

loss_fn = FocalLoss(
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    label_smoothing: float = 0.0
)

loss = loss_fn(predictions, targets)
```

---

#### `CombinedLoss`

Combined Focal + Cross-Entropy loss.

```python
from hvit.training import CombinedLoss

loss_fn = CombinedLoss(
    focal_weight: float = 0.5,
    gamma: float = 2.0,
    label_smoothing: float = 0.1
)

loss = loss_fn(predictions, targets)
```

---

## Utilities

### `hvit.utils`

#### `EMA`

Exponential Moving Average for model weights.

```python
from hvit.utils import EMA

ema = EMA(model: nn.Module, decay: float = 0.9999)

# During training
ema.update()

# For evaluation
ema.apply_shadow()  # Apply EMA weights
model.eval()
# ... evaluate ...
ema.restore()  # Restore original weights
```

**Methods:**

| Method | Description |
|--------|-------------|
| `update()` | Update shadow weights |
| `apply_shadow()` | Apply EMA weights to model |
| `restore()` | Restore original weights |
| `state_dict()` | Get EMA state dict |
| `load_state_dict(state_dict)` | Load EMA state dict |

---

#### `compute_metrics`

Compute classification metrics.

```python
from hvit.utils import compute_metrics

metrics = compute_metrics(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]
```

**Returns:**

| Key | Description |
|-----|-------------|
| `accuracy` | Overall accuracy |
| `f1_macro` | Macro-averaged F1 |
| `f1_weighted` | Weighted F1 |
| `precision_macro` | Macro-averaged precision |
| `recall_macro` | Macro-averaged recall |
| `f1_{class_name}` | Per-class F1 scores |

---

#### `setup_logging`

Configure logging for training scripts.

```python
from hvit.utils import setup_logging

logger = setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger
```

---

## Quick Reference

### Import Cheatsheet

```python
# Models
from hvit.models import (
    DiseaseAwareHVT,
    create_disease_aware_hvt,
    DiseaseFocusedCrossAttention,
)

# Data
from hvit.data import (
    SARCLD2024Dataset,
    SARCLD_CLASSES,
    SimCLRAugmentation,
    get_train_transforms,
    get_val_transforms,
)

# Training
from hvit.training import (
    Pretrainer,
    EnhancedFinetuner,
    InfoNCELoss,
    FocalLoss,
    CombinedLoss,
)

# Utilities
from hvit.utils import (
    EMA,
    compute_metrics,
    setup_logging,
)
```

### Complete Example

```python
import torch
from hvit.models import create_disease_aware_hvt
from hvit.data import SARCLD2024Dataset, get_train_transforms
from hvit.training import EnhancedFinetuner

# Create model
model = create_disease_aware_hvt(
    current_img_size=(256, 256),
    num_classes=7,
    model_params_dict={
        "embed_dim_rgb": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
    }
)

# Create dataset
transform = get_train_transforms((256, 256), severity='moderate')
dataset = SARCLD2024Dataset('/path/to/data', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Train
config = {
    'training': {'epochs': 100, 'learning_rate': 1e-4},
    'device': 'cuda'
}
trainer = EnhancedFinetuner(model, loader, loader, config, 'outputs/')
trainer.run()
```
