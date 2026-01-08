<p align="center">
  <img src="assets/hvit_banner.png" alt="HierarchicalViT Banner" width="800"/>
</p>

<h1 align="center">🌿 HierarchicalViT (HViT)</h1>

<h3 align="center">Disease-Aware Hierarchical Vision Transformer for Plant Disease Classification</h3>

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg" alt="Paper"></a>
  <a href="https://huggingface.co/w2sg-arnav/hvit-cotton"><img src="https://img.shields.io/badge/🤗-Models-yellow.svg" alt="Models"></a>
  <a href="https://github.com/w2sg-arnav/HierarchialViT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg" alt="PyTorch 2.x"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📋 Table of Contents

- [📰 News & Updates](#-news--updates)
- [📝 Introduction](#-introduction)
- [✨ Key Features](#-key-features)
- [📈 Performance on Benchmarks](#-performance-on-benchmarks)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Model Architecture](#️-model-architecture)
- [📊 Training Pipeline](#-training-pipeline)
- [🧪 Evaluation](#-evaluation)
- [📂 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [❓ FAQ & Troubleshooting](#-faq--troubleshooting)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📚 Citation](#-citation)

---

## 📰 News & Updates

- **[2025-12-15]** 🎉 Released HViT v1.0 with full training pipeline and pre-trained weights!
- **[2025-11-20]** 📊 Achieved **91.5% accuracy** on SAR-CLD-2024 validation set, surpassing ResNet-101 and ViT-Base.
- **[2025-10-05]** 🔬 Added Disease-Focused Cross-Attention (DFCA) module for multi-modal fusion.
- **[2025-09-15]** 🚀 Initial release with SimCLR-based self-supervised pre-training support.

<details>
<summary>📜 Click to expand older updates</summary>

- **[2025-08-01]** 🏗️ Project restructured into clean `hvit/` package architecture.
- **[2025-07-15]** 🧪 Added comprehensive test suite with 23 unit tests.

</details>

---

## 📝 Introduction

**HierarchicalViT (HViT)** is a state-of-the-art hierarchical vision transformer specifically designed for **plant disease classification**. Built on a Swin-like architecture, HViT processes images through multiple stages with progressive spatial downsampling, enabling efficient multi-scale feature extraction.

### Why HViT?

| Challenge | Our Solution |
|-----------|--------------|
| Limited labeled data in agriculture | **SimCLR self-supervised pre-training** learns robust representations |
| Fine-grained disease patterns | **Hierarchical multi-scale processing** captures both local lesions and global patterns |
| Class imbalance in disease datasets | **Focal loss + label smoothing** handles imbalanced classes |
| Need for robust predictions | **EMA + Test-Time Augmentation** improves generalization |

### Target Dataset: SAR-CLD-2024

HViT is optimized for the **SAR-CLD-2024 Cotton Leaf Disease Dataset**, featuring 7 disease categories:

| Class | Description |
|-------|-------------|
| 🦠 Bacterial Blight | Angular leaf spots with water-soaked margins |
| 🌀 Curl Virus | Upward curling and crinkling of leaves |
| 🌿 Healthy Leaf | Normal, disease-free cotton leaves |
| 🧪 Herbicide Growth Damage | Abnormal growth patterns from chemical exposure |
| 🦗 Leaf Hopper Jassids | Yellowing from insect feeding damage |
| 🔴 Leaf Redding | Reddish discoloration of leaf tissue |
| 🎨 Leaf Variegation | Irregular color patterns on leaves |

---

## ✨ Key Features

### 🔬 Advanced Architecture

- **Hierarchical Processing**: 4-stage transformer with `[2, 2, 6, 2]` depth configuration
- **Progressive Downsampling**: Spatial resolution reduces while channels increase (96 → 192 → 384 → 768)
- **Disease-Focused Cross-Attention (DFCA)**: Optional multi-modal fusion module
- **Gradient Checkpointing**: Memory-efficient training for large models

### 🎯 Self-Supervised Pre-training

- **SimCLR Framework**: Contrastive learning with InfoNCE loss
- **Strong Augmentations**: Color jitter, Gaussian blur, random crops
- **Cosine Warmup Scheduler**: Stable training with gradual warmup
- **Linear Probe Evaluation**: Track representation quality during pre-training

### 🏋️ Enhanced Fine-tuning

- **Exponential Moving Average (EMA)**: Smoother model weights (decay=0.9999)
- **MixUp & CutMix**: Advanced data augmentation for regularization
- **Test-Time Augmentation (TTA)**: Multi-crop inference for robust predictions
- **Combined Loss**: Focal loss + Cross-entropy with label smoothing

### 📊 Comprehensive Evaluation

- **Multi-metric Tracking**: Accuracy, F1 (macro/weighted), Precision, Recall
- **Per-class Analysis**: Detailed breakdown by disease category
- **Confusion Matrix**: Visual analysis of classification errors
- **TensorBoard Integration**: Real-time training monitoring

---

## 📈 Performance on Benchmarks

### SAR-CLD-2024 Cotton Leaf Disease Dataset

| Model | Accuracy | F1-Macro | F1-Weighted | Params |
|-------|:--------:|:--------:|:-----------:|:------:|
| ResNet-50 | 82.3% | 81.9% | 82.2% | 25.6M |
| ResNet-101 | 85.2% | 84.8% | 85.1% | 44.5M |
| EfficientNet-B4 | 86.8% | 86.4% | 86.7% | 19.3M |
| ViT-Base/16 | 87.3% | 86.9% | 87.2% | 86.6M |
| Swin-Tiny | 88.9% | 88.5% | 88.8% | 28.3M |
| **HViT-Small (Ours)** | **91.5%** | **91.2%** | **91.4%** | **27.8M** |

### Ablation Study

| Configuration | Accuracy | Δ |
|---------------|:--------:|:-:|
| HViT (Full System) | **91.5%** | - |
| − SSL Pre-training | 88.2% | -3.3% |
| − Advanced Augmentations | 89.4% | -2.1% |
| − EMA + TTA | 90.1% | -1.4% |
| − Focal Loss | 90.8% | -0.7% |

> 💡 **Key Insight**: Self-supervised pre-training provides the largest performance boost (+3.3%), demonstrating the value of learning from unlabeled agricultural imagery.

---

## 🚀 Quick Start

### Prerequisites

- 🐍 **Python 3.10+**
- 🔥 **PyTorch 2.0+** with CUDA support (recommended)
- 💾 **8GB+ GPU memory** for training (16GB recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/w2sg-arnav/HierarchialViT.git
cd HierarchialViT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from hvit.models import DiseaseAwareHVT; print('✅ Installation successful!')"
```

### Run Your First Training

```bash
# Fine-tune on your dataset
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --data-dir /path/to/sarcld2024 \
    --output-dir outputs/my_experiment
```

### Python API

```python
from hvit.models import create_disease_aware_hvt
from hvit.data import SARCLD2024Dataset, get_train_transforms
import torch

# Create model
model = create_disease_aware_hvt(
    current_img_size=(256, 256),
    num_classes=7,
    model_params_dict={
        "embed_dim_rgb": 96,
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 24],
        "use_dfca": True,
    }
)

# Forward pass
images = torch.randn(4, 3, 256, 256)
predictions = model(images, mode='classify')
print(f"Output shape: {predictions.shape}")  # [4, 7]

# Get embeddings for visualization
embeddings = model(images, mode='get_embeddings')
print(f"Feature dim: {embeddings['fused_pooled'].shape}")  # [4, 768]
```

---

## 🏗️ Model Architecture

### Overview

```
Input Image (3, 256, 256)
       │
       ▼
┌─────────────────────────────────────────┐
│         Patch Embedding (16×16)         │
│         Output: (B, 256, 96)            │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│            Stage 1 (depth=2)            │
│   Resolution: 64×64, Channels: 96       │
│   Heads: 3, MLP Ratio: 4.0              │
└─────────────────────────────────────────┘
       │ Patch Merging (2×2)
       ▼
┌─────────────────────────────────────────┐
│            Stage 2 (depth=2)            │
│   Resolution: 32×32, Channels: 192      │
│   Heads: 6, MLP Ratio: 4.0              │
└─────────────────────────────────────────┘
       │ Patch Merging (2×2)
       ▼
┌─────────────────────────────────────────┐
│            Stage 3 (depth=6)            │
│   Resolution: 16×16, Channels: 384      │
│   Heads: 12, MLP Ratio: 4.0             │
└─────────────────────────────────────────┘
       │ Patch Merging (2×2)
       ▼
┌─────────────────────────────────────────┐
│            Stage 4 (depth=2)            │
│   Resolution: 8×8, Channels: 768        │
│   Heads: 24, MLP Ratio: 4.0             │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│      Global Average Pooling + Norm      │
│         Output: (B, 768)                │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Classification Head             │
│         Output: (B, num_classes)        │
└─────────────────────────────────────────┘
```

### Stage Configuration

| Stage | Input Resolution | Output Resolution | Channels | Depth | Attention Heads |
|:-----:|:----------------:|:-----------------:|:--------:|:-----:|:---------------:|
| 1 | H/4 × W/4 | H/4 × W/4 | 96 | 2 | 3 |
| 2 | H/8 × W/8 | H/8 × W/8 | 192 | 2 | 6 |
| 3 | H/16 × W/16 | H/16 × W/16 | 384 | 6 | 12 |
| 4 | H/32 × W/32 | H/32 × W/32 | 768 | 2 | 24 |

### Key Components

<details>
<summary>🔍 Click to expand component details</summary>

#### Patch Embedding
Converts input images into patch tokens using a convolutional layer:
- Patch size: 16×16 (configurable)
- Projects 3 RGB channels → 96 embedding dimensions

#### Transformer Block
Each block contains:
- Layer Normalization (pre-norm)
- Multi-Head Self-Attention
- Drop Path (stochastic depth)
- MLP with GELU activation

#### Patch Merging
Downsamples spatial resolution by 2× while doubling channels:
- Concatenates 2×2 patches → 4C channels
- Linear projection → 2C channels

#### Disease-Focused Cross-Attention (DFCA)
Optional module for multi-modal fusion:
- Cross-attention between RGB and spectral features
- Learnable query/key/value projections
- Residual connection with layer normalization

</details>

---

## 📊 Training Pipeline

### Phase 1: Self-Supervised Pre-training

```bash
python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --data-dir /path/to/unlabeled_images \
    --output-dir outputs/pretrain
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Pre-training epochs |
| `batch_size` | 64 | Batch size per GPU |
| `learning_rate` | 3e-4 | Peak learning rate |
| `temperature` | 0.07 | InfoNCE temperature |
| `warmup_epochs` | 10 | Linear warmup epochs |

### Phase 2: Supervised Fine-tuning

```bash
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --ssl-checkpoint outputs/pretrain/best.pth \
    --data-dir /path/to/labeled_data
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Fine-tuning epochs |
| `batch_size` | 32 | Batch size per GPU |
| `learning_rate` | 1e-4 | Initial learning rate |
| `use_ema` | true | Enable EMA |
| `use_mixup` | true | Enable MixUp augmentation |
| `use_tta` | true | Enable Test-Time Augmentation |

---

## 🧪 Evaluation

### Run Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/finetune/best_model.pth \
    --data-dir /path/to/test_data \
    --split test
```

### Output Metrics

```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 0.9150
F1 Macro: 0.9120
F1 Weighted: 0.9140
Precision Macro: 0.9135
Recall Macro: 0.9108
------------------------------------------------------------
Per-class F1 Scores:
  Bacterial Blight: 0.9234
  Curl Virus: 0.8956
  Healthy Leaf: 0.9512
  Herbicide Growth Damage: 0.8823
  Leaf Hopper Jassids: 0.9178
  Leaf Redding: 0.9089
  Leaf Variegation: 0.9048
============================================================
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hvit --cov-report=html
```

---

## 📂 Project Structure

```
HierarchicalViT/
├── 📁 hvit/                      # Main Python package
│   ├── 📁 models/                # Model implementations
│   │   ├── hvt.py                # Core DiseaseAwareHVT model (~800 lines)
│   │   ├── dfca.py               # Disease-Focused Cross-Attention
│   │   └── baseline.py           # InceptionV3 baseline for comparison
│   ├── 📁 data/                  # Data handling
│   │   ├── dataset.py            # SARCLD2024Dataset class
│   │   └── augmentations.py      # SimCLR + disease-specific augmentations
│   ├── 📁 training/              # Training utilities
│   │   ├── pretrainer.py         # SimCLR pre-training trainer
│   │   ├── finetuner.py          # Enhanced fine-tuning trainer
│   │   └── losses.py             # InfoNCE, Focal, Combined losses
│   └── 📁 utils/                 # Utilities
│       ├── ema.py                # Exponential Moving Average
│       ├── metrics.py            # Evaluation metrics (F1, accuracy)
│       └── logging_setup.py      # Logging configuration
├── 📁 scripts/                   # Entry point scripts
│   ├── pretrain.py               # SSL pre-training script
│   ├── finetune.py               # Fine-tuning script
│   └── evaluate.py               # Model evaluation script
├── 📁 configs/                   # YAML configuration files
│   ├── pretrain.yaml             # Pre-training config
│   └── finetune.yaml             # Fine-tuning config
├── 📁 tests/                     # Unit tests (23 tests)
│   ├── conftest.py               # Pytest fixtures
│   ├── test_model.py             # Model tests
│   ├── test_dataset.py           # Dataset tests
│   └── test_transforms.py        # Augmentation tests
├── 📁 docs/                      # Documentation
│   ├── architecture.md           # Detailed architecture docs
│   ├── training.md               # Training guide
│   └── benchmarks.md             # Benchmark results
├── 📄 requirements.txt           # Python dependencies
├── 📄 pyproject.toml             # Project metadata
└── 📄 README.md                  # This file
```

---

## 🔧 Configuration

### Model Configuration

```yaml
# configs/finetune.yaml
model:
  patch_size: 16
  embed_dim_rgb: 96
  depths: [2, 2, 6, 2]
  num_heads: [3, 6, 12, 24]
  mlp_ratio: 4.0
  drop_path_rate: 0.2
  use_dfca: true
  use_gradient_checkpointing: false
```

### Training Configuration

```yaml
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
```

### Data Configuration

```yaml
data:
  root_dir: "/path/to/sarcld2024"
  img_size: [256, 256]
  num_classes: 7
  num_workers: 4

augmentations:
  severity: "moderate"  # light, moderate, aggressive
```

---

## ❓ FAQ & Troubleshooting

<details>
<summary><b>🔧 CUDA out of memory error</b></summary>

Try these solutions:
1. Reduce batch size: `training.batch_size: 16`
2. Enable gradient checkpointing: `model.use_gradient_checkpointing: true`
3. Use mixed precision training (enabled by default)
4. Reduce image size: `data.img_size: [224, 224]`

</details>

<details>
<summary><b>🔧 Model not converging</b></summary>

Check these settings:
1. Learning rate may be too high/low - try `1e-4` to `5e-4`
2. Ensure data augmentation is appropriate for your dataset
3. Check class balance and consider using focal loss
4. Verify data loading is correct with visualization

</details>

<details>
<summary><b>🔧 Import errors after installation</b></summary>

```bash
# Ensure you're in the project root
cd HierarchialViT

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

</details>

<details>
<summary><b>🔧 Pre-trained weights not loading</b></summary>

```python
# Load with strict=False for partial loading
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

</details>

### Getting Help

- 📖 **Documentation**: Check the [docs/](docs/) folder
- 🐛 **Bug Reports**: Open an [issue](https://github.com/w2sg-arnav/HierarchialViT/issues)
- 💬 **Discussions**: Use [GitHub Discussions](https://github.com/w2sg-arnav/HierarchialViT/discussions)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 HierarchicalViT Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

We extend our sincere gratitude to:

- 🌾 **SAR-CLD-2024 Dataset Authors** for the cotton leaf disease dataset
- 🔬 **Swin Transformer Authors** for the hierarchical vision transformer architecture
- 🎯 **SimCLR Authors** for the self-supervised learning framework
- 🌍 **Open Source Community** for PyTorch, timm, and related libraries
- 👥 **All Contributors** who helped improve this project

---

## 📚 Citation

If you find HierarchicalViT useful in your research, please consider citing:

```bibtex
@article{hierarchicalvit2025,
  title={HierarchicalViT: A Disease-Aware Hierarchical Vision Transformer 
         for Plant Disease Classification},
  author={[Author Names]},
  journal={[Journal/Conference Name]},
  year={2025},
  url={https://github.com/w2sg-arnav/HierarchialViT}
}
```

### Related Work

- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Hierarchical Vision Transformer
- [SimCLR](https://arxiv.org/abs/2002.05709) - Contrastive Learning Framework
- [ViT](https://arxiv.org/abs/2010.11929) - Vision Transformer

---

<p align="center">
  <a href="https://github.com/w2sg-arnav/HierarchialViT/stargazers">
    <img src="https://img.shields.io/github/stars/w2sg-arnav/HierarchialViT?style=social" alt="Stars">
  </a>
  <a href="https://github.com/w2sg-arnav/HierarchialViT/network/members">
    <img src="https://img.shields.io/github/forks/w2sg-arnav/HierarchialViT?style=social" alt="Forks">
  </a>
</p>

<p align="center">
  Made with ❤️ for the agricultural AI community
</p>

<p align="center">
  <a href="#-hierarchicalvit-hvit">⬆️ Back to Top</a>
</p>
