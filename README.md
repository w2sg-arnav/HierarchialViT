# HierarchicalViT (HViT): Disease-Aware Vision Transformer for Plant Disease Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

HierarchicalViT (HViT) is a novel hierarchical vision transformer architecture designed for plant disease classification, specifically optimized for the SAR-CLD-2024 cotton leaf disease dataset. The architecture features:

- **Hierarchical Multi-Stage Processing** with progressive spatial downsampling
- **Disease-Focused Cross-Attention (DFCA)** for multi-modal fusion
- **SimCLR-based Self-Supervised Pre-training** for robust feature learning
- **Advanced Fine-tuning** with EMA, MixUp, CutMix, and Test-Time Augmentation

## Key Features

- 🌱 **7-Class Cotton Disease Classification**: Bacterial Blight, Curl Virus, Healthy Leaf, Herbicide Growth Damage, Leaf Hopper Jassids, Leaf Redding, Leaf Variegation
- 🔬 **Hierarchical Architecture**: 4-stage transformer with [2, 2, 6, 2] depth configuration
- 🎯 **Self-Supervised Pre-training**: SimCLR with InfoNCE loss and cosine warmup scheduler
- 📊 **State-of-the-Art Results**: Outperforms ResNet-101 and ViT-Base baselines
- ⚡ **Efficient Design**: Gradient checkpointing and mixed-precision training support

## Project Structure

```
HierarchicalViT/
├── hvit/                    # Main package
│   ├── models/              # Model implementations
│   │   ├── hvt.py           # Core DiseaseAwareHVT model
│   │   ├── dfca.py          # Disease-Focused Cross-Attention
│   │   └── baseline.py      # InceptionV3 baseline
│   ├── data/                # Data handling
│   │   ├── dataset.py       # SARCLD2024Dataset
│   │   └── augmentations.py # SimCLR and disease-specific augmentations
│   ├── training/            # Training utilities
│   │   ├── pretrainer.py    # Self-supervised pre-trainer
│   │   ├── finetuner.py     # Enhanced fine-tuning
│   │   └── losses.py        # InfoNCE, Focal, Combined losses
│   └── utils/               # Utilities
│       ├── ema.py           # Exponential Moving Average
│       ├── metrics.py       # Evaluation metrics
│       └── logging_setup.py # Logging configuration
├── scripts/                 # Entry point scripts
│   ├── pretrain.py          # SSL pre-training script
│   ├── finetune.py          # Fine-tuning script
│   └── evaluate.py          # Evaluation script
├── configs/                 # Configuration files
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/w2sg-arnav/HierarchialViT.git
cd HierarchialViT
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Pre-training (Self-Supervised)
```bash
python scripts/pretrain.py --config configs/pretrain.yaml --data-dir /path/to/data
```

### Fine-tuning
```bash
python scripts/finetune.py --config configs/finetune.yaml --ssl-checkpoint outputs/pretrain/best.pth
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint outputs/finetune/best_model.pth --split val
```

## Python API

```python
from hvit.models import create_disease_aware_hvt, DiseaseAwareHVT
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

# Forward pass
output = model(images, mode='classify')
```

## Model Architecture

The DiseaseAwareHVT follows a Swin-like hierarchical design:

| Stage | Resolution | Channels | Depth | Heads |
|-------|------------|----------|-------|-------|
| 1     | H/4 × W/4  | 96       | 2     | 3     |
| 2     | H/8 × W/8  | 192      | 2     | 6     |
| 3     | H/16 × W/16| 384      | 6     | 12    |
| 4     | H/32 × W/32| 768      | 2     | 24    |

For detailed architecture, see [docs/architecture.md](docs/architecture.md).

## Results

| Model | Accuracy | F1-Macro | F1-Weighted |
|-------|----------|----------|-------------|
| ResNet-101 | 85.2% | 84.8% | 85.1% |
| ViT-Base | 87.3% | 86.9% | 87.2% |
| **HViT (Ours)** | **91.5%** | **91.2%** | **91.4%** |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed results.

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hierarchicalvit2025,
  title={HierarchicalViT: A Disease-Aware Hierarchical Vision Transformer for Plant Disease Classification},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
