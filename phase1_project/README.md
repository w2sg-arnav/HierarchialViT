# Phase 1: Initial Project Setup and Baseline Implementation

This phase contains the foundational implementation of the HierarchialViT project.

## Directory Structure

```
phase1_project/
├── config.py          # Configuration and hyperparameters
├── data_utils.py      # Data loading and processing utilities
├── dataset.py         # Dataset implementations
├── main.py           # Main training script
├── models.py         # Model architecture implementations
├── progression.py    # Training progression tracking
├── train.py         # Training loop implementation
└── transforms.py     # Data augmentation transformations
```

## Components

### Configuration (`config.py`)
Defines all hyperparameters and configuration options for the model and training process.

### Data Utilities (`data_utils.py`)
- Image loading and preprocessing
- Batch processing utilities
- Data pipeline helpers

### Dataset Implementation (`dataset.py`)
- Custom dataset classes
- Data loading and augmentation pipeline
- Support for various input formats

### Models (`models.py`)
Base implementation of the vision transformer architecture including:
- Patch embedding
- Multi-head attention
- Transformer blocks
- Position embedding

### Training (`train.py`)
Implementation of the training loop with:
- Forward/backward passes
- Loss computation
- Optimization steps
- Validation process

### Transforms (`transforms.py`)
Custom data augmentation implementations:
- Random cropping
- Color jittering
- Random flipping
- Normalization

## Usage

```bash
# Train the base model
python main.py --config configs/base_config.yaml

# Evaluate the model
python main.py --config configs/base_config.yaml --evaluate
```

## Configuration Options

Example configuration in `config.py`:
```python
model_config = {
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4,
    'qkv_bias': True,
}

training_config = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 0.05,
    'epochs': 300,
}
```
