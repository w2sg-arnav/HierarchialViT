# Cotton Leaf Disease Detection Research

## Project Overview

This research project implements a novel hierarchical vision transformer architecture for cotton leaf disease detection, incorporating multi-modal fusion, self-supervised learning, and few-shot capabilities.

## Technical Components

### 1. Hierarchical Vision Transformer (HViT)
- Multi-scale feature processing
- Disease-specific attention mechanisms
- Progressive dimension reduction
- Cross-stage feature fusion

### 2. Self-Supervised Learning
- Masked Image Modeling (MIM)
- Contrastive Learning
- Disease-aware pretext tasks

### 3. Multi-Modal Fusion
- RGB-Spectral fusion
- Attention-guided modality weighting
- Cross-modal consistency

### 4. Few-Shot Learning
- Meta-learning framework
- Prototypical networks
- Uncertainty calibration

## Directory Structure

```
research/
├── architectures/       # Model architectures
├── experiments/        # Experiment configurations
├── datasets/          # Dataset handling
├── evaluations/       # Evaluation protocols
├── analysis/         # Result analysis
└── visualizations/   # Visualization tools
```

## Experiment Tracking

All experiments are tracked using Weights & Biases with:
- Hyperparameter configurations
- Training metrics
- Evaluation results
- Visualizations

## Getting Started

1. Set up environment:
```bash
mamba env create -p ./env -f environment.yml
```

2. Prepare datasets:
```bash
python research/datasets/prepare_data.py
```

3. Run experiments:
```bash
python research/experiments/run_experiment.py --config configs/experiments/baseline.yaml
```

## Results

Current results on the SAR-CLD-2024 dataset:
- Accuracy: XX%
- F1-Score: XX%
- Early Detection Rate: XX%

For detailed results and analysis, see [results.md](./results.md).
