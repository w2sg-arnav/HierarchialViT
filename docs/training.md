# Training Instructions

This document provides detailed instructions for training the HierarchialViT model.

## Pre-training (Phase 3)

1. **Data Preparation**
   - Prepare your dataset according to the format in `phase3_pretraining/dataset.py`
   - Set the data paths in configuration

2. **Configuration**
   - Modify `phase3_pretraining/config.py` for:
     - Model architecture
     - Training hyperparameters
     - Data augmentation settings

3. **Running Pre-training**
   ```bash
   cd phase3_pretraining
   python run_ssl_pretraining.py
   ```

## Fine-tuning (Phase 4)

1. **Dataset Setup**
   - Prepare your target dataset
   - Update paths in `phase4_finetuning/config.yaml`

2. **Fine-tuning Process**
   ```bash
   cd phase4_finetuning
   python main.py --config config.yaml
   ```

3. **Monitoring**
   - Training progress is logged to tensorboard
   - Check logs in `logs_finetune/` directory

## Evaluation and Analysis (Phase 5)

Various scripts are provided for model analysis:

- `analyze_best_model.py`: Comprehensive model evaluation
- `test_robustness.py`: Test model robustness
- `visualize_attention.py`: Visualize attention patterns
- `test_adversarial.py`: Test against adversarial attacks
