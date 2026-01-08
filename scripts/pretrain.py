#!/usr/bin/env python
# scripts/pretrain.py
"""Self-supervised pre-training script for HViT.

This script runs SimCLR-based self-supervised pre-training on the
SAR-CLD-2024 dataset to learn robust visual representations.

Usage:
    python scripts/pretrain.py --config configs/pretrain.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hvit.models import DiseaseAwareHVT, create_disease_aware_hvt
from hvit.data import SARCLD2024Dataset, SimCLRAugmentation, get_val_transforms
from hvit.training import Pretrainer, InfoNCELoss
from hvit.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="HViT Self-Supervised Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pretrain",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Starting HViT Self-Supervised Pre-training")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    config['device'] = args.device
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    img_size = tuple(config['data'].get('img_size', (224, 224)))
    
    # For pre-training, we use a simple transform that converts to tensor
    import torchvision.transforms.v2 as T_v2
    base_transform = T_v2.Compose([
        T_v2.Resize(img_size, antialias=True),
        T_v2.ToImage(),
        T_v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_dataset = SARCLD2024Dataset(
        root_dir=config['data']['root_dir'],
        img_size=img_size,
        split='train',
        transform=base_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Create augmentation pipeline
    augmentations = SimCLRAugmentation(
        img_size=img_size,
        s=config.get('simclr_s', 1.0),
        p_grayscale=config.get('simclr_p_grayscale', 0.2),
        p_gaussian_blur=config.get('simclr_p_gaussian_blur', 0.5)
    )
    
    # Create model
    from hvit.models.hvt import DiseaseAwareHVT
    import torch.nn as nn
    
    # Create a wrapper for pre-training
    class HVTForPretraining(nn.Module):
        def __init__(self, hvt_model, projection_dim=128):
            super().__init__()
            self.backbone = hvt_model
            self.backbone_init_config = hvt_model.hvt_params
            
            # Get feature dimension
            feature_dim = hvt_model.final_encoded_dim_rgb
            
            # Projection head for SimCLR
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, projection_dim)
            )
        
        def forward(self, rgb_img, mode='pretrain'):
            # Get encoded features
            x_rgb_encoded, _, _, _ = self.backbone.forward_features_encoded(rgb_img)
            
            # Global average pooling
            features = x_rgb_encoded.mean(dim=1)
            
            if mode == 'pretrain':
                return self.projection_head(features)
            elif mode == 'probe_extract':
                return features
            else:
                return self.backbone(rgb_img, mode=mode)
    
    hvt_backbone = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=config['data'].get('num_classes', 7),
        model_params_dict=config['model']
    )
    
    model = HVTForPretraining(
        hvt_backbone,
        projection_dim=config.get('projection_dim', 128)
    )
    
    # Create loss function
    loss_fn = InfoNCELoss(temperature=config.get('temperature', 0.07))
    
    # Create trainer
    trainer = Pretrainer(
        model=model,
        augmentations=augmentations,
        loss_fn=loss_fn,
        config=config['training'],
        device=args.device
    )
    
    # Training loop
    total_epochs = config['training']['epochs']
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    best_probe_acc = -1.0
    
    for epoch in range(1, total_epochs + 1):
        # Train one epoch
        avg_loss = trainer.train_one_epoch(train_loader, epoch)
        
        # Evaluate with linear probe periodically
        if epoch % config.get('probe_every', 10) == 0:
            probe_acc = trainer.evaluate_linear_probe(epoch)
            
            if probe_acc > best_probe_acc:
                best_probe_acc = probe_acc
                trainer.save_checkpoint(
                    epoch,
                    str(checkpoint_dir),
                    best_probe_accuracy=probe_acc,
                    is_best=True
                )
        
        # Regular checkpoint
        if epoch % config.get('save_every', 25) == 0:
            trainer.save_checkpoint(
                epoch,
                str(checkpoint_dir),
                best_probe_accuracy=best_probe_acc,
                is_best=False
            )
    
    # Final checkpoint
    trainer.save_checkpoint(
        total_epochs,
        str(checkpoint_dir),
        best_probe_accuracy=best_probe_acc,
        is_best=False
    )
    
    logger.info(f"Pre-training complete. Best probe accuracy: {best_probe_acc:.2f}%")


if __name__ == "__main__":
    main()
