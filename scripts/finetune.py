#!/usr/bin/env python
# scripts/finetune.py
"""Fine-tuning script for HViT classification.

This script fine-tunes a pre-trained HViT model on the SAR-CLD-2024
dataset for cotton disease classification.

Usage:
    python scripts/finetune.py --config configs/finetune.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hvit.models import create_disease_aware_hvt
from hvit.data import SARCLD2024Dataset, get_train_transforms, get_val_transforms
from hvit.training import EnhancedFinetuner
from hvit.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="HViT Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
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
        default=None,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--ssl-checkpoint",
        type=str,
        default=None,
        help="Path to SSL pre-trained checkpoint"
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
        help="Path to fine-tune checkpoint to resume"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Starting HViT Fine-tuning")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    if args.ssl_checkpoint:
        config['model']['ssl_pretrained_path'] = args.ssl_checkpoint
    if args.resume:
        config['model']['resume_finetune_path'] = args.resume
    
    config['device'] = args.device
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs/finetune") / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Setup data
    img_size = tuple(config['data'].get('img_size', (256, 256)))
    
    train_transform = get_train_transforms(
        img_size=img_size,
        severity=config.get('augmentations', {}).get('severity', 'moderate')
    )
    val_transform = get_val_transforms(img_size=img_size)
    
    train_dataset = SARCLD2024Dataset(
        root_dir=config['data']['root_dir'],
        img_size=img_size,
        split='train',
        transform=train_transform
    )
    
    val_dataset = SARCLD2024Dataset(
        root_dir=config['data']['root_dir'],
        img_size=img_size,
        split='val',
        transform=val_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training'].get('val_batch_size', config['training']['batch_size']),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=config['data']['num_classes'],
        model_params_dict=config['model']
    )
    
    # Create trainer
    trainer = EnhancedFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=str(output_dir)
    )
    
    # Run training
    trainer.run()
    
    logger.info(f"Fine-tuning complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
