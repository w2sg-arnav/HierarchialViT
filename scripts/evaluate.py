#!/usr/bin/env python
# scripts/evaluate.py
"""Model evaluation script for HViT.

This script evaluates a trained HViT model on a test/validation set
and generates detailed metrics and visualizations.

Usage:
    python scripts/evaluate.py --checkpoint outputs/finetune/best_model.pth
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hvit.models import create_disease_aware_hvt
from hvit.data import SARCLD2024Dataset, get_val_transforms, SARCLD_CLASSES
from hvit.utils import setup_logging, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="HViT Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("Starting HViT Evaluation")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Override with CLI args
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    
    # Setup data
    img_size = tuple(config.get('data', {}).get('img_size', (256, 256)))
    num_classes = config.get('data', {}).get('num_classes', 7)
    
    val_transform = get_val_transforms(img_size=img_size)
    
    dataset = SARCLD2024Dataset(
        root_dir=config['data']['root_dir'],
        img_size=img_size,
        split=args.split,
        transform=val_transform
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Evaluating on {len(dataset)} samples from '{args.split}' split")
    
    # Create model
    model_params = config.get('model', {})
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=num_classes,
        model_params_dict=model_params
    )
    
    # Load weights
    state_dict_key = 'model_state_dict'
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
        logger.info("Using EMA weights for evaluation")
        state_dict_key = 'ema_state_dict'
    
    model.load_state_dict(checkpoint[state_dict_key], strict=False)
    model = model.to(args.device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(args.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    class_names = dataset.get_class_names()
    metrics = compute_metrics(
        all_preds,
        all_labels,
        num_classes=num_classes,
        class_names=class_names
    )
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision Macro: {metrics['precision_macro']:.4f}")
    logger.info(f"Recall Macro: {metrics['recall_macro']:.4f}")
    logger.info("-" * 60)
    logger.info("Per-class F1 Scores:")
    for class_name in class_names:
        key = f"f1_{class_name.replace(' ', '_')}"
        if key in metrics:
            logger.info(f"  {class_name}: {metrics[key]:.4f}")
    logger.info("=" * 60)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"metrics_{args.split}.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
