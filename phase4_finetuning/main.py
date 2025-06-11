# phase4_finetuning/main.py

import torch
import yaml
import argparse
import os
import random
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler

# Use relative imports to access modules within the same package
from .finetune.trainer import EnhancedFinetuner
from .utils.logging_setup import setup_logging
from .utils.augmentations import create_cotton_leaf_augmentation
from .dataset import SARCLD2024Dataset
from phase2_model.models.hvt import create_disease_aware_hvt

def run(cfg):
    """ Main execution function """
    # --- Setup ---
    run_name = f"{cfg['run_name_prefix']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(cfg['PACKAGE_ROOT_PATH'], 'logs_finetune', run_name)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir=log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting fine-tuning run: {run_name}")
    logger.info(f"Full configuration:\n{yaml.dump(cfg, indent=2)}")

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
        if cfg['cudnn_benchmark']:
            torch.backends.cudnn.benchmark = True

    # --- Datasets and Dataloaders ---
    logger.info("Setting up datasets and dataloaders...")
    img_size = tuple(cfg['data']['img_size'])

    # Create augmentation pipelines
    train_augs = create_cotton_leaf_augmentation(
        strategy=cfg['augmentations']['strategy'],
        img_size=img_size,
        severity=cfg['augmentations']['severity']
    ) if cfg['augmentations']['enable'] else create_cotton_leaf_augmentation(strategy='minimal', img_size=img_size)

    val_augs = create_cotton_leaf_augmentation(
        strategy='minimal',
        img_size=img_size
    )

    # Initialize datasets with explicit arguments
    train_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="train",
        transform=train_augs,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name']
    )
    val_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="val",
        transform=val_augs,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name']
    )

    cfg['data']['num_classes'] = len(train_dataset.get_class_names())
    logger.info(f"Discovered {cfg['data']['num_classes']} classes.")

    sampler = None
    if cfg['data']['use_weighted_sampler']:
        targets = train_dataset.get_targets()
        class_counts = np.bincount(targets, minlength=cfg['data']['num_classes'])
        class_counts = class_counts + 1e-9
        weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = weights[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_args = {
        'num_workers': cfg['data']['num_workers'],
        'pin_memory': True,
        'persistent_workers': cfg['data']['num_workers'] > 0,
        'prefetch_factor': cfg['data']['prefetch_factor']
    }
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], sampler=sampler, shuffle=(sampler is None), drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'] * 2, shuffle=False, **loader_args)

    # --- Model ---
    logger.info("Creating HVT model for fine-tuning...")
    model = create_disease_aware_hvt(
        current_img_size=img_size,
        num_classes=cfg['data']['num_classes'],
        model_params_dict=cfg['model']['hvt_params']
    )

    # --- Trainer ---
    logger.info("Initializing EnhancedFinetuner...")
    trainer = EnhancedFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=log_dir
    )

    # --- Run Training ---
    logger.info("Starting the training and validation loop...")
    trainer.run()
    logger.info(f"Fine-tuning run '{run_name}' finished successfully.")

if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Advanced Fine-tuning for Cotton Leaf Disease Detection")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config['PACKAGE_ROOT_PATH'] = current_dir
    config['PROJECT_ROOT_PATH'] = project_root

    run(config)