# phase4_finetuning/main.py

import torch
import yaml
import argparse
import os
import random
import numpy as np
import logging
import timm # <--- ADD THIS IMPORT
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler

# --- Explicit Path Setup to make the project root known ---
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- CORRECTED IMPORTS ---
# Use absolute imports from the project root now that the path is set.
from phase4_finetuning.finetune.trainer import EnhancedFinetuner
from phase4_finetuning.utils.logging_setup import setup_logging
from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
from phase4_finetuning.dataset import SARCLD2024Dataset
from phase2_model.models.hvt import create_disease_aware_hvt

def run(cfg):
    """ Main execution function """
    # --- Setup ---
    run_name = f"{cfg['run_name_prefix']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(cfg['PACKAGE_ROOT_PATH'], 'logs_finetune', run_name)
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir=log_dir, log_file_name=f"{run_name}.log")
    
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
    )
    val_augs = create_cotton_leaf_augmentation(
        strategy='minimal',
        img_size=img_size
    )

    # Initialize datasets
    train_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="train",
        transform=train_augs,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],
        random_seed=cfg['seed']
    )
    val_dataset = SARCLD2024Dataset(
        root_dir=cfg['data']['root_dir'],
        split="val",
        transform=val_augs,
        img_size=img_size,
        train_split_ratio=cfg['data']['train_split_ratio'],
        original_dataset_name=cfg['data']['original_dataset_name'],
        augmented_dataset_name=cfg['data']['augmented_dataset_name'],
        random_seed=cfg['seed']
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
        logger.info("Using WeightedRandomSampler for training.")

    loader_args = {'num_workers': cfg['data']['num_workers'], 'pin_memory': True, 'persistent_workers': cfg['data']['num_workers'] > 0}
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], sampler=sampler, shuffle=(sampler is None), drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'] * 2, shuffle=False, **loader_args)

    # --- Model Creation (Now Configurable) ---
    # This is the new, flexible model creation block.
    if 'model_override' in cfg and cfg['model_override'] is not None:
        override_cfg = cfg['model_override']
        model_type = override_cfg.get('type', 'timm')
        model_name = override_cfg.get('name')
        
        logger.info(f"--- MODEL OVERRIDE IN EFFECT ---")
        logger.info(f"Loading baseline model -> Type: {model_type}, Name: {model_name}")

        if model_type.lower() == 'timm':
            # Create a standard model from timm for SOTA comparison
            model = timm.create_model(
                model_name,
                pretrained=True, # Always use ImageNet pre-trained weights for baselines
                num_classes=cfg['data']['num_classes'],
                # For ViT models, we might need to specify the image size if it's not 224
                # For CNNs, this is usually not necessary
                img_size=img_size if 'vit' in model_name else None 
            )
            # The finetuner will handle loading SSL weights (or not) based on the config,
            # but for baselines, `ssl_pretrained_path` should be null.
        else:
            raise ValueError(f"Unknown model_override type specified in config: '{model_type}'")
    else:
        # Default behavior: create our custom HVT model
        logger.info("Creating custom HVT model for fine-tuning...")
        model = create_disease_aware_hvt(
            current_img_size=img_size,
            num_classes=cfg['data']['num_classes'],
            model_params_dict=cfg['model']['hvt_params']
        )
    # --- End of New Model Creation Block ---

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
    try:
        logger.info("Starting the training and validation loop...")
        trainer.run()
        logger.info(f"Fine-tuning run '{run_name}' finished successfully.")
    except Exception as e:
        logger.critical(f"A critical error occurred during the training run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Fine-tuning for Cotton Leaf Disease Detection")
    # Default path assumes running from project root
    parser.add_argument("--config", type=str, default="phase4_finetuning/config.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # The config file path is now relative to where you run the script
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found at '{args.config}'")
        sys.exit(1)

    with open(args.config, 'r') as f:
        config_from_file = yaml.safe_load(f)

    # Add runtime paths to the config
    config_from_file['PACKAGE_ROOT_PATH'] = _current_dir
    config_from_file['PROJECT_ROOT_PATH'] = _project_root
    
    run(config_from_file)