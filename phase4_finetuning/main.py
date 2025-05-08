# phase4_finetuning/main.py
from collections import OrderedDict 
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LinearLR, SequentialLR
import argparse
import yaml
import torch.nn.functional as F 
import math 

# --- Path Setup ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added project root to sys.path: {project_root}")

# --- Project Imports ---
from phase4_finetuning.config import config as default_config 
from phase4_finetuning.dataset import SARCLD2024Dataset
from phase4_finetuning.utils.augmentations import FinetuneAugmentation
from phase4_finetuning.utils.logging_setup import setup_logging
from phase4_finetuning.finetune.trainer import Finetuner
from phase2_model.models.hvt import DiseaseAwareHVT 

# --- Helper Functions --- 
def load_config_yaml(config_path=None):
    # (Keep as is)
    config = default_config.copy() 
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config: config.update(yaml_config)
            print(f"Loaded configuration from YAML: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load/parse YAML {config_path}. Error: {e}. Using defaults.")
    else:
         print("No config file path provided or file not found. Using default/base config.")
    return config

def parse_args():
    # (Keep as is)
    parser = argparse.ArgumentParser(description="Fine-tuning script for DiseaseAwareHVT")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    return parser.parse_args()

def set_seed(seed):
    # (Keep as is)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def _interpolate_pos_embed(checkpoint_pos_embed: torch.Tensor, 
                           model_pos_embed: nn.Parameter, 
                           patch_size: int) -> torch.Tensor:
    # (Keep as is -unchanged from previous correction)
    N_src = checkpoint_pos_embed.shape[1]
    N_tgt = model_pos_embed.shape[1]
    C = model_pos_embed.shape[2] 
    if N_src == N_tgt and checkpoint_pos_embed.shape[2] == C: return checkpoint_pos_embed
    logger = logging.getLogger(__name__) # Get logger inside function
    logger.info(f"Interpolating positional embedding from {N_src} to {N_tgt} patches.")
    if math.isqrt(N_src)**2 == N_src: H0 = W0 = math.isqrt(N_src)
    else: 
        if N_src == 196 and patch_size == 16: H0, W0 = 14, 14; logger.info(f"Inferred source grid {H0}x{W0}")
        else: logger.error(f"Cannot infer src grid from N_src={N_src}."); return model_pos_embed.data 
    if math.isqrt(N_tgt)**2 == N_tgt: H_tgt = W_tgt = math.isqrt(N_tgt)
    else:
         target_H_img = default_config['img_size'][0]; target_W_img = default_config['img_size'][1]
         H_tgt = target_H_img // patch_size; W_tgt = target_W_img // patch_size
         if H_tgt * W_tgt != N_tgt: logger.error(f"Inferred target grid {H_tgt}x{W_tgt} != N_tgt={N_tgt}."); return model_pos_embed.data
    pos_embed_reshaped = checkpoint_pos_embed.reshape(1, H0, W0, C).permute(0, 3, 1, 2) 
    pos_embed_interpolated = F.interpolate(pos_embed_reshaped, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False)
    pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).flatten(1, 2) 
    if pos_embed_interpolated.shape != model_pos_embed.shape: logger.error(f"Interpolated shape {pos_embed_interpolated.shape} != model shape {model_pos_embed.shape}."); return model_pos_embed.data
    logger.info(f"Positional embedding interpolated successfully.")
    return pos_embed_interpolated


def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, config: dict):
    """ Loads weights from pre-trained checkpoint, handling pos embed interpolation and head mismatch. """
    logger = logging.getLogger(__name__)
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}. Training from scratch.")
        return model

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint state_dict from: {checkpoint_path}")

        current_model_dict = model.state_dict()
        
        # --- State Dict Modification ---
        new_state_dict = OrderedDict()
        loaded_keys = set(checkpoint.keys())
        model_keys = set(current_model_dict.keys())
        
        for k, v in checkpoint.items():
            if k not in model_keys:
                # logger.debug(f"Skipping key from checkpoint not in model: {k}")
                continue # Skip keys not present in the current model structure

            # Handle Positional Embedding interpolation
            if k in ["rgb_pos_embed", "spectral_pos_embed"]:
                if v.shape != current_model_dict[k].shape:
                    logger.info(f"Shape mismatch for {k}. Attempting interpolation.")
                    interpolated_embed = _interpolate_pos_embed(
                        v, current_model_dict[k], config['hvt_patch_size']
                    )
                    if interpolated_embed.shape == current_model_dict[k].shape:
                        new_state_dict[k] = interpolated_embed
                    else:
                        logger.warning(f"Interpolation failed or resulted in wrong shape for {k}. Skipping this weight.")
                else:
                    new_state_dict[k] = v # Shapes match, copy directly
                continue # Move to next key after handling pos embed

            # Handle Classification Head mismatch (explicitly skip loading these keys)
            if k.startswith("head."):
                logger.info(f"Skipping classification head weight from checkpoint: {k}")
                continue

            # If shapes match for other keys, copy them
            if v.shape == current_model_dict[k].shape:
                new_state_dict[k] = v
            else:
                logger.warning(f"Skipping key '{k}' due to shape mismatch: ckpt {v.shape} vs model {current_model_dict[k].shape}")

        # Load the carefully filtered and potentially modified state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        logger.info("Attempted loading pre-trained weights with filtering.")
        
        # Report missing/unexpected keys (should be more informative now)
        if missing_keys:
             # Keys expected to be missing if only backbone was saved are head.* / head_norm.*
             # Since we explicitly removed 'head.*' from new_state_dict, these shouldn't appear here
             # unless head_norm wasn't in the checkpoint either.
             logger.warning(f"Weights not found in checkpoint for model keys: {missing_keys}")
        if unexpected_keys:
            # This list contains keys from new_state_dict that are *not* in the model's state_dict.
            # Should be empty given our initial filtering `if k not in model_keys`.
            logger.error(f"Logic error: Unexpected keys found when loading filtered state_dict: {unexpected_keys}")

        logger.info("Successfully processed pre-trained backbone weights loading.")
        return model

    except Exception as e:
        logger.error(f"Error loading pretrained checkpoint from {checkpoint_path}: {e}", exc_info=True)
        logger.warning("Could not load pretrained weights. Training from scratch.")
        return model

# --- Main Fine-tuning Function ---
def main():
    # (Keep dataset, model init, optimizer, scheduler, trainer init as is)
    # ... (previous code from main() up to the training loop) ...
    args = parse_args()
    config = load_config_yaml(args.config)

    log_file = config.get("log_file_finetune", "finetune.log")
    log_dir = config.get("log_dir", "logs")
    setup_logging(log_file_name=log_file, log_dir=log_dir, log_level=logging.INFO, logger_name=None) 
    logger = logging.getLogger(__name__) 

    set_seed(config["seed"])
    logger.info("Starting fine-tuning process...")
    logger.info(f"Loaded configuration: {config}")
    device = config["device"]
    logger.info(f"Using device: {device}")
    if device == "cuda": logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Datasets and DataLoaders ---
    logger.info("Setting up datasets and dataloaders...")
    train_dataset = SARCLD2024Dataset(
        root_dir=config["data_root"], img_size=config["img_size"], split="train", 
        train_split_ratio=config["train_split"], normalize_for_model=config["normalize_data"],
        random_seed=config["seed"] 
    )
    val_dataset = SARCLD2024Dataset(
        root_dir=config["data_root"], img_size=config["img_size"], split="val", 
        train_split_ratio=config["train_split"], normalize_for_model=config["normalize_data"],
        random_seed=config["seed"] 
    )
    
    sampler = None; class_weights = None # Initialize
    if config.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights() 
        if class_weights is not None:
            logger.info("Using WeightedRandomSampler for training.")
            train_indices = train_dataset.current_indices
            train_labels = train_dataset.labels[train_indices]
            sample_weights = torch.zeros(len(train_labels)) 
            for i in range(config["num_classes"]):
                 if i < len(class_weights): sample_weights[train_labels == i] = class_weights[i]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        else: logger.warning("Could not compute class weights, disabling WeightedRandomSampler.")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], 
        sampler=sampler, shuffle=(sampler is None), 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=False
    )
    logger.info(f"Train loader: {len(train_loader)} batches. Validation loader: {len(val_loader)} batches.")
    class_names = train_dataset.get_class_names()

    # --- Model Selection and Initialization ---
    logger.info(f"Initializing model: {config['model_name']}")
    model = DiseaseAwareHVT(
        img_size=config["img_size"],
        patch_size=config["hvt_patch_size"],
        num_classes=config["num_classes"],
        embed_dim_rgb=config["hvt_embed_dim_rgb"],
        embed_dim_spectral=config["hvt_embed_dim_spectral"],
        spectral_channels=config["hvt_spectral_channels"],
        depths=config["hvt_depths"],
        num_heads=config["hvt_num_heads"],
        mlp_ratio=config["hvt_mlp_ratio"],
        qkv_bias=config["hvt_qkv_bias"],
        drop_rate=config["hvt_model_drop_rate"],
        attn_drop_rate=config["hvt_attn_drop_rate"],
        drop_path_rate=config["hvt_drop_path_rate"],
        use_dfca=config["hvt_use_dfca"],
    )

    if config["load_pretrained"]:
        model = load_pretrained_backbone(model, config["pretrained_checkpoint_path"], config) # Pass config
    
    model = model.to(device)

    # --- Augmentations, Loss, Optimizer, Scheduler ---
    augmentations = FinetuneAugmentation(config["img_size"]) if config["augmentations_enabled"] else None
    loss_weights = class_weights.to(device) if class_weights is not None and not config.get("use_weighted_sampler", False) else None
    if loss_weights is not None: logger.info("Using weighted CrossEntropyLoss.")
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=config["loss_label_smoothing"])
    if config["optimizer"].lower() == "adamw": optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], **config.get("optimizer_params", {}))
    else: optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]) # Default AdamW

    scheduler = None; lr_reducer = None
    if config.get("scheduler"):
        if config["warmup_epochs"] > 0: warmup = LinearLR(optimizer, start_factor=config.get("warmup_lr_init_factor", 0.1), total_iters=config["warmup_epochs"])
        else: warmup = None
        sched_type = config["scheduler"].lower(); main_sched = None
        if sched_type == "cosineannealingwarmrestarts": main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=config["cosine_t_0"], T_mult=config["cosine_t_mult"], eta_min=config["eta_min"]); logger.info("Using CosineAnnealingWarmRestarts scheduler.")
        elif sched_type == "reducelronplateau": lr_reducer = ReduceLROnPlateau(optimizer, mode='max', factor=config["reducelr_factor"], patience=config["reducelr_patience"], verbose=True); logger.info("Using ReduceLROnPlateau scheduler.")
        else: logger.warning(f"Unsupported scheduler type: {config['scheduler']}.")
        schedulers_to_combine = [s for s in [warmup, main_sched] if s is not None]
        if len(schedulers_to_combine) > 1: scheduler = SequentialLR(optimizer, schedulers=schedulers_to_combine, milestones=[config["warmup_epochs"]]); logger.info("Combined Warmup and Main scheduler.")
        elif len(schedulers_to_combine) == 1: scheduler = schedulers_to_combine[0]
        elif warmup is None and main_sched is None and lr_reducer is None: logger.info("No valid scheduler configured.")

    scaler = GradScaler(enabled=config["amp_enabled"])

    # --- Initialize Trainer ---
    trainer = Finetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler,
        scheduler=scheduler, accumulation_steps=config["accumulation_steps"],
        clip_grad_norm=config["clip_grad_norm"], augmentations=augmentations,
        num_classes=config["num_classes"]
    )

    # --- Training Loop ---
    # (Keep training loop logic as is)
    best_val_metric = 0.0; metric_to_monitor = 'accuracy'; patience_counter = 0
    logger.info(f"Starting fine-tuning loop for {config['epochs']} epochs...")
    for epoch in range(1, config["epochs"] + 1):
        avg_train_loss = trainer.train_one_epoch(train_loader, epoch, config["epochs"])
        if epoch % config["evaluate_every_n_epochs"] == 0:
            avg_val_loss, val_metrics = trainer.validate_one_epoch(val_loader, class_names=class_names)
            current_val_metric = val_metrics.get(metric_to_monitor, 0.0)
            if lr_reducer: lr_reducer.step(current_val_metric)
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                trainer.save_model_checkpoint(config["best_model_path"])
                logger.info(f"Epoch {epoch}: New best model! Val {metric_to_monitor}: {best_val_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch}: Val {metric_to_monitor} ({current_val_metric:.4f}) no improve. Patience: {patience_counter}/{config.get('early_stopping_patience', float('inf'))}")
            if patience_counter >= config.get("early_stopping_patience", float('inf')):
                logger.info(f"Early stopping triggered.")
                break
        if scheduler and not lr_reducer: scheduler.step()

    logger.info(f"Fine-tuning finished. Best validation {metric_to_monitor}: {best_val_metric:.4f}")
    trainer.save_model_checkpoint(config["final_model_path"])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__) 
        logger.exception(f"An critical error occurred during fine-tuning main execution: {e}")
        sys.exit(1)