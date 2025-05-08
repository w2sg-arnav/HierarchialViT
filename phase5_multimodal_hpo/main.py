# phase5_multimodal_hpo/main.py
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
from collections import OrderedDict 

# --- Path Setup ---
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../phase5_multimodal_hpo
project_root = os.path.dirname(current_dir) # .../cvpr25
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added project root to sys.path: {project_root}")

# --- Project Imports ---
# Use ABSOLUTE imports from project root perspective
from phase5_multimodal_hpo.config import config as default_config # Import config from THIS phase
from phase5_multimodal_hpo.dataset import SARCLD2024Dataset   # Import dataset from THIS phase
from phase5_multimodal_hpo.utils.augmentations import FinetuneAugmentation # Import utils from THIS phase
from phase5_multimodal_hpo.utils.logging_setup import setup_logging
from phase5_multimodal_hpo.finetune.trainer import Finetuner # Import trainer from THIS phase

# Use ABSOLUTE imports for models from other packages
from phase2_model.models.hvt import DiseaseAwareHVT # Use canonical HVT from Phase 2

# --- Helper Functions --- 
def load_config_yaml(config_path=None):
    """ Loads configuration from YAML file, falling back to defaults. """
    config = default_config.copy() # Start with Phase 5 defaults
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config: 
                    config.update(yaml_config) # Override defaults with YAML values
            print(f"Loaded configuration from YAML: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load/parse YAML config file {config_path}. Error: {e}. Using default/base config.")
    else:
         print("No config file path provided or file not found. Using default/base config.")
    return config

def parse_args():
    """ Parses command line arguments. """
    parser = argparse.ArgumentParser(description="Fine-tuning script for DiseaseAwareHVT (Phase 5 - Multi-Modal)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    # Add HPO overrides here if running single trials via command line
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--wd", type=float, default=None, help="Override weight decay")
    parser.add_argument("--ls", type=float, default=None, help="Override label smoothing")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    return parser.parse_args()

def set_seed(seed):
    """ Sets random seed for reproducibility. """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def _interpolate_pos_embed(chkpt_embed: torch.Tensor, 
                           model_embed: nn.Parameter, 
                           patch_size: int,
                           target_img_size_from_config: tuple) -> torch.Tensor: # Pass target size explicitly
    """ Helper function to interpolate positional embeddings. """
    logger = logging.getLogger(__name__) 
    N_src, C_src = chkpt_embed.shape[1], chkpt_embed.shape[2]
    N_tgt, C_tgt = model_embed.shape[1], model_embed.shape[2]

    if N_src == N_tgt and C_src == C_tgt: return chkpt_embed
    if C_src != C_tgt: logger.warning(f"Pos embed dim mismatch: {C_src} vs {C_tgt}. Cannot load."); return model_embed.data

    logger.info(f"Interpolating positional embedding from {N_src} to {N_tgt} patches.")
    
    if math.isqrt(N_src)**2 == N_src: H0 = W0 = math.isqrt(N_src)
    elif N_src == 196 and patch_size == 16: H0, W0 = 14, 14; logger.info(f"Inferred source grid {H0}x{W0}.")
    else: logger.error(f"Cannot infer source grid from N_src={N_src}."); return model_embed.data 

    # Infer target grid from passed target_img_size_from_config
    H_tgt = target_img_size_from_config[0] // patch_size
    W_tgt = target_img_size_from_config[1] // patch_size
    if H_tgt * W_tgt != N_tgt: logger.error(f"Target grid {H_tgt}x{W_tgt} != N_tgt={N_tgt}."); return model_embed.data
        
    pos_embed_reshaped = chkpt_embed.reshape(1, H0, W0, C_tgt).permute(0, 3, 1, 2) 
    pos_embed_interp = F.interpolate(pos_embed_reshaped, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False)
    pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).flatten(1, 2) 
    if pos_embed_interp.shape != model_embed.shape: logger.error(f"Interpolated shape {pos_embed_interp.shape} != model shape {model_embed.shape}."); return model_embed.data
    logger.info(f"Pos embed interpolated successfully.")
    return pos_embed_interp

def load_pretrained_backbone(model: nn.Module, checkpoint_path: str, config: dict):
    """ Loads weights, handles pos embed interpolation and head mismatch. """
    logger = logging.getLogger(__name__)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path) # Assumes relative to project root
        logger.info(f"Resolved relative checkpoint path to: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Pretrained checkpoint not found: {checkpoint_path}. Training from scratch.")
        return model

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint state_dict from: {checkpoint_path}")
        current_model_dict = model.state_dict()
        new_state_dict = OrderedDict() 
        
        for k, v in checkpoint.items():
            if k not in current_model_dict: continue
            if k in ["rgb_pos_embed", "spectral_pos_embed"]:
                model_param = current_model_dict.get(k)
                if model_param is not None and v.shape != model_param.shape:
                    # Pass necessary config values for interpolation
                    interpolated_embed = _interpolate_pos_embed(
                        v, model_param, config['hvt_patch_size'], config['img_size']
                    )
                    if interpolated_embed.shape == model_param.shape: new_state_dict[k] = interpolated_embed
                    else: logger.warning(f"Interpolation failed/wrong shape for {k}. Skipping.")
                elif model_param is not None: new_state_dict[k] = v 
                continue 
            if k.startswith("head."): logger.info(f"Skipping classification head weight: {k}"); continue
            if k in current_model_dict and v.shape == current_model_dict[k].shape: new_state_dict[k] = v
            else: logger.warning(f"Skipping key '{k}' due to shape mismatch: ckpt {v.shape} vs model {current_model_dict.get(k, 'MISSING').shape}")

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        logger.info("Attempted loading pre-trained weights with filtering and interpolation.")
        expected_missing_prefixes = ('head.', 'head_norm.') 
        truly_missing = [k for k in missing_keys if not k.startswith(expected_missing_prefixes)]
        if truly_missing: logger.warning(f"  Weights MISSING for model keys: {truly_missing}")
        else: logger.info(f"  Missing keys only in classification head (expected): {missing_keys}")
        if unexpected_keys: logger.error(f"  Error: UNEXPECTED keys found in filtered state_dict but not in model: {unexpected_keys}")
        logger.info("Successfully processed pre-trained backbone weights loading.")
        return model
    except Exception as e:
        logger.error(f"Error loading pretrained checkpoint from {checkpoint_path}: {e}", exc_info=True)
        logger.warning("Could not load pretrained weights. Training from scratch.")
        return model

# --- Main Training Function ---
def run_training_session(config: dict) -> float:
    """ Runs a full training and validation loop, returns best validation metric. """
    
    logger = logging.getLogger(__name__) 
    set_seed(config["seed"])
    device = config["device"]
    
    # --- Datasets ---
    logger.info("Setting up datasets...")
    train_dataset = SARCLD2024Dataset(
        root_dir=config["data_root"], 
        img_size=config["img_size"], 
        split="train", 
        train_split_ratio=config["train_split_ratio"], # <<<--- CORRECTED KEY
        normalize_for_model=config["normalize_data"],
        use_spectral=(config["hvt_spectral_channels"] > 0), 
        spectral_channels=config["hvt_spectral_channels"],
        random_seed=config["seed"] 
    )
    val_dataset = SARCLD2024Dataset(
        root_dir=config["data_root"], 
        img_size=config["img_size"], 
        split="val", 
        train_split_ratio=config["train_split_ratio"], # <<<--- CORRECTED KEY
        normalize_for_model=config["normalize_data"],
        use_spectral=(config["hvt_spectral_channels"] > 0), 
        spectral_channels=config["hvt_spectral_channels"],
        random_seed=config["seed"] 
    )
    
    sampler = None; class_weights = None
    if config.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights() 
        if class_weights is not None:
            logger.info("Using WeightedRandomSampler.")
            train_labels = train_dataset.current_split_labels
            sample_weights = torch.zeros(len(train_labels)) 
            for i in range(config["num_classes"]):
                 if i < len(class_weights): sample_weights[train_labels == i] = class_weights[i]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        else: logger.warning("Disabling WeightedRandomSampler (could not get weights).")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    class_names = train_dataset.get_class_names()
    logger.info(f"Train loader: {len(train_loader)} batches. Validation loader: {len(val_loader)} batches.")


    # --- Model ---
    logger.info(f"Initializing model: {config['model_name']}")
    model = DiseaseAwareHVT(
        img_size=config["img_size"], patch_size=config["hvt_patch_size"],
        num_classes=config["num_classes"], embed_dim_rgb=config["hvt_embed_dim_rgb"],
        embed_dim_spectral=config["hvt_embed_dim_spectral"], spectral_channels=config["hvt_spectral_channels"],
        depths=config["hvt_depths"], num_heads=config["hvt_num_heads"],
        mlp_ratio=config["hvt_mlp_ratio"], qkv_bias=config["hvt_qkv_bias"],
        drop_rate=config["hvt_model_drop_rate"], attn_drop_rate=config["hvt_attn_drop_rate"],
        drop_path_rate=config["hvt_drop_path_rate"], use_dfca=config["hvt_use_dfca"],
    )
    if config["load_pretrained"]: model = load_pretrained_backbone(model, config["pretrained_checkpoint_path"], config)
    model = model.to(device)

    # --- Trainer Components ---
    augmentations = FinetuneAugmentation(config["img_size"]) if config["augmentations_enabled"] else None
    loss_weights = class_weights.to(device) if class_weights is not None and not config.get("use_weighted_sampler", False) else None
    if loss_weights is not None: logger.info("Using weighted CrossEntropyLoss.")
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=config["loss_label_smoothing"])
    
    if config["optimizer"].lower() == "adamw": optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], **config.get("optimizer_params", {}))
    else: optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    scheduler = None; lr_reducer = None
    if config.get("scheduler"):
        if config["warmup_epochs"] > 0: warmup = LinearLR(optimizer, start_factor=config.get("warmup_lr_init_factor", 0.1), total_iters=config["warmup_epochs"])
        else: warmup = None
        sched_type = config["scheduler"].lower(); main_sched = None
        if sched_type == "cosineannealingwarmrestarts": main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=config["cosine_t_0"], T_mult=config["cosine_t_mult"], eta_min=config["eta_min"]); logger.info("Using CosineAnnealingWarmRestarts scheduler.")
        elif sched_type == "reducelronplateau": lr_reducer = ReduceLROnPlateau(optimizer, mode='max', factor=config["reducelr_factor"], patience=config["reducelr_patience"], verbose=False); logger.info("Using ReduceLROnPlateau scheduler.")
        else: logger.warning(f"Unsupported scheduler type: {config['scheduler']}.")
        schedulers_to_combine = [s for s in [warmup, main_sched] if s is not None]
        if len(schedulers_to_combine) > 1: scheduler = SequentialLR(optimizer, schedulers=schedulers_to_combine, milestones=[config["warmup_epochs"]]); logger.info("Combined Warmup+Scheduler.")
        elif len(schedulers_to_combine) == 1: scheduler = schedulers_to_combine[0]
        elif lr_reducer is None: logger.info("No valid scheduler configured.")

    scaler = GradScaler(enabled=config["amp_enabled"])
    trainer = Finetuner(model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, scheduler=scheduler, accumulation_steps=config["accumulation_steps"], clip_grad_norm=config["clip_grad_norm"], augmentations=augmentations, num_classes=config["num_classes"])

    # --- Training Loop ---
    best_val_metric = -1.0 # Initialize lower than any possible metric
    metric_to_monitor = config.get("metric_to_monitor", 'f1_weighted') 
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", float('inf'))

    logger.info(f"Starting training loop for {config['epochs']} epochs... Monitoring: {metric_to_monitor}") 
    
    for epoch in range(1, config["epochs"] + 1):
        avg_train_loss = trainer.train_one_epoch(train_loader, epoch, config["epochs"])
        
        # --- Validation and Checkpointing ---
        if epoch % config["evaluate_every_n_epochs"] == 0:
            avg_val_loss, val_metrics = trainer.validate_one_epoch(val_loader, class_names=class_names)
            current_val_metric = val_metrics.get(metric_to_monitor, 0.0)
            
            # Ensure checkpoint dir exists
            best_model_dir = os.path.dirname(config["best_model_path"])
            if best_model_dir and not os.path.exists(best_model_dir): os.makedirs(best_model_dir)

            if lr_reducer: lr_reducer.step(current_val_metric) # Step reducer based on metric

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                trainer.save_model_checkpoint(config["best_model_path"]) 
                logger.info(f"Epoch {epoch}: New best val {metric_to_monitor}: {best_val_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch}: Val {metric_to_monitor} ({current_val_metric:.4f}) no improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
        
        # Step other schedulers (if they are not ReduceLROnPlateau)
        if scheduler and not lr_reducer: 
            scheduler.step()

    logger.info(f"Training finished. Best validation {metric_to_monitor}: {best_val_metric:.4f}")
    # Ensure final model save dir exists
    final_model_dir = os.path.dirname(config["final_model_path"])
    if final_model_dir and not os.path.exists(final_model_dir): os.makedirs(final_model_dir)
    trainer.save_model_checkpoint(config["final_model_path"])
    return best_val_metric 


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    config = load_config_yaml(args.config)
    
    log_file = config.get("log_file_finetune", "finetune.log")
    log_dir = config.get("log_dir", "logs")
    setup_logging(log_file_name=log_file, log_dir=log_dir, log_level=logging.INFO, logger_name=None) 
    logger = logging.getLogger(__name__)

    # HPO logic / Single run logic
    if config.get("hpo_enabled", False):
        logger.warning("HPO is enabled in config, but this script runs a single train session.")
        logger.warning("Please run hpo.py for hyperparameter optimization.")
        # Optional: Placeholder if you want this script to trigger HPO eventually
        # run_hpo(config) 
    else:
        try:
            if args.lr is not None: config['learning_rate'] = args.lr; logger.info(f"Overriding LR: {args.lr}")
            if args.wd is not None: config['weight_decay'] = args.wd; logger.info(f"Overriding WD: {args.wd}")
            if args.ls is not None: config['loss_label_smoothing'] = args.ls; logger.info(f"Overriding LS: {args.ls}")
            if args.epochs is not None: config['epochs'] = args.epochs; logger.info(f"Overriding Epochs: {args.epochs}")
            if args.batch_size is not None: config['batch_size'] = args.batch_size; logger.info(f"Overriding Batch Size: {args.batch_size}")
            
            run_training_session(config)

        except Exception as e:
            logger.exception(f"An critical error occurred during fine-tuning main execution: {e}")
            sys.exit(1)