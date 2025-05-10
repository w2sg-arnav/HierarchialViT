# phase4_finetuning/main.py
from collections import OrderedDict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LinearLR, SequentialLR, CosineAnnealingLR
import argparse
import yaml
import torch.nn.functional as F
import math
import sys # For sys.exit and path

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # phase4_finetuning
project_root = os.path.dirname(current_dir) # cvpr25
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG (phase4_main): Added project root to sys.path: {project_root}")

# --- Project Imports ---
try:
    from phase4_finetuning.config import config as base_finetune_config # The dict
    from phase4_finetuning.dataset import SARCLD2024Dataset
    from phase4_finetuning.utils.augmentations import FinetuneAugmentation
    from phase4_finetuning.utils.logging_setup import setup_logging
    from phase4_finetuning.finetune.trainer import Finetuner
    # Import HVT from phase2_model.models.hvt, it will use its internal config logic
    # which should pick up phase3_pretraining.config for XL params if that's how it's set up.
    from phase2_model.models.hvt import create_disease_aware_hvt_from_config
    # For baseline models, if used
    # from phase4_finetuning.models.baseline import InceptionV3Baseline
    # from phase4_finetuning.models.efficientnet import EfficientNetBaseline
except ImportError as e:
    print(f"CRITICAL ERROR in phase4_main.py: Failed to import modules: {e}")
    print("Ensure PYTHONPATH includes project root and all packages have __init__.py files.")
    sys.exit(1)


def load_config_yaml(config_path=None):
    config_to_use = base_finetune_config.copy() # Start with defaults from phase4_finetuning.config.py
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config_to_use.update(yaml_config) # Override defaults with YAML values
                    print(f"Successfully loaded and merged configuration from YAML: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load/parse YAML {config_path}. Error: {e}. Using defaults from config.py.")
    else:
         print("No YAML config file path provided or file not found. Using defaults from config.py.")
    return config_to_use

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for HVT models")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file to override defaults.")
    return parser.parse_args()

def set_seed(seed_value: int):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    # Potentially: random.seed(seed_value) if using stdlib random
    # Ensuring determinism can be harder, e.g. cudnn.deterministic = True, but often slower
    logger.info(f"Global random seed set to: {seed_value}")


def _interpolate_pos_embed(checkpoint_pos_embed: torch.Tensor,
                           model_pos_embed_param: nn.Parameter, # model's actual nn.Parameter
                           patch_size: int,
                           expected_img_size_for_model: Tuple[int, int]) -> torch.Tensor:
    logger = logging.getLogger(__name__) # Get logger if not global
    N_src = checkpoint_pos_embed.shape[1]
    N_tgt = model_pos_embed_param.shape[1]
    C_src = checkpoint_pos_embed.shape[2]
    C_tgt = model_pos_embed_param.shape[2]

    if C_src != C_tgt:
        logger.error(f"Positional embedding C mismatch: ckpt {C_src} vs model {C_tgt}. Cannot interpolate.")
        return model_pos_embed_param.data # Return original model data

    if N_src == N_tgt:
        logger.info(f"Positional embedding num_patches match ({N_src}). Using checkpointed embedding directly.")
        return checkpoint_pos_embed.view_as(model_pos_embed_param.data)


    logger.info(f"Interpolating positional embedding from {N_src} patches to {N_tgt} patches (C={C_src}).")

    # Infer H0, W0 from N_src
    if math.isqrt(N_src)**2 == N_src:
        H0 = W0 = math.isqrt(N_src)
    else:
        # Attempt to infer from common pre-training sizes if N_src is known, e.g., 196 for 224px/P16 or 256 for 256px/P16
        # This part is heuristic and might need adjustment based on your pre-training image size
        if N_src == (224//16)**2 and patch_size == 16: H0, W0 = 224//16, 224//16
        elif N_src == (384//patch_size)**2 : H0, W0 = 384//patch_size, 384//patch_size
        elif N_src == (448//patch_size)**2 : H0, W0 = 448//patch_size, 448//patch_size # For XL pretraining
        else:
            logger.error(f"Cannot reliably infer source grid H0, W0 from N_src={N_src} for patch_size={patch_size}. Returning original model embedding.")
            return model_pos_embed_param.data
    logger.info(f"Inferred source grid: {H0}x{W0}")

    # Target grid size for current model's image size
    H_tgt = expected_img_size_for_model[0] // patch_size
    W_tgt = expected_img_size_for_model[1] // patch_size
    if H_tgt * W_tgt != N_tgt:
        logger.error(f"Calculated target grid {H_tgt}x{W_tgt} ({H_tgt*W_tgt} patches) != model's N_tgt ({N_tgt}). Cannot interpolate. Returning original.")
        return model_pos_embed_param.data
    logger.info(f"Target grid for interpolation: {H_tgt}x{W_tgt}")

    try:
        pos_embed_reshaped = checkpoint_pos_embed.reshape(1, H0, W0, C_src).permute(0, 3, 1, 2) # B, C, H, W
        pos_embed_interpolated = F.interpolate(pos_embed_reshaped, size=(H_tgt, W_tgt), mode='bicubic', align_corners=False)
        pos_embed_interpolated = pos_embed_interpolated.permute(0, 2, 3, 1).flatten(1, 2) # B, N_tgt, C

        if pos_embed_interpolated.shape != model_pos_embed_param.shape:
            logger.error(f"Interpolated shape {pos_embed_interpolated.shape} != model target shape {model_pos_embed_param.shape}. Returning original.")
            return model_pos_embed_param.data

        logger.info("Positional embedding interpolated successfully.")
        return pos_embed_interpolated
    except Exception as e_interp:
        logger.error(f"Error during positional embedding interpolation: {e_interp}", exc_info=True)
        return model_pos_embed_param.data


def load_pretrained_hvt_backbone(model: nn.Module, checkpoint_path: str, config_dict: dict):
    logger = logging.getLogger(__name__) # Local logger
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Pretrained HVT checkpoint not found: {checkpoint_path}. Training from scratch or using ImageNet if baselines.")
        return model

    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        # Check if it's a dict and has 'model_state_dict' (from phase3 trainer) or is just a state_dict
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            pretrained_state_dict = checkpoint_data['model_state_dict']
            logger.info(f"Loaded 'model_state_dict' from checkpoint: {checkpoint_path}")
            if 'model_init_config' in checkpoint_data:
                logger.info(f"Checkpoint also contains model_init_config: {checkpoint_data['model_init_config']}")
        else:
            pretrained_state_dict = checkpoint_data # Assume it's just the state_dict
            logger.info(f"Loaded raw state_dict from checkpoint: {checkpoint_path}")


        current_model_dict = model.state_dict()
        new_state_dict = OrderedDict()

        for k_ckpt, v_ckpt in pretrained_state_dict.items():
            if k_ckpt not in current_model_dict:
                logger.debug(f"Skipping key from checkpoint (not in current model): {k_ckpt}")
                continue

            # Handle positional embedding interpolation specifically for HVT
            # This assumes your HVT model parameters are named 'rgb_pos_embed' and 'spectral_pos_embed'
            is_pos_embed = k_ckpt in ["rgb_pos_embed", "spectral_pos_embed"]
            if is_pos_embed and v_ckpt.shape != current_model_dict[k_ckpt].shape:
                logger.info(f"Shape mismatch for positional embedding '{k_ckpt}'. Checkpoint shape: {v_ckpt.shape}, Model shape: {current_model_dict[k_ckpt].shape}. Attempting interpolation.")
                interpolated_embed = _interpolate_pos_embed(
                    v_ckpt, current_model_dict[k_ckpt],
                    config_dict['hvt_patch_size'],
                    config_dict['img_size'] # Current fine-tuning image size
                )
                if interpolated_embed.shape == current_model_dict[k_ckpt].shape:
                    new_state_dict[k_ckpt] = interpolated_embed
                    logger.info(f"Successfully interpolated {k_ckpt}.")
                else:
                    logger.warning(f"Interpolation failed for {k_ckpt} or resulted in wrong shape. Skipping this weight. Model will use its initialized embedding.")
            elif k_ckpt.startswith("head.") or k_ckpt.startswith("head_norm."): # Skip HVT's own head/head_norm
                logger.info(f"Skipping HVT's classification head/norm weight from pre-trained checkpoint: {k_ckpt}")
            elif v_ckpt.shape == current_model_dict[k_ckpt].shape:
                new_state_dict[k_ckpt] = v_ckpt
            else:
                logger.warning(f"Shape mismatch for key '{k_ckpt}': Checkpoint shape {v_ckpt.shape} vs Model shape {current_model_dict[k_ckpt].shape}. Skipping this weight.")

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        if missing_keys:
            # Filter out head keys as they are expected to be missing for fine-tuning
            missing_backbone_keys = [k for k in missing_keys if not (k.startswith("head.") or k.startswith("head_norm."))]
            if missing_backbone_keys:
                logger.warning(f"Weights not found in checkpoint for these backbone model keys: {missing_backbone_keys}")
            else:
                logger.info("All backbone weights found or handled. Head layers are expected to be missing/reinitialized.")
        if unexpected_keys:
            logger.error(f"Unexpected keys found in checkpoint that are not in the current model structure: {unexpected_keys}. This should not happen with current filtering.")

        logger.info(f"Successfully loaded and processed pre-trained HVT backbone weights from {checkpoint_path}.")
        return model

    except Exception as e:
        logger.error(f"Error loading HVT backbone checkpoint from {checkpoint_path}: {e}", exc_info=True)
        logger.warning("Could not load HVT backbone weights. Training from scratch or with HVT's default init.")
        return model


def main():
    args = parse_args()
    # Load config: base_finetune_config (from .py) updated by YAML file if provided
    cfg = load_config_yaml(args.config) # Renamed to avoid conflict with module name

    # Setup logging using the final config
    log_file_main = cfg.get("log_file_finetune", "finetune_main.log")
    log_dir_main = cfg.get("log_dir", "logs_finetune")
    if not os.path.isabs(log_dir_main): log_dir_main = os.path.join(current_dir, log_dir_main)
    os.makedirs(log_dir_main, exist_ok=True)
    # Re-initialize logger for main after config is fully loaded
    if logging.getLogger().hasHandlers(): # Clear existing handlers if any, to avoid duplicate logs if run multiple times in a session
        for handler in logging.getLogger().handlers[:]: logging.getLogger().removeHandler(handler)
    setup_logging(log_file_name=log_file_main, log_dir=log_dir_main, log_level=logging.INFO, logger_name=None) # Setup root logger
    logger = logging.getLogger(__name__) # Get logger for this main script

    set_seed(cfg["seed"])
    logger.info("Starting fine-tuning process for HVT-XL...")
    logger.info(f"Final effective configuration: {cfg}")
    device = cfg["device"]
    logger.info(f"Using device: {device}")
    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        if cfg.get("cudnn_benchmark", False): torch.backends.cudnn.benchmark = True
        if cfg.get("matmul_precision") and hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(cfg["matmul_precision"])


    logger.info("Setting up datasets and dataloaders...")
    # Pass relevant config values directly to dataset
    train_dataset = SARCLD2024Dataset(root_dir=cfg["data_root"], img_size=tuple(cfg["img_size"]), split="train",
                                     train_split_ratio=cfg["train_split_ratio"], normalize_for_model=cfg["normalize_data"],
                                     original_dataset_name=cfg["original_dataset_name"],
                                     augmented_dataset_name=cfg["augmented_dataset_name"],
                                     random_seed=cfg["seed"])
    val_dataset = SARCLD2024Dataset(root_dir=cfg["data_root"], img_size=tuple(cfg["img_size"]), split="val",
                                   train_split_ratio=cfg["train_split_ratio"], normalize_for_model=cfg["normalize_data"],
                                   original_dataset_name=cfg["original_dataset_name"],
                                   augmented_dataset_name=cfg["augmented_dataset_name"],
                                   random_seed=cfg["seed"])

    sampler = None
    if cfg.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights()
        if class_weights is not None and len(train_dataset.current_split_labels) > 0:
            logger.info("Using WeightedRandomSampler for training.")
            sample_weights = torch.zeros(len(train_dataset.current_split_labels))
            for i, label_idx in enumerate(train_dataset.current_split_labels):
                sample_weights[i] = class_weights[label_idx]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        else: logger.warning("Could not compute class weights or train dataset empty, disabling WeightedRandomSampler.")

    # DataLoader common args
    loader_num_workers = cfg.get('num_workers', 0) # Default to 0 if not set for finetuning
    loader_prefetch = cfg.get('prefetch_factor', 2) if loader_num_workers > 0 else None
    loader_persistent = (device == 'cuda' and loader_num_workers > 0 and torch.cuda.is_available())
    
    dl_common_args = {"num_workers": loader_num_workers, "pin_memory": (device == 'cuda'),
                      "persistent_workers": loader_persistent, "prefetch_factor": loader_prefetch}
    if loader_num_workers == 0: dl_common_args["persistent_workers"] = False; dl_common_args["prefetch_factor"] = None

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None),
                              drop_last=True, **dl_common_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **dl_common_args)
    logger.info(f"Train loader: {len(train_loader)} batches. Val loader: {len(val_loader)} batches. Num workers: {loader_num_workers}")
    class_names = train_dataset.get_class_names()

    logger.info(f"Initializing model: {cfg.get('model_architecture', 'DiseaseAwareHVT_XL')}")
    # Use create_disease_aware_hvt_from_config which uses its own config mechanism
    # but ensure the parameters passed to it align with the HVT-XL from phase3 config
    # The phase2_model.models.hvt.py should be set up to read phase3_pretraining.config for XL params.
    # Here, we instantiate based on *this phase4_finetuning config* which should hold the XL params.
    hvt_init_params = {
        "img_size": tuple(cfg["img_size"]), "patch_size": cfg["hvt_patch_size"], "num_classes": cfg["num_classes"],
        "embed_dim_rgb": cfg["hvt_embed_dim_rgb"], "embed_dim_spectral": cfg["hvt_embed_dim_spectral"],
        "spectral_channels": cfg["hvt_spectral_channels"], "depths": cfg["hvt_depths"],
        "num_heads": cfg["hvt_num_heads"], "mlp_ratio": cfg["hvt_mlp_ratio"],
        "qkv_bias": cfg["hvt_qkv_bias"], "drop_rate": cfg["hvt_model_drop_rate"],
        "attn_drop_rate": cfg["hvt_attn_drop_rate"], "drop_path_rate": cfg["hvt_drop_path_rate"],
        "use_dfca": cfg["hvt_use_dfca"], "dfca_num_heads": cfg.get("hvt_dfca_heads"), # Use .get for optional ones
        "dfca_drop_rate": cfg.get("dfca_drop_rate"),
        "dfca_use_disease_mask": cfg.get("dfca_use_disease_mask"),
        "use_gradient_checkpointing": cfg.get("use_gradient_checkpointing", False),
        # SSL flags for HVT (usually False for finetuning the head)
        "ssl_enable_mae": False, "ssl_enable_contrastive": False,
        "enable_consistency_loss_heads": False # Not used in standard finetuning
    }
    # This way, DiseaseAwareHVT is built with params from *this* config
    model = DiseaseAwareHVT(**hvt_init_params)


    if cfg["load_pretrained_backbone"]:
        model = load_pretrained_hvt_backbone(model, cfg["pretrained_checkpoint_path"], cfg) # Pass current config for img_size, patch_size

    # Freeze backbone layers if specified
    if cfg.get("freeze_backbone_epochs", 0) > 0:
        logger.info(f"Freezing backbone for the first {cfg['freeze_backbone_epochs']} epochs.")
        for name, param in model.named_parameters():
            if not name.startswith("head."): # Assuming 'head' is the prefix for the classification head
                param.requires_grad = False

    model = model.to(device)

    # Compile model if enabled
    if cfg.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = cfg.get("torch_compile_mode", "reduce-overhead")
        logger.info(f"Attempting to compile model for fine-tuning with torch.compile(mode='{compile_mode}')...")
        try: model = torch.compile(model, mode=compile_mode); logger.info("Fine-tuning model compiled.")
        except Exception as e: logger.warning(f"torch.compile() for fine-tuning failed: {e}. No compilation.", exc_info=True)


    augmentations = FinetuneAugmentation(tuple(cfg["img_size"])) if cfg["augmentations_enabled"] else None
    
    # Weighted loss (alternative to sampler)
    loss_class_weights = None
    if cfg.get("use_weighted_loss", False) and not cfg.get("use_weighted_sampler", False): # Only if sampler not used
        loss_class_weights = train_dataset.get_class_weights()
        if loss_class_weights is not None: loss_class_weights = loss_class_weights.to(device)
        logger.info(f"Using weighted CrossEntropyLoss with weights: {loss_class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=loss_class_weights, label_smoothing=cfg.get("loss_label_smoothing", 0.0))

    # Optimizer: Differentiate LR for backbone and head if unfreezing later
    if cfg.get("freeze_backbone_epochs", 0) > 0 and cfg.get("unfreeze_backbone_lr_factor", 1.0) != 1.0:
        # Setup parameter groups for differential learning rates later
        # This example assumes immediate full unfreezing, adjust if staged unfreezing needed
        head_params = [p for n, p in model.named_parameters() if n.startswith("head.") and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.") and p.requires_grad]
        optimizer_grouped_parameters = [
            {'params': backbone_params, 'lr': cfg["learning_rate"] * cfg.get("unfreeze_backbone_lr_factor", 0.1)},
            {'params': head_params, 'lr': cfg["learning_rate"]}
        ]
        logger.info(f"Using differential LR for backbone (factor {cfg.get('unfreeze_backbone_lr_factor', 0.1)}) and head.")
        if cfg["optimizer"].lower() == "adamw":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=cfg["weight_decay"], **cfg.get("optimizer_params", {}))
        else: # Default or other optimizer
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, momentum=0.9, weight_decay=cfg["weight_decay"]) # Example SGD
            logger.warning(f"Differential LR with {cfg['optimizer']} not fully implemented, using SGD for backbone group as example.")
    else: # Standard optimizer setup
        if cfg["optimizer"].lower() == "adamw":
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], **cfg.get("optimizer_params", {}))
        elif cfg["optimizer"].lower() == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], momentum=cfg.get("momentum",0.9), weight_decay=cfg["weight_decay"])
        else:
            logger.warning(f"Unsupported optimizer {cfg['optimizer']}. Defaulting to AdamW.")
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    logger.info(f"Optimizer: {optimizer.__class__.__name__}, Base LR: {cfg['learning_rate']}")


    scheduler = None; lr_reducer_on_plateau = None # Specific name for clarity
    if cfg.get("scheduler"):
        sched_type = cfg["scheduler"].lower()
        total_steps_for_sched = cfg["epochs"] * len(train_loader) // cfg.get("accumulation_steps",1) # total optimizer steps
        warmup_steps_for_sched = cfg.get("warmup_epochs",0) * len(train_loader) // cfg.get("accumulation_steps",1)

        main_sched = None
        if sched_type == "warmupcosine":
            if warmup_steps_for_sched >= total_steps_for_sched and total_steps_for_sched > 0:
                warmup_steps_for_sched = max(1, int(0.1 * total_steps_for_sched))
            main_sched = get_cosine_schedule_with_warmup( # Assuming get_cosine_schedule_with_warmup is defined globally or imported
                optimizer, num_warmup_steps=warmup_steps_for_sched, num_training_steps=total_steps_for_sched
            )
            logger.info(f"Using WarmupCosine scheduler: WarmupSteps={warmup_steps_for_sched}, TotalSteps={total_steps_for_sched}")
            scheduler = main_sched # This scheduler is per-step
        elif sched_type == "cosineannealinglr": # Epoch based
            t_max_epochs = cfg["epochs"] - cfg.get("warmup_epochs",0)
            if t_max_epochs <=0: t_max_epochs = 1
            main_sched = CosineAnnealingLR(optimizer, T_max=t_max_epochs, eta_min=cfg.get("eta_min_lr", 1e-6))
            logger.info(f"Using CosineAnnealingLR scheduler: T_max={t_max_epochs} epochs (after warmup).")
        elif sched_type == "reducelronplateau":
            lr_reducer_on_plateau = ReduceLROnPlateau(optimizer, mode='max', factor=cfg["reducelr_factor"], patience=cfg["reducelr_patience"], verbose=True)
            logger.info("Using ReduceLROnPlateau scheduler.")
        else: logger.warning(f"Unsupported main scheduler type: {config['scheduler']}.")

        # Handle linear warmup separately if main scheduler is epoch-based (like CosineAnnealingLR) and warmup_epochs > 0
        if cfg.get("warmup_epochs", 0) > 0 and main_sched and sched_type != "warmupcosine":
            warmup_scheduler = LinearLR(optimizer, start_factor=cfg.get("warmup_lr_init_factor", 0.1), total_iters=cfg["warmup_epochs"]) # Epoch-based warmup
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_sched], milestones=[cfg["warmup_epochs"]])
            logger.info(f"Combined epoch-based Linear Warmup ({cfg['warmup_epochs']} epochs) with {sched_type}.")
        elif main_sched and sched_type != "warmupcosine": # No separate warmup, main_sched is the scheduler
            scheduler = main_sched


    scaler = GradScaler(enabled=cfg["amp_enabled"])
    trainer = Finetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler,
        scheduler=scheduler,
        lr_scheduler_on_batch=(scheduler is not None and sched_type == "warmupcosine"), # Only WarmupCosine is per-step here
        accumulation_steps=cfg["accumulation_steps"],
        clip_grad_norm=cfg["clip_grad_norm"], augmentations=augmentations,
        num_classes=cfg["num_classes"]
    )

    best_val_metric = 0.0
    metric_to_monitor = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    early_stopping_patience = cfg.get("early_stopping_patience", float('inf'))
    patience_counter = 0

    logger.info(f"Starting fine-tuning loop for {cfg['epochs']} epochs. Monitoring '{metric_to_monitor}' for early stopping (patience={early_stopping_patience}).")
    for epoch in range(1, cfg["epochs"] + 1):
        # Unfreeze backbone after specified epochs
        if cfg.get("freeze_backbone_epochs", 0) > 0 and epoch == cfg.get("freeze_backbone_epochs", 0) + 1:
            logger.info(f"Epoch {epoch}: Unfreezing backbone layers.")
            for name, param in model.named_parameters():
                if not name.startswith("head."):
                    param.requires_grad = True
            # Recreate optimizer with all parameters now trainable, potentially with different LR for backbone
            # This part needs careful handling of param groups if differential LR is used.
            # For simplicity here, we assume optimizer was already created with groups or will be re-created.
            # If optimizer_grouped_parameters was used, it's fine. Otherwise, re-init optimizer.
            logger.info("Optimizer parameters might need to be re-evaluated if not using param_groups.")


        avg_train_loss = trainer.train_one_epoch(train_loader, epoch, cfg["epochs"])

        if epoch % cfg["evaluate_every_n_epochs"] == 0 or epoch == cfg["epochs"]:
            avg_val_loss, val_metrics = trainer.validate_one_epoch(val_loader, class_names=class_names)
            current_val_metric = val_metrics.get(metric_to_monitor, 0.0)

            if lr_reducer_on_plateau: # ReduceLROnPlateau steps on metric
                lr_reducer_on_plateau.step(current_val_metric)
            elif scheduler and not trainer.lr_scheduler_on_batch: # Other epoch-based schedulers
                scheduler.step()


            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                trainer.save_model_checkpoint(cfg["best_model_path"])
                logger.info(f"Epoch {epoch}: New best model! Val {metric_to_monitor}: {best_val_metric:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch}: Val {metric_to_monitor} ({current_val_metric:.4f}) no improve. Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
        # If scheduler is per-batch and not ReduceLROnPlateau, it's handled in trainer.train_one_epoch

    logger.info(f"Fine-tuning finished. Best validation {metric_to_monitor}: {best_val_metric:.4f}")
    trainer.save_model_checkpoint(cfg["final_model_path"])

# Helper for WarmupCosine scheduler if not in trainer
# def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
#    # ... (implementation) ...

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure logger is available for the final error message
        if 'logger' not in globals(): # Fallback if logger wasn't initialized due to early error
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
        logger.exception(f"An critical error occurred during fine-tuning main execution: {e}")
        sys.exit(1)