# phase4_finetuning/main.py
from collections import OrderedDict
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, LinearLR,
    SequentialLR, CosineAnnealingLR, LambdaLR
)
import argparse
import yaml
import torch.nn.functional as F
import math
import sys
from typing import Tuple, Optional, Dict, Any
import traceback

# --- Global Logger ---
# Logger will be initialized in main()
logger: Optional[logging.Logger] = None

# --- Path Setup ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path: sys.path.insert(0, project_root)
    if current_dir not in sys.path: sys.path.insert(0, current_dir)
except Exception as e:
    print(f"CRITICAL ERROR during path setup in phase4_main.py: {e}", file=sys.stderr); sys.exit(1)

# --- Project Imports ---
try:
    from phase4_finetuning.config import config as base_finetune_config
    from phase4_finetuning.dataset import SARCLD2024Dataset
    from phase4_finetuning.utils.augmentations import FinetuneAugmentation
    from phase4_finetuning.utils.logging_setup import setup_logging # Ensure this is correct
    from phase4_finetuning.finetune.trainer import Finetuner
    from phase2_model.models.hvt import DiseaseAwareHVT
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import modules for phase4_main.py: {e}", file=sys.stderr); sys.exit(1)


def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    _local_logger = logging.getLogger(f"{__name__}.get_cosine_schedule_with_warmup")
    if num_warmup_steps < 0: raise ValueError("num_warmup_steps must be non-negative.")
    if num_training_steps <= num_warmup_steps and num_training_steps > 0:
        _local_logger.warning(f"num_training_steps ({num_training_steps}) <= num_warmup_steps ({num_warmup_steps}). Scheduler might behave unexpectedly.")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        eff_training_steps = num_training_steps - num_warmup_steps
        if eff_training_steps <=0: return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, eff_training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_config_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    _local_logger = logging.getLogger(f"{__name__}.load_config_yaml")
    config_to_use = base_finetune_config.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_to_use.update(yaml_config)
        except Exception as e:
            _local_logger.warning(f"Could not load or parse YAML {config_path}: {e}. Using defaults.")
    return config_to_use

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning script for HVT models")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    return parser.parse_args()

def set_seed(seed_value: int):
    global logger
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

def _interpolate_pos_embed(ckpt_pe: torch.Tensor, model_pe_param: nn.Parameter,
                           patch_size_cfg: int, current_img_size: Tuple[int, int],
                           cfg: Dict[str, Any]) -> torch.Tensor:
    _local_logger = logging.getLogger(f"{__name__}._interpolate_pos_embed")
    N_s_total, C_s = ckpt_pe.shape[1], ckpt_pe.shape[2]
    N_t_total, C_t = model_pe_param.shape[1], model_pe_param.shape[2]

    if C_s != C_t:
        _local_logger.error(f"Positional Embedding C dimension mismatch: ckpt {C_s} vs model {C_t}. Cannot interpolate.")
        return model_pe_param.data
    if N_s_total == N_t_total:
        return ckpt_pe.view_as(model_pe_param.data)

    if math.isqrt(N_s_total)**2 == N_s_total:
        H0 = W0 = math.isqrt(N_s_total)
    else:
        source_patch_size_hint = cfg.get('hvt_source_model_patch_size', cfg.get('hvt_patch_size', 14))
        if N_s_total == (448//source_patch_size_hint)**2: H0,W0 = 448//source_patch_size_hint,448//source_patch_size_hint
        elif N_s_total == (384//source_patch_size_hint)**2: H0,W0 = 384//source_patch_size_hint,384//source_patch_size_hint
        elif N_s_total == (224//16)**2 and source_patch_size_hint==16 : H0,W0 = 224//16, 224//16
        else:
            _local_logger.error(f"Cannot infer source PE grid dimensions from N_s_total={N_s_total} with patch_hint={source_patch_size_hint}. Interpolation failed.")
            return model_pe_param.data

    H_t = current_img_size[0] // patch_size_cfg
    W_t = current_img_size[1] // patch_size_cfg
    if H_t * W_t != N_t_total:
        _local_logger.error(f"Target PE grid {H_t}x{W_t} (={H_t*W_t}) does not match N_t_total ({N_t_total}). Interpolation failed.")
        return model_pe_param.data

    try:
        pe_reshaped = ckpt_pe.reshape(1, H0, W0, C_s).permute(0, 3, 1, 2)
        pe_interpolated = F.interpolate(pe_reshaped, size=(H_t, W_t), mode='bicubic', align_corners=False)
        pe_interpolated = pe_interpolated.permute(0, 2, 3, 1).flatten(1, 2)

        if pe_interpolated.shape == model_pe_param.shape:
            return pe_interpolated
        else:
            _local_logger.error(f"Interpolated PE shape {pe_interpolated.shape} does not match model PE shape {model_pe_param.shape}. Interpolation failed.")
            return model_pe_param.data
    except Exception as e:
        _local_logger.error(f"Error during PE interpolation: {e}", exc_info=True)
        return model_pe_param.data

def load_pretrained_hvt_backbone(model: DiseaseAwareHVT, checkpoint_path: str, cfg: Dict[str, Any]):
    _local_logger = logging.getLogger(f"{__name__}.load_pretrained_hvt_backbone")
    if not (checkpoint_path and os.path.exists(checkpoint_path)):
        _local_logger.warning(f"Pretrained HVT checkpoint path is invalid or not found: '{checkpoint_path}'. Model will use initial random weights for backbone.")
        return model

    _local_logger.info(f"Attempting to load HVT backbone weights from: {checkpoint_path}")
    try:
        ckpt_data = torch.load(checkpoint_path, map_location='cpu')
        pretrained_sd = ckpt_data.get('model_state_dict', ckpt_data if isinstance(ckpt_data, (OrderedDict, dict)) else None)

        if pretrained_sd is None:
            _local_logger.error(f"Checkpoint format not recognized at {checkpoint_path}. Could not find 'model_state_dict' key. Backbone not loaded.")
            return model

        current_sd = model.state_dict()
        new_sd = OrderedDict()
        head_module_actual_name = "head"
        head_norm_module_actual_name = "head_norm"
        loaded_count = 0; pe_interpolated_count = 0; head_skipped_count = 0; shape_mismatch_count = 0

        for k_ckpt, v_ckpt in pretrained_sd.items():
            if k_ckpt not in current_sd:
                continue
            is_head_param = k_ckpt.startswith(head_module_actual_name + ".") or k_ckpt.startswith(head_norm_module_actual_name + ".")
            if is_head_param:
                head_skipped_count += 1
                continue
            is_pe_key = k_ckpt in ["rgb_pos_embed", "spectral_pos_embed"]
            model_param_for_pe = getattr(model, k_ckpt, None)
            if is_pe_key and isinstance(model_param_for_pe, nn.Parameter) and v_ckpt.shape != model_param_for_pe.shape:
                v_ckpt_3d = v_ckpt.unsqueeze(0) if v_ckpt.ndim == 2 else v_ckpt
                interp_pe = _interpolate_pos_embed(v_ckpt_3d, model_param_for_pe, cfg['hvt_patch_size'], tuple(cfg['img_size']), cfg)
                if interp_pe.shape == model_param_for_pe.shape: new_sd[k_ckpt] = interp_pe; pe_interpolated_count += 1
            elif v_ckpt.shape == current_sd[k_ckpt].shape: new_sd[k_ckpt] = v_ckpt; loaded_count += 1
            else:
                _local_logger.warning(f"Shape mismatch for '{k_ckpt}': Ckpt {v_ckpt.shape} vs Model {current_sd[k_ckpt].shape}. Skipping this weight.")
                shape_mismatch_count += 1

        missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)
        _local_logger.info(f"Pretrained weights loaded: {loaded_count} direct loads, {pe_interpolated_count} PEs interpolated, {head_skipped_count} head/norm params skipped, {shape_mismatch_count} shape mismatches.")
        missing_backbone_keys = [k for k in missing_keys if not (k.startswith(head_module_actual_name + ".") or k.startswith(head_norm_module_actual_name + "."))]
        if missing_backbone_keys:
            _local_logger.warning(f"Weights NOT LOADED for these backbone keys: {missing_backbone_keys}")
        if unexpected_keys:
            _local_logger.error(f"Unexpected keys found when loading state_dict (should be empty if new_sd is subset of model_sd): {unexpected_keys}.")
        _local_logger.info(f"Successfully processed pretrained HVT backbone weights from {checkpoint_path}.")
        return model
    except FileNotFoundError:
        _local_logger.error(f"Pretrained checkpoint file not found at {checkpoint_path}. Backbone not loaded.")
        return model
    except Exception as e:
        _local_logger.error(f"Error loading HVT backbone checkpoint {checkpoint_path}: {e}", exc_info=True)
        return model

def main():
    global logger
    args = parse_args()
    temp_cfg_for_log_setup = load_config_yaml(args.config)
    log_file_main = temp_cfg_for_log_setup.get("log_file_finetune", "finetune_main.log")
    log_dir_main = temp_cfg_for_log_setup.get("log_dir", "logs_finetune_prod")
    if not os.path.isabs(log_dir_main): log_dir_main = os.path.join(current_dir, log_dir_main)
    os.makedirs(log_dir_main, exist_ok=True)
    root_logger_obj = logging.getLogger()
    if root_logger_obj.hasHandlers():
        for handler in root_logger_obj.handlers[:]: root_logger_obj.removeHandler(handler); handler.close()
    
    setup_logging(log_file_name=log_file_main, log_dir=log_dir_main,
                  log_level=logging.DEBUG, # For file
                  logger_name=None) # Removed console_level
    logger = logging.getLogger(__name__)

    cfg = load_config_yaml(args.config)
    set_seed(cfg["seed"])
    logger.info(f"Starting fine-tuning run for model: {cfg.get('model_architecture', cfg.get('model_name', 'HVT'))}")
    device = cfg["device"]
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; cfg['device'] = 'cpu'; logger.warning("CUDA was requested but not available. Switched to CPU.")
    logger.info(f"Using device: {device}")
    if device == 'cuda': logger.info(f"GPU: {torch.cuda.get_device_name(0)}; CUDA Version: {torch.version.cuda}")
    if cfg.get("cudnn_benchmark", True) and device == 'cuda': torch.backends.cudnn.benchmark = True
    if cfg.get("matmul_precision") and hasattr(torch,'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(cfg["matmul_precision"])
        except Exception as e: logger.warning(f"Failed to set matmul_precision: {e}")

    dataset_args = {"root_dir": cfg["data_root"], "img_size": tuple(cfg["img_size"]), "train_split_ratio": cfg["train_split_ratio"], "normalize_for_model": cfg["normalize_data"], "original_dataset_name": cfg["original_dataset_name"], "augmented_dataset_name": cfg["augmented_dataset_name"], "random_seed": cfg["seed"]}
    train_dataset = SARCLD2024Dataset(**dataset_args, split="train"); val_dataset = SARCLD2024Dataset(**dataset_args, split="val")
    sampler = None
    if cfg.get("use_weighted_sampler", False):
        class_weights = train_dataset.get_class_weights()
        if class_weights is not None and len(train_dataset.current_split_labels) > 0:
            sample_weights = torch.zeros(len(train_dataset.current_split_labels));
            for i, label_idx in enumerate(train_dataset.current_split_labels): sample_weights[i] = class_weights[label_idx]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader_num_workers = cfg.get('num_workers', 4); loader_prefetch = cfg.get('prefetch_factor', 2) if loader_num_workers > 0 else None
    loader_persistent = (device == 'cuda' and loader_num_workers > 0 and torch.cuda.is_available())
    dl_common_args = {"num_workers": loader_num_workers, "pin_memory": (device == 'cuda')}
    if loader_num_workers > 0: dl_common_args["prefetch_factor"] = loader_prefetch; dl_common_args["persistent_workers"] = loader_persistent
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None), drop_last=True, **dl_common_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **dl_common_args)
    logger.info(f"Dataloaders: Train batches={len(train_loader)} (samples={len(train_dataset)}), Val batches={len(val_loader)} (samples={len(val_dataset)}). Num workers: {loader_num_workers}")
    class_names = train_dataset.get_class_names()

    hvt_init_args = {"img_size": tuple(cfg["img_size"]), "patch_size": cfg["hvt_patch_size"], "num_classes": cfg["num_classes"], "embed_dim_rgb": cfg["hvt_embed_dim_rgb"], "embed_dim_spectral": cfg["hvt_embed_dim_spectral"], "spectral_channels": cfg["hvt_spectral_channels"], "depths": cfg["hvt_depths"], "num_heads": cfg["hvt_num_heads"], "mlp_ratio": cfg["hvt_mlp_ratio"], "qkv_bias": cfg["hvt_qkv_bias"], "drop_rate": cfg["hvt_model_drop_rate"], "attn_drop_rate": cfg["hvt_attn_drop_rate"], "drop_path_rate": cfg["hvt_drop_path_rate"], "use_dfca": cfg["hvt_use_dfca"], "dfca_num_heads": cfg.get("hvt_dfca_heads"), "dfca_drop_rate": cfg.get("dfca_drop_rate"), "dfca_use_disease_mask": cfg.get("dfca_use_disease_mask"), "use_gradient_checkpointing": cfg.get("use_gradient_checkpointing", False), "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False}
    model = DiseaseAwareHVT(**hvt_init_args)
    if cfg.get("load_pretrained_backbone", False):
        checkpoint_path_from_config = cfg.get("pretrained_checkpoint_path")
        if checkpoint_path_from_config:
            resolved_checkpoint_path = os.path.join(project_root, checkpoint_path_from_config) if not os.path.isabs(checkpoint_path_from_config) else checkpoint_path_from_config
            if not os.path.exists(resolved_checkpoint_path): logger.error(f"CRITICAL: Pretrained backbone checkpoint NOT FOUND at: {resolved_checkpoint_path}"); logger.warning("Proceeding with randomly initialized model weights for the backbone.")
            else: model = load_pretrained_hvt_backbone(model, resolved_checkpoint_path, cfg)
        else: logger.warning("load_pretrained_backbone is True, but no pretrained_checkpoint_path provided. Backbone will be randomly initialized.")
    current_freeze_epoch_target = cfg.get("freeze_backbone_epochs", 0)
    head_module_actual_name = "head"; head_norm_module_actual_name = "head_norm"
    if current_freeze_epoch_target > 0:
        logger.info(f"Freezing backbone (params not starting with '{head_module_actual_name}.' or '{head_norm_module_actual_name}.') for the first {current_freeze_epoch_target} epochs.")
        for name, param in model.named_parameters(): param.requires_grad = name.startswith(head_module_actual_name + ".") or name.startswith(head_norm_module_actual_name + ".")
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad); total_params_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: Total={total_params_count:,}, Trainable={trainable_params_count:,}")
    if trainable_params_count == 0: logger.critical("CRITICAL: No parameters are set to trainable! Check freezing logic and model structure."); sys.exit(1)
    model = model.to(device)
    if cfg.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = cfg.get("torch_compile_mode", "reduce-overhead"); logger.info(f"Attempting to compile model for fine-tuning (mode='{compile_mode}')...")
        try: model = torch.compile(model, mode=compile_mode); logger.info("Fine-tuning model compiled successfully.")
        except Exception as e: logger.warning(f"torch.compile() failed: {e}. Proceeding without compilation.", exc_info=True)
    augmentations = FinetuneAugmentation(tuple(cfg["img_size"])) if cfg.get("augmentations_enabled", True) else None
    loss_class_weights = None
    if cfg.get("use_weighted_loss", False) and not cfg.get("use_weighted_sampler", False):
        loss_class_weights = train_dataset.get_class_weights()
        if loss_class_weights is not None: loss_class_weights = loss_class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_class_weights, label_smoothing=cfg.get("loss_label_smoothing", 0.0))
    lr_main = cfg["learning_rate"]; lr_head_actual = lr_main * cfg.get("head_lr_multiplier", 1.0)
    optimizer_param_groups = []
    head_params_for_opt = [p for n,p in model.named_parameters() if (n.startswith(head_module_actual_name + ".") or n.startswith(head_norm_module_actual_name + ".")) and p.requires_grad]
    if head_params_for_opt: optimizer_param_groups.append({'params': head_params_for_opt, 'lr': lr_head_actual, 'name': 'head'})
    all_backbone_params = [p for n,p in model.named_parameters() if not (n.startswith(head_module_actual_name + ".") or n.startswith(head_norm_module_actual_name + "."))]
    if all_backbone_params: optimizer_param_groups.append({'params': all_backbone_params, 'lr': lr_main, 'name': 'backbone'})
    if not optimizer_param_groups: logger.critical("CRITICAL: No parameter groups for optimizer!"); sys.exit(1)
    optim_name_cfg = cfg.get("optimizer", "adamw").lower(); opt_kwargs = {'weight_decay': cfg.get("weight_decay", 0.01)}; opt_kwargs.update(cfg.get("optimizer_params", {}))
    if optim_name_cfg == "adamw": optimizer = torch.optim.AdamW(optimizer_param_groups, **opt_kwargs)
    elif optim_name_cfg == "sgd": opt_kwargs.setdefault('momentum', 0.9); optimizer = torch.optim.SGD(optimizer_param_groups, **opt_kwargs)
    else: logger.warning(f"Unsupported optimizer '{cfg['optimizer']}'. Defaulting to AdamW."); opt_kwargs.pop('momentum', None); optimizer = torch.optim.AdamW(optimizer_param_groups, **opt_kwargs)
    scheduler = None; lr_reducer_on_plateau = None; main_lr_sched_obj = None
    sched_config_name = cfg.get("scheduler", "None").lower(); warmup_epochs_cfg = cfg.get("warmup_epochs", 0)
    eff_accumulation_steps = max(1, cfg.get("accumulation_steps", 1))
    if sched_config_name != "none":
        steps_per_epoch_sched = len(train_loader) // eff_accumulation_steps
        if steps_per_epoch_sched == 0 and len(train_loader.dataset) > 0: steps_per_epoch_sched = 1; logger.warning("Calculated steps_per_epoch_sched for scheduler is 0, setting to 1.")
        if steps_per_epoch_sched > 0:
            if sched_config_name == "warmupcosine":
                total_training_steps = cfg["epochs"] * steps_per_epoch_sched; warmup_steps = warmup_epochs_cfg * steps_per_epoch_sched
                if warmup_steps >= total_training_steps and total_training_steps > 0: warmup_steps = max(1, int(0.1 * total_training_steps)); logger.warning(f"Warmup steps was >= total steps. Adjusted to {warmup_steps}.")
                if total_training_steps > 0: main_lr_sched_obj = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_training_steps, num_cycles=cfg.get("cosine_num_cycles", 0.5))
                if main_lr_sched_obj: logger.info(f"Using WarmupCosine scheduler (per-step): WU steps={warmup_steps}, Total steps={total_training_steps}")
            elif sched_config_name == "cosineannealinglr":
                epochs_for_cosine = cfg["epochs"];
                if warmup_epochs_cfg > 0 and cfg.get("scheduler_warmup_type", "sequential") == "sequential": epochs_for_cosine -= warmup_epochs_cfg
                epochs_for_cosine = max(1, epochs_for_cosine)
                main_lr_sched_obj = CosineAnnealingLR(optimizer, T_max=epochs_for_cosine, eta_min=cfg.get("eta_min_lr", 1e-7))
                logger.info(f"Using CosineAnnealingLR (per-epoch post-warmup): T_max_epochs={epochs_for_cosine}, Eta_min={cfg.get('eta_min_lr', 1e-7)}.")
            elif sched_config_name == "reducelronplateau":
                rop_mode = 'max' if cfg.get("metric_to_monitor_early_stopping", "f1_macro") != "loss" else 'min'
                lr_reducer_on_plateau = ReduceLROnPlateau(optimizer,mode=rop_mode,factor=cfg.get("reducelr_factor",0.1),patience=cfg.get("reducelr_patience",10),verbose=False)
                logger.info(f"Using ReduceLROnPlateau scheduler (on validation metric). Mode: {rop_mode}, Factor: {cfg.get('reducelr_factor',0.1)}, Patience: {cfg.get('reducelr_patience',10)}")
            if lr_reducer_on_plateau: scheduler = lr_reducer_on_plateau
            elif main_lr_sched_obj: scheduler = main_lr_sched_obj
            if warmup_epochs_cfg > 0 and scheduler and scheduler == main_lr_sched_obj and sched_config_name not in ["warmupcosine", "reducelronplateau"] and cfg.get("scheduler_warmup_type", "sequential") == "sequential":
                warmup_sched_epoch = LinearLR(optimizer, start_factor=cfg.get("warmup_lr_init_factor", 0.01), total_iters=warmup_epochs_cfg)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_sched_epoch, scheduler], milestones=[warmup_epochs_cfg])
                logger.info(f"Combined epoch-based Linear Warmup ({warmup_epochs_cfg} epochs) with {sched_config_name} scheduler.")
            if sched_config_name == "warmupcosine" and steps_per_epoch_sched == 0 : sched_config_name = "none"
    scaler = GradScaler(enabled=(cfg.get("amp_enabled", True) and device == 'cuda'))
    trainer = Finetuner(model=model, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, scheduler=scheduler, lr_scheduler_on_batch=(scheduler is not None and sched_config_name == "warmupcosine"), accumulation_steps=eff_accumulation_steps, clip_grad_norm=cfg.get("clip_grad_norm"), augmentations=augmentations, num_classes=cfg["num_classes"])
    best_val_metric = 0.0 if cfg.get("metric_to_monitor_early_stopping", "f1_macro") != "loss" else float('inf')
    metric_to_monitor = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    early_stopping_patience_val = cfg.get("early_stopping_patience", float('inf'))
    patience_counter = 0; last_completed_epoch = 0; current_val_metric_for_saving: Optional[float] = None
    logger.info(f"Starting fine-tuning for {cfg['epochs']} epochs. Monitoring '{metric_to_monitor}' for best model and early stopping (patience: {early_stopping_patience_val}).")
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            last_completed_epoch = epoch - 1
            if current_freeze_epoch_target > 0 and epoch == current_freeze_epoch_target + 1:
                logger.info(f"Epoch {epoch}: Unfreezing backbone layers.")
                unfrozen_count = 0; model_to_unfreeze = model._orig_mod if hasattr(model, '_orig_mod') and model._orig_mod is not None else model
                for name, param in model_to_unfreeze.named_parameters():
                    if not (name.startswith(head_module_actual_name + ".") or name.startswith(head_norm_module_actual_name + ".")) and not param.requires_grad:
                        param.requires_grad = True; unfrozen_count +=1
                if unfrozen_count > 0: logger.info(f"{unfrozen_count} backbone parameters unfrozen and set to requires_grad=True.")
                for group in optimizer.param_groups:
                    if group.get('name') == 'backbone': group['lr'] = cfg["learning_rate"] * cfg.get("unfreeze_backbone_lr_factor", 1.0); logger.info(f"Optimizer group '{group['name']}' (backbone) LR set to {group['lr']:.2e} after unfreezing.")
                trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad); logger.info(f"Model parameters post-unfreeze: Total={total_params_count:,}, Trainable={trainable_params_count:,}")
            
            avg_train_loss = trainer.train_one_epoch(train_loader, epoch, cfg["epochs"])
            
            val_metrics: Dict[str, float] = {}; avg_val_loss: float = float('inf'); current_val_metric_for_decision: Optional[float] = None
            if epoch % cfg.get("evaluate_every_n_epochs", 1) == 0 or epoch == cfg["epochs"]:
                avg_val_loss, val_metrics = trainer.validate_one_epoch(val_loader, class_names=class_names)
                current_val_metric_for_saving = avg_val_loss
                default_for_monitored = avg_val_loss if metric_to_monitor == "loss" else (0.0 if metric_to_monitor != "loss" else float('inf'))
                current_val_metric_for_decision = val_metrics.get(metric_to_monitor, default_for_monitored)
                if metric_to_monitor != "loss" and current_val_metric_for_decision == float('inf'): current_val_metric_for_decision = 0.0
                current_val_metric_for_saving = current_val_metric_for_decision
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_val_metric_for_decision)
                elif scheduler and not trainer.lr_scheduler_on_batch:
                    if isinstance(scheduler, SequentialLR):
                        active_sub_scheduler_stepped = False
                        if scheduler._milestones and len(scheduler._milestones) > 0 and epoch >= scheduler._milestones[0] and len(scheduler._schedulers) > 1 : scheduler._schedulers[1].step(); active_sub_scheduler_stepped = True
                        elif len(scheduler._schedulers) > 0: scheduler._schedulers[0].step(); active_sub_scheduler_stepped = True
                        if not active_sub_scheduler_stepped and hasattr(scheduler, 'step'): scheduler.step()
                    else: scheduler.step()
                improved = False
                if current_val_metric_for_decision is not None:
                    if metric_to_monitor != "loss": improved = current_val_metric_for_decision > best_val_metric
                    else: improved = current_val_metric_for_decision < best_val_metric
                if improved:
                    best_val_metric = current_val_metric_for_decision
                    if cfg.get("best_model_path"):
                        trainer.save_model_checkpoint(cfg["best_model_path"])
                    logger.info(f"Epoch {epoch}: New best model! Val {metric_to_monitor}: {best_val_metric:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= early_stopping_patience_val: logger.info(f"Early stopping triggered at epoch {epoch}."); break
            last_completed_epoch = epoch
    except KeyboardInterrupt:
        logger.warning(f"Fine-tuning interrupted by user at epoch {last_completed_epoch + 1}.")
    except Exception as e:
        logger.critical(f"Critical error during fine-tuning at epoch {last_completed_epoch + 1}: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"Fine-tuning process ended. Last completed epoch: {last_completed_epoch}.")
        if 'trainer' in locals() and trainer is not None and cfg.get("final_model_path"):
            trainer.save_model_checkpoint(cfg["final_model_path"])
        logger.info(f"Fine-tuning summary: Best validation '{metric_to_monitor}' achieved: {best_val_metric:.4f}")
        if early_stopping_patience_val != float('inf') and patience_counter >= early_stopping_patience_val:
             logger.info(f"Early stopping was triggered after {last_completed_epoch} epochs.")
        elif last_completed_epoch < cfg["epochs"] and not (patience_counter >= early_stopping_patience_val) :
             logger.info(f"Training stopped before completing all {cfg['epochs']} epochs (e.g., user interruption or error).")


if __name__ == "__main__":
    # Set console logging to WARNING for minimalist output before main's logger setup
    # This basicConfig will be overridden by setup_logging in main() for the file handler,
    # but it sets the default for console messages if setup_logging doesn't explicitly add a console handler
    # or if an error occurs before setup_logging.
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    try:
        main()
    except SystemExit as se:
        exit_logger = logging.getLogger(__name__) if logger and logger.hasHandlers() else logging.getLogger("main_fallback")
        if se.code != 0 and se.code is not None :
            # Use info for exit codes, as critical/error would be logged by main if it reached there
            exit_logger.info(f"Application exited with status code {se.code}.")
        sys.exit(se.code if se.code is not None else 0) # Propagate exit code
    except Exception as e:
        final_logger = logging.getLogger(__name__) if logger and logger.hasHandlers() else logging.getLogger("main_fallback")
        has_effective_handlers = any(h for h in final_logger.handlers) or \
                                 (final_logger.parent and any(h for h in final_logger.parent.handlers))
        
        if not has_effective_handlers: # If no logger is truly configured to output anywhere
            print(f"CRITICAL UNHANDLED EXCEPTION IN __main__ (fallback print): {e}\n{traceback.format_exc()}", file=sys.stderr)
        final_logger.critical(f"Unhandled exception in __main__ execution: {e}", exc_info=True)
        sys.exit(1)