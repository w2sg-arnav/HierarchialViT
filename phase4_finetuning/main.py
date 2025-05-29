# phase4_finetuning/main.py
from collections import OrderedDict, Counter as PyCounter
import os
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, LinearLR, ReduceLROnPlateau
import argparse
import yaml
import torch.nn.functional as F
import math
import sys
from typing import Tuple, Optional, Dict, Any, List
import traceback
from datetime import datetime
import time

logger: Optional[logging.Logger] = None

try:
    from .config import config as base_finetune_config
    from .dataset import SARCLD2024Dataset
    from .utils.augmentations import create_augmentation # UPDATED IMPORT
    from .utils.logging_setup import setup_logging
    from .finetune.trainer import Finetuner
    from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt
except ImportError as e_imp:
    print(f"CRITICAL IMPORT ERROR: {e_imp}. Check paths and ensure Phase 2 models are accessible.", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

def get_cosine_schedule_with_warmup_step(optimizer: torch.optim.Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
    _local_logger_sched = logging.getLogger(f"{__name__}.get_cosine_schedule_with_warmup_step")
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
        eff_training_steps = num_training_steps - num_warmup_steps
        if eff_training_steps <=0: return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, eff_training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def load_config_from_yaml_or_default(yaml_config_path: Optional[str] = None) -> Dict[str, Any]:
    config_to_use = base_finetune_config.copy()
    if yaml_config_path and os.path.exists(yaml_config_path):
        try:
            with open(yaml_config_path, 'r') as f: yaml_override_config = yaml.safe_load(f)
            if yaml_override_config and isinstance(yaml_override_config, dict):
                config_to_use.update(yaml_override_config)
                logging.info(f"Loaded and applied config overrides from YAML: {yaml_config_path}")
        except Exception as e_yaml: logging.warning(f"Could not load/parse YAML {yaml_config_path}: {e_yaml}. Using defaults.")
    elif yaml_config_path: logging.warning(f"Specified YAML config {yaml_config_path} not found. Using defaults.")
    return config_to_use

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning script for HVT models.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    return parser.parse_args()

def set_global_seed(seed_value: int):
    torch.manual_seed(seed_value); np.random.seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)
    if logger: logger.info(f"Global random seed set to: {seed_value}")
    else: print(f"Global random seed set to: {seed_value} (logger not yet init)")

def _interpolate_positional_embedding(
    checkpoint_pos_embed: torch.Tensor, model_pos_embed_param: nn.Parameter,
    hvt_patch_size: int, target_img_size_for_finetune: Tuple[int, int],
    ssl_img_size_for_pos_embed: Tuple[int, int]) -> Tuple[torch.Tensor, bool]:
    _local_logger_pe = logging.getLogger(f"{__name__}._interpolate_pe")
    if checkpoint_pos_embed.ndim == 2: checkpoint_pos_embed = checkpoint_pos_embed.unsqueeze(0)

    if checkpoint_pos_embed.ndim == 2:
        num_tokens_ckpt, C_ckpt = checkpoint_pos_embed.shape
    elif checkpoint_pos_embed.ndim == 3:
        num_tokens_ckpt, C_ckpt = checkpoint_pos_embed.shape[1], checkpoint_pos_embed.shape[2]
    else:
        _local_logger_pe.error(f"PE Interp: Unexpected checkpoint_pos_embed ndim: {checkpoint_pos_embed.ndim}. No interp."); return model_pos_embed_param.data, False

    num_tokens_model, C_model = model_pos_embed_param.shape[1], model_pos_embed_param.shape[2]

    if C_ckpt != C_model:
        _local_logger_pe.error(f"PE C-dim mismatch (ckpt:{C_ckpt} vs model:{C_model}). No interpolation."); return model_pos_embed_param.data, False
    
    if num_tokens_ckpt == num_tokens_model:
        _local_logger_pe.debug(f"PE N-dim and C-dim match ({num_tokens_ckpt} tokens, C={C_ckpt}). Using ckpt PE directly.");
        return checkpoint_pos_embed.view_as(model_pos_embed_param.data), True

    H0_patches = ssl_img_size_for_pos_embed[0] // hvt_patch_size
    W0_patches = ssl_img_size_for_pos_embed[1] // hvt_patch_size
    if H0_patches * W0_patches != num_tokens_ckpt:
        _local_logger_pe.warning(f"PE Interp: num_tokens_ckpt ({num_tokens_ckpt}) does not match SSL grid ({H0_patches}x{W0_patches} from {ssl_img_size_for_pos_embed} / {hvt_patch_size}). Trying sqrt inference.")
        if math.isqrt(num_tokens_ckpt)**2 == num_tokens_ckpt:
            H0_patches = W0_patches = math.isqrt(num_tokens_ckpt)
            _local_logger_pe.info(f"PE Interp: Inferred source grid as {H0_patches}x{W0_patches} from num_tokens_ckpt {num_tokens_ckpt}.")
        else:
            _local_logger_pe.error(f"PE Interp: Cannot infer source grid for num_tokens_ckpt={num_tokens_ckpt}. No interp."); return model_pos_embed_param.data, False

    Ht_patches = target_img_size_for_finetune[0] // hvt_patch_size
    Wt_patches = target_img_size_for_finetune[1] // hvt_patch_size
    if Ht_patches * Wt_patches != num_tokens_model:
        _local_logger_pe.error(f"PE Interp: Target model patch count num_tokens_model={num_tokens_model} does not match target grid ({Ht_patches}x{Wt_patches} from {target_img_size_for_finetune} / {hvt_patch_size}). No interp."); return model_pos_embed_param.data, False

    _local_logger_pe.info(f"PE Interpolating: {num_tokens_ckpt} patches ({H0_patches}x{W0_patches}) from SSL to {num_tokens_model} patches ({Ht_patches}x{Wt_patches}) for finetune.")
    try:
        pe_to_interp = checkpoint_pos_embed.reshape(1, H0_patches, W0_patches, C_ckpt).permute(0, 3, 1, 2)
        pe_interpolated = F.interpolate(pe_to_interp, size=(Ht_patches, Wt_patches), mode='bicubic', align_corners=False)
        pe_interpolated = pe_interpolated.permute(0, 2, 3, 1).flatten(1, 2)
        
        if pe_interpolated.shape == model_pos_embed_param.shape:
            _local_logger_pe.info("PE Interpolation successful.")
            return pe_interpolated, True
        else:
            _local_logger_pe.error(f"PE Interp: Final shape mismatch after interpolation {pe_interpolated.shape} vs {model_pos_embed_param.shape}. No interp."); return model_pos_embed_param.data, False
    except Exception as e:
        _local_logger_pe.error(f"PE Interp Exception: {e}", exc_info=True); return model_pos_embed_param.data, False

def load_and_prepare_hvt_model(hvt_model_instance: DiseaseAwareHVT, cfg: Dict[str, Any], device: str) -> DiseaseAwareHVT:
    _local_logger_load = logging.getLogger(f"{__name__}.load_and_prepare_hvt_model")
    hvt_model_instance.cpu()

    if cfg.get("load_pretrained_backbone", False):
        ssl_ckpt_path_from_cfg = cfg.get("pretrained_checkpoint_path")
        resolved_ssl_ckpt_path = ssl_ckpt_path_from_cfg
        project_root_for_path = cfg.get("PROJECT_ROOT_PATH")
        if project_root_for_path is None and 'PACKAGE_ROOT_PATH' in cfg:
            project_root_for_path = os.path.dirname(cfg['PACKAGE_ROOT_PATH'])

        if not os.path.isabs(resolved_ssl_ckpt_path) and project_root_for_path and ssl_ckpt_path_from_cfg:
             resolved_ssl_ckpt_path = os.path.join(project_root_for_path, ssl_ckpt_path_from_cfg)

        if resolved_ssl_ckpt_path and os.path.exists(resolved_ssl_ckpt_path):
            _local_logger_load.info(f"Loading SSL backbone weights from: {resolved_ssl_ckpt_path}")
            checkpoint = torch.load(resolved_ssl_ckpt_path, map_location='cpu')
            ssl_backbone_sd = checkpoint.get('model_backbone_state_dict')

            if ssl_backbone_sd:
                current_model_sd = hvt_model_instance.state_dict()
                new_sd_for_backbone = OrderedDict()
                ssl_run_cfg_snapshot = checkpoint.get('run_config_snapshot', {})
                ssl_img_size_val = cfg.get("ssl_pretrain_img_size_fallback",
                                           ssl_run_cfg_snapshot.get('pretrain_img_size', cfg["img_size"]))
                ssl_img_size_tuple = tuple(ssl_img_size_val)

                if tuple(ssl_img_size_tuple) == tuple(cfg["img_size"]):
                    _local_logger_load.info(f"SSL and Finetune img_size match: {ssl_img_size_tuple}.")
                else:
                    _local_logger_load.info(f"SSL img_size: {ssl_img_size_tuple}, Finetune img_size: {cfg['img_size']}. PE interpolation may occur.")

                loaded_count, pe_interp_successful_count, head_skip_count = 0,0,0
                for k_ckpt, v_ckpt in ssl_backbone_sd.items():
                    if k_ckpt not in current_model_sd: continue
                    if k_ckpt.startswith("classifier_head."): head_skip_count += 1; continue

                    is_pe, target_pe_p = (k_ckpt in ["rgb_pos_embed", "spectral_pos_embed"]), getattr(hvt_model_instance, k_ckpt, None)

                    if is_pe and target_pe_p is not None and isinstance(target_pe_p, nn.Parameter):
                        ckpt_pe_c_dim = v_ckpt.shape[-1]
                        model_pe_c_dim = target_pe_p.shape[-1]

                        if ckpt_pe_c_dim != model_pe_c_dim:
                            _local_logger_load.warning(f"PE {k_ckpt}: C-dim mismatch (ckpt:{ckpt_pe_c_dim} vs model:{model_pe_c_dim}). Model's original PE will be used.")
                            continue

                        ckpt_pe_n_tokens = v_ckpt.shape[-2] if v_ckpt.ndim > 1 else v_ckpt.shape[0] # Handle (N,C) or (1,N,C)
                        model_pe_n_tokens = target_pe_p.shape[-2]

                        if ckpt_pe_n_tokens != model_pe_n_tokens:
                            _local_logger_load.info(f"PE {k_ckpt}: C-dim matches ({ckpt_pe_c_dim}), N-dim differs (ckpt_tokens:{ckpt_pe_n_tokens} vs model_tokens:{model_pe_n_tokens}). Attempting PE interpolation.")
                            interp_pe_val, interp_success = _interpolate_positional_embedding(
                                v_ckpt, target_pe_p,
                                cfg['hvt_params_for_model_init']['patch_size'],
                                tuple(cfg["img_size"]),
                                ssl_img_size_tuple
                            )
                            if interp_success:
                                new_sd_for_backbone[k_ckpt] = interp_pe_val
                                pe_interp_successful_count +=1 # Count successful interpolations
                            else:
                                _local_logger_load.warning(f"PE {k_ckpt} interpolation failed. Model's original PE will be used.")
                        elif v_ckpt.shape == target_pe_p.shape or \
                             (v_ckpt.ndim == 2 and target_pe_p.ndim == 3 and v_ckpt.shape == target_pe_p.shape[1:]):
                             _local_logger_load.info(f"PE {k_ckpt}: Shape matches. Loading directly.")
                             new_sd_for_backbone[k_ckpt] = v_ckpt.view_as(target_pe_p.data)
                             loaded_count +=1 # Counts as direct load if shapes fully match
                        else:
                             _local_logger_load.warning(f"PE {k_ckpt}: Unhandled shape scenario ({v_ckpt.shape} vs {target_pe_p.shape}). Model's original PE will be used.")
                    
                    elif v_ckpt.shape == current_model_sd[k_ckpt].shape:
                        new_sd_for_backbone[k_ckpt] = v_ckpt; loaded_count +=1
                    else: _local_logger_load.warning(f"Shape mismatch for {k_ckpt}: ckpt {v_ckpt.shape} vs model {current_model_sd[k_ckpt].shape}. Skipping.")

                msg = hvt_model_instance.load_state_dict(new_sd_for_backbone, strict=False)
                _local_logger_load.info(f"SSL Backbone weights loaded: {loaded_count} direct, {pe_interp_successful_count} PE interpolated, {head_skip_count} head layers skipped.")
                if msg.missing_keys: _local_logger_load.warning(f"Missing keys when loading SSL backbone: {msg.missing_keys}")
                if msg.unexpected_keys: _local_logger_load.warning(f"Unexpected keys in SSL backbone load: {msg.unexpected_keys}")

                if hasattr(hvt_model_instance, 'classifier_head') and isinstance(hvt_model_instance.classifier_head, nn.Linear):
                    in_feats_current_model = hvt_model_instance.classifier_head.in_features
                    final_norm_layer_key_rgb = 'norm_rgb_final_encoder.weight'
                    final_norm_layer_key_classifier = 'classifier_head_norm.weight'
                    expected_in_features = None
                    if final_norm_layer_key_classifier in current_model_sd:
                        expected_in_features = current_model_sd[final_norm_layer_key_classifier].shape[0]
                    elif final_norm_layer_key_rgb in current_model_sd:
                         expected_in_features = current_model_sd[final_norm_layer_key_rgb].shape[0]
                    
                    if expected_in_features is not None and in_feats_current_model != expected_in_features:
                        _local_logger_load.warning(f"Classifier head in_features ({in_feats_current_model}) differs from expected from norm ({expected_in_features}). Re-initializing with {expected_in_features}.")
                        in_feats_current_model = expected_in_features
                    elif expected_in_features is None:
                         _local_logger_load.warning(f"Could not determine expected in_features for classifier head from norm layers. Using existing {in_feats_current_model}.")

                    hvt_model_instance.classifier_head = nn.Linear(in_feats_current_model, cfg["num_classes"])
                    _local_logger_load.info(f"Re-initialized HVT classifier_head for {cfg['num_classes']} classes (in_features={in_feats_current_model}).")
                else: _local_logger_load.error("Could not re-initialize classifier_head. VERIFY HVT MODEL STRUCTURE.")
            else: _local_logger_load.error(f"Key 'model_backbone_state_dict' not found in checkpoint {resolved_ssl_ckpt_path}. Backbone will be randomly initialized.")
        elif ssl_ckpt_path_from_cfg: _local_logger_load.warning(f"Specified SSL checkpoint path '{resolved_ssl_ckpt_path}' not found. Backbone will be randomly initialized.")
        else: _local_logger_load.info("No SSL checkpoint path provided ('pretrained_checkpoint_path' is empty or None). Backbone will be randomly initialized.")
    else: _local_logger_load.info("'load_pretrained_backbone' is False. Backbone will be randomly initialized.")
    return hvt_model_instance.to(device)

def main_execution_logic():
    global logger
    args = parse_arguments()
    cfg = load_config_from_yaml_or_default(args.config)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    abs_pkg_root_for_logs = cfg.get("PACKAGE_ROOT_PATH", os.path.dirname(os.path.abspath(__file__)))
    abs_log_dir = os.path.join(abs_pkg_root_for_logs, cfg.get("log_dir", "logs_finetune_phase4"))
    log_file_base = os.path.splitext(cfg.get("log_file_finetune", "finetune_run.log"))[0]
    final_log_filename = f"{log_file_base}_{run_ts}.log"

    root_logger_obj = logging.getLogger();
    if root_logger_obj.hasHandlers():
        for handler in root_logger_obj.handlers[:]: root_logger_obj.removeHandler(handler); handler.close()
    setup_logging(log_file_name=final_log_filename, log_dir=abs_log_dir, log_level=logging.DEBUG, run_timestamp=run_ts)
    logger = logging.getLogger(__name__)

    logger.info(f"======== Starting Phase 4: HVT Fine-tuning (Run ID: {run_ts}) ========")
    logger.info(f"Full run configuration: {cfg}")
    set_global_seed(cfg["seed"])
    device = cfg["device"]
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; cfg['device'] = 'cpu'; logger.warning("CUDA unavailable. Using CPU.")
    logger.info(f"Using device: {device}. GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    if cfg.get("cudnn_benchmark", True) and device == 'cuda': torch.backends.cudnn.benchmark = True
    if cfg.get("matmul_precision") and hasattr(torch,'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(cfg["matmul_precision"])
        except Exception as e: logger.warning(f"Failed to set matmul_precision: {e}")

    dataset_args = {
        "root_dir": cfg["data_root"], "img_size": tuple(cfg["img_size"]),
        "train_split_ratio": cfg["train_split_ratio"], "normalize_for_model": cfg["normalize_data"],
        "original_dataset_name": cfg["original_dataset_name"],
        "augmented_dataset_name": cfg.get("augmented_dataset_name", None), "random_seed": cfg["seed"]
    }
    train_dataset = SARCLD2024Dataset(**dataset_args, split="train"); val_dataset = SARCLD2024Dataset(**dataset_args, split="val")
    class_names = train_dataset.get_class_names()

    sampler = None
    if cfg.get("use_weighted_sampler", False):
        if not hasattr(train_dataset, 'current_split_labels') or len(train_dataset.current_split_labels) == 0:
            logger.warning("WeightedRandomSampler requested, but train_dataset has no current_split_labels or it's empty.")
        else:
            train_labels_np = train_dataset.current_split_labels; class_counts_counter = PyCounter(train_labels_np)
            num_classes = cfg.get("num_classes", train_dataset.num_classes); sampler_mode = cfg.get("weighted_sampler_mode", "inv_freq")
            per_class_weight = torch.ones(num_classes, dtype=torch.float)
            if sampler_mode == "sqrt_inv_count":
                for i in range(num_classes):
                    count = class_counts_counter.get(i, 0)
                    if count > 0: per_class_weight[i] = 1.0 / math.sqrt(float(count))
                    else: per_class_weight[i] = 1.0
                logger.info(f"Using WeightedRandomSampler with 'sqrt_inv_count' mode. Per-class weights: {per_class_weight.numpy().round(3)}")
            elif sampler_mode == "inv_freq": # Fallback to inv_freq if needed
                retrieved_weights = train_dataset.get_class_weights()
                if retrieved_weights is not None and len(retrieved_weights) == num_classes: per_class_weight = retrieved_weights
                else: logger.error(f"Failed to get valid class weights for sampler (inv_freq mode).")
                logger.info(f"Using WeightedRandomSampler with 'inv_freq' mode. Per-class weights: {per_class_weight.numpy().round(3)}")
            sample_weights = torch.zeros(len(train_labels_np), dtype=torch.float)
            for i, label_idx in enumerate(train_labels_np):
                if 0 <= label_idx < len(per_class_weight): sample_weights[i] = per_class_weight[label_idx]
                else: logger.warning(f"Label index {label_idx} out of bounds. Using default weight 1.0."); sample_weights[i] = 1.0
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            logger.info(f"WeightedRandomSampler enabled (mode: {sampler_mode}).")

    loader_args = {"num_workers": cfg.get('num_workers',4), "pin_memory": (device=='cuda'), "persistent_workers": (device=='cuda' and cfg.get('num_workers',4)>0)}
    if cfg.get('num_workers',4) > 0 and cfg.get('prefetch_factor') is not None : loader_args["prefetch_factor"] = cfg['prefetch_factor']
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None), drop_last=True, **loader_args)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, drop_last=False, **loader_args)
    logger.info(f"Dataloaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model_hvt_instance = create_disease_aware_hvt(current_img_size=tuple(cfg["img_size"]), num_classes=cfg["num_classes"], model_params_dict=cfg['hvt_params_for_model_init'])
    model = load_and_prepare_hvt_model(model_hvt_instance, cfg, device)
    total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model ready. Total params: {total_params:,}, Trainable params: {trainable_params:,}")

    if cfg.get("enable_torch_compile", False) and hasattr(torch, 'compile'):
        compile_mode = cfg.get("torch_compile_mode", "reduce-overhead"); logger.info(f"Attempting torch.compile (mode='{compile_mode}')...")
        try: model = torch.compile(model, mode=compile_mode); logger.info("Model compiled.")
        except Exception as e: logger.warning(f"torch.compile() failed: {e}. No compilation.", exc_info=True)
    
    augmentations_pipeline = None
    if cfg.get("augmentations_enabled", True):
        augmentations_pipeline = create_augmentation(
            strategy=cfg.get('augmentation_strategy', 'stable_enhanced'),
            img_size=tuple(cfg["img_size"]),
            severity=cfg.get('augmentation_severity', 'mild')
        )
    
    loss_fn_weights = None
    if cfg.get("use_weighted_loss", False):
        retrieved_loss_weights = train_dataset.get_class_weights()
        if retrieved_loss_weights is not None:
            loss_fn_weights = retrieved_loss_weights.to(device)
            logger.info(f"Using weighted CrossEntropyLoss. Weights: {loss_fn_weights.cpu().numpy().round(3)}")
        else: logger.warning("use_weighted_loss is True, but could not get class weights. Using unweighted loss.")
    criterion = nn.CrossEntropyLoss(weight=loss_fn_weights, label_smoothing=cfg.get("loss_label_smoothing", 0.1))

    head_module_name = "classifier_head"
    all_head_params = [p for n, p in model.named_parameters() if n.startswith(head_module_name + ".")]
    all_backbone_params = [p for n, p in model.named_parameters() if not n.startswith(head_module_name + ".")]
    freeze_backbone_until_epoch_cfg = cfg.get("freeze_backbone_epochs", 0)
    
    if freeze_backbone_until_epoch_cfg > 0: 
        initial_lr_head = cfg['lr_head_frozen_phase']; initial_lr_backbone = 0.0 
        logger.info(f"Optimizer initial setup: Backbone frozen. Head LR: {initial_lr_head:.2e}, Backbone LR: {initial_lr_backbone:.2e}")
    else: 
        initial_lr_head = cfg['lr_head_unfrozen_phase']; initial_lr_backbone = cfg['lr_backbone_unfrozen_phase']
        logger.info(f"Optimizer initial setup: Training full model. Head LR: {initial_lr_head:.2e}, Backbone LR: {initial_lr_backbone:.2e}")

    param_groups = []
    if all_head_params: param_groups.append({'params': all_head_params, 'name': 'head', 'lr': initial_lr_head})
    if all_backbone_params: param_groups.append({'params': all_backbone_params, 'name': 'backbone', 'lr': initial_lr_backbone})
    
    optim_name = cfg.get("optimizer", "AdamW").lower()
    default_opt_constructor_lr = max(cfg.get('lr_head_unfrozen_phase', 1e-4), cfg.get('lr_backbone_unfrozen_phase', 1e-5))
    opt_kwargs = cfg.get("optimizer_params", {}).copy(); opt_kwargs['weight_decay'] = cfg.get("weight_decay", 0.05)
    opt_kwargs['lr'] = default_opt_constructor_lr 

    if optim_name == "adamw": optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    elif optim_name == "sgd": opt_kwargs.pop('eps', None); optimizer = torch.optim.SGD(param_groups, **opt_kwargs)
    else: logger.warning(f"Optimizer '{optim_name}' not handled. Defaulting to AdamW."); optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__} created with overall defaults: {optimizer.defaults}")
    for pg_idx, pg in enumerate(optimizer.param_groups):
        logger.info(f"  Group '{pg.get('name', f'UnnamedGroup{pg_idx}')}': Configured LR {pg.get('lr'):.2e}, WD {pg.get('weight_decay'):.2e}")

    scheduler, lr_scheduler_on_batch_flag = None, False
    sched_cfg_name = cfg.get("scheduler", "None").lower(); warmup_epochs_cfg = cfg.get("warmup_epochs", 0)
    total_epochs_for_sched = cfg["epochs"]; eff_accum_steps_sched = max(1, cfg.get("accumulation_steps", 1))
    steps_per_epoch_for_sched = len(train_loader) // eff_accum_steps_sched
    if steps_per_epoch_for_sched == 0 and len(train_loader.dataset) > 0: steps_per_epoch_for_sched = 1
    
    if sched_cfg_name != "none" and steps_per_epoch_for_sched > 0 :
        if sched_cfg_name == "warmupcosine":
            total_sched_steps = total_epochs_for_sched * steps_per_epoch_for_sched
            warmup_sched_steps = warmup_epochs_cfg * steps_per_epoch_for_sched
            if warmup_sched_steps >= total_sched_steps and total_sched_steps > 0:
                warmup_sched_steps = max(0, int(0.1 * total_sched_steps))
                logger.warning(f"Warmup steps adjusted to {warmup_sched_steps}.")
            if total_sched_steps > 0 : 
                scheduler = get_cosine_schedule_with_warmup_step(optimizer, warmup_sched_steps, total_sched_steps, num_cycles=0.5, last_epoch=-1)
                lr_scheduler_on_batch_flag = True
            if scheduler: logger.info(f"Scheduler: WarmupCosine (per-step). WU Steps: {warmup_sched_steps}, Total Steps: {total_sched_steps}")
        elif sched_cfg_name == "cosineannealinglr":
            cosine_t_max = total_epochs_for_sched - warmup_epochs_cfg if warmup_epochs_cfg > 0 else total_epochs_for_sched
            if cosine_t_max <=0: cosine_t_max = 1 
            main_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=cfg.get("eta_min_lr", 0))
            if warmup_epochs_cfg > 0:
                warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs_cfg)
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs_cfg])
                logger.info(f"Scheduler: SequentialLR(Warmup({warmup_epochs_cfg}eps) -> CosineAnnealingLR(T_max={cosine_t_max})).")
            else: scheduler = main_scheduler; logger.info(f"Scheduler: CosineAnnealingLR (per-epoch). T_max={cosine_t_max}")
            lr_scheduler_on_batch_flag = False 
    
    scaler_obj = GradScaler(enabled=(cfg.get("amp_enabled", True) and device == 'cuda'))
    finetuner_obj = Finetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        scaler=scaler_obj, scheduler=scheduler, lr_scheduler_on_batch=lr_scheduler_on_batch_flag,
        accumulation_steps=eff_accum_steps_sched, 
        clip_grad_norm=cfg.get("clip_grad_norm"),
        augmentations=augmentations_pipeline, 
        num_classes=cfg["num_classes"],
        tta_enabled_val=cfg.get("tta_enabled_val", False),
        debug_nan_detection=cfg.get("debug_nan_detection", False),
        stop_on_nan_threshold=cfg.get("stop_on_nan_threshold", 5),
        monitor_gradients=cfg.get("monitor_gradients", False),
        gradient_log_interval=cfg.get("gradient_log_interval", 50)
    )

    best_val_metric_val = 0.0 if cfg.get("metric_to_monitor_early_stopping", "f1_macro") != "val_loss" else float('inf')
    metric_to_watch = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    patience_val = cfg.get("early_stopping_patience", float('inf'))
    patience_count = 0; last_completed_ep_main = 0
    
    abs_ckpt_save_dir = os.path.join(abs_log_dir, cfg.get("checkpoint_save_dir_name", "checkpoints"))
    os.makedirs(abs_ckpt_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved in: {abs_ckpt_save_dir}")
    
    if scheduler :
        logger.info("LRs after scheduler initialization:")
        for i, opt_g in enumerate(optimizer.param_groups):
            sched_base_lr_str = ""
            if hasattr(scheduler, 'base_lrs') and i < len(scheduler.base_lrs):
                 sched_base_lr_str = f", Scheduler Base LR {scheduler.base_lrs[i]:.2e}"
            logger.info(f"  Opt Group {i} ('{opt_g.get('name')}'): Current LR in optimizer {opt_g['lr']:.2e}{sched_base_lr_str}")

    freeze_backbone_until_epoch_from_cfg = cfg.get("freeze_backbone_epochs", 0)
    logger.info(f"Starting fine-tuning for {cfg['epochs']} epochs. Monitor: '{metric_to_watch}'. Backbone frozen for initial {freeze_backbone_until_epoch_from_cfg} epochs.")
    nan_threshold_exceeded_flag = False
    try:
        for epoch_1_based in range(1, cfg["epochs"] + 1):
            last_completed_ep_main = epoch_1_based -1
            epoch_start_time = time.time()

            is_backbone_frozen_this_epoch = epoch_1_based <= freeze_backbone_until_epoch_from_cfg
            model_to_operate_on = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model
            
            backbone_target_requires_grad = not is_backbone_frozen_this_epoch
            newly_unfrozen_this_epoch_flag = False

            for name, param in model_to_operate_on.named_parameters():
                if name.startswith(head_module_name + "."): param.requires_grad = True 
                else: 
                    if param.requires_grad != backbone_target_requires_grad: 
                        if backbone_target_requires_grad and not param.requires_grad : newly_unfrozen_this_epoch_flag = True
                        param.requires_grad = backbone_target_requires_grad
            
            if epoch_1_based == 1:
                current_phase_msg = "Starting in Frozen Backbone phase." if is_backbone_frozen_this_epoch else "Starting Full Model Training phase."
                logger.info(f"Epoch {epoch_1_based}: {current_phase_msg}")
            
            elif newly_unfrozen_this_epoch_flag and (epoch_1_based == freeze_backbone_until_epoch_from_cfg + 1):
                logger.info(f"Epoch {epoch_1_based}: Backbone UNPROZEN. Updating LRs.")
                lr_head_target_unfrozen = cfg['lr_head_unfrozen_phase']; lr_backbone_target_unfrozen = cfg['lr_backbone_unfrozen_phase']
                for p_group_idx, p_group in enumerate(optimizer.param_groups):
                    group_name = p_group.get('name')
                    target_lr_for_group = None
                    if group_name == 'head': target_lr_for_group = lr_head_target_unfrozen
                    elif group_name == 'backbone': target_lr_for_group = lr_backbone_target_unfrozen
                    if target_lr_for_group is not None:
                        if p_group['lr'] != target_lr_for_group:
                            logger.info(f"  Updating {group_name} group optimizer LR to {target_lr_for_group:.2e}")
                            p_group['lr'] = target_lr_for_group
                        if hasattr(scheduler, 'base_lrs') and scheduler.base_lrs and p_group_idx < len(scheduler.base_lrs):
                            if scheduler.base_lrs[p_group_idx] != target_lr_for_group:
                                logger.info(f"    Updating Scheduler's base_lr for group '{group_name}' to {target_lr_for_group:.2e}")
                                scheduler.base_lrs[p_group_idx] = target_lr_for_group
                        # ... (Logic for SequentialLR if used)
                logger.info("  Effective LRs for scheduler after unfreezing update:")
                for i, pg in enumerate(optimizer.param_groups): logger.info(f"    Opt Group {i} ('{pg.get('name')}'): LR {pg.get('lr'):.3e}, Scheduler base {scheduler.base_lrs[i]:.3e}" if hasattr(scheduler, 'base_lrs') else "")

            avg_train_loss, nan_threshold_exceeded_flag = finetuner_obj.train_one_epoch(train_loader, epoch_1_based, cfg["epochs"])
            if nan_threshold_exceeded_flag:
                logger.error(f"NaN threshold exceeded during epoch {epoch_1_based}. Stopping training.")
                break
            
            current_val_metric_for_decision = None
            if epoch_1_based % cfg.get("evaluate_every_n_epochs", 1) == 0 or epoch_1_based == cfg["epochs"]:
                avg_val_loss, val_metrics_dict = finetuner_obj.validate_one_epoch(val_loader, class_names=class_names)
                if metric_to_watch == "val_loss": current_val_metric_for_decision = avg_val_loss
                else: current_val_metric_for_decision = val_metrics_dict.get(metric_to_watch)

                if isinstance(finetuner_obj.scheduler, ReduceLROnPlateau) and current_val_metric_for_decision is not None: 
                    finetuner_obj.scheduler.step(current_val_metric_for_decision)
                elif finetuner_obj.scheduler and not finetuner_obj.lr_scheduler_on_batch: 
                    finetuner_obj.scheduler.step()

                if current_val_metric_for_decision is not None:
                    is_better = (current_val_metric_for_decision > best_val_metric_val) if metric_to_watch != "val_loss" else \
                                (current_val_metric_for_decision < best_val_metric_val)
                    if is_better:
                        best_val_metric_val = current_val_metric_for_decision
                        if cfg.get("best_model_filename"): 
                            finetuner_obj.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, cfg["best_model_filename"]))
                        logger.info(f"E{epoch_1_based}: New best! Val {metric_to_watch}: {best_val_metric_val:.4f}")
                        patience_count = 0
                    else: 
                        patience_count += 1; logger.info(f"E{epoch_1_based}: Val {metric_to_watch} ({current_val_metric_for_decision:.4f}) not better. Patience: {patience_count}/{patience_val}")
                else: logger.warning(f"E{epoch_1_based}: Metric '{metric_to_watch}' not found. Cannot make early stopping decision.")
            
            if cfg.get("save_checkpoint_every_n_epochs", 0) > 0 and \
               epoch_1_based % cfg["save_checkpoint_every_n_epochs"] == 0:
                periodic_ckpt_name = f"checkpoint_epoch_{epoch_1_based}.pth"
                finetuner_obj.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, periodic_ckpt_name))


            if patience_count >= patience_val: 
                logger.info(f"Early stopping at E{epoch_1_based}. Patience ({patience_val}) exceeded for '{metric_to_watch}'.")
                break
            last_completed_ep_main = epoch_1_based
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch_1_based} completed in {epoch_duration:.2f}s.")
            if device == 'cuda': logger.debug(f"CUDA Mem E{epoch_1_based} End: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB, Reserved {torch.cuda.memory_reserved(0)/1024**2:.1f}MB")

    except KeyboardInterrupt: logger.warning(f"Fine-tuning interrupted. Last completed epoch: {last_completed_ep_main}.")
    except Exception as e_fatal: logger.critical(f"Critical error at epoch {last_completed_ep_main + 1}: {e_fatal}", exc_info=True); sys.exit(1)
    finally:
        logger.info(f"Fine-tuning ended. Last completed epoch: {last_completed_ep_main}.")
        if nan_threshold_exceeded_flag: logger.error("Training stopped prematurely due to excessive NaN losses.")
        if 'finetuner_obj' in locals() and finetuner_obj is not None and cfg.get("final_model_filename"):
            finetuner_obj.save_model_checkpoint(os.path.join(abs_ckpt_save_dir, cfg["final_model_filename"]))
        logger.info(f"Fine-tuning summary: Best validation '{metric_to_watch}': {best_val_metric_val:.4f}.")

if __name__ == "__main__":
    try: main_execution_logic()
    except SystemExit as se:
        final_logger_exit = logger if logger and (logger.hasHandlers() or (logger.parent and logger.parent.hasHandlers())) else logging.getLogger("main_fallback_exit")
        if not (final_logger_exit.hasHandlers() or (final_logger_exit.parent and final_logger_exit.parent.hasHandlers())): logging.basicConfig(level=logging.INFO)
        if se.code != 0 and se.code is not None: final_logger_exit.info(f"Application exited with status code {se.code}.")
        elif se.code is None : final_logger_exit.info(f"Application exited normally.")
        sys.exit(se.code if se.code is not None else 0)
    except Exception as e_unhandled:
        final_logger_unhandled = logger if logger and (logger.hasHandlers() or (logger.parent and logger.parent.hasHandlers())) else logging.getLogger("main_fallback_unhandled")
        if not (final_logger_unhandled.hasHandlers() or (final_logger_unhandled.parent and final_logger_unhandled.parent.hasHandlers())): logging.basicConfig(level=logging.ERROR)
        final_logger_unhandled.critical(f"Unhandled CRITICAL exception in __main__: {e_unhandled}", exc_info=True)
        sys.exit(1)