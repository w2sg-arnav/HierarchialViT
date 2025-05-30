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
    from .utils.augmentations import create_augmentation
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
        if num_warmup_steps > 0 and current_step < num_warmup_steps: # only apply warmup if num_warmup_steps > 0
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # If no warmup, or warmup is done
        eff_training_steps = num_training_steps - num_warmup_steps
        if eff_training_steps <=0 : # If total steps is less than or equal to warmup, or no training steps
             # This case means we either only warmup, or total steps are very few.
             # If only warmup, LR stays at peak. If eff_training_steps is 0 after warmup, it should also be peak.
            return 1.0 # Maintain peak LR from warmup, or initial LR if no warmup.

        progress_step = current_step - num_warmup_steps if num_warmup_steps > 0 else current_step
        progress = float(progress_step) / float(max(1, eff_training_steps))
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
    elif checkpoint_pos_embed.ndim == 1: 
        _local_logger_pe.warning("PE Interp: Checkpoint PE is 1D. This is unusual. Assuming it's (N*C) and trying to infer N if C matches.")
        if model_pos_embed_param.ndim == 3 and model_pos_embed_param.shape[0] == 1: 
            C_model = model_pos_embed_param.shape[2]
            if checkpoint_pos_embed.shape[0] % C_model == 0:
                N_ckpt_inferred = checkpoint_pos_embed.shape[0] // C_model
                checkpoint_pos_embed = checkpoint_pos_embed.view(1, N_ckpt_inferred, C_model)
                _local_logger_pe.info(f"PE Interp: Reshaped 1D PE to {checkpoint_pos_embed.shape}")
            else:
                 _local_logger_pe.error(f"PE Interp: Cannot reshape 1D PE. No interp."); return model_pos_embed_param.data, False
        else:
            _local_logger_pe.error(f"PE Interp: Cannot handle 1D PE with model PE shape {model_pos_embed_param.shape}. No interp."); return model_pos_embed_param.data, False


    if checkpoint_pos_embed.ndim == 2: 
         num_tokens_ckpt, C_ckpt = checkpoint_pos_embed.shape[0], checkpoint_pos_embed.shape[1]
         checkpoint_pos_embed = checkpoint_pos_embed.unsqueeze(0) 
    elif checkpoint_pos_embed.ndim == 3: 
        if checkpoint_pos_embed.shape[0] != 1:
             _local_logger_pe.warning(f"PE Interp: Checkpoint PE has batch size {checkpoint_pos_embed.shape[0]} > 1. Using first element.")
             checkpoint_pos_embed = checkpoint_pos_embed[0].unsqueeze(0)
        num_tokens_ckpt, C_ckpt = checkpoint_pos_embed.shape[1], checkpoint_pos_embed.shape[2]
    else:
        _local_logger_pe.error(f"PE Interp: Unexpected checkpoint_pos_embed ndim: {checkpoint_pos_embed.ndim}. No interp."); return model_pos_embed_param.data, False

    if model_pos_embed_param.ndim == 3 and model_pos_embed_param.shape[0] == 1: 
        num_tokens_model, C_model = model_pos_embed_param.shape[1], model_pos_embed_param.shape[2]
    else: 
        _local_logger_pe.warning(f"PE Interp: Model PE param has unusual shape {model_pos_embed_param.shape}. Assuming (N,C) if 2D.")
        if model_pos_embed_param.ndim == 2:
            num_tokens_model, C_model = model_pos_embed_param.shape[0], model_pos_embed_param.shape[1]
        else:
            _local_logger_pe.error(f"PE Interp: Cannot handle model PE shape {model_pos_embed_param.shape}. No interp."); return model_pos_embed_param.data, False


    if C_ckpt != C_model:
        _local_logger_pe.error(f"PE C-dim mismatch (ckpt:{C_ckpt} vs model:{C_model}). No interpolation."); return model_pos_embed_param.data, False
    
    if num_tokens_ckpt == num_tokens_model:
        _local_logger_pe.debug(f"PE N-dim and C-dim match ({num_tokens_ckpt} tokens, C={C_ckpt}). Using ckpt PE directly.");
        return checkpoint_pos_embed.view_as(model_pos_embed_param.data), True

    H0_patches = ssl_img_size_for_pos_embed[0] // hvt_patch_size
    W0_patches = ssl_img_size_for_pos_embed[1] // hvt_patch_size
    if H0_patches * W0_patches != num_tokens_ckpt:
        _local_logger_pe.warning(f"PE Interp: num_tokens_ckpt ({num_tokens_ckpt}) does not match SSL grid ({H0_patches}x{W0_patches} from {ssl_img_size_for_pos_embed} / {hvt_patch_size}). Trying sqrt inference.")
        sqrt_num_tokens_ckpt = math.isqrt(num_tokens_ckpt)
        if sqrt_num_tokens_ckpt * sqrt_num_tokens_ckpt == num_tokens_ckpt:
            H0_patches = W0_patches = sqrt_num_tokens_ckpt
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
        
        if pe_interpolated.shape == model_pos_embed_param.data.shape: 
            _local_logger_pe.info("PE Interpolation successful.")
            return pe_interpolated, True
        else:
            _local_logger_pe.error(f"PE Interp: Final shape mismatch after interpolation {pe_interpolated.shape} vs {model_pos_embed_param.data.shape}. No interp."); return model_pos_embed_param.data, False
    except Exception as e:
        _local_logger_pe.error(f"PE Interp Exception: {e}", exc_info=True); return model_pos_embed_param.data, False

def load_and_prepare_hvt_model(hvt_model_instance: DiseaseAwareHVT, cfg: Dict[str, Any], device: str) -> DiseaseAwareHVT:
    _local_logger_load = logging.getLogger(f"{__name__}.load_and_prepare_hvt_model")
    hvt_model_instance.cpu()

    if not cfg.get("load_pretrained_backbone", False):
        _local_logger_load.info("'load_pretrained_backbone' is False. SSL Backbone will not be loaded by this function.")
        if hasattr(hvt_model_instance, 'classifier_head') and isinstance(hvt_model_instance.classifier_head, nn.Linear):
            _local_logger_load.info(f"Classifier head already exists. Assuming it's correctly initialized for {cfg['num_classes']} classes.")
        return hvt_model_instance.to(device)

    ssl_ckpt_path_from_cfg = cfg.get("pretrained_checkpoint_path") 
    if not ssl_ckpt_path_from_cfg:
        _local_logger_load.info("No SSL 'pretrained_checkpoint_path' provided. SSL Backbone will not be loaded.")
        return hvt_model_instance.to(device)

    resolved_ssl_ckpt_path = ssl_ckpt_path_from_cfg
    project_root_for_path = cfg.get("PROJECT_ROOT_PATH")
    if project_root_for_path is None and 'PACKAGE_ROOT_PATH' in cfg:
        project_root_for_path = os.path.dirname(cfg['PACKAGE_ROOT_PATH'])

    if not os.path.isabs(resolved_ssl_ckpt_path) and project_root_for_path:
            resolved_ssl_ckpt_path = os.path.join(project_root_for_path, ssl_ckpt_path_from_cfg)

    if not os.path.exists(resolved_ssl_ckpt_path):
        _local_logger_load.warning(f"Specified SSL checkpoint path '{resolved_ssl_ckpt_path}' not found. SSL Backbone will not be loaded.")
        return hvt_model_instance.to(device)

    _local_logger_load.info(f"Loading SSL backbone weights from: {resolved_ssl_ckpt_path}")
    checkpoint = torch.load(resolved_ssl_ckpt_path, map_location='cpu')
    ssl_backbone_sd = checkpoint.get('model_backbone_state_dict', checkpoint.get('model_state_dict', checkpoint.get('state_dict')))


    if not ssl_backbone_sd:
        _local_logger_load.error(f"Relevant state_dict not found in SSL checkpoint {resolved_ssl_ckpt_path}. SSL Backbone will not be loaded.")
        return hvt_model_instance.to(device)

    current_model_sd = hvt_model_instance.state_dict()
    new_sd_for_backbone = OrderedDict()
    
    ssl_run_cfg_snapshot = checkpoint.get('run_config_snapshot', {})
    ssl_pretrain_img_size_config = cfg.get("ssl_pretrain_img_size_fallback", (224,224)) 
    if 'pretrain_img_size' in ssl_run_cfg_snapshot: 
        ssl_pretrain_img_size_config = ssl_run_cfg_snapshot['pretrain_img_size']
    ssl_img_size_tuple = tuple(ssl_pretrain_img_size_config)


    if tuple(ssl_img_size_tuple) == tuple(cfg["img_size"]):
        _local_logger_load.info(f"SSL and Finetune img_size match: {ssl_img_size_tuple}.")
    else:
        _local_logger_load.info(f"SSL img_size: {ssl_img_size_tuple}, Finetune img_size: {cfg['img_size']}. PE interpolation may occur for SSL backbone.")

    loaded_count, pe_interp_successful_count, head_skip_count = 0,0,0
    
    simclr_prefix = "backbone."
    mae_prefix = "encoder." 
    
    filtered_ssl_sd = OrderedDict()
    for k_ckpt, v_ckpt in ssl_backbone_sd.items():
        k_to_check = k_ckpt
        if k_ckpt.startswith(simclr_prefix):
            k_to_check = k_ckpt[len(simclr_prefix):]
        elif k_ckpt.startswith(mae_prefix): 
             k_to_check = k_ckpt[len(mae_prefix):]
        
        if k_to_check in current_model_sd:
            filtered_ssl_sd[k_to_check] = v_ckpt
        elif k_ckpt in current_model_sd: 
            filtered_ssl_sd[k_ckpt] = v_ckpt


    for k_ckpt_mapped, v_ckpt in filtered_ssl_sd.items():
        if k_ckpt_mapped not in current_model_sd: 
            _local_logger_load.debug(f"Skipping {k_ckpt_mapped} from SSL checkpoint as it's not in current model structure.")
            continue
        if k_ckpt_mapped.startswith("classifier_head.") or k_ckpt_mapped.startswith("head."): 
            head_skip_count += 1; continue

        target_pe_p = getattr(hvt_model_instance, k_ckpt_mapped, None)
        is_pe = (k_ckpt_mapped in ["rgb_pos_embed", "spectral_pos_embed"]) and \
                target_pe_p is not None and isinstance(target_pe_p, nn.Parameter)

        if is_pe:
            interp_pe_val, interp_success = _interpolate_positional_embedding(
                v_ckpt, target_pe_p, 
                cfg['hvt_params_for_model_init']['patch_size'],
                tuple(cfg["img_size"]), 
                ssl_img_size_tuple      
            )
            if interp_success:
                new_sd_for_backbone[k_ckpt_mapped] = interp_pe_val
                pe_interp_successful_count +=1
            else:
                _local_logger_load.warning(f"PE {k_ckpt_mapped} interpolation failed. Model's original PE will be used.")
        elif v_ckpt.shape == current_model_sd[k_ckpt_mapped].shape:
            new_sd_for_backbone[k_ckpt_mapped] = v_ckpt; loaded_count +=1
        else: 
            _local_logger_load.warning(f"Shape mismatch for {k_ckpt_mapped}: ckpt {v_ckpt.shape} vs model {current_model_sd[k_ckpt_mapped].shape}. Skipping.")

    if new_sd_for_backbone:
        msg = hvt_model_instance.load_state_dict(new_sd_for_backbone, strict=False)
        _local_logger_load.info(f"SSL Backbone weights loaded: {loaded_count} direct, {pe_interp_successful_count} PE interpolated, {head_skip_count} head layers skipped from SSL.")
        if msg.missing_keys: _local_logger_load.warning(f"Missing keys when loading SSL backbone: {msg.missing_keys}")
        if msg.unexpected_keys: _local_logger_load.warning(f"Unexpected keys in SSL backbone load: {msg.unexpected_keys}")
    else:
        _local_logger_load.warning("No weights from SSL backbone were loaded into the model (new_sd_for_backbone is empty).")

    if hasattr(hvt_model_instance, 'classifier_head') and isinstance(hvt_model_instance.classifier_head, nn.Linear):
        current_in_features = hvt_model_instance.classifier_head.in_features
        expected_in_features = current_in_features 
        
        if hasattr(hvt_model_instance, 'hvt_rgb') and hasattr(hvt_model_instance.hvt_rgb, 'embed_dim') and isinstance(hvt_model_instance.hvt_rgb.embed_dim, list) and hvt_model_instance.hvt_rgb.embed_dim:
             expected_in_features = hvt_model_instance.hvt_rgb.embed_dim[-1] 
        elif 'embed_dim_rgb' in cfg['hvt_params_for_model_init']:
            # This part is tricky and highly dependent on HVT architecture.
            # The 'in_features' for the final nn.Linear head is what create_disease_aware_hvt determined.
            # We should trust that unless there's a strong reason SSL model structure dictates otherwise.
            # For now, assume create_disease_aware_hvt sets it correctly.
            expected_in_features = hvt_model_instance.classifier_head.in_features


        if current_in_features != expected_in_features: # This condition might be redundant if expected_in_features is set as above
            _local_logger_load.warning(f"Classifier head in_features ({current_in_features}) differs from expected backbone output ({expected_in_features}). Re-initializing head.")
        
        hvt_model_instance.classifier_head = nn.Linear(expected_in_features, cfg["num_classes"])
        _local_logger_load.info(f"Re-initialized HVT classifier_head for {cfg['num_classes']} classes (in_features={expected_in_features}).")
    else: 
        _local_logger_load.error("Could not find or re-initialize classifier_head. VERIFY HVT MODEL STRUCTURE for finetuning.")
        
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
    setup_logging(log_file_name=final_log_filename, log_dir=abs_log_dir, log_level=logging.INFO, run_timestamp=run_ts) 
    logger = logging.getLogger(__name__)

    logger.info(f"======== Starting Phase 4: HVT Fine-tuning (Run ID: {run_ts}) ========")
    logger.info(f"Full run configuration loaded. Log dir: {abs_log_dir}")
    if cfg.get("debug_nan_detection", False) or cfg.get("monitor_gradients", False):
        logger.warning("Detailed debugging (NaNs, Gradients) is ON. Log level might be verbose.")
        logging.getLogger().setLevel(logging.DEBUG) 

    logger.debug(f"Full config: {cfg}") 
    set_global_seed(cfg["seed"])
    device = cfg["device"]
    if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'; cfg['device'] = 'cpu'; logger.warning("CUDA unavailable. Using CPU.")
    logger.info(f"Using device: {device}. GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    if cfg.get("cudnn_benchmark", True) and device == 'cuda': torch.backends.cudnn.benchmark = True
    if cfg.get("matmul_precision") and hasattr(torch,'set_float32_matmul_precision'):
        try: torch.set_float32_matmul_precision(cfg["matmul_precision"]); logger.info(f"Set matmul_precision to {cfg['matmul_precision']}")
        except Exception as e: logger.warning(f"Failed to set matmul_precision: {e}")

    dataset_args = {
        "root_dir": cfg["data_root"], "img_size": tuple(cfg["img_size"]),
        "train_split_ratio": cfg["train_split_ratio"], "normalize_for_model": cfg["normalize_data"],
        "original_dataset_name": cfg["original_dataset_name"],
        "augmented_dataset_name": cfg.get("augmented_dataset_name", None), "random_seed": cfg["seed"]
    }
    train_dataset = SARCLD2024Dataset(**dataset_args, split="train"); val_dataset = SARCLD2024Dataset(**dataset_args, split="val")
    class_names = train_dataset.get_class_names()
    logger.info(f"Datasets loaded. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}. Classes: {class_names}")

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
                    else: per_class_weight[i] = 1e-3 
                logger.info(f"Using WeightedRandomSampler with 'sqrt_inv_count' mode. Per-class weights: {per_class_weight.numpy().round(3)}")
            elif sampler_mode == "inv_freq": 
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
    
    resuming_finetune_succeeded_model_load = False
    finetune_resume_path = cfg.get("resume_from_checkpoint")
    checkpoint_format_is_new = False # Flag to indicate if loaded checkpoint is new format

    if finetune_resume_path and os.path.exists(finetune_resume_path):
        logger.info(f"Attempting to load finetuning checkpoint from: {finetune_resume_path}")
        try:
            checkpoint_content = torch.load(finetune_resume_path, map_location='cpu')
            if isinstance(checkpoint_content, dict) and 'model_state_dict' in checkpoint_content:
                logger.info("Checkpoint is a dictionary with 'model_state_dict' (new format). Loading model weights.")
                missing_keys, unexpected_keys = model_hvt_instance.load_state_dict(checkpoint_content['model_state_dict'], strict=False)
                if missing_keys: logger.warning(f"Resuming model (new format): Missing keys: {missing_keys}")
                if unexpected_keys: logger.warning(f"Resuming model (new format): Unexpected keys: {unexpected_keys}")
                resuming_finetune_succeeded_model_load = True
                checkpoint_format_is_new = True 
                cfg['load_pretrained_backbone'] = False 
                logger.info("Model state loaded from new format finetuning checkpoint. SSL backbone loading will be skipped.")
            elif isinstance(checkpoint_content, (OrderedDict, dict)): 
                logger.info("Checkpoint appears to be a model state_dict (old format). Loading model weights.")
                missing_keys, unexpected_keys = model_hvt_instance.load_state_dict(checkpoint_content, strict=False)
                if missing_keys: logger.warning(f"Resuming model (old format): Missing keys: {missing_keys}")
                if unexpected_keys: logger.warning(f"Resuming model (old format): Unexpected keys: {unexpected_keys}")
                resuming_finetune_succeeded_model_load = True
                checkpoint_format_is_new = False
                cfg['load_pretrained_backbone'] = False
                logger.info("Model state loaded from old format finetuning checkpoint. SSL backbone loading will be skipped.")
            else:
                logger.error(f"Finetuning checkpoint {finetune_resume_path} is not a recognized format.")

        except Exception as e_load_ckpt:
            logger.error(f"Error loading finetuning checkpoint {finetune_resume_path}: {e_load_ckpt}. Proceeding without resuming model weights from it.", exc_info=True)
            resuming_finetune_succeeded_model_load = False

    if resuming_finetune_succeeded_model_load:
        model = model_hvt_instance.to(device)
    else: 
        if finetune_resume_path and not os.path.exists(finetune_resume_path) : logger.warning(f"Specified resume_from_checkpoint path not found: {finetune_resume_path}")
        logger.info("Proceeding with SSL backbone loading (if configured) or random init.")
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
    
    # Use the LRs from config as the new target base LRs
    lr_head_target = cfg['lr_head_unfrozen_phase']
    lr_backbone_target = cfg['lr_backbone_unfrozen_phase']
    
    param_groups = []
    if all_head_params: param_groups.append({'params': all_head_params, 'name': 'head', 'lr': lr_head_target})
    else: logger.warning("No parameters found for 'head' group.")
    if all_backbone_params: param_groups.append({'params': all_backbone_params, 'name': 'backbone', 'lr': lr_backbone_target})
    else: logger.warning("No parameters found for 'backbone' group.")
    
    optim_name = cfg.get("optimizer", "AdamW").lower()
    default_opt_constructor_lr = max(lr_head_target, lr_backbone_target) # Use new target LRs
    opt_kwargs = cfg.get("optimizer_params", {}).copy(); opt_kwargs['weight_decay'] = cfg.get("weight_decay", 0.05)
    opt_kwargs['lr'] = default_opt_constructor_lr 

    if not param_groups: 
        logger.warning("No param groups defined (head/backbone not found). Using all model parameters for optimizer.")
        param_groups = model.parameters() # This would not have named groups or per-group LRs

    if optim_name == "adamw": optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    elif optim_name == "sgd": opt_kwargs.pop('eps', None); optimizer = torch.optim.SGD(param_groups, **opt_kwargs)
    else: logger.warning(f"Optimizer '{optim_name}' not handled. Defaulting to AdamW."); optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)
    
    logger.info(f"Optimizer: {optimizer.__class__.__name__} created. Configured LRs (pre-resume):")
    for pg_idx, pg in enumerate(optimizer.param_groups):
         if isinstance(pg, dict): 
            logger.info(f"  Group '{pg.get('name', f'UnnamedGroup{pg_idx}')}': LR {pg.get('lr'):.2e}, WD {pg.get('weight_decay'):.2e}")

    scheduler, lr_scheduler_on_batch_flag = None, False
    sched_cfg_name = cfg.get("scheduler", "None").lower(); warmup_epochs_cfg = cfg.get("warmup_epochs", 0)
    total_epochs_for_sched = cfg["epochs"]; eff_accum_steps_sched = max(1, cfg.get("accumulation_steps", 1))
    steps_per_epoch_for_sched = len(train_loader) // eff_accum_steps_sched
    if steps_per_epoch_for_sched == 0 and len(train_loader.dataset) > 0 : steps_per_epoch_for_sched = 1 

    # last_epoch_for_scheduler will be updated if scheduler state is loaded
    # If scheduler state is NOT loaded (e.g. old checkpoint or no scheduler in ckpt),
    # it starts from -1, and the warmup_epochs_cfg from current config applies.
    last_epoch_for_scheduler = -1 

    if sched_cfg_name != "none" and steps_per_epoch_for_sched > 0 :
        if sched_cfg_name == "warmupcosine":
            total_sched_steps = total_epochs_for_sched * steps_per_epoch_for_sched # Based on total epochs in *this* run
            warmup_sched_steps = warmup_epochs_cfg * steps_per_epoch_for_sched
            
            # Adjust total_sched_steps if resuming to reflect remaining steps,
            # but this is complex if scheduler state is loaded.
            # Simpler: LambdaLR's last_epoch handles continuation if loaded.
            # If not loaded, it uses current config's warmup_epochs_cfg.
            scheduler = get_cosine_schedule_with_warmup_step(optimizer, warmup_sched_steps, total_sched_steps, num_cycles=0.5, last_epoch=last_epoch_for_scheduler) 
            lr_scheduler_on_batch_flag = True
            logger.info(f"Scheduler: WarmupCosine (per-step). Configured WU Steps: {warmup_sched_steps}, Total Steps for this run: {total_sched_steps}")
        elif sched_cfg_name == "cosineannealinglr":
            # T_max is epochs for this run. last_epoch handles continuation.
            cosine_t_max_epochs = total_epochs_for_sched - warmup_epochs_cfg if warmup_epochs_cfg > 0 else total_epochs_for_sched
            if cosine_t_max_epochs <=0: cosine_t_max_epochs = 1 
            main_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max_epochs, eta_min=cfg.get("eta_min_lr", 1e-7), last_epoch=last_epoch_for_scheduler)
            if warmup_epochs_cfg > 0:
                # For SequentialLR, milestones are relative to the start of *this usage* of SequentialLR
                # If resuming, the individual schedulers' last_epoch will be set.
                warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs_cfg) # last_epoch not directly set here for LinearLR component
                scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs_cfg], last_epoch=last_epoch_for_scheduler) 
                logger.info(f"Scheduler: SequentialLR(Warmup({warmup_epochs_cfg}eps) -> CosineAnnealingLR(T_max={cosine_t_max_epochs})).")
            else: 
                scheduler = main_scheduler
                logger.info(f"Scheduler: CosineAnnealingLR (per-epoch). T_max={cosine_t_max_epochs}")
            lr_scheduler_on_batch_flag = False
        # ... other schedulers if any ...
    
    scaler_obj = GradScaler(enabled=(cfg.get("amp_enabled", True) and device == 'cuda'))
    
    finetuner_obj = Finetuner(
        model=model, optimizer=optimizer, criterion=criterion, device=device,
        scaler=scaler_obj, scheduler=scheduler, lr_scheduler_on_batch=lr_scheduler_on_batch_flag,
        # ... other finetuner args ...
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

    start_epoch = 1
    metric_to_watch = cfg.get("metric_to_monitor_early_stopping", "f1_macro")
    best_val_metric_val = 0.0 if metric_to_watch != "val_loss" else float('inf')
    
    if resuming_finetune_succeeded_model_load and checkpoint_format_is_new:
        logger.info(f"Finetune checkpoint {finetune_resume_path} is new format. Loading full trainer state (optimizer, scheduler, epoch...).")
        start_epoch, best_val_metric_val = finetuner_obj.load_checkpoint(finetune_resume_path, metric_to_watch)
        
        # --- AFTER loading checkpoint, explicitly set LRs in optimizer groups and scheduler base_lrs to NEW config values ---
        logger.info(f"Applying NEW configured LRs after resuming checkpoint: Head LR={lr_head_target:.2e}, Backbone LR={lr_backbone_target:.2e}")
        for p_group in optimizer.param_groups:
            group_name = p_group.get('name')
            new_lr_for_group = None
            if group_name == 'head': new_lr_for_group = lr_head_target
            elif group_name == 'backbone': new_lr_for_group = lr_backbone_target
            
            if new_lr_for_group is not None:
                logger.info(f"  Updating optimizer group '{group_name}' LR from {p_group['lr']:.3e} to {new_lr_for_group:.3e}")
                p_group['lr'] = new_lr_for_group
        
        if scheduler:
            # For LambdaLR (like WarmupCosine), changing optimizer LR directly affects it.
            # For other schedulers (CosineAnnealingLR, SequentialLR), we need to update their base_lrs.
            if hasattr(scheduler, 'base_lrs'):
                current_base_lrs = scheduler.base_lrs
                new_base_lrs = []
                for i, p_group in enumerate(optimizer.param_groups): # Iterate through optimizer's groups to map to scheduler's base_lrs
                    group_name = p_group.get('name')
                    new_base_lr_for_group = current_base_lrs[i] # Default to existing
                    if group_name == 'head': new_base_lr_for_group = lr_head_target
                    elif group_name == 'backbone': new_base_lr_for_group = lr_backbone_target
                    new_base_lrs.append(new_base_lr_for_group)

                if tuple(current_base_lrs) != tuple(new_base_lrs): # Check if update is needed
                    logger.info(f"  Updating scheduler base_lrs from {current_base_lrs} to {new_base_lrs}")
                    scheduler.base_lrs = new_base_lrs
                    # For SequentialLR, need to update base_lrs of internal schedulers too
                    if isinstance(scheduler, SequentialLR):
                        for internal_sched_idx, internal_sched in enumerate(scheduler.schedulers):
                            if hasattr(internal_sched, 'base_lrs'):
                                # Map optimizer groups to internal scheduler's param groups (can be complex if optimizer had different groups than scheduler expects)
                                # Assuming 1-to-1 mapping of base_lrs for simplicity here.
                                internal_new_base_lrs = []
                                for i, p_group_opt in enumerate(optimizer.param_groups):
                                    group_name_opt = p_group_opt.get('name')
                                    new_base_lr_val = internal_sched.base_lrs[i] # Default
                                    if group_name_opt == 'head': new_base_lr_val = lr_head_target
                                    elif group_name_opt == 'backbone': new_base_lr_val = lr_backbone_target
                                    internal_new_base_lrs.append(new_base_lr_val)
                                
                                if tuple(internal_sched.base_lrs) != tuple(internal_new_base_lrs):
                                    logger.info(f"    Updating internal scheduler {internal_sched_idx} base_lrs to {internal_new_base_lrs}")
                                    internal_sched.base_lrs = internal_new_base_lrs
            # If scheduler is LambdaLR, it uses optimizer's LR directly, so above optimizer update is enough.
            # Call scheduler.step() with epoch-1 to "fast-forward" it if epoch-based and start_epoch > 1
            # This is tricky as finetuner_obj.load_checkpoint already sets scheduler's last_epoch.
            # The next call to scheduler.step() in the loop should use the correct (resumed) step/epoch.
            logger.info(f"Scheduler state loaded. Next step will use its internal state (last_epoch={scheduler.last_epoch if hasattr(scheduler,'last_epoch') else 'N/A'}).")

    elif resuming_finetune_succeeded_model_load and not checkpoint_format_is_new:
        logger.warning(f"Finetune checkpoint {finetune_resume_path} was old format (model weights only). Optimizer, scheduler, and epoch not resumed. Starting fresh for these aspects with NEWLY configured LRs and scheduler settings.")
        # LRs in optimizer are already set to new config values. Scheduler will start fresh.

    elif cfg.get("resume_from_checkpoint") and not os.path.exists(cfg.get("resume_from_checkpoint")):
        logger.error(f"Resume checkpoint specified ({cfg.get('resume_from_checkpoint')}) but not found. Training from scratch or SSL backbone.")


    freeze_backbone_until_epoch_from_cfg = cfg.get("freeze_backbone_epochs", 0) # Should be 0 for this run
    model_to_operate_on = model._orig_mod if hasattr(model, '_orig_mod') and isinstance(model._orig_mod, nn.Module) else model

    # Ensure backbone is unfrozen if start_epoch is past freeze_backbone_epochs (which is 0 here)
    if start_epoch > freeze_backbone_until_epoch_from_cfg :
        logger.info(f"Ensuring backbone is unfrozen as start_epoch ({start_epoch}) > freeze_backbone_epochs_cfg ({freeze_backbone_until_epoch_from_cfg}).")
        for name, param in model_to_operate_on.named_parameters():
            param.requires_grad = True # Unfreeze everything
        logger.info("All model parameters set to requires_grad=True.")
    
    patience_val = cfg.get("early_stopping_patience", float('inf'))
    patience_count = 0; last_completed_ep_main = start_epoch - 1
    
    abs_ckpt_save_dir = os.path.join(abs_log_dir, cfg.get("checkpoint_save_dir_name", "checkpoints"))
    os.makedirs(abs_ckpt_save_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved in: {abs_ckpt_save_dir}")
    
    logger.info(f"Starting fine-tuning from epoch {start_epoch} up to {cfg['epochs']} epochs. Monitor: '{metric_to_watch}'. Initial Best Metric Value (resumed or default): {best_val_metric_val:.4f}")
    logger.info("Optimizer LRs at start of training loop:")
    for i, pg_check in enumerate(optimizer.param_groups):
        if isinstance(pg_check, dict): logger.info(f"  Opt Group {i} ('{pg_check.get('name')}'): LR {pg_check['lr']:.3e}")

    nan_threshold_exceeded_flag = False
    try:
        for epoch_1_based in range(start_epoch, cfg["epochs"] + 1):
            last_completed_ep_main = epoch_1_based -1
            epoch_start_time = time.time()

            # Backbone freezing logic is simplified as freeze_backbone_epochs is 0
            # All params should be trainable from the start of this run.
            # The check above (start_epoch > freeze_backbone_until_epoch_from_cfg) handles this.

            avg_train_loss, nan_threshold_exceeded_flag = finetuner_obj.train_one_epoch(train_loader, epoch_1_based, cfg["epochs"])
            if nan_threshold_exceeded_flag:
                logger.error(f"NaN threshold exceeded during epoch {epoch_1_based}. Stopping training.")
                break
            
            current_val_metric_for_decision = None
            if epoch_1_based % cfg.get("evaluate_every_n_epochs", 1) == 0 or epoch_1_based == cfg["epochs"]:
                avg_val_loss, val_metrics_dict = finetuner_obj.validate_one_epoch(val_loader, class_names=class_names)
                if metric_to_watch == "val_loss": current_val_metric_for_decision = avg_val_loss
                else: current_val_metric_for_decision = val_metrics_dict.get(metric_to_watch)

                if finetuner_obj.scheduler:
                    if isinstance(finetuner_obj.scheduler, ReduceLROnPlateau) and current_val_metric_for_decision is not None: 
                        finetuner_obj.scheduler.step(current_val_metric_for_decision)
                    elif not finetuner_obj.lr_scheduler_on_batch and not isinstance(finetuner_obj.scheduler, ReduceLROnPlateau): 
                        # For epoch-based schedulers like CosineAnnealingLR or SequentialLR components
                        finetuner_obj.scheduler.step()


                if current_val_metric_for_decision is not None:
                    is_better = (current_val_metric_for_decision > best_val_metric_val) if metric_to_watch != "val_loss" else \
                                (current_val_metric_for_decision < best_val_metric_val)
                    if is_better and not math.isnan(current_val_metric_for_decision): 
                        best_val_metric_val = current_val_metric_for_decision
                        if cfg.get("best_model_filename"): 
                            finetuner_obj.save_checkpoint(
                                os.path.join(abs_ckpt_save_dir, cfg["best_model_filename"]),
                                epoch_1_based, best_val_metric_val, metric_to_watch
                            )
                        logger.info(f"E{epoch_1_based}: New best! Val {metric_to_watch}: {best_val_metric_val:.4f}")
                        patience_count = 0
                    elif not math.isnan(current_val_metric_for_decision):
                        patience_count += 1; logger.info(f"E{epoch_1_based}: Val {metric_to_watch} ({current_val_metric_for_decision:.4f}) not better. Patience: {patience_count}/{patience_val}")
                    else: 
                        logger.warning(f"E{epoch_1_based}: Metric '{metric_to_watch}' is NaN. Cannot make early stopping decision or save best model based on this.")
                        patience_count +=1 
                else: 
                    logger.warning(f"E{epoch_1_based}: Metric '{metric_to_watch}' not found or is None. Cannot make early stopping decision.")
                    patience_count += 1
            
            if cfg.get("save_checkpoint_every_n_epochs", 0) > 0 and \
               epoch_1_based % cfg["save_checkpoint_every_n_epochs"] == 0:
                periodic_ckpt_name = f"checkpoint_epoch_{epoch_1_based}.pth"
                finetuner_obj.save_checkpoint(
                    os.path.join(abs_ckpt_save_dir, periodic_ckpt_name),
                    epoch_1_based, best_val_metric_val, metric_to_watch # Save with the current overall best metric
                )

            if patience_count >= patience_val: 
                logger.info(f"Early stopping at E{epoch_1_based}. Patience ({patience_val}) exceeded for '{metric_to_watch}'.")
                break
            last_completed_ep_main = epoch_1_based
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch_1_based} completed in {epoch_duration:.2f}s.")
            if device == 'cuda': logger.debug(f"CUDA Mem E{epoch_1_based} End: Alloc {torch.cuda.memory_allocated(0)/1024**2:.1f}MB, Reserved {torch.cuda.memory_reserved(0)/1024**2:.1f}MB")

    except KeyboardInterrupt: logger.warning(f"Fine-tuning interrupted by user. Last completed epoch: {last_completed_ep_main}.")
    except Exception as e_fatal: logger.critical(f"Critical error during training at epoch {last_completed_ep_main + 1}: {e_fatal}", exc_info=True); sys.exit(1)
    finally:
        logger.info(f"Fine-tuning ended. Last completed epoch: {last_completed_ep_main}.")
        if nan_threshold_exceeded_flag: logger.error("Training stopped prematurely due to excessive NaN losses.")
        # Save final model using current overall best_val_metric_val
        if 'finetuner_obj' in locals() and finetuner_obj is not None and cfg.get("final_model_filename"):
            finetuner_obj.save_checkpoint(
                os.path.join(abs_ckpt_save_dir, cfg["final_model_filename"]),
                last_completed_ep_main, best_val_metric_val, metric_to_watch
            )
        logger.info(f"Fine-tuning summary: Best validation '{metric_to_watch}': {best_val_metric_val:.4f} (achieved over {last_completed_ep_main - (start_epoch -1)} trained epochs in this run).")

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