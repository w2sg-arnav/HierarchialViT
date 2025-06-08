# phase4_finetuning/config.py
import os
import torch
import logging

_config_module_logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

try:
    PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH = os.path.dirname(PACKAGE_ROOT)
except NameError:
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25"
    PACKAGE_ROOT = os.path.join(PROJECT_ROOT_PATH, "phase4_finetuning")
    _config_module_logger.warning(f"Guessed PROJECT_ROOT_PATH & PACKAGE_ROOT.")

DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
DEFAULT_NUM_CLASSES = 7
DEFAULT_ORIGINAL_DATASET_NAME = "Original Dataset"
DEFAULT_AUGMENTED_DATASET_NAME = "Augmented Dataset"
DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE = 0.85 # From Guide Phase 3 Opt
DEFAULT_IMG_SIZE_FINETUNE = (512, 512) # Guide Phase 1 Opt

DEFAULT_ENABLE_TORCH_COMPILE = False # Safer for T4
DEFAULT_TORCH_COMPILE_MODE = "reduce-overhead"
DEFAULT_MATMUL_PRECISION = 'high'
DEFAULT_CUDNN_BENCHMARK = True

# Fallback HVT Architecture if Phase 3 config is missing or doesn't have HVT params.
# Importantly, set patch_size compatible with default fine-tuning img_size (512x512)
DEFAULT_HVT_ARCH_PARAMS_FALLBACK = {
    "patch_size": 16, # 512 / 16 = 32 (divisible by 8 for 4 stages)
    "embed_dim_rgb": 192, "spectral_channels": 0, # Assuming RGB-only from SSL
    "depths": [3, 6, 24, 3], # Example HVT-XL like depths
    "num_heads": [6, 12, 24, 48], # For embed_dim 192 base (head_dim=32)
    "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.1, "attn_drop_rate": 0.0, "drop_path_rate": 0.1,
    "norm_layer_name": "LayerNorm", "use_dfca": False,
    "use_gradient_checkpointing": True, # Essential for T4 with large models
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}

phase3_cfg_imported_dict = {}
try:
    from phase3_pretraining.config import config as imported_phase3_config
    phase3_cfg_imported_dict = imported_phase3_config.copy()
    _config_module_logger.info("Successfully imported base config from 'phase3_pretraining.config'.")
except ImportError: _config_module_logger.warning("Could not import config from 'phase3_pretraining.config'.")

# --- Resolve Parameters for Fine-tuning ---
SEED_final = phase3_cfg_imported_dict.get('seed', DEFAULT_SEED)
DEVICE_final = phase3_cfg_imported_dict.get('device', DEFAULT_DEVICE) # Use device from Phase 3 if available
DATA_ROOT_final = phase3_cfg_imported_dict.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_final = phase3_cfg_imported_dict.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_final = phase3_cfg_imported_dict.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)

ENABLE_TORCH_COMPILE_final = phase3_cfg_imported_dict.get('enable_torch_compile', DEFAULT_ENABLE_TORCH_COMPILE)
TORCH_COMPILE_MODE_final = phase3_cfg_imported_dict.get('torch_compile_mode', DEFAULT_TORCH_COMPILE_MODE)
MATMUL_PRECISION_final = phase3_cfg_imported_dict.get('matmul_precision', DEFAULT_MATMUL_PRECISION)
CUDNN_BENCHMARK_final = phase3_cfg_imported_dict.get('cudnn_benchmark', DEFAULT_CUDNN_BENCHMARK)

# HVT Architecture: MUST match SSL pre-trained model for weight loading
HVT_PARAMS_FOR_MODEL_INIT_final = phase3_cfg_imported_dict.get('hvt_params_for_backbone', DEFAULT_HVT_ARCH_PARAMS_FALLBACK).copy()
if HVT_PARAMS_FOR_MODEL_INIT_final == DEFAULT_HVT_ARCH_PARAMS_FALLBACK and phase3_cfg_imported_dict and 'hvt_params_for_backbone' not in phase3_cfg_imported_dict:
    _config_module_logger.warning("HVT params ('hvt_params_for_backbone') not in imported Phase 3 config. Using Phase 4 fallback HVT arch. VERIFY THIS!")
elif not phase3_cfg_imported_dict:
    _config_module_logger.warning("Phase 3 config not imported. Using Phase 4 fallback HVT arch. VERIFY THIS!")
else:
    _config_module_logger.info("Using HVT architecture parameters sourced from Phase 3 config for fine-tuning base.")

# --- Apply Optimization Guide Tweaks to the sourced/fallback HVT Arch ---
# Phase 1/2 Guide: Image Size 512x512, Patch Size 16
config_img_size_ft = tuple(main_config_from_phase4.get("img_size", DEFAULT_IMG_SIZE_FINETUNE)) # Get target FT img_size from *this* file's dict later
guide_patch_size = 16 # As per guide for 512px or if changing capacity
if config_img_size_ft[0] % guide_patch_size != 0 or config_img_size_ft[1] % guide_patch_size != 0:
    _config_module_logger.warning(f"Target fine-tune image size {config_img_size_ft} is not divisible by guide patch_size {guide_patch_size}. This may cause issues. Consider adjusting 'img_size' or HVT 'patch_size'.")
HVT_PARAMS_FOR_MODEL_INIT_final['patch_size'] = guide_patch_size # Critical for compatibility with 512px
# Guide Phase 2 Arch Tweaks (if applying them *on top* of SSL arch)
# HVT_PARAMS_FOR_MODEL_INIT_final['embed_dim_rgb'] = 256 # Example
# HVT_PARAMS_FOR_MODEL_INIT_final['num_heads'] = [8, 16, 32, 64] # Example
# HVT_PARAMS_FOR_MODEL_INIT_final['depths'] = [2,2,18,2] # Example
HVT_PARAMS_FOR_MODEL_INIT_final['model_drop_rate'] = 0.1 # Guide default for fine-tuning
HVT_PARAMS_FOR_MODEL_INIT_final['drop_path_rate'] = 0.1  # Guide default for fine-tuning (guide Phase 2 suggests 0.05)
HVT_PARAMS_FOR_MODEL_INIT_final['attn_drop_rate'] = HVT_PARAMS_FOR_MODEL_INIT_final.get('attn_drop_rate', 0.0) # Keep if from SSL
HVT_PARAMS_FOR_MODEL_INIT_final['use_gradient_checkpointing'] = True # For T4
HVT_PARAMS_FOR_MODEL_INIT_final['spectral_channels'] = 0 # Ensure RGB for this fine-tuning
HVT_PARAMS_FOR_MODEL_INIT_final['use_dfca'] = False

# SSL Pretrained Checkpoint Path (absolute)
SSL_PRETRAINED_PATH_final = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_best_probe.pth"
if not os.path.exists(SSL_PRETRAINED_PATH_final): _config_module_logger.error(f"CRITICAL: SSL Checkpoint NOT FOUND: '{SSL_PRETRAINED_PATH_final}'.")

config = {
    "seed": SEED_final, "device": DEVICE_final,
    "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH, "PACKAGE_ROOT_PATH": PACKAGE_ROOT,
    "run_name_suffix": "ft_guide_p1_v3_fix",
    "log_dir": "logs_finetune_guide", "log_file_finetune_base": "finetune_hvt",
    "best_model_filename_base": "best_finetuned_hvt", "final_model_filename_base": "final_finetuned_hvt",
    "checkpoint_save_dir_name": "checkpoints",

    "enable_torch_compile": ENABLE_TORCH_COMPILE_final, "torch_compile_mode": TORCH_COMPILE_MODE_final,
    "matmul_precision": MATMUL_PRECISION_final, "cudnn_benchmark": CUDNN_BENCHMARK_final,

    "ssl_pretrained_backbone_path": SSL_PRETRAINED_PATH_final,
    "load_pretrained_backbone_from_ssl": True if SSL_PRETRAINED_PATH_final and os.path.exists(SSL_PRETRAINED_PATH_final) else False,
    "resume_finetune_checkpoint_path": None, # Set path to resume a previous FT run
    "load_optimizer_scheduler_on_resume": True,
    "ssl_pretrain_img_size_fallback": (448, 448), # Image size of the actual SSL PE

    "data_root": DATA_ROOT_final, "original_dataset_name": ORIGINAL_DATASET_NAME_final,
    "augmented_dataset_name": None, # Guide doesn't specify using this for FT
    "img_size": (512, 512), # Guide Phase 1
    "num_classes": NUM_CLASSES_final,
    "train_split_ratio": 0.85, # Guide Phase 3 (can start with 0.8 if preferred)
    "normalize_data": True,    # Guide Phase 1: CRITICAL
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "sqrt_inv_count", # Guide Phase 3
    "use_weighted_loss": True, # Often combined with sampler or as alternative

    "num_workers": 6, # From example config in guide
    "prefetch_factor": 3 if DEVICE_final == 'cuda' and 6 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_Finetuned_Guide_P1P2_V3",
    "hvt_params_for_model_init": HVT_PARAMS_FOR_MODEL_INIT_final,
    "hvt_head_module_name": "classifier_head",

    "epochs": 150, # Guide Phase 2
    "batch_size": 12,  # Guide Phase 1 (for 512px on T4)
    "accumulation_steps": 4,  # Guide Phase 1 (Effective BS = 48)
    "amp_enabled": True, "clip_grad_norm": 1.0, # Guide: 0.5 if unstable
    "log_interval": 10,

    "optimizer": "AdamW", "weight_decay": 0.01, # Guide "Optimized Config"
    "optimizer_params": {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": True}, # Guide "Optimized Config"

    "freeze_backbone_epochs": 5, # Guide Phase 1
    "lr_head_frozen_phase": 1e-3,       # Guide Phase 1
    "lr_backbone_unfrozen_phase": 5e-5, # Guide Phase 1
    "lr_head_unfrozen_phase": 5e-4,     # Guide Phase 1

    # Scheduler: Guide suggests OneCycleLR for "Optimized Config"
    "scheduler": "OneCycleLR",
    "onecycle_max_lr": 1e-3, # This will be a list per param group if LLRD not used, or global max
    "onecycle_pct_start": 0.1,
    "onecycle_div_factor": 25,
    "onecycle_final_div_factor": 1e4,
    "warmup_epochs": 10, # Guide Phase 2 (relevant if WarmupCosine is chosen)
    "eta_min_lr": 1e-8, # For WarmupCosine

    "loss_function": "combined", # Guide "Optimized Config"
    "loss_label_smoothing": 0.15, # Guide "Optimized Config"
    "loss_weights": {"ce_weight": 0.7, "focal_weight": 0.3}, # Guide "Optimized Config"
    "focal_loss_alpha": 0.25, "focal_loss_gamma": 2.0,

    "augmentations_enabled": True,
    "augmentation_strategy": "aggressive_medical", # Guide Phase 2
    "augmentation_severity": "high", # Guide Phase 2

    "mixup_alpha": 0.4, "cutmix_alpha": 1.0, "cutmix_prob": 0.5, # From Guide
    # rand_augment_n, rand_augment_m - these would be used by a RandAugment transform

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 25, # Guide Phase 2 (was 40, 25 is good)
    "metric_to_monitor_early_stopping": "f1_macro",
    "min_delta_early_stopping": 1e-4,

    "tta_enabled_val": True, # Guide Phase 3
    "tta_transforms": 8, # Guide Phase 3

    "use_ema": True, "ema_decay": 0.9999, # From guide
    "use_swa": True, "swa_start_epoch": int(150 * 0.8), "swa_lr": 1e-5, # Guide SWA start
}
NUM_CLASSES = config['num_classes']
_config_module_logger.info(f"--- Phase 4 Config (Guide - Quick Wins + Phase 2 Target) ---")
_config_module_logger.info(f"Image Size: {config['img_size']}, HVT Patch Size: {config['hvt_params_for_model_init']['patch_size']}")
_config_module_logger.info(f"Scheduler: {config['scheduler']}")