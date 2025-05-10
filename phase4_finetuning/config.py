# phase4_finetuning/config.py
import os
import torch
import logging

# --- Define PROJECT_ROOT_PATH correctly at the top ---
# This assumes config.py is in phase4_finetuning, and phase4_finetuning is in cvpr25
try:
    CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../phase4_finetuning
    PACKAGE_ROOT = CURRENT_FILE_DIR
    PROJECT_ROOT_PATH = os.path.dirname(PACKAGE_ROOT) # .../cvpr25
except NameError: # Fallback if __file__ is not defined (e.g. interactive session)
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # MUST BE CORRECT FOR YOUR ENV
    print(f"Warning (phase4_config): Guessed PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH}")


# --- Base values (some might be overridden by phase3_base_config_dict) ---
RANDOM_SEED_default = 42
DEVICE_default = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_BASE_PATH_default = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
NUM_CLASSES_default = 7
ORIGINAL_DATASET_NAME_default = "Original Dataset"
AUGMENTED_DATASET_NAME_default = "Augmented Dataset"
TRAIN_SPLIT_RATIO_default = 0.8

# --- Attempt to Import and Use base configurations from Phase 3 ---
phase3_imported_successfully = False
try:
    from phase3_pretraining.config import config as phase3_base_config_dict
    print("INFO (phase4_config): Successfully imported base config dict from phase3_pretraining.config")
    phase3_imported_successfully = True

    RANDOM_SEED_base = phase3_base_config_dict.get('seed', RANDOM_SEED_default)
    DEVICE_base = phase3_base_config_dict.get('device', DEVICE_default)
    DATASET_BASE_PATH_base = phase3_base_config_dict.get('data_root', DATASET_BASE_PATH_default)
    NUM_CLASSES_base = phase3_base_config_dict.get('num_classes', NUM_CLASSES_default)
    ORIGINAL_DATASET_NAME_base = phase3_base_config_dict.get('original_dataset_name', ORIGINAL_DATASET_NAME_default)
    AUGMENTED_DATASET_NAME_base = phase3_base_config_dict.get('augmented_dataset_name', AUGMENTED_DATASET_NAME_default)

    # Construct the path to the Phase 3 HVT-XL pre-trained checkpoint
    phase3_ckpt_dir_rel_to_proj = phase3_base_config_dict.get('checkpoint_dir', "pretrain_checkpoints_h100_xl") # Relative path from project root
    # Ensure it's an absolute path starting from PROJECT_ROOT_PATH
    phase3_ckpt_dir_abs = os.path.join(PROJECT_ROOT_PATH, os.path.basename(phase3_ckpt_dir_rel_to_proj.strip('/\\'))) # Use basename to avoid double project_root

    phase3_ckpt_filename = "diseaseawarehvt_xl_h100_prod_pretrain_best_probe.pth" # Your confirmed filename
    PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT = os.path.join(phase3_ckpt_dir_abs, phase3_ckpt_filename)
    print(f"INFO (phase4_config): Defaulting pretrained_checkpoint_path (absolute) to: {PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT}")

    HVT_XL_PATCH_SIZE = phase3_base_config_dict.get('hvt_patch_size', 14)
    HVT_XL_EMBED_DIM_RGB = phase3_base_config_dict.get('hvt_embed_dim_rgb', 192)
    HVT_XL_EMBED_DIM_SPECTRAL = phase3_base_config_dict.get('hvt_embed_dim_spectral', 192)
    HVT_XL_SPECTRAL_CHANNELS = phase3_base_config_dict.get('hvt_spectral_channels', 3)
    HVT_XL_DEPTHS = phase3_base_config_dict.get('hvt_depths', [3, 6, 24, 3])
    HVT_XL_NUM_HEADS = phase3_base_config_dict.get('hvt_num_heads', [6, 12, 24, 48])
    HVT_XL_MLP_RATIO = phase3_base_config_dict.get('hvt_mlp_ratio', 4.0)
    HVT_XL_QKV_BIAS = phase3_base_config_dict.get('hvt_qkv_bias', True)
    HVT_XL_PRETRAIN_DROP_PATH_RATE = phase3_base_config_dict.get('hvt_drop_path_rate', 0.2)
    HVT_XL_USE_DFCA = phase3_base_config_dict.get('hvt_use_dfca', True)
    HVT_XL_DFCA_HEADS = phase3_base_config_dict.get('hvt_dfca_heads', 32)
    HVT_XL_DFCA_DROP_RATE = phase3_base_config_dict.get('dfca_drop_rate', 0.1)
    HVT_XL_DFCA_USE_DISEASE_MASK = phase3_base_config_dict.get('dfca_use_disease_mask', True)
    HVT_XL_GRAD_CKPT_PRETRAIN = phase3_base_config_dict.get('use_gradient_checkpointing', True)

    ENABLE_TORCH_COMPILE_base = phase3_base_config_dict.get('enable_torch_compile', True)
    TORCH_COMPILE_MODE_base = phase3_base_config_dict.get('torch_compile_mode', "reduce-overhead")
    MATMUL_PRECISION_base = phase3_base_config_dict.get('matmul_precision', 'high')
    CUDNN_BENCHMARK_base = phase3_base_config_dict.get('cudnn_benchmark', True)
    NUM_WORKERS_base = phase3_base_config_dict.get('num_workers', 16)
    PREFETCH_FACTOR_base = phase3_base_config_dict.get('prefetch_factor', 2) if NUM_WORKERS_base > 0 else None

except ImportError as e:
    print(f"Warning (phase4_config): Could not import config from phase3_pretraining: {e}. Using fallback defaults for Phase 4.")
    RANDOM_SEED_base = RANDOM_SEED_default; DEVICE_base = DEVICE_default; DATASET_BASE_PATH_base = DATASET_BASE_PATH_default
    NUM_CLASSES_base = NUM_CLASSES_default; ORIGINAL_DATASET_NAME_base = ORIGINAL_DATASET_NAME_default
    AUGMENTED_DATASET_NAME_base = AUGMENTED_DATASET_NAME_default; TRAIN_SPLIT_RATIO_base = TRAIN_SPLIT_RATIO_default # Use fine-tune default split ratio
    # IMPORTANT: User must verify this fallback path if Phase 3 import fails
    PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT = os.path.join(PROJECT_ROOT_PATH, "pretrain_checkpoints_h100_xl", "diseaseawarehvt_xl_h100_prod_pretrain_best_probe.pth")
    print(f"Phase4 Config Fallback: Using hardcoded pretrained_checkpoint_path: {PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT}")
    HVT_XL_PATCH_SIZE = 14; HVT_XL_EMBED_DIM_RGB = 192; HVT_XL_EMBED_DIM_SPECTRAL = 192; HVT_XL_SPECTRAL_CHANNELS = 3
    HVT_XL_DEPTHS = [3, 6, 24, 3]; HVT_XL_NUM_HEADS = [6, 12, 24, 48]; HVT_XL_MLP_RATIO = 4.0; HVT_XL_QKV_BIAS = True
    HVT_XL_PRETRAIN_DROP_PATH_RATE = 0.2; HVT_XL_USE_DFCA = True; HVT_XL_DFCA_HEADS = 32; HVT_XL_DFCA_DROP_RATE = 0.1; HVT_XL_DFCA_USE_DISEASE_MASK = True
    HVT_XL_GRAD_CKPT_PRETRAIN = True
    ENABLE_TORCH_COMPILE_base = True; TORCH_COMPILE_MODE_base = "reduce-overhead"; MATMUL_PRECISION_base = 'high'; CUDNN_BENCHMARK_base = True
    NUM_WORKERS_base = 4; PREFETCH_FACTOR_base = 2


# --- Fine-tuning Specific Configurations (this is the exported 'config' dict) ---
config = {
    # General
    "seed": RANDOM_SEED_base,
    "device": DEVICE_base,
    "log_dir": "logs_finetune_xl_prod", # Relative to phase4_finetuning directory
    "log_file_finetune": "finetune_hvt_xl_prod.log",
    "best_model_path": "best_finetuned_hvt_xl_prod.pth", # Relative to log_dir
    "final_model_path": "final_finetuned_hvt_xl_prod.pth", # Relative to log_dir

    # Data
    "data_root": DATASET_BASE_PATH_base,
    "original_dataset_name": ORIGINAL_DATASET_NAME_base,
    "augmented_dataset_name": AUGMENTED_DATASET_NAME_base,
    "img_size": (448, 448),
    "num_classes": NUM_CLASSES_base,
    "train_split_ratio": TRAIN_SPLIT_RATIO_default, # Fine-tuning specific split
    "normalize_data": True,
    "use_weighted_sampler": True,
    "num_workers": NUM_WORKERS_base,
    "prefetch_factor": PREFETCH_FACTOR_base,

    # Model - HVT-XL Specific
    "model_architecture": "DiseaseAwareHVT_XL",
    "pretrained_checkpoint_path": PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT, # This should now be absolute
    "load_pretrained_backbone": True,
    "freeze_backbone_epochs": 5,
    "unfreeze_backbone_lr_factor": 0.1,

    # HVT-XL Architecture (values sourced from Phase 3 or fallbacks)
    "hvt_patch_size": HVT_XL_PATCH_SIZE,
    "hvt_embed_dim_rgb": HVT_XL_EMBED_DIM_RGB,
    "hvt_embed_dim_spectral": HVT_XL_EMBED_DIM_SPECTRAL,
    "hvt_spectral_channels": HVT_XL_SPECTRAL_CHANNELS,
    "hvt_depths": HVT_XL_DEPTHS,
    "hvt_num_heads": HVT_XL_NUM_HEADS,
    "hvt_mlp_ratio": HVT_XL_MLP_RATIO,
    "hvt_qkv_bias": HVT_XL_QKV_BIAS,
    "hvt_model_drop_rate": 0.1, # Fine-tuning specific dropout
    "hvt_attn_drop_rate": 0.0,
    "hvt_drop_path_rate": 0.1,  # Fine-tuning specific drop path (can be HVT_XL_PRETRAIN_DROP_PATH_RATE or different)
    "hvt_use_dfca": HVT_XL_USE_DFCA,
    "hvt_dfca_heads": HVT_XL_DFCA_HEADS,
    "dfca_drop_rate": HVT_XL_DFCA_DROP_RATE,
    "dfca_use_disease_mask": HVT_XL_DFCA_USE_DISEASE_MASK,
    "use_gradient_checkpointing": False,

    # PyTorch Performance Settings
    "enable_torch_compile": ENABLE_TORCH_COMPILE_base,
    "torch_compile_mode": TORCH_COMPILE_MODE_base,
    "matmul_precision": MATMUL_PRECISION_base,
    "cudnn_benchmark": CUDNN_BENCHMARK_base,

    # Training Loop
    "epochs": 50,
    "batch_size": 32,
    "accumulation_steps": 1,
    "amp_enabled": True,
    "clip_grad_norm": 1.0,
    "log_interval": 20,

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 3e-5,
    "head_lr_multiplier": 1.0,
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)},

    # Schedulers
    "scheduler": "WarmupCosine",
    "warmup_epochs": 5,
    "eta_min_lr": 1e-6,

    # Loss Function
    "loss_label_smoothing": 0.1,

    # Augmentations
    "augmentations_enabled": True,

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 10,
    "metric_to_monitor_early_stopping": "f1_macro",
}

# Export NUM_CLASSES for baseline models if they import directly from this config file
NUM_CLASSES = config['num_classes'] # Make sure config dict has 'num_classes'
# It does, as it's inherited from NUM_CLASSES_base or fallback

_config_module_logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_config_module_logger.info(f"Phase 4 Fine-tuning Configuration Defaults Loaded (config.py). Model arch: {config.get('model_architecture')}")
_config_module_logger.info(f"Default pretrained checkpoint path set to: {config.get('pretrained_checkpoint_path')}")