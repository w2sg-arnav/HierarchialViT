# phase4_finetuning/config.py
import os
import torch
import logging

# --- Logger for this config file ---
_config_module_logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Project Path Configuration ---
try:
    PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH = os.path.dirname(PACKAGE_ROOT)
except NameError:
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25"
    PACKAGE_ROOT = os.path.join(PROJECT_ROOT_PATH, "phase4_finetuning")
    _config_module_logger.warning(f"Guessed PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH} and PACKAGE_ROOT: {PACKAGE_ROOT}")

# --- Default Values ---
DEFAULT_RANDOM_SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
DEFAULT_NUM_CLASSES = 7
DEFAULT_ORIGINAL_DATASET_NAME = "Original Dataset"
DEFAULT_AUGMENTED_DATASET_NAME = "Augmented Dataset"
DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE = 0.8

# --- Attempt to Import and Use Base Configurations from Phase 3 ---
phase3_config_imported = False
phase3_cfg = {}
try:
    from phase3_pretraining.config import config as phase3_base_config_dict_imported
    phase3_cfg = phase3_base_config_dict_imported.copy()
    phase3_config_imported = True
    _config_module_logger.info("Successfully imported base config dict from phase3_pretraining.config")
except ImportError as e:
    _config_module_logger.warning(f"Could not import config from phase3_pretraining: {e}. Phase 4 will use its own defaults for HVT arch and checkpoint path.")
except Exception as e_other:
    _config_module_logger.error(f"Unexpected error importing Phase 3 config: {e_other}", exc_info=True)

# --- Resolve Key Parameters ---
SEED = phase3_cfg.get('seed', DEFAULT_RANDOM_SEED)
DEVICE_RESOLVED = phase3_cfg.get('device', DEFAULT_DEVICE)
DATA_ROOT_RESOLVED = phase3_cfg.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_RESOLVED = phase3_cfg.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_RESOLVED = phase3_cfg.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)
AUGMENTED_DATASET_NAME_RESOLVED = phase3_cfg.get('augmented_dataset_name', DEFAULT_AUGMENTED_DATASET_NAME)

default_phase3_ckpt_filename = f"{phase3_cfg.get('model_arch_name_for_ckpt', 'hvt_xl_simclr')}_best_probe.pth"
default_phase3_ckpt_path = None
if phase3_config_imported:
    phase3_pkg_root = phase3_cfg.get('PACKAGE_ROOT_PATH', os.path.join(PROJECT_ROOT_PATH, 'phase3_pretraining'))
    phase3_ckpt_dir_name = phase3_cfg.get('checkpoint_dir_name', 'pretrain_checkpoints_hvt_xl') # Default from your prev config
    default_phase3_ckpt_path = os.path.join(phase3_pkg_root, phase3_ckpt_dir_name, default_phase3_ckpt_filename)
else:
    # THIS PATH MUST BE SET MANUALLY IF PHASE 3 IMPORT FAILS
    default_phase3_ckpt_path = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_best_probe.pth" # Manually set based on your info
    _config_module_logger.warning(f"Phase 3 config import failed. PRETRAINED_HVT_CHECKPOINT_PATH set to: {default_phase3_ckpt_path}. VERIFY THIS PATH.")

# HVT Architecture Parameters
# Define the fallback dictionary separately
_fallback_hvt_arch_params = {
    "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True,
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}

HVT_ARCH_PARAMS_FOR_FINETUNE = phase3_cfg.get('hvt_params_for_backbone', None)
if HVT_ARCH_PARAMS_FOR_FINETUNE is None:
    _config_module_logger.warning("Using FALLBACK HVT architecture parameters for fine-tuning because 'hvt_params_for_backbone' was not found in Phase 3 config. VERIFY THESE!")
    HVT_ARCH_PARAMS_FOR_FINETUNE = _fallback_hvt_arch_params
else:
    _config_module_logger.info("Using HVT architecture parameters sourced from Phase 3 config for fine-tuning.")


# For fine-tuning, some HVT params might change (e.g., drop_rate, grad_ckpt)
HVT_ARCH_PARAMS_FOR_FINETUNE['model_drop_rate'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('model_drop_rate_finetune', 0.1) # Allow override or default
HVT_ARCH_PARAMS_FOR_FINETUNE['drop_path_rate'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('drop_path_rate_finetune', 0.1)
HVT_ARCH_PARAMS_FOR_FINETUNE['use_gradient_checkpointing'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('use_gradient_checkpointing_finetune', False) # Default to False for finetuning if enough VRAM

# --- Fine-tuning Specific Configurations (this is the exported 'config' dict) ---
config = {
    # General
    "seed": SEED,
    "device": DEVICE_RESOLVED,
    "log_dir": "logs_finetune_phase4", # Relative to PACKAGE_ROOT (phase4_finetuning/)
    "log_file_finetune": "finetune_hvt_xl.log",
    "best_model_filename": "best_finetuned_hvt_xl.pth",
    "final_model_filename": "final_finetuned_hvt_xl.pth",
    "checkpoint_save_dir_name": "checkpoints",

    # Data
    "data_root": DATA_ROOT_RESOLVED,
    "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": AUGMENTED_DATASET_NAME_RESOLVED,
    "img_size": (448, 448),
    "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE,
    "normalize_data": True,
    "use_weighted_sampler": True,
    "num_workers": phase3_cfg.get('num_workers', 4), # Inherit from phase3 or use default
    "prefetch_factor": (phase3_cfg.get('prefetch_factor', 2)
                        if phase3_cfg.get('num_workers', 4) > 0 else None),

    # Model - HVT-XL Fine-tuning
    "model_architecture_name": "DiseaseAwareHVT_SSL_Finetuned",
    "pretrained_checkpoint_path": default_phase3_ckpt_path, # This is now correctly an absolute path or a verified manual one
    "load_pretrained_backbone": True,
    "freeze_backbone_epochs": 0, # Example: Start with full model unfrozen
    "unfreeze_backbone_lr_factor": 0.1,

    # HVT Architecture (already defined in HVT_ARCH_PARAMS_FOR_FINETUNE)
    "hvt_params_for_model_init": HVT_ARCH_PARAMS_FOR_FINETUNE,

    # PyTorch Performance Settings
    "enable_torch_compile": phase3_cfg.get('enable_torch_compile', False),
    "torch_compile_mode": phase3_cfg.get('torch_compile_mode', "reduce-overhead"),
    "matmul_precision": phase3_cfg.get('matmul_precision', 'high'),
    "cudnn_benchmark": phase3_cfg.get('cudnn_benchmark', True),

    # Training Loop
    "epochs": 30,
    "batch_size": 16, # For T4 with HVT-XL 448px
    "accumulation_steps": 2, # Effective BS = 32
    "amp_enabled": True,
    "clip_grad_norm": 1.0,
    "log_interval": 10, # Log more frequently for smaller batches

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 3e-5,
    "head_lr_multiplier": 1.0, # Can set to >1 if only head is trained initially
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)},

    # Schedulers
    "scheduler": "WarmupCosine",
    "warmup_epochs": 3,
    "eta_min_lr": 1e-7,

    # Loss Function
    "loss_label_smoothing": 0.1,

    # Augmentations
    "augmentations_enabled": True,

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 10,
    "metric_to_monitor_early_stopping": "f1_macro",
}

NUM_CLASSES = config['num_classes']

_config_module_logger.info(f"Phase 4 Fine-tuning Configuration Loaded. Model: {config.get('model_architecture_name')}")
_config_module_logger.info(f"Pretrained Checkpoint to Load: {config.get('pretrained_checkpoint_path')}")
_config_module_logger.info(f"Fine-tuning LR: {config['learning_rate']}, Epochs: {config['epochs']}, Eff. BS: {config['batch_size']*config['accumulation_steps']}")
if config['hvt_params_for_model_init']['use_gradient_checkpointing']:
    _config_module_logger.info("Gradient Checkpointing for HVT: ENABLED during fine-tuning.")
else:
    _config_module_logger.info("Gradient Checkpointing for HVT: DISABLED during fine-tuning.")