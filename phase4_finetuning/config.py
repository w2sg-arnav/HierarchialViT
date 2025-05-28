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
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # MUST BE CORRECT
    PACKAGE_ROOT = os.path.join(PROJECT_ROOT_PATH, "phase4_finetuning")
    _config_module_logger.warning(f"Guessed PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH} and PACKAGE_ROOT: {PACKAGE_ROOT}")

DEFAULT_RANDOM_SEED = 42; DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
DEFAULT_NUM_CLASSES = 7; DEFAULT_ORIGINAL_DATASET_NAME = "Original Dataset"; DEFAULT_AUGMENTED_DATASET_NAME = "Augmented Dataset"
DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE = 0.8

phase3_config_imported = False; phase3_cfg = {}
try:
    from phase3_pretraining.config import config as phase3_base_config_dict_imported
    phase3_cfg = phase3_base_config_dict_imported.copy()
    phase3_config_imported = True
    _config_module_logger.info("Successfully imported base config dict from phase3_pretraining.config")
except ImportError as e: _config_module_logger.warning(f"Could not import config from phase3_pretraining: {e}. Using fallbacks.")
except Exception as e_other: _config_module_logger.error(f"Unexpected error importing Phase 3 config: {e_other}", exc_info=True)

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
    phase3_ckpt_dir_name = phase3_cfg.get('checkpoint_dir_name', 'pretrain_checkpoints_hvt_xl')
    default_phase3_ckpt_path = os.path.join(phase3_pkg_root, phase3_ckpt_dir_name, default_phase3_ckpt_filename)
else:
    default_phase3_ckpt_path = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_best_probe.pth"
    _config_module_logger.warning(f"Using hardcoded SSL checkpoint path: {default_phase3_ckpt_path}. VERIFY THIS.")

_fallback_hvt_arch_params = {
    "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True,
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}
HVT_ARCH_PARAMS_FOR_FINETUNE = phase3_cfg.get('hvt_params_for_backbone', None)
if HVT_ARCH_PARAMS_FOR_FINETUNE is None:
    _config_module_logger.warning("Using FALLBACK HVT architecture parameters for fine-tuning. VERIFY THESE!")
    HVT_ARCH_PARAMS_FOR_FINETUNE = _fallback_hvt_arch_params.copy() # Use a copy
else:
    HVT_ARCH_PARAMS_FOR_FINETUNE = HVT_ARCH_PARAMS_FOR_FINETUNE.copy() # Use a copy
    _config_module_logger.info("Using HVT architecture parameters sourced from Phase 3 config for fine-tuning.")

# Fine-tuning specific HVT param overrides
HVT_ARCH_PARAMS_FOR_FINETUNE['model_drop_rate'] = 0.1
HVT_ARCH_PARAMS_FOR_FINETUNE['drop_path_rate'] = 0.1
HVT_ARCH_PARAMS_FOR_FINETUNE['use_gradient_checkpointing'] = False # <<< Try False for T4 fine-tuning if memory permits

# --- Fine-tuning Specific Configurations ---
config = {
    "seed": SEED, "device": DEVICE_RESOLVED,
    "log_dir": "logs_finetune_full_v1", # New log dir
    "log_file_finetune": "finetune_hvt_full.log",
    "best_model_filename": "best_finetuned_hvt_full.pth",
    "final_model_filename": "final_finetuned_hvt_full.pth",
    "checkpoint_save_dir_name": "checkpoints_full_v1",

    "data_root": DATA_ROOT_RESOLVED, "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": AUGMENTED_DATASET_NAME_RESOLVED, "img_size": (448, 448),
    "num_classes": NUM_CLASSES_RESOLVED, "train_split_ratio": DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE,
    "normalize_data": True, "use_weighted_sampler": True,
    "num_workers": phase3_cfg.get('num_workers', 4),
    "prefetch_factor": phase3_cfg.get('prefetch_factor', 2) if phase3_cfg.get('num_workers', 4) > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_FullFinetune",
    "pretrained_checkpoint_path": default_phase3_ckpt_path,
    "load_pretrained_backbone": True,
    
    "freeze_backbone_epochs": 0,      # <<< NO STAGED FREEZING, train all from start
    "unfreeze_backbone_lr_factor": 0.1, # Irrelevant if freeze_backbone_epochs is 0 for initial setup

    "hvt_params_for_model_init": HVT_ARCH_PARAMS_FOR_FINETUNE,

    "enable_torch_compile": False, # Keep False for T4 for stability
    "torch_compile_mode": "reduce-overhead", "matmul_precision": "high", "cudnn_benchmark": True,

    "epochs": 90, # Increased total epochs for full fine-tuning
    "batch_size": 16, # T4
    "accumulation_steps": 2, # Effective BS = 32
    "amp_enabled": True, "clip_grad_norm": 1.0, "log_interval": 10,

    "optimizer": "AdamW",
    "learning_rate": 2e-5, # <<< BASE LR, will be applied to backbone
    "head_lr_multiplier": 5.0, # <<< Head LR = 2e-5 * 5.0 = 1e-4
    "weight_decay": 0.05, "optimizer_params": {"betas": (0.9, 0.999)},

    "scheduler": "WarmupCosine",
    "warmup_epochs": 5, # 10% of 50 epochs
    "eta_min_lr": 1e-7,

    "loss_label_smoothing": 0.1, "augmentations_enabled": True,

    "evaluate_every_n_epochs": 1, "early_stopping_patience": 10,
    "metric_to_monitor_early_stopping": "f1_macro",
    "ssl_pretrain_img_size_fallback": tuple(phase3_cfg.get('pretrain_img_size', (448,448))),
}
NUM_CLASSES = config['num_classes']

_config_module_logger.info(f"Phase 4 Config (Full Fine-tune) Loaded. Model: {config.get('model_architecture_name')}")
_config_module_logger.info(f"Pretrained Checkpoint: {config.get('pretrained_checkpoint_path')}")
_config_module_logger.info(f"Base LR (Backbone): {config['learning_rate']:.2e}, Head Multiplier: {config['head_lr_multiplier']}, Head LR: {config['learning_rate']*config['head_lr_multiplier']:.2e}")
_config_module_logger.info(f"Freeze Backbone Epochs: {config['freeze_backbone_epochs']}")
_config_module_logger.info(f"HVT Grad Ckpt for Finetune: {config['hvt_params_for_model_init']['use_gradient_checkpointing']}")