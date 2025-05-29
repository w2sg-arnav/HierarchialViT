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
    _config_module_logger.warning(f"Guessed PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH} and PACKAGE_ROOT: {PACKAGE_ROOT}")

DEFAULT_RANDOM_SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
DEFAULT_NUM_CLASSES = 7
DEFAULT_ORIGINAL_DATASET_NAME = "Original Dataset"
DEFAULT_AUGMENTED_DATASET_NAME = "Augmented Dataset" 
DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE = 0.8 

phase3_config_imported = False
phase3_cfg = {}
try:
    from phase3_pretraining.config import config as phase3_base_config_dict_imported
    phase3_cfg = phase3_base_config_dict_imported.copy()
    phase3_config_imported = True
    _config_module_logger.info("Attempted to import base config from phase3_pretraining.config")
except Exception as e: 
    _config_module_logger.warning(f"Could not import or process config from phase3_pretraining: {e}. Using Phase 4 defaults for inherited values.")
    phase3_cfg = {} # Ensure phase3_cfg is empty on failure to avoid using stale/partial data

SEED = phase3_cfg.get('seed', DEFAULT_RANDOM_SEED)
DEVICE_RESOLVED = phase3_cfg.get('device', DEFAULT_DEVICE)
DATA_ROOT_RESOLVED = phase3_cfg.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_RESOLVED = phase3_cfg.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_RESOLVED = phase3_cfg.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)
# Note: augmented_dataset_name will be set directly in the config dict below for this strategy

resolved_ssl_checkpoint_path = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"
if not os.path.exists(resolved_ssl_checkpoint_path):
    _config_module_logger.error(f"CRITICAL: SSL Checkpoint path not found: {resolved_ssl_checkpoint_path}. Please verify this path.")

# Default HVT architecture, attempts to be overridden by phase3_cfg
_default_hvt_arch = {
    "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.1, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True,
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}
HVT_ARCH_PARAMS_FOR_FINETUNE = phase3_cfg.get('hvt_params_for_backbone', _default_hvt_arch).copy() # Important: use .copy()
if HVT_ARCH_PARAMS_FOR_FINETUNE == _default_hvt_arch and phase3_config_imported: # Only warn if phase3 was imported but params were still default
     _config_module_logger.warning("Using FALLBACK HVT architecture parameters from Phase 4 for fine-tuning, as Phase 3 config did not provide 'hvt_params_for_backbone' or it was identical to fallback. VERIFY THESE!")
elif not phase3_config_imported:
    _config_module_logger.warning("Phase 3 config not imported. Using FALLBACK HVT architecture parameters for fine-tuning. VERIFY THESE!")


config = {
    "seed": SEED, "device": DEVICE_RESOLVED,
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT, "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH,

    "log_dir": "logs_finetune_ssl_nonorm_lowlr_v1", 
    "log_file_finetune": "finetune_hvt_ssl_nonorm_lowlr_v1.log",
    "best_model_filename": "best_finetuned_hvt_ssl_nonorm_lowlr_v1.pth",
    "final_model_filename": "final_finetuned_hvt_ssl_nonorm_lowlr_v1.pth",
    "checkpoint_save_dir_name": "checkpoints_ssl_nonorm_lowlr_v1",

    "data_root": DATA_ROOT_RESOLVED,
    "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": "Augmented Dataset", # Explicitly use augmented data
    "img_size": (448, 448), 
    "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE,
    "normalize_data": False, # KEY CHANGE: No ImageNet normalization for fine-tuning
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "inv_freq", # Uses dataset.get_class_weights which is inv_freq
    "use_weighted_loss": True, # Also uses dataset.get_class_weights

    "num_workers": 4,
    "prefetch_factor": 2 if DEVICE_RESOLVED == 'cuda' and 4 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_SSL_NoNorm_LowLR_Finetune_v1",
    "pretrained_checkpoint_path": resolved_ssl_checkpoint_path,
    "load_pretrained_backbone": True,
    
    "hvt_params_for_model_init": {
        **HVT_ARCH_PARAMS_FOR_FINETUNE, # Base params from phase3 or fallback
        "model_drop_rate": 0.1, # Fine-tuning specific override
        "drop_path_rate": 0.1,  # Fine-tuning specific override
        "use_gradient_checkpointing": True, # Keep True for T4 memory if HVT-XL
    },

    "enable_torch_compile": False, "torch_compile_mode": "reduce-overhead",
    "matmul_precision": 'high', "cudnn_benchmark": True,

    "freeze_backbone_epochs": 10, 
    "epochs": 100,
    "batch_size": 16, 
    "accumulation_steps": 2, # Effective BS 32

    "amp_enabled": True,
    "clip_grad_norm": 1.0, 
    "log_interval": 20,

    "optimizer": "AdamW",
    "optimizer_params": { 
        "betas": (0.9, 0.999),
        "eps": 1e-8  
    },
    "weight_decay": 0.05, 

    "lr_head_frozen_phase": 5e-5,       
    "lr_backbone_unfrozen_phase": 1e-5, 
    "lr_head_unfrozen_phase": 2e-5,     
    
    "scheduler": "WarmupCosine",
    "warmup_epochs": 5, 
    "eta_min_lr": 1e-7, 

    "loss_label_smoothing": 0.1, 
    "augmentations_enabled": True,

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 15, 
    "metric_to_monitor_early_stopping": "f1_macro",

    "tta_enabled_val": False, # Keep False for this diagnostic run
    "ssl_pretrain_img_size_fallback": tuple(phase3_cfg.get('pretrain_img_size', (448,448))), # For PE interpolation if needed
}

NUM_CLASSES = config['num_classes'] 

_config_module_logger.info(f"Phase 4 Config (SSL_NoNorm_LowLR_v1 Strategy) Loaded.")
_config_module_logger.info(f"Data Normalization (ImageNet stats): {config['normalize_data']}")
_config_module_logger.info(f"Augmented Dataset: {config['augmented_dataset_name']}")
_config_module_logger.info(f"Total Epochs: {config['epochs']}, Effective BS: {config['batch_size'] * config['accumulation_steps']}")
_config_module_logger.info(f"LRs - Frozen Head: {config['lr_head_frozen_phase']:.1e}; Unfrozen BB: {config['lr_backbone_unfrozen_phase']:.1e}, Head: {config['lr_head_unfrozen_phase']:.1e}")