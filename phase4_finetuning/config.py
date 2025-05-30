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
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # Fallback, adjust if necessary
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
    phase3_cfg = {}

SEED = phase3_cfg.get('seed', DEFAULT_RANDOM_SEED)
DEVICE_RESOLVED = phase3_cfg.get('device', DEFAULT_DEVICE)
DATA_ROOT_RESOLVED = phase3_cfg.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_RESOLVED = phase3_cfg.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_RESOLVED = phase3_cfg.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)

# Path to the SSL pretrained backbone (if used for initial training before the checkpoint we are resuming)
# This path is relevant if `resume_from_checkpoint` is NOT set, or if the resumed checkpoint was only partial.
# For full finetune resume, this specific SSL checkpoint might be less directly used if the finetune checkpoint has all weights.
resolved_ssl_checkpoint_path_for_initial_pretraining = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"
if not os.path.exists(resolved_ssl_checkpoint_path_for_initial_pretraining):
    _config_module_logger.warning(f"SSL Checkpoint path for initial pretraining not found: {resolved_ssl_checkpoint_path_for_initial_pretraining}.")

# Path to the finetuning checkpoint to resume from
# THIS IS THE KEY FOR RESUMING THE FINETUNING PROCESS
RESUME_FROM_FINETUNE_CHECKPOINT_PATH = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune_optimized_strategy_v1_stable/checkpoints_optimized_strategy_v1_stable/best_finetuned_hvt_optimized_strategy_v1_stable.pth"
if not os.path.exists(RESUME_FROM_FINETUNE_CHECKPOINT_PATH):
    _config_module_logger.warning(f"Target resume checkpoint NOT FOUND: {RESUME_FROM_FINETUNE_CHECKPOINT_PATH}. Training will start from scratch or SSL backbone if configured.")
    # Set to None if not found, so main.py logic for resuming is skipped.
    # RESUME_FROM_FINETUNE_CHECKPOINT_PATH = None # Or handle this in main.py

_default_hvt_arch_for_ssl_ckpt = {
    "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.1, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True,
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}
HVT_ARCH_PARAMS_FROM_SSL_SOURCE = phase3_cfg.get('hvt_params_for_backbone', _default_hvt_arch_for_ssl_ckpt).copy()

if HVT_ARCH_PARAMS_FROM_SSL_SOURCE == _default_hvt_arch_for_ssl_ckpt and phase3_config_imported and 'hvt_params_for_backbone' not in phase3_cfg:
     _config_module_logger.warning("Phase 3 config imported but 'hvt_params_for_backbone' was missing. Using Phase 4's _default_hvt_arch_for_ssl_ckpt. VERIFY THESE for SSL checkpoint compatibility!")
elif not phase3_config_imported:
    _config_module_logger.warning("Phase 3 config not imported. Using Phase 4's _default_hvt_arch_for_ssl_ckpt. VERIFY THESE for SSL checkpoint compatibility!")


config = {
    "seed": SEED, "device": DEVICE_RESOLVED,
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT, "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH,

    "log_dir": "logs_finetune_optimized_strategy_v1_stable_resumed", # Suffix for resumed run
    "log_file_finetune": "finetune_hvt_optimized_strategy_v1_stable_resumed.log",
    "best_model_filename": "best_finetuned_hvt_optimized_strategy_v1_stable_resumed.pth",
    "final_model_filename": "final_finetuned_hvt_optimized_strategy_v1_stable_resumed.pth",
    "checkpoint_save_dir_name": "checkpoints_optimized_strategy_v1_stable_resumed",

    "resume_from_checkpoint": RESUME_FROM_FINETUNE_CHECKPOINT_PATH, # Key for resuming

    "data_root": DATA_ROOT_RESOLVED,
    "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": "Augmented Dataset",
    "img_size": (448, 448), # Ensure this matches the checkpoint's image size expectation
    "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": 0.85,
    "normalize_data": True,
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "sqrt_inv_count",
    "use_weighted_loss": True,

    "num_workers": 4,
    "prefetch_factor": 2 if DEVICE_RESOLVED == 'cuda' and 4 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_Optimized_Strategy_Finetune_v1_Stable_Resumed",
    # `pretrained_checkpoint_path` is for SSL backbone if NOT resuming a full finetune checkpoint or if it's a partial one.
    # If `resume_from_checkpoint` is a full finetune state, this SSL path might be secondary.
    "pretrained_checkpoint_path": resolved_ssl_checkpoint_path_for_initial_pretraining,
    "load_pretrained_backbone": True, # Can be set to False in main.py if resume_from_checkpoint is successful with full model state

    "hvt_params_for_model_init": {
        **HVT_ARCH_PARAMS_FROM_SSL_SOURCE, # Base architecture
        # Overrides for finetuning (should match the model structure of the checkpoint being resumed)
        "model_drop_rate": 0.1,
        "drop_path_rate": 0.05,
        "use_gradient_checkpointing": True,
        "spectral_channels": HVT_ARCH_PARAMS_FROM_SSL_SOURCE.get("spectral_channels", 0),
    },

    "enable_torch_compile": False, "torch_compile_mode": "reduce-overhead",
    "matmul_precision": 'high', "cudnn_benchmark": True,

    "freeze_backbone_epochs": 5, # This needs to be handled WRT resumed epoch
    "epochs": 200, # Increased total epochs for resumed training

    "batch_size": 8,
    "accumulation_steps": 6, # Effective BS 48

    "amp_enabled": False, # Keep False to maintain stability from previous config
    "clip_grad_norm": 0.5,

    "optimizer": "AdamW",
    "optimizer_params": {
        "betas": (0.9, 0.999),
        "eps": 1e-7
    },
    "weight_decay": 0.05,

    # LRs might be effectively superseded by scheduler state if resuming
    "lr_head_frozen_phase": 1e-4,
    "lr_backbone_unfrozen_phase": 1e-5,
    "lr_head_unfrozen_phase": 1e-4,

    "scheduler": "WarmupCosine",
    "warmup_epochs": 15, # If scheduler is reset, this applies. If resumed, scheduler state dictates.
    "eta_min_lr": 1e-7,

    "loss_label_smoothing": 0.1,
    "augmentations_enabled": True,
    "augmentation_strategy": "stable_enhanced",
    "augmentation_severity": "moderate", # Changed from "mild" to "moderate"

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 30, # Increased patience
    "metric_to_monitor_early_stopping": "f1_macro",

    "tta_enabled_val": True,
    "ssl_pretrain_img_size_fallback": tuple(phase3_cfg.get('pretrain_img_size', (448,448))),

    "debug_nan_detection": True,
    "stop_on_nan_threshold": 5,
    "monitor_gradients": True,
    "gradient_log_interval": 50,
    "save_checkpoint_every_n_epochs": 10,
}

NUM_CLASSES = config['num_classes']

_config_module_logger.info(f"Phase 4 Config (Optimized_Strategy_v1, Stable Variant - RESUMED) Loaded.")
if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
    _config_module_logger.info(f"Attempting to resume from: {config['resume_from_checkpoint']}")
else:
    _config_module_logger.info(f"No valid resume_from_checkpoint specified or path not found. Will train from scratch or SSL backbone.")
_config_module_logger.info(f"AMP Enabled: {config['amp_enabled']}")
_config_module_logger.info(f"LRs (initial if not resuming scheduler) - Frozen Head: {config['lr_head_frozen_phase']:.1e}; Unfrozen BB: {config['lr_backbone_unfrozen_phase']:.1e}, Head: {config['lr_head_unfrozen_phase']:.1e}")
_config_module_logger.info(f"Augmentation Strategy: {config['augmentation_strategy']}, Severity: {config.get('augmentation_severity','N/A')}")
_config_module_logger.info(f"Total epochs set to: {config['epochs']}")