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
    phase3_cfg = {}

SEED = phase3_cfg.get('seed', DEFAULT_RANDOM_SEED) # Or set a new seed if desired for this run: e.g., 43
DEVICE_RESOLVED = phase3_cfg.get('device', DEFAULT_DEVICE)
DATA_ROOT_RESOLVED = phase3_cfg.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_RESOLVED = phase3_cfg.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_RESOLVED = phase3_cfg.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)

resolved_ssl_checkpoint_path_for_initial_pretraining = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth"

# --- IMPORTANT: Path to the finetuning checkpoint to resume from ---
# This should be the BEST checkpoint from the PREVIOUS run (e.g., the one from epoch 14 that hit f1_macro 0.8200)
RESUME_FROM_FINETUNE_CHECKPOINT_PATH = "/teamspace/studios/this_studio/cvpr25/phase4_finetuning/logs_finetune_optimized_strategy_v1_stable_resumed/checkpoints_optimized_strategy_v1_stable_resumed/best_finetuned_hvt_optimized_strategy_v1_stable_resumed.pth"
# Verify this path carefully. If the best was saved as checkpoint_epoch_14.pth, use that.
if not os.path.exists(RESUME_FROM_FINETUNE_CHECKPOINT_PATH):
    _config_module_logger.error(f"CRITICAL: Target resume checkpoint NOT FOUND: {RESUME_FROM_FINETUNE_CHECKPOINT_PATH}. Please verify this path.")
    # RESUME_FROM_FINETUNE_CHECKPOINT_PATH = None # Or exit

_default_hvt_arch_for_ssl_ckpt = {
    "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
    "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
    "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.1, "norm_layer_name": "LayerNorm",
    "use_dfca": False, "use_gradient_checkpointing": True,
    "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
}
HVT_ARCH_PARAMS_FROM_SSL_SOURCE = phase3_cfg.get('hvt_params_for_backbone', _default_hvt_arch_for_ssl_ckpt).copy()


config = {
    "seed": SEED, "device": DEVICE_RESOLVED,
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT, "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH,

    "log_dir": "logs_finetune_optimized_strategy_v1_stable_resumed2", # Suffix for this new resumed run
    "log_file_finetune": "finetune_hvt_optimized_strategy_v1_stable_resumed2.log",
    "best_model_filename": "best_finetuned_hvt_optimized_strategy_v1_stable_resumed2.pth",
    "final_model_filename": "final_finetuned_hvt_optimized_strategy_v1_stable_resumed2.pth",
    "checkpoint_save_dir_name": "checkpoints_optimized_strategy_v1_stable_resumed2",

    "resume_from_checkpoint": RESUME_FROM_FINETUNE_CHECKPOINT_PATH,

    "data_root": DATA_ROOT_RESOLVED,
    "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "img_size": (448, 448), 
    "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": 0.85, # Keep consistent with previous dataset split if using same data
    "normalize_data": True,
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "sqrt_inv_count",
    "use_weighted_loss": True,

    "num_workers": 4,
    "prefetch_factor": 2 if DEVICE_RESOLVED == 'cuda' and 4 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_Optimized_Strategy_Finetune_v1_Stable_Resumed2",
    "pretrained_checkpoint_path": resolved_ssl_checkpoint_path_for_initial_pretraining, # SSL backbone, not used if resume_from_checkpoint is successful
    "load_pretrained_backbone": True, # Will be set to False in main.py if full finetune ckpt is loaded

    "hvt_params_for_model_init": {
        **HVT_ARCH_PARAMS_FROM_SSL_SOURCE, 
        "model_drop_rate": 0.1, 
        "drop_path_rate": 0.05, 
        "use_gradient_checkpointing": True,
        "spectral_channels": HVT_ARCH_PARAMS_FROM_SSL_SOURCE.get("spectral_channels", 0),
    },

    "enable_torch_compile": False, "torch_compile_mode": "reduce-overhead",
    "matmul_precision": 'high', "cudnn_benchmark": True,

    "freeze_backbone_epochs": 0, # Already unfrozen, set to 0 to ensure no re-freezing logic interferes
    "epochs": 200, # Total epochs for this run (will start from resumed_epoch + 1)

    "batch_size": 8,
    "accumulation_steps": 6, 

    "amp_enabled": False, 
    "clip_grad_norm": 0.5,

    "optimizer": "AdamW",
    "optimizer_params": {
        "betas": (0.9, 0.999),
        "eps": 1e-7 
    },
    "weight_decay": 0.05,

    # --- CRITICAL LR ADJUSTMENTS FOR RESUMING FINER TUNING ---
    "lr_head_frozen_phase": 1e-5,       # Lowered, though not directly used if backbone already unfrozen
    "lr_backbone_unfrozen_phase": 2e-6, # Significantly Lowered (was 1e-5)
    "lr_head_unfrozen_phase": 1e-5,     # Significantly Lowered (was 1e-4)

    "scheduler": "WarmupCosine", # Scheduler type will be loaded from checkpoint if available
    "warmup_epochs": 1, # Very short warmup if scheduler is reset, or ignored if scheduler state is loaded
                      # The main effect comes from lower base LRs.
    "eta_min_lr": 5e-8, # Slightly lower eta_min as well (was 1e-7)

    "loss_label_smoothing": 0.1,
    "augmentations_enabled": True,
    "augmentation_strategy": "stable_enhanced",
    "augmentation_severity": "moderate", 

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 30, # Keep or slightly increase
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

_config_module_logger.info(f"Phase 4 Config (Optimized_Strategy_v1, Stable Variant - RESUMED2) Loaded.")
if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
    _config_module_logger.info(f"Attempting to resume from: {config['resume_from_checkpoint']}")
else:
    _config_module_logger.warning(f"Resume checkpoint not found or not specified. Training will start based on 'load_pretrained_backbone'.")

_config_module_logger.info(f"NEW LRs for this run - Unfrozen BB: {config['lr_backbone_unfrozen_phase']:.1e}, Head: {config['lr_head_unfrozen_phase']:.1e}")
_config_module_logger.info(f"NEW Warmup Epochs: {config['warmup_epochs']}")