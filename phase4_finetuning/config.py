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

    "log_dir": "logs_finetune_high_performance_v2", # New experiment name
    "log_file_finetune": "finetune_hvt_high_performance_v2.log",
    "best_model_filename": "best_finetuned_hvt_high_performance_v2.pth",
    "final_model_filename": "final_finetuned_hvt_high_performance_v2.pth",
    "checkpoint_save_dir_name": "checkpoints_high_performance_v2",

    "resume_from_checkpoint": RESUME_FROM_FINETUNE_CHECKPOINT_PATH, # Key for resuming

    "data_root": DATA_ROOT_RESOLVED,
    "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": "Augmented Dataset",
    "img_size": (512, 512), # Increased resolution for better feature extraction
    "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": 0.9, # Use more data for training
    "normalize_data": True,
    "use_weighted_sampler": True,
    "weighted_sampler_mode": "inv_count", # More aggressive class balancing
    "use_weighted_loss": True,
    "focal_loss_alpha": 0.25, # Add focal loss parameters
    "focal_loss_gamma": 2.0,

    "num_workers": 6, # Increased for better data loading
    "prefetch_factor": 3 if DEVICE_RESOLVED == 'cuda' and 6 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_HighPerformance_v2",
    # `pretrained_checkpoint_path` is for SSL backbone if NOT resuming a full finetune checkpoint or if it's a partial one.
    # If `resume_from_checkpoint` is a full finetune state, this SSL path might be secondary.
    "pretrained_checkpoint_path": resolved_ssl_checkpoint_path_for_initial_pretraining,
    "load_pretrained_backbone": True, # Can be set to False in main.py if resume_from_checkpoint is successful with full model state

    "hvt_params_for_model_init": {
        **HVT_ARCH_PARAMS_FROM_SSL_SOURCE, # Base architecture
        # Overrides for finetuning (should match the model structure of the checkpoint being resumed)
        "model_drop_rate": 0.15, # Increased regularization
        "drop_path_rate": 0.2, # Increased stochastic depth
        "attn_drop_rate": 0.1, # Add attention dropout
        "use_gradient_checkpointing": True,
        "spectral_channels": HVT_ARCH_PARAMS_FROM_SSL_SOURCE.get("spectral_channels", 0),
        # Add layer-wise learning rate decay
        "layer_wise_lr_decay": 0.8,
    },

    "enable_torch_compile": True, # Enable for better performance
    "torch_compile_mode": "max-autotune", # More aggressive optimization
    "matmul_precision": 'high', 
    "cudnn_benchmark": True,

    # Progressive unfreezing strategy
    "freeze_backbone_epochs": 3, # Shorter initial freeze
    "progressive_unfreezing": True, # Enable progressive unfreezing
    "unfreezing_schedule": [3, 8, 15, 20], # Epochs to unfreeze layers
    
    "epochs": 250, # More epochs for better convergence

    "batch_size": 6, # Adjusted for higher resolution
    "accumulation_steps": 8, # Effective BS 48

    "amp_enabled": True, # Enable mixed precision for efficiency
    "amp_opt_level": "O1", # Conservative mixed precision
    "clip_grad_norm": 1.0, # Increased gradient clipping

    "optimizer": "AdamW",
    "optimizer_params": {
        "betas": (0.9, 0.999),
        "eps": 1e-8, # More stable epsilon
        "amsgrad": True # Use AMSGrad variant
    },
    "weight_decay": 0.01, # Reduced weight decay

    # Improved learning rate schedule
    "lr_head_frozen_phase": 3e-4, # Higher initial LR for head
    "lr_backbone_unfrozen_phase": 5e-5, # Higher backbone LR
    "lr_head_unfrozen_phase": 2e-4, # Higher head LR
    
    # Layer-wise learning rates
    "use_layer_wise_lr": True,
    "layer_wise_lr_multipliers": {
        "head": 1.0,
        "layer_4": 0.9,
        "layer_3": 0.8,
        "layer_2": 0.7,
        "layer_1": 0.6
    },

    "scheduler": "OneCycleLR", # Better scheduler for higher accuracy
    "onecycle_max_lr": 1e-3, # Peak learning rate
    "onecycle_pct_start": 0.1, # 10% warmup
    "onecycle_div_factor": 25, # Initial LR = max_lr / div_factor
    "onecycle_final_div_factor": 1e4, # Final LR = initial_lr / final_div_factor
    
    # Backup cosine scheduler parameters
    "warmup_epochs": 10,
    "eta_min_lr": 1e-8,

    # Advanced loss configuration
    "loss_function": "combined", # Use combined loss
    "loss_label_smoothing": 0.15, # Increased label smoothing
    "loss_weights": {
        "ce_weight": 0.7,
        "focal_weight": 0.3
    },

    # Enhanced augmentation strategy
    "augmentations_enabled": True,
    "augmentation_strategy": "aggressive_medical", # New aggressive strategy
    "augmentation_severity": "high", # Increased severity
    "mixup_alpha": 0.4, # Enable mixup
    "cutmix_alpha": 1.0, # Enable cutmix
    "cutmix_prob": 0.5, # Probability of applying cutmix
    "rand_augment_n": 3, # RandAugment operations
    "rand_augment_m": 12, # RandAugment magnitude
    
    # Test-time augmentation
    "tta_enabled_val": True,
    "tta_transforms": 8, # More TTA transforms
    
    # Advanced training techniques
    "use_ema": True, # Exponential moving average
    "ema_decay": 0.9999,
    "use_swa": True, # Stochastic weight averaging
    "swa_start_epoch": 200, # Start SWA in last 50 epochs
    "swa_lr": 1e-5,
    
    # Knowledge distillation (if you have a teacher model)
    "use_knowledge_distillation": False, # Set to True if available
    "teacher_model_path": None,
    "distillation_alpha": 0.7,
    "distillation_temperature": 4.0,

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 40, # Increased patience
    "metric_to_monitor_early_stopping": "f1_macro",
    "min_delta_early_stopping": 1e-4, # Minimum improvement threshold

    "ssl_pretrain_img_size_fallback": tuple(phase3_cfg.get('pretrain_img_size', (512,512))), # Updated

    # Enhanced debugging and monitoring
    "debug_nan_detection": True,
    "stop_on_nan_threshold": 3, # More strict NaN detection
    "monitor_gradients": True,
    "gradient_log_interval": 25, # More frequent logging
    "save_checkpoint_every_n_epochs": 5, # More frequent checkpointing
    
    # Learning rate finder
    "use_lr_finder": False, # Set True for initial LR search
    "lr_finder_start_lr": 1e-7,
    "lr_finder_end_lr": 1e-1,
    "lr_finder_num_iter": 100,
    
    # Advanced validation
    "cross_validation_folds": 0, # Set > 0 for k-fold CV
    "stratified_sampling": True,
    
    # Model ensembling
    "save_top_k_models": 5, # Save top 5 models for ensembling
    "ensemble_inference": False, # Set True for final inference
    
    # Additional regularization
    "dropout_schedule": { # Progressive dropout scheduling
        0: 0.1,
        50: 0.15,
        100: 0.2,
        150: 0.15,
        200: 0.1
    },
    
    # Memory optimization
    "max_memory_usage": 0.9, # 90% GPU memory limit
    "empty_cache_frequency": 10, # Clear cache every N batches
}

NUM_CLASSES = config['num_classes']

_config_module_logger.info(f"Phase 4 Config (High Performance v2) Loaded.")
if config['resume_from_checkpoint'] and os.path.exists(config['resume_from_checkpoint']):
    _config_module_logger.info(f"Attempting to resume from: {config['resume_from_checkpoint']}")
else:
    _config_module_logger.info(f"No valid resume_from_checkpoint specified or path not found. Will train from scratch or SSL backbone.")

_config_module_logger.info(f"Key Performance Optimizations Enabled:")
_config_module_logger.info(f"  - Higher Resolution: {config['img_size']}")
_config_module_logger.info(f"  - Progressive Unfreezing: {config['progressive_unfreezing']}")
_config_module_logger.info(f"  - OneCycle LR Scheduler: {config['scheduler']}")
_config_module_logger.info(f"  - Mixed Precision: {config['amp_enabled']}")
_config_module_logger.info(f"  - Advanced Augmentations: {config['augmentation_strategy']}")
_config_module_logger.info(f"  - EMA: {config['use_ema']}, SWA: {config['use_swa']}")
_config_module_logger.info(f"  - Combined Loss Function: {config['loss_function']}")
_config_module_logger.info(f"AMP Enabled: {config['amp_enabled']}, Opt Level: {config.get('amp_opt_level', 'N/A')}")
_config_module_logger.info(f"LRs - Frozen Head: {config['lr_head_frozen_phase']:.1e}; Unfrozen BB: {config['lr_backbone_unfrozen_phase']:.1e}, Head: {config['lr_head_unfrozen_phase']:.1e}")
_config_module_logger.info(f"Augmentation Strategy: {config['augmentation_strategy']}, Severity: {config.get('augmentation_severity','N/A')}")
_config_module_logger.info(f"Total epochs set to: {config['epochs']}")
_config_module_logger.info(f"Effective batch size: {config['batch_size'] * config['accumulation_steps']}")
_config_module_logger.info(f"Image resolution: {config['img_size']}")