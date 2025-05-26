# phase3_pretraining/config.py
import torch
import os
import logging # Import logging to use the logger instance here

# --- Absolute Project Root Path ---
# IMPORTANT: Set this to the absolute path of your "cvpr25" directory.
# Example: "/home/user/projects/cvpr25" or "/teamspace/studios/this_studio/cvpr25"
PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # <<< MODIFY THIS AS NEEDED!

# --- Path Construction (relative to PROJECT_ROOT_PATH) ---
DATASET_BASE_DIR_NAME = "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, DATASET_BASE_DIR_NAME)

PACKAGE_ROOT_PATH = os.path.join(PROJECT_ROOT_PATH, "phase3_pretraining") # Absolute path to this package
LOG_DIR_NAME = "logs" # Log directory will be phase3_pretraining/logs/
CHECKPOINT_DIR_NAME = "pretrain_checkpoints_hvt_xl" # Checkpoints will be in phase3_pretraining/pretrain_checkpoints_hvt_xl/

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PyTorch Performance Settings (e.g., for H100 or similar) ---
ENABLE_TORCH_COMPILE = False # Set to True if using PyTorch 2.0+ and want to test compilation
TORCH_COMPILE_MODE = "reduce-overhead" # Options: "default", "reduce-overhead", "max-autotune"
MATMUL_PRECISION = 'high' # 'high' or 'highest' for TF32 on Ampere/Hopper GPUs
CUDNN_BENCHMARK = True    # Enable cuDNN auto-tuner

# --- Configuration Dictionary ---
# This dict holds all parameters for the pre-training run.
config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH, # For reference within other modules if needed
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT_PATH, # For reference
    "log_dir_name": LOG_DIR_NAME, # Name of the log directory within PACKAGE_ROOT_PATH
    "log_file_pretrain": "phase3_simclr_hvt_xl.log",
    "checkpoint_dir_name": CHECKPOINT_DIR_NAME, # Name of checkpoint dir within PACKAGE_ROOT_PATH
    "enable_torch_compile": ENABLE_TORCH_COMPILE,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "matmul_precision": MATMUL_PRECISION,
    "cudnn_benchmark": CUDNN_BENCHMARK,

    # Dataset Config
    "data_root": DATA_ROOT, # Absolute path to dataset parent directory
    "original_dataset_name": "Original Dataset", # Subfolder name for original images
    "augmented_dataset_name": "Augmented Dataset",# Subfolder name for augmented images (if used)
    "train_split_ratio": 0.95, # Proportion of data for training (remainder for validation in probe)
    "num_classes": 7,          # For the linear probe evaluation
    "num_workers": 4,          # DataLoader workers (set to 0 for easier debugging if issues arise)
    "prefetch_factor": 2 if DEVICE == 'cuda' and 4 > 0 else None, # For num_workers > 0

    # HVT Backbone Parameters (e.g., for an "HVT-XL" variant)
    # These parameters will be passed to the Phase 2 HVT factory.
    # Ensure these are compatible with the HVT architecture from Phase 2.
    # The HVT from Phase 2 should NOT have its own SSL components enabled if this wrapper is used.
    "hvt_params_for_backbone": {
        # Phase 2 HVT constructor arguments:
        # "img_size" and "num_classes" are passed by the wrapper/trainer dynamically.
        "patch_size": 14, # Example for an "XL" model that might use 448px input
        "embed_dim_rgb": 192,
        "embed_dim_spectral": 192,
        "spectral_channels": 0,      # Set to 0 for SimCLR pre-training on RGB only
        "depths": [3, 6, 24, 3],   # Example: Deeper HVT-XL
        "num_heads": [6, 12, 24, 48], # Ensure embed_dim is divisible by num_heads at each stage
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "model_drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2,
        "norm_layer_name": "LayerNorm",
        "use_dfca": False,           # No DFCA if spectral_channels is 0
        "dfca_embed_dim_match_rgb": True,
        "dfca_num_heads": 32,        # Irrelevant if use_dfca is False or spectral_channels=0
        "dfca_drop_rate": 0.1,
        "dfca_use_disease_mask": True,
        "use_gradient_checkpointing": True, # Recommended for large models

        # Disable HVT's internal SSL components if using this external SimCLR wrapper
        "ssl_enable_mae": False,
        "ssl_enable_contrastive": False,
        "enable_consistency_loss_heads": False,
        # Other SSL params for HVT internals are not strictly needed here if above are False
        "ssl_mae_mask_ratio": 0.75,
        "ssl_mae_decoder_dim": 64,
        "ssl_mae_norm_pix_loss": True,
        "ssl_contrastive_projector_dim": 128,
        "ssl_contrastive_projector_depth": 2,
    },

    # SimCLR Pre-training Specific
    "pretrain_img_size": (448, 448), # Image size for SimCLR pre-training
    "pretrain_epochs": 80,          # Number of epochs for SimCLR
    "pretrain_batch_size": 32,       # Effective batch size (adjust with accumulation)
    "accumulation_steps": 2,         # Gradient accumulation (effective_batch_size = 32*2=64)
    "pretrain_lr": 5e-4,             # Base learning rate for SimCLR
    "pretrain_optimizer": "AdamW",   # "AdamW" or "SGD"
    "pretrain_scheduler": "WarmupCosine", # "WarmupCosine", "CosineAnnealingLR", or "None"
    "warmup_epochs": 10,             # For WarmupCosine or manual warmup with CosineAnnealingLR
    "eta_min_lr": 1e-6,              # Minimum LR for CosineAnnealingLR (if used)
    "pretrain_weight_decay": 0.05,
    "pretrain_momentum": 0.9,        # For SGD optimizer
    "temperature": 0.1,              # InfoNCE loss temperature
    "projection_dim": 256,           # Output dimension of SimCLR projection head
    "projection_hidden_dim": 4096,   # Hidden dimension of SimCLR projection head

    # SimCLR Augmentations (parameters for SimCLRAugmentation class)
    "simclr_s": 1.0,                 # Strength of color jitter
    "simclr_p_grayscale": 0.2,
    "simclr_p_gaussian_blur": 0.5,
    "simclr_rrc_scale_min": 0.08,    # Min scale for RandomResizedCrop (SimCLR default)

    # Linear Probing Configuration (for evaluating features during pre-training)
    "evaluate_every_n_epochs": 10,    # How often to run linear probing
    "linear_probe_epochs": 20,       # Epochs to train the linear probe classifier
    "linear_probe_lr": 0.1,
    "probe_optimizer": "SGD",        # "SGD" or "AdamW" for probe
    "probe_momentum": 0.9,           # For SGD probe optimizer
    "probe_weight_decay": 0.0,       # Typically no WD for linear probe
    "probe_batch_size": 64,

    # Checkpointing
    "save_every_n_epochs": 20,       # How often to save a regular checkpoint
    "model_arch_name_for_ckpt": "hvt_xl_simclr", # Used in checkpoint filenames
    "clip_grad_norm": 1.0,           # Max norm for gradient clipping (set to None to disable)
}

# --- Basic Log after config dict is defined ---
# This logger instance is for config.py itself.
# Other modules will get their own logger instances.
_config_logger = logging.getLogger(__name__)
_config_logger.info(f"Phase 3 Configuration Loaded. Project Root: {PROJECT_ROOT_PATH}")
_config_logger.info(f"Data Root: {DATA_ROOT}, Device: {DEVICE}")
_config_logger.info(f"Pre-training image size: {config['pretrain_img_size']}, Effective Batch Size (w/ accum): {config['pretrain_batch_size'] * config['accumulation_steps']}")
_config_logger.info(f"HVT Backbone spectral channels for pre-training: {config['hvt_params_for_backbone']['spectral_channels']}")
if config['hvt_params_for_backbone']['spectral_channels'] > 0 and config['hvt_params_for_backbone']['use_dfca']:
    _config_logger.info("DFCA fusion in backbone: Potentially active (depends on spectral_channels > 0).")
else:
    _config_logger.info("DFCA fusion in backbone: Inactive (spectral_channels=0 or use_dfca=False).")