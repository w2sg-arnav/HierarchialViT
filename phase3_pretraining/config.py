# phase3_pretraining/config.py
import torch
import os
import logging

# --- Absolute Project Root Path ---
# CRITICAL: Set this to the absolute path of your "cvpr25" directory.
# This directory should contain 'phase2_model' and 'phase3_pretraining'.
PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25"  # <<< MODIFY THIS AS NEEDED!

# --- Path Construction (using absolute paths for clarity and robustness) ---
# Ensure these directories exist or can be created by the scripts.
DATASET_BASE_DIR_NAME = "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, DATASET_BASE_DIR_NAME)

# Package root is where this config file resides, useful for relative pathing within the package
PACKAGE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) # Absolute path to phase3_pretraining/

LOG_DIR_NAME = "logs_h100_run" # Log directory will be phase3_pretraining/logs_h100_run/
CHECKPOINT_DIR_NAME = "checkpoints_hvt_xl_h100_run" # Checkpoints dir: phase3_pretraining/checkpoints_hvt_xl_h100_run/

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PyTorch Performance Settings for H100 (or similar powerful GPUs) ---
ENABLE_TORCH_COMPILE = False # Set to True if using PyTorch 2.0+ AND after initial debugging.
                             # It can significantly speed up training but adds compilation overhead.
TORCH_COMPILE_MODE = "reduce-overhead" # Good balance: "default", "reduce-overhead", "max-autotune"
MATMUL_PRECISION = 'high' # Enables TF32 tensor cores on Ampere/Hopper GPUs for matmuls.
CUDNN_BENCHMARK = True    # Allows cuDNN to find the best algorithms for the hardware.

# --- Configuration Dictionary ---
# This dictionary centralizes all parameters for the pre-training run.
config = {
    # General Setup & Paths
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH, # Stored for reference
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT_PATH, # Stored for reference
    "log_dir_name": LOG_DIR_NAME,           # Name of the log directory (relative to PACKAGE_ROOT_PATH)
    "log_file_pretrain": "ssl_pretrain.log",# Base name for the log file (timestamp will be added)
    "checkpoint_dir_name": CHECKPOINT_DIR_NAME, # Name of checkpoint dir (relative to PACKAGE_ROOT_PATH)

    # PyTorch Optimizations
    "enable_torch_compile": ENABLE_TORCH_COMPILE,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "matmul_precision": MATMUL_PRECISION,
    "cudnn_benchmark": CUDNN_BENCHMARK,

    # --- Resuming Training ---
    # Set to the *absolute path* of your checkpoint, or None/empty string to train from scratch.
    # Example for resuming from best probe:
    "resume_checkpoint_path": "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_epoch_20.pth",
    # Example for resuming from a specific epoch:
    # "resume_checkpoint_path": os.path.join(PACKAGE_ROOT_PATH, CHECKPOINT_DIR_NAME, "hvt_xl_simclr_h100_run_epoch_30.pth"),
    # To train from scratch, set to None or comment out:
    # "resume_checkpoint_path": None,

    # Dataset Configuration
    "data_root": DATA_ROOT, # Absolute path to dataset parent directory
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset", # Currently not used if only original is scanned
    "train_split_ratio": 0.95, # 95% for SSL pre-training, 5% for probe validation
    "num_classes": 7,          # Number of classes for the linear probe evaluation
    "num_workers": 8,          # Increased for H100. Adjust based on CPU cores available.
                               # Start with 4 if unsure, then increase.
    "prefetch_factor": 2 if DEVICE == 'cuda' and 8 > 0 else None,

    # HVT Backbone Parameters (e.g., for an "HVT-XL" variant)
    # CRITICAL: If resuming, these parameters MUST MATCH the architecture of the loaded checkpoint.
    "hvt_params_for_backbone": {
        "patch_size": 14,        # Results in (448/14)=32 initial patches per dimension. 32 is divisible by 2^3=8.
        "embed_dim_rgb": 192,
        "embed_dim_spectral": 192, # Not used if spectral_channels is 0
        "spectral_channels": 0,    # SimCLR pre-training currently on RGB only
        "depths": [3, 6, 24, 3], # HVT-XL: Total 36 Transformer blocks
        "num_heads": [6, 12, 24, 48], # Head dims: 192/6=32, 384/12=32, 768/24=32, 1536/48=32
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "model_drop_rate": 0.0,    # Dropout in HVT model (embeddings, MLP)
        "attn_drop_rate": 0.0,     # Attention dropout in HVT
        "drop_path_rate": 0.2,     # Stochastic depth rate
        "norm_layer_name": "LayerNorm",
        "use_dfca": False,         # No DFCA if spectral_channels is 0
        "dfca_embed_dim_match_rgb": True, # Irrelevant if use_dfca is False
        "dfca_num_heads": 32,             # Irrelevant
        "dfca_drop_rate": 0.1,            # Irrelevant
        "dfca_use_disease_mask": True,    # Irrelevant
        "use_gradient_checkpointing": True, # KEEP TRUE for HVT-XL on T4-like GPUs.
                                            # On H100 80GB, can *try* False with large batch sizes, but monitor VRAM.
                                            # For safety and broader compatibility, keeping True is safer.
        # Disable HVT's internal SSL components if using this external SimCLR wrapper
        "ssl_enable_mae": False,
        "ssl_enable_contrastive": False,
        "enable_consistency_loss_heads": False,
        # Below are defaults for HVT's internal SSL, not strictly needed if above are False
        "ssl_mae_mask_ratio": 0.75, "ssl_mae_decoder_dim": 64, "ssl_mae_norm_pix_loss": True,
        "ssl_contrastive_projector_dim": 128, "ssl_contrastive_projector_depth": 2,
    },

    # SimCLR Pre-training Specific Configuration
    "pretrain_img_size": (448, 448), # Image size for SimCLR pre-training
    "pretrain_epochs": 80,           # TOTAL desired number of pre-training epochs
    "pretrain_batch_size": 32,       # Per-GPU batch size before accumulation
    "accumulation_steps": 2,         # Effective batch size = pretrain_batch_size * accumulation_steps (e.g., 32*2=64)
                                     # For H100 80GB, you can try larger pretrain_batch_size (e.g., 64, 128) and accumulation_steps=1
                                     # if use_gradient_checkpointing=False fits.
    "pretrain_lr": 5e-4,             # Base learning rate. If changing effective batch size, may need to scale this.
    "pretrain_optimizer": "AdamW",   # Options: "AdamW", "SGD"
    "pretrain_scheduler": "WarmupCosine", # Options: "WarmupCosine", "CosineAnnealingLR", "None"
    "warmup_epochs": 10,             # Number of epochs for LR warmup (used by WarmupCosine or manual for CosineAnnealingLR)
    "eta_min_lr": 1e-6,              # Minimum LR for CosineAnnealingLR scheduler
    "pretrain_weight_decay": 0.05,
    "pretrain_momentum": 0.9,        # For SGD optimizer if used

    # SimCLR Loss and Projection Head
    "temperature": 0.1,              # InfoNCE loss temperature
    "projection_dim": 256,           # Output dimension of SimCLR projection head
    "projection_hidden_dim": 4096,   # Hidden dimension of SimCLR projection head

    # SimCLR Augmentations (parameters for SimCLRAugmentation class)
    "simclr_s": 1.0,                 # Strength of color jitter
    "simclr_p_grayscale": 0.2,       # Probability of applying grayscale
    "simclr_p_gaussian_blur": 0.5,   # Probability of applying Gaussian blur
    "simclr_rrc_scale_min": 0.08,    # Min scale for RandomResizedCrop (SimCLR default)

    # Linear Probing Configuration (for evaluating features during pre-training)
    "evaluate_every_n_epochs": 10,   # How often to run linear probing
    "linear_probe_epochs": 10,       # Epochs to train the linear probe classifier (can be reduced for speed)
    "linear_probe_lr": 0.1,          # LR for probe optimizer
    "probe_optimizer": "SGD",        # Optimizer for probe
    "probe_momentum": 0.9,           # Momentum for SGD probe optimizer
    "probe_weight_decay": 0.0,       # Weight decay for probe (typically 0)
    "probe_batch_size": 64,          # Batch size for linear probing (adjust based on memory)

    # Checkpointing and Model Naming
    "save_every_n_epochs": 10,        # Save a regular checkpoint every N epochs
    "model_arch_name_for_ckpt": "hvt_xl_simclr_h100_run", # Base name for saved models
    "clip_grad_norm": 1.0,            # Max norm for gradient clipping (set to None or 0 to disable)
}

# --- Logging after config dict is defined ---
# This ensures that this logger is configured after basicConfig might have been called by setup_logging.
_config_logger = logging.getLogger(__name__)
_config_logger.info(f"Phase 3 Configuration (H100 Optimized) Loaded. Run will use settings from: {__file__}")
_config_logger.info(f"Device: {DEVICE}, Num Classes for Probe: {config['num_classes']}")
_config_logger.info(f"Pre-training Img Size: {config['pretrain_img_size']}, Effective Batch Size (w/ accum): {config['pretrain_batch_size'] * config['accumulation_steps']}")
_config_logger.info(f"Target Total Pre-training Epochs: {config['pretrain_epochs']}")
_config_logger.info(f"HVT Spectral Channels (for backbone): {config['hvt_params_for_backbone']['spectral_channels']}")
if config['hvt_params_for_backbone']['use_gradient_checkpointing']:
    _config_logger.info("Gradient Checkpointing for HVT Backbone: ENABLED")
else:
    _config_logger.info("Gradient Checkpointing for HVT Backbone: DISABLED (Attempting to fit full activations in VRAM)")
if config['enable_torch_compile']:
    _config_logger.info(f"Torch Compile: ENABLED (Mode: {config['torch_compile_mode']})")
else:
    _config_logger.info("Torch Compile: DISABLED")
_config_logger.info(f"Resume Checkpoint Path: {config.get('resume_checkpoint_path', 'None (training from scratch)')}")