# phase3_pretraining/config.py
import torch
import os
import logging

# --- Absolute Project Root Path ---
PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25"

# --- Path Construction ---
DATASET_BASE_DIR_NAME = "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
DATA_ROOT = os.path.join(PROJECT_ROOT_PATH, DATASET_BASE_DIR_NAME)
PACKAGE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Using the *original* checkpoint directory where your best_probe.pth is located
# If you want to save *new* checkpoints from the resumed run to a different dir, change CHECKPOINT_DIR_NAME here.
# For simplicity, let's assume we continue saving to the same directory structure.
ORIGINAL_CHECKPOINT_DIR_NAME = "pretrain_checkpoints_hvt_xl" # From your path
LOG_DIR_NAME = "logs_t4_ssl_resumed_from_best_probe"
CHECKPOINT_DIR_NAME = ORIGINAL_CHECKPOINT_DIR_NAME # Save new checkpoints to the same dir structure

# --- Core Settings ---
RANDOM_SEED = 42 # Keep seed consistent with the initial run for any new random initializations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PyTorch Performance Settings (for T4) ---
ENABLE_TORCH_COMPILE = False
TORCH_COMPILE_MODE = "reduce-overhead"
MATMUL_PRECISION = 'high'
CUDNN_BENCHMARK = True

# --- Configuration Dictionary ---
config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH,
    "PACKAGE_ROOT_PATH": PACKAGE_ROOT_PATH,
    "log_dir_name": LOG_DIR_NAME,
    "log_file_pretrain": "ssl_t4_resumed_best_probe.log", # New log file
    "checkpoint_dir_name": CHECKPOINT_DIR_NAME, # Where new checkpoints will be saved by trainer
    "enable_torch_compile": ENABLE_TORCH_COMPILE,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "matmul_precision": MATMUL_PRECISION,
    "cudnn_benchmark": CUDNN_BENCHMARK,

    # --- Resuming Training ---
    # Path to the checkpoint you want to resume from.
    "resume_checkpoint_path": "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_best_probe.pth", # <<< YOUR SPECIFIED PATH

    # Dataset Config (should match initial run)
    "data_root": DATA_ROOT,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.95,
    "num_classes": 7,
    "num_workers": 4,
    "prefetch_factor": 2 if DEVICE == 'cuda' and 4 > 0 else None,

    # HVT Backbone Parameters (MUST MATCH THE ARCHITECTURE OF THE LOADED CHECKPOINT)
    "hvt_params_for_backbone": {
        "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192,
        "spectral_channels": 0, "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48],
        "mlp_ratio": 4.0, "qkv_bias": True, "model_drop_rate": 0.0, "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm", "use_dfca": False,
        "use_gradient_checkpointing": True, # CRITICAL for T4 with HVT-XL
        "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
        # Irrelevant defaults
        "dfca_embed_dim_match_rgb": True, "dfca_num_heads": 32, "dfca_drop_rate": 0.1, "dfca_use_disease_mask": True,
        "ssl_mae_mask_ratio": 0.75, "ssl_mae_decoder_dim": 64, "ssl_mae_norm_pix_loss": True,
        "ssl_contrastive_projector_dim": 128, "ssl_contrastive_projector_depth": 2,
    },

    # SimCLR Pre-training Specific Configuration
    # These should ideally match the settings of the run you are resuming for optimizer/scheduler state to be valid.
    "pretrain_img_size": (448, 448),
    "pretrain_epochs": 80,        # TOTAL desired epochs. If best_probe was after epoch 20, it will run for 60 more.
    "pretrain_batch_size": 32,    # Keep consistent with the run being resumed
    "accumulation_steps": 2,      # Keep consistent (Effective batch size = 64)
    "pretrain_lr": 5e-4,          # Must be the SAME base LR used to save the checkpoint for scheduler to resume correctly.
    "pretrain_optimizer": "AdamW",# Must be the SAME optimizer
    "pretrain_scheduler": "WarmupCosine", # Must be the SAME scheduler
    "warmup_epochs": 10,          # Must be the SAME warmup epochs
    "eta_min_lr": 1e-6,
    "pretrain_weight_decay": 0.05,
    "temperature": 0.1,           # Keep consistent with the run being resumed

    # SimCLR Projection Head (dimensions should match the saved projection head)
    "projection_dim": 256,
    "projection_hidden_dim": 4096,

    # SimCLR Augmentations (should match the run being resumed)
    "simclr_s": 1.0, "simclr_p_grayscale": 0.2, "simclr_p_gaussian_blur": 0.5, "simclr_rrc_scale_min": 0.08,

    # Linear Probing Configuration
    "evaluate_every_n_epochs": 10,
    "linear_probe_epochs": 10,      # Or 20, as per your original run
    "linear_probe_lr": 0.1,
    "probe_optimizer": "SGD", "probe_momentum": 0.9, "probe_weight_decay": 0.0,
    "probe_batch_size": 64,         # Keep as it was for T4

    # Checkpointing
    "save_every_n_epochs": 20,      # Or your original save frequency
    "model_arch_name_for_ckpt": "hvt_xl_simclr_t4_resumed", # New name for checkpoints from this resumed run
    "clip_grad_norm": 1.0,
}

# --- Logging after config dict is defined ---
_config_logger = logging.getLogger(__name__)
_config_logger.info(f"Phase 3 Configuration (Resuming on T4 GPU) Loaded from: {__file__}")
_config_logger.info(f"Resume Checkpoint Path: {config.get('resume_checkpoint_path')}")
_config_logger.info(f"Target Total Pre-training Epochs: {config['pretrain_epochs']}")
_config_logger.info(f"Device: {DEVICE}, Effective Batch Size: {config['pretrain_batch_size'] * config['accumulation_steps']}")
_config_logger.info(f"Gradient Checkpointing for HVT Backbone: {config['hvt_params_for_backbone']['use_gradient_checkpointing']}")