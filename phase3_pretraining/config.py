# phase3_pretraining/config.py
import torch
import logging
import os

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs" # Relative to package_root (phase3_pretraining)
PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # Make sure this is correct
DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
CHECKPOINT_DIR = "pretrain_checkpoints_h100_xl" # Relative to PROJECT_ROOT_PATH

# --- PyTorch Performance Settings for H100 ---
ENABLE_TORCH_COMPILE = False         # ENSURE THIS IS TRUE AND WORKING (PyTorch 2.0+ needed)
TORCH_COMPILE_MODE = "reduce-overhead" # "max-autotune" for best runtime after longer compile, "reduce-overhead" for faster compile
MATMUL_PRECISION = 'high'         # 'high' or 'highest' for TF32 on H100
CUDNN_BENCHMARK = True

config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "PROJECT_ROOT_PATH": PROJECT_ROOT_PATH, # Store for consistent path construction
    "log_dir": LOG_DIR,
    "log_file_pretrain": "phase3_pretrain_hvt_xl_h100_lr1e-4_aug_v2.log", # Reflect new LR and aug change
    "checkpoint_dir": CHECKPOINT_DIR,
    "enable_torch_compile": ENABLE_TORCH_COMPILE,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "matmul_precision": MATMUL_PRECISION,
    "cudnn_benchmark": CUDNN_BENCHMARK,

    # Dataset Config
    "data_root": DATASET_BASE_PATH,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.95,
    "num_classes": 7,
    "num_workers": 32, # Keep high for data loading if torch.compile works
    "prefetch_factor": 4,

    # --- HVT-XL Model Configurations ---
    "model_name": "DiseaseAwareHVT_XL_H100_Prod_lr1e-4_aug_v2", # Reflect new LR and aug change
    "use_gradient_checkpointing": True,
    "hvt_patch_size": 14,
    "hvt_embed_dim_rgb": 192,
    "hvt_embed_dim_spectral": 192,
    "hvt_spectral_channels": 3,
    "hvt_depths": [3, 6, 24, 3],
    "hvt_num_heads": [6, 12, 24, 48],
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.0,
    "hvt_attn_drop_rate": 0.0,
    "hvt_drop_path_rate": 0.2,
    "hvt_use_dfca": True,
    "hvt_dfca_heads": 32,
    "dfca_drop_rate": 0.1,
    "dfca_use_disease_mask": True,
    "ssl_enable_mae": False, # As per current SimCLR setup
    "ssl_enable_contrastive": False, # As per current SimCLR setup
    "ssl_mae_mask_ratio": 0.75, # Default, not used by SimCLR wrapper
    "ssl_mae_decoder_dim": 64,  # Default, not used by SimCLR wrapper
    "ssl_mae_norm_pix_loss": True, # Default, not used by SimCLR wrapper
    "ssl_contrastive_projector_dim": 128, # Default, not used by SimCLR wrapper
    "ssl_contrastive_projector_depth": 2, # Default, not used by SimCLR wrapper
    "enable_consistency_loss_heads": False, # Default, not used by SimCLR wrapper


    # --- Pre-training Specific (SimCLR) ---
    "pretrain_img_size": (448, 448),
    "pretrain_epochs": 160, # Observe for at least 30-50 epochs with new settings
    "pretrain_batch_size": 64,
    "accumulation_steps": 1,
    "pretrain_lr": 1e-4,  # Primary Change: Significantly Reduced Learning Rate
    "pretrain_optimizer": "AdamW",
    "pretrain_scheduler": "WarmupCosine",
    "warmup_epochs": 10, # Keep warmup; helps stabilize early training with new LR
    "pretrain_weight_decay": 0.05,
    "temperature": 0.1, # Standard, can be tuned later if needed
    "projection_dim": 256,
    "projection_hidden_dim": 4096,

    # --- Pre-training Augmentations ---
    # These values are used by SimCLRAugmentation class if it reads from config,
    # or directly if passed. The SimCLRAugmentation class itself has defaults.
    # We will adjust RandomResizedCrop directly in augmentations.py for this experiment.
    "augmentations_enabled": True,
    "simclr_s": 1.0, # Color jitter strength (standard)
    "simclr_p_grayscale": 0.2,
    "simclr_p_gaussian_blur": 0.5,
    # Adding a key for RRC scale to be potentially used by augmentations.py if we modify it to read from config
    "simclr_rrc_scale_min": 0.08, # More aggressive cropping, as in original SimCLR paper

    # --- Linear Probing ---
    "linear_probe_epochs": 20,
    "linear_probe_lr": 0.1,
    "probe_optimizer": "SGD",
    "probe_momentum": 0.9,
    "probe_batch_size": 256,

    # --- Checkpointing ---
    "evaluate_every_n_epochs": 10,
    "save_every_n_epochs": 20,
    "pretrain_final_checkpoint_name": "hvt_xl_h100_prod_lr1e-4_aug_v2_final.pth", # Reflect changes
    "clip_grad_norm": 1.0,
}

# Ensure checkpoint directory exists (can be created by pretrain.py too)
_abs_checkpoint_dir = config["checkpoint_dir"]
if not os.path.isabs(_abs_checkpoint_dir):
    _abs_checkpoint_dir = os.path.join(config["PROJECT_ROOT_PATH"], config["checkpoint_dir"])

if not os.path.exists(_abs_checkpoint_dir):
    os.makedirs(_abs_checkpoint_dir, exist_ok=True)
    # This print might be redundant if pretrain.py also logs directory creation
    # print(f"Config.py: Checkpoint directory created/ensured at {_abs_checkpoint_dir}")