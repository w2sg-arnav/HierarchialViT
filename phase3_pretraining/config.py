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
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT_PATH, "pretrain_checkpoints_h100_xl") # Specific for this config

# --- PyTorch Performance Settings for H100 ---
ENABLE_TORCH_COMPILE = False         # Highly recommended for PyTorch 2.0+ on H100
TORCH_COMPILE_MODE = "reduce-overhead" # "max-autotune" for best runtime after longer compile, "reduce-overhead" for faster compile
MATMUL_PRECISION = 'high'         # 'high' or 'highest' for TF32 on H100
CUDNN_BENCHMARK = True

config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_pretrain": "phase3_pretrain_hvt_xl_h100_final.log", # Log for this specific run
    "checkpoint_dir": CHECKPOINT_DIR,
    "enable_torch_compile": ENABLE_TORCH_COMPILE,
    "torch_compile_mode": TORCH_COMPILE_MODE,
    "matmul_precision": MATMUL_PRECISION,
    "cudnn_benchmark": CUDNN_BENCHMARK,

    # Dataset Config
    "data_root": DATASET_BASE_PATH,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.95, # Use a large portion for pre-training
    "num_classes": 7,          # For linear probe and potential downstream tasks
    "num_workers": 32,         # START HERE for H100. Monitor CPU. Can go to 48, 64, 96 depending on server CPU.
    "prefetch_factor": 4,      # Good with higher num_workers.

    # --- HVT-XL Model Configurations (Target "Good" Large Model for H100) ---
    "model_name": "DiseaseAwareHVT_XL_H100_Prod", # Production HVT-XL for H100
    "use_gradient_checkpointing": True, # RECOMMENDED for HVT-XL to save memory, allows larger batches/models
                                        # If disabled, training will be faster per step but use more memory.

    "hvt_patch_size": 14,                 # Results in 32x32 patches for 448px images
    "hvt_embed_dim_rgb": 192,             # Increased from 128 for a larger model (e.g., ~ViT-Base/Large embed start)
    "hvt_embed_dim_spectral": 192,        # Match RGB embed dim
    "hvt_spectral_channels": 3,           # As per previous successful run
    "hvt_depths": [3, 6, 24, 3],          # Deeper configuration (e.g., similar to Swin-B/L ideas)
                                          # Total blocks: 3+6+24+3 = 36
    "hvt_num_heads": [6, 12, 24, 48],     # Heads: embed_dim / 64 or / 32. Here, 192/6=32, 384/12=32, 768/24=32, 1536/48=32
                                          # Adjusted to common head dimension of 32 or 64.
                                          # Let's go with embed_dim / num_heads = 32 as a common practice.
                                          # Stage 0: 192 / 6 = 32
                                          # Stage 1: 384 / 12 = 32
                                          # Stage 2: 768 / 24 = 32
                                          # Stage 3: 1536 / 48 = 32
    "hvt_mlp_ratio": 4.0,                 # Standard MLP ratio
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.0,           # No dropout on embeddings/MLPs during SSL usually
    "hvt_attn_drop_rate": 0.0,            # No attention dropout during SSL usually
    "hvt_drop_path_rate": 0.2,            # Stochastic depth, can be beneficial (0.1-0.3 range)

    # DFCA (Ensure these are aligned with what hvt.py expects if merging)
    "hvt_use_dfca": True,
    "hvt_dfca_heads": 32, # e.g., embed_dim_final_rgb (1536) / 48, or a fixed number like 16 or 24.
                         # Let's make it match the last stage's head count for consistency if it uses final embed dim.
                         # Final RGB encoded dim before DFCA would be 192 * 2^3 = 1536.
                         # If DFCA operates on this: 1536 / num_dfca_heads = head_dim.
                         # Let's use a reasonable number, e.g., 24 or 32 heads.
                         # The DFCA in phase2_model uses embed_dim of the *fused* stream, which is final_encoded_dim_rgb.
                         # So, 1536 / 32 heads = 48 head_dim. Or 1536 / 24 heads = 64 head_dim. Let's use 24.
    "dfca_drop_rate": 0.1,                # Dropout within DFCA
    "dfca_use_disease_mask": True,        # As per your dfca.py

    # SSL flags for the base DiseaseAwareHVT (usually False if wrapper handles SSL)
    # These will be picked up by hvt.py's cfg_module merging if not overridden by phase3_config
    "ssl_enable_mae": False, # Base HVT's MAE not used in SimCLR pre-training via wrapper
    "ssl_enable_contrastive": False, # Base HVT's projector not used

    # --- Pre-training Specific (SimCLR) ---
    "pretrain_img_size": (448, 448),        # Good size for HVT-XL
    "pretrain_epochs": 160,                 # SSL often needs many epochs (100-800)
    "pretrain_batch_size": 64,             # Target this for H100 80GB. Adjust if OOM. Maximize this.
                                            # Effective batch size is pretrain_batch_size * accumulation_steps
    "accumulation_steps": 1,                # Try 1 if batch_size is large enough. Increase if OOM with bs=64.
                                            # If bs=32, accum=2. If bs=16, accum=4 for effective_bs=64.
    "pretrain_lr": 1e-3,                    # Starting LR for AdamW, adjust with batch size (sqrt scaling or linear)
                                            # For SimCLR, LARS optimizer with LR like 0.3 * batch_size / 256 is common.
    "pretrain_optimizer": "AdamW",          # "AdamW" or "LARS" (if using very large effective batches)
    "pretrain_scheduler": "WarmupCosine",   # "WarmupCosine" or "CosineAnnealingLR"
    "warmup_epochs": 10,                    # For "WarmupCosine" scheduler
    "pretrain_weight_decay": 0.05,          # For AdamW
    "temperature": 0.1,                     # InfoNCE temperature
    "projection_dim": 256,                  # Output of SimCLR projection head
    "projection_hidden_dim": 4096,          # Hidden layer in SimCLR projection head (often larger, e.g., 2048 or 4096)

    # --- Pre-training Augmentations ---
    "augmentations_enabled": True,
    "simclr_s": 1.0, "simclr_p_grayscale": 0.2, "simclr_p_gaussian_blur": 0.5,

    # --- Linear Probing ---
    "linear_probe_epochs": 20,              # More epochs for probe on a larger model
    "linear_probe_lr": 0.1,                # Higher LR common for linear probe (SGD with momentum)
    "probe_optimizer": "SGD",               # SGD often used for linear probe
    "probe_momentum": 0.9,
    "probe_batch_size": 256,                # Can often use larger batch for linear probe

    # --- Checkpointing ---
    "evaluate_every_n_epochs": 10,          # Evaluate probe every 10-20 epochs
    "save_every_n_epochs": 20,              # Save checkpoints periodically
    "pretrain_final_checkpoint_name": "hvt_xl_h100_prod_final.pth",
    "clip_grad_norm": 1.0,                  # Optional: gradient clipping
}

# Create checkpoint directory if it doesn't exist
if not os.path.exists(config["checkpoint_dir"]):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

# Optional: Log that this specific config is loaded
# This logging might be duplicated if pretrain.py also logs.
# Can be useful for confirming which config file is active.
# import logging # Already imported at top
# logger = logging.getLogger(__name__)
# if not logger.hasHandlers(): # Avoid configuring root logger multiple times
#    logging.basicConfig(level=logging.INFO)
# logger.info(f"Phase 3 HVT-XL Production Configuration Loaded (config.py). Model: {config['model_name']}")