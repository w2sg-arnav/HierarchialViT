# phase4_finetuning/config.py
import os
import logging
import torch
# Import base configurations from Phase 3 (which should include Phase 2 settings)
# Ensure the path is correct relative to where python -m is run (e.g., from cvpr25 root)
try:
    from phase3_pretraining.config import * 
except ImportError:
    print("Warning: Could not import config from phase3_pretraining. Define defaults here.")
    # Define necessary fallback defaults if import fails
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    ORIGINAL_DATASET_NAME = "Original Dataset"
    AUGMENTED_DATASET_NAME = "Augmented Dataset"
    TRAIN_SPLIT_RATIO = 0.8
    NUM_CLASSES = 7
    PRETRAIN_CHECKPOINT_DIR = "pretrain_checkpoints" # Needed to find default pretrained path

# --- Fine-tuning Specific Configurations ---
# Define parameters specific to the fine-tuning stage.
# These can be overridden by a YAML config file via command line args in main.py.

config = {
    # General
    "seed": RANDOM_SEED if 'RANDOM_SEED' in globals() else 42,
    "device": DEVICE if 'DEVICE' in globals() else ("cuda" if torch.cuda.is_available() else "cpu"),
    "log_dir": "logs",
    "log_file_finetune": "finetune.log",
    "best_model_path": "best_finetuned_hvt.pth", # Where to save the best model
    "final_model_path": "final_finetuned_hvt.pth", # Where to save the final model

    # Data
    "data_root": DATASET_BASE_PATH if 'DATASET_BASE_PATH' in globals() else "/path/to/dataset",
    "img_size": (384, 384), # Target image size for fine-tuning (e.g., largest progressive size)
    "num_classes": NUM_CLASSES if 'NUM_CLASSES' in globals() else 7,
    "train_split": TRAIN_SPLIT_RATIO if 'TRAIN_SPLIT_RATIO' in globals() else 0.8,
    "normalize_data": True, # Apply ImageNet normalization for fine-tuning
    "use_weighted_sampler": True, # Address class imbalance

    # Model
    "model_name": "DiseaseAwareHVT", # Identifier for the model to load/use
    "pretrained_checkpoint_path": os.path.join(PRETRAIN_CHECKPOINT_DIR if 'PRETRAIN_CHECKPOINT_DIR' in globals() else "pretrain_checkpoints", 
                                               "hvt_pretrain_epoch_final.pth"), # Path to Phase 3 backbone weights
    "load_pretrained": True, # Flag to load weights from the path above
    # HVT specific params (should match Phase 2/3 config ideally, ensure consistency)
    "hvt_patch_size": 16,
    "hvt_embed_dim_rgb": 96,
    "hvt_embed_dim_spectral": 96,
    "hvt_spectral_channels": 1,
    "hvt_depths": [2, 2, 6, 2],
    "hvt_num_heads": [3, 6, 12, 24],
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.0, # Dropout during fine-tuning (can differ from pretrain)
    "hvt_attn_drop_rate": 0.0,
    "hvt_drop_path_rate": 0.1, # Stochastic depth during fine-tuning
    "hvt_use_dfca": True,
    # Add other model params if comparing (e.g., effnet_variant)

    # Training Loop
    "epochs": 30,
    "batch_size": 16, # Adjust based on GPU memory for fine-tuning image size
    "accumulation_steps": 2, # Accumulate gradients over 2 steps (effective batch size = 16*2=32)
    "amp_enabled": True, # Use Automatic Mixed Precision
    "clip_grad_norm": 1.0, 
    "log_interval": 50, # Log training loss every N batches

    # Optimizer
    "optimizer": "AdamW", # Options: AdamW, SGD, etc.
    "learning_rate": 1e-4, # Starting LR for fine-tuning (often lower than pretrain)
    "weight_decay": 0.05,
    "optimizer_params": {}, # Extra params like betas for AdamW can go here if needed

    # Schedulers
    "scheduler": "CosineAnnealingWarmRestarts", # Options: CosineAnnealingWarmRestarts, ReduceLROnPlateau, StepLR, None
    "warmup_epochs": 3,
    "warmup_lr_init_factor": 0.1,
    "cosine_t_0": 10, # Restart interval for CosineAnnealingWarmRestarts
    "cosine_t_mult": 1,
    "eta_min": 1e-6, # Minimum LR for Cosine
    "reducelr_factor": 0.2, # Factor for ReduceLROnPlateau
    "reducelr_patience": 5, # Patience for ReduceLROnPlateau

    # Loss Function
    "loss_label_smoothing": 0.1,

    # Augmentations
    "augmentations_enabled": True,
    # Add specific augmentation parameters here if needed

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1, # Evaluate on validation set every epoch
    "early_stopping_patience": 10 # Stop if validation metric doesn't improve for N epochs
}

# Helper to ensure os is imported if needed by path joins
import os 