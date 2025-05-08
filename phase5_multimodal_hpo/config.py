# phase5_multimodal_hpo/config.py
import torch
import logging
import os

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
PRETRAIN_CHECKPOINT_DIR_REL = "pretrain_checkpoints" 
CHECKPOINT_DIR = "phase5_checkpoints" # Specific checkpoints for this phase

# --- Configuration Dictionary ---
config = {
    # General Reproducibility & Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_finetune": "phase5_finetune_high_acc.log", # Log for this advanced run
    "best_model_path": os.path.join(CHECKPOINT_DIR, "phase5_best_multimodal_hvt_high_acc.pth"), 
    "final_model_path": os.path.join(CHECKPOINT_DIR, "phase5_final_multimodal_hvt_high_acc.pth"), 
    "checkpoint_dir": CHECKPOINT_DIR, 

    # Dataset Config
    "data_root": DATASET_BASE_PATH,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.8, 
    "num_classes": 7, 
    "num_workers": 8, # Adjusted based on warning/H100 capability

    # Model Configurations
    "model_name": "DiseaseAwareHVT", 
    "pretrained_checkpoint_path": os.path.join("..", PRETRAIN_CHECKPOINT_DIR_REL, "hvt_pretrain_epoch_final.pth"), # Path relative to project root
    "load_pretrained": True, 

    # Core HVT parameters (ensure match pre-trained model's backbone structure)
    "hvt_patch_size": 16,
    "hvt_embed_dim_rgb": 96,      
    "hvt_embed_dim_spectral": 96, 
    "hvt_spectral_channels": 1, 
    "hvt_depths": [2, 2, 6, 2], # Example: Swin-T depth
    "hvt_num_heads": [3, 6, 12, 24], # Example: Swin-T heads
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.0,   # Model dropout during fine-tuning
    "hvt_attn_drop_rate": 0.0,    # Attention dropout
    "hvt_drop_path_rate": 0.2,    # Stochastic depth rate
    "hvt_use_dfca": True,         # Use Disease Focused Cross Attention

    # Fine-tuning Specific
    "img_size": (384, 384), 
    "epochs": 60, # Total epochs for fine-tuning
    "batch_size": 32, # Increased for H100 (adjust if OOM)
    "accumulation_steps": 1, # Reduced (due to larger base batch size)
    "amp_enabled": True, 
    "clip_grad_norm": 1.0, 
    "log_interval": 50, 
    "normalize_data": True, # Use ImageNet normalization

    # Optimizer & Freeze/Unfreeze Strategy
    "optimizer": "AdamW", 
    "head_lr": 1e-3, # LR for training ONLY the head initially
    "learning_rate": 3e-5, # Base LR for fine-tuning the whole model (after head training)
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)}, 
    "freeze_backbone_epochs": 5, # Number of epochs to train only the head
    "use_llrd": True, # Enable Layer-wise Learning Rate Decay after unfreezing
    "llrd_rate": 0.9, # Decay rate (less aggressive)

    # Schedulers
    "scheduler": "CosineAnnealingWarmRestarts", 
    "warmup_epochs": 5, # Warmup applied over first few epochs (incl. freeze phase)
    "warmup_lr_init_factor": 0.1,
    # T_0 should reflect cycle length *after* warmup and freeze phase
    "cosine_t_0": 15, # Restart cosine cycle every 15 epochs (adjust based on total unfreeze epochs)
    "cosine_t_mult": 1,
    "eta_min": 1e-6, 
    "reducelr_factor": 0.2, # For ReduceLROnPlateau if used instead
    "reducelr_patience": 7, 

    # Loss Function
    "loss_label_smoothing": 0.1, # Re-enabled
    "use_weighted_sampler": True, 

    # Augmentations (Applied in Trainer)
    "augmentations_enabled": True,
    "mixup_alpha": 0.2, # Re-enabled Mixup (start lower)
    "cutmix_alpha": 1.0, # Re-enabled Cutmix
    "mixup_cutmix_prob": 0.5, # Probability of applying Mixup OR Cutmix

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1, 
    "early_stopping_patience": 20, # More patience
    "metric_to_monitor": 'f1_weighted', 
    "tta_enabled": True, 
    "tta_augmentations": ['hflip', 'vflip'], # TTA views

    # Hyperparameter Optimization (HPO) - Not used when running main.py directly
    "hpo_enabled": False, 
    "hpo_n_trials": 50, 
    "hpo_study_name": "hvt_finetune_hpo_adv", 
    "hpo_storage_db": "sqlite:///hpo_study_adv.db", 
    "hpo_lr_low": 1e-5, "hpo_lr_high": 8e-5, 
    "hpo_wd_low": 0.01, "hpo_wd_high": 0.1,
    "hpo_label_smoothing_low": 0.0, "hpo_label_smoothing_high": 0.2,
    "hpo_llrd_rate_low": 0.7, "hpo_llrd_rate_high": 0.95, 
    "hpo_mixup_alpha_low": 0.1, "hpo_mixup_alpha_high": 1.0, 
    "hpo_cutmix_alpha_low": 0.1, "hpo_cutmix_alpha_high": 1.0, 
}