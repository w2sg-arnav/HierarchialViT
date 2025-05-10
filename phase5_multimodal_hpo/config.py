# phase5_multimodal_hpo/config.py
import torch
import logging
import os

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
# --- IMPORTANT: VERIFY THIS BASE PATH ---
PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # Define project root explicitly
# --- ---
DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
PRETRAIN_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT_PATH, "pretrain_checkpoints") # Absolute path
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT_PATH, "phase5_multimodal_hpo", "phase5_checkpoints") # Absolute path

# --- Configuration Dictionary ---
config = {
    # General Reproducibility & Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_finetune": "phase5_finetune_pt_load_test.log", # New log file name
    "best_model_path": os.path.join(CHECKPOINT_DIR, "phase5_best_pt_test.pth"), 
    "final_model_path": os.path.join(CHECKPOINT_DIR, "phase5_final_pt_test.pth"), 
    "checkpoint_dir": CHECKPOINT_DIR, 

    # Dataset Config
    "data_root": DATASET_BASE_PATH,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.8, 
    "num_classes": 7, 
    "num_workers": 8, # <<<--- ADJUSTED based on warning

    # Model Configurations
    "model_name": "DiseaseAwareHVT", 
    # --- IMPORTANT: VERIFY this checkpoint exists ---
    "pretrained_checkpoint_path": os.path.join(PRETRAIN_CHECKPOINT_DIR, "hvt_pretrain_epoch_final.pth"), 
    "load_pretrained": True, # Ensure this is True

    # Core HVT parameters (MUST match pre-trained model's backbone structure)
    "hvt_patch_size": 16, "hvt_embed_dim_rgb": 96, "hvt_embed_dim_spectral": 96, 
    "hvt_spectral_channels": 1, "hvt_depths": [2, 2, 6, 2], "hvt_num_heads": [3, 6, 12, 24],
    "hvt_mlp_ratio": 4.0, "hvt_qkv_bias": True, "hvt_model_drop_rate": 0.0,   
    "hvt_attn_drop_rate": 0.0, "hvt_drop_path_rate": 0.1, # Moderate drop path
    "hvt_use_dfca": True,         

    # Fine-tuning Specific
    "img_size": (384, 384), 
    "epochs": 60, # Can reduce for this test run, e.g., 30
    "batch_size": 32, 
    "accumulation_steps": 1, 
    "amp_enabled": True, 
    "clip_grad_norm": 1.0, 
    "log_interval": 50, 
    "normalize_data": True, 

    # Optimizer & Freeze/Unfreeze Strategy
    "optimizer": "AdamW", 
    "head_lr": 1e-3, # Not used if freeze_backbone_epochs=0
    "learning_rate": 5e-5, # <<<--- Start fine-tuning LR here (adjust based on results)
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)}, 
    "freeze_backbone_epochs": 0, # <<<--- DISABLED freeze phase
    "use_llrd": True, 
    "llrd_rate": 0.9, 

    # Schedulers
    "scheduler": "CosineAnnealingWarmRestarts", 
    "warmup_epochs": 5, 
    "warmup_lr_init_factor": 0.1,
    "cosine_t_0": 15, 
    "cosine_t_mult": 1,
    "eta_min": 1e-6, 
    "reducelr_factor": 0.2, 
    "reducelr_patience": 7, 

    # Loss Function
    "loss_label_smoothing": 0.1, # Keep moderate label smoothing
    "use_weighted_sampler": True, 

    # Augmentations (Applied in Trainer)
    "augmentations_enabled": True,
    "mixup_alpha": 0.0, # <<<--- Disabled Mixup
    "cutmix_alpha": 0.0, # <<<--- Disabled Cutmix
    "mixup_cutmix_prob": 0.0, # <<<--- Disabled 

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1, 
    "early_stopping_patience": 15, 
    "metric_to_monitor": 'f1_weighted', 
    "tta_enabled": True, 
    "tta_augmentations": ['hflip', 'vflip'], 

    # Multi-Scale Training DISABLED for now
    "multi_scale_training_epoch_interval": 0, 
    "multi_scale_training_factors": [0.8, 1.0, 1.1], 
    "multi_scale_max_factor_after_unfreeze": 1.0,

    # HPO Configs (remain for reference)
    "hpo_enabled": False, 

    # Regularization
    "ema_decay": 0.999,

    # Schedulers
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 15,
    "warmup_lr_init_factor": 0.01,
    "min_lr": 1e-6,
    "cosine_t_0": 10,
    "cosine_t_mult": 2,
    "reducelr_factor": 0.2,
    "reducelr_patience": 10,

    # Evaluation
    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 25,
    "metric_to_monitor": 'accuracy',
    "tta_enabled": True,
    "tta_augmentations": ['hflip', 'vflip', 'rotate'],

    # HPO
    "hpo_enabled": False,
    "hpo_n_trials": 150,
    "hpo_study_name": "disease_hvt_xl_phase5_hpo",
    "hpo_storage_db": "sqlite:///hpo_phase5_study.db",
    "hpo_params_ranges": {
        "learning_rate": (1e-5, 1e-3, "log"),
        "weight_decay": (1e-3, 1e-1, "log"),
        "llrd_rate": (0.7, 0.95, "uniform"),
        "mixup_alpha": (0.1, 0.8, "uniform"),
        "label_smoothing": (0.0, 0.2, "uniform"),
        "hvt_drop_path_rate": (0.1, 0.5, "uniform"),
        "focal_gamma": (1.0, 3.0, "uniform"),
    }
}

if not os.path.exists(CHECKPOINT_DIR_BASE):
    os.makedirs(CHECKPOINT_DIR_BASE, exist_ok=True)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)