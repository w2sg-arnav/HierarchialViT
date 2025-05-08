# phase5_multimodal_hpo/config.py
import torch
import logging
import os

# --- Core Settings (can be accessed directly if needed) ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
PRETRAIN_CHECKPOINT_DIR_REL = "pretrain_checkpoints" # Relative path for default

# --- Configuration Dictionary ---
# All parameters that scripts should use are defined within this dictionary.
config = {
    # General Reproducibility & Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_finetune": "phase5_finetune.log",
    "best_model_path": "phase5_best_multimodal_hvt.pth", 
    "final_model_path": "phase5_final_multimodal_hvt.pth", 
    "checkpoint_dir": "phase5_checkpoints", 

    # Dataset Config
    "data_root": DATASET_BASE_PATH,
    "original_dataset_name": "Original Dataset",
    "augmented_dataset_name": "Augmented Dataset",
    "train_split_ratio": 0.8, 
    "num_classes": 7, 

    # Model Configurations
    "model_name": "DiseaseAwareHVT", 
    # Path to backbone weights PRE-TRAINED in Phase 3
    "pretrained_checkpoint_path": os.path.join("..", PRETRAIN_CHECKPOINT_DIR_REL, "hvt_pretrain_epoch_final.pth"), 
    "load_pretrained": True, 

    # Core HVT parameters 
    "hvt_patch_size": 16,
    "hvt_embed_dim_rgb": 96,      
    "hvt_embed_dim_spectral": 96, 
    "hvt_spectral_channels": 1, 
    "hvt_depths": [2, 2, 6, 2],
    "hvt_num_heads": [3, 6, 12, 24],
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.0,   
    "hvt_attn_drop_rate": 0.0,    
    "hvt_drop_path_rate": 0.1,    
    "hvt_use_dfca": True, 

    # Fine-tuning Specific
    "img_size": (384, 384), 
    "epochs": 30,
    "batch_size": 16, 
    "accumulation_steps": 2, 
    "amp_enabled": True, 
    "clip_grad_norm": 1.0, 
    "log_interval": 50, 
    "normalize_data": True, # Apply ImageNet normalization for fine-tuning

    # Optimizer
    "optimizer": "AdamW", 
    "learning_rate": 5e-5, 
    "weight_decay": 0.05,
    "optimizer_params": {}, 

    # Schedulers
    "scheduler": "CosineAnnealingWarmRestarts", 
    "warmup_epochs": 3,
    "warmup_lr_init_factor": 0.1,
    "cosine_t_0": 10, 
    "cosine_t_mult": 1,
    "eta_min": 1e-6, 
    "reducelr_factor": 0.2, 
    "reducelr_patience": 5, 

    # Loss Function
    "loss_label_smoothing": 0.1,
    "use_weighted_sampler": True, 

    # Augmentations
    "augmentations_enabled": True,

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1, 
    "early_stopping_patience": 10,
    "metric_to_monitor": 'f1_weighted', 

    # Hyperparameter Optimization (HPO)
    "hpo_enabled": False, 
    "hpo_n_trials": 50, 
    "hpo_study_name": "hvt_finetune_hpo",
    "hpo_storage_db": "sqlite:///hpo_study.db", 
    # Ranges for HPO
    "hpo_lr_low": 1e-5,
    "hpo_lr_high": 1e-4,
    "hpo_wd_low": 0.01,
    "hpo_wd_high": 0.1,
    "hpo_label_smoothing_low": 0.0,
    "hpo_label_smoothing_high": 0.2,
}

# --- Optional: Setup basic logging config here if run standalone ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')