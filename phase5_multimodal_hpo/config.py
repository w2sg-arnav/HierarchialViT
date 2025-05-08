import torch
import logging
import os

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
PRETRAIN_CHECKPOINT_DIR_REL = "pretrain_checkpoints"
CHECKPOINT_DIR = "phase5_checkpoints"

# --- Enhanced Configuration ---
config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_finetune": "phase5_high_acc.log",
    "best_model_path": os.path.join(CHECKPOINT_DIR, "phase5_best_90percent.pth"),
    "final_model_path": os.path.join(CHECKPOINT_DIR, "phase5_final_90percent.pth"),
    
    # Dataset
    "data_root": DATASET_BASE_PATH,
    "train_split_ratio": 0.85,  # More training data
    "num_classes": 7,
    "num_workers": 12,  # Faster data loading
    
    # Enhanced Model Architecture
    "model_name": "DiseaseAwareHVT_XL",
    "pretrained_checkpoint_path": os.path.join("..", PRETRAIN_CHECKPOINT_DIR_REL, "hvt_xl_pretrain.pth"),
    "load_pretrained": True,
    
    # XL Architecture Parameters
    "hvt_patch_size": 14,
    "hvt_embed_dim_rgb": 128,
    "hvt_embed_dim_spectral": 128,
    "hvt_spectral_channels": 3,  # Use 3-channel spectral
    "hvt_depths": [3, 6, 18, 3],  # Deeper architecture
    "hvt_num_heads": [4, 8, 16, 32],
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.2,
    "hvt_attn_drop_rate": 0.1,
    "hvt_drop_path_rate": 0.3,  # Increased stochastic depth
    "hvt_use_dfca": True,
    "hvt_dfca_heads": 16,  # More attention heads
    
    # Training Params
    "img_size": (512, 512),  # Higher resolution
    "epochs": 200,  # Longer training
    "batch_size": 64,  # Larger batch
    "accumulation_steps": 2,
    "amp_enabled": True,
    "clip_grad_norm": 3.0,
    
    # Optimizer
    "optimizer": "LAMB",
    "learning_rate": 3e-4,
    "weight_decay": 0.02,
    "layer_decay": 0.75,  # For LAMB optimizer
    
    # Advanced Training
    "freeze_backbone_epochs": 10,
    "use_llrd": True,
    "llrd_rate": 0.85,
    
    # Loss Function
    "loss_type": "focal",  # Focal loss
    "focal_gamma": 2.0,
    "label_smoothing": 0.15,
    
    # Augmentations
    "augmentations_enabled": True,
    "mixup_alpha": 0.4,
    "cutmix_alpha": 0.4,
    "mixup_cutmix_prob": 0.7,
    "random_erase_prob": 0.3,
    
    # Regularization
    "ema_decay": 0.999,  # EMA model
    "auto_augment": True,
    
    # Schedulers
    "scheduler": "CosineAnnealingLR",
    "warmup_epochs": 15,
    "warmup_lr_init_factor": 0.01,
    "min_lr": 1e-6,
    
    # Evaluation
    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 25,
    "metric_to_monitor": 'accuracy',
    "tta_enabled": True,
    "tta_augmentations": ['hflip', 'vflip', 'rotate'],
    
    # HPO
    "hpo_enabled": False,
    "hpo_n_trials": 150,
}