import torch
import os

# --- Core Settings ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
PRETRAIN_CHECKPOINT_DIR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrain_checkpoints"))
CHECKPOINT_DIR_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "phase5_checkpoints"))


# --- Enhanced Configuration ---
config = {
    # General Setup
    "seed": RANDOM_SEED,
    "device": DEVICE,
    "log_dir": LOG_DIR,
    "log_file_finetune": "phase5_high_acc.log",
    "checkpoint_dir": CHECKPOINT_DIR_BASE,
    "best_model_path": os.path.join(CHECKPOINT_DIR_BASE, "phase5_best_90percent.pth"),
    "final_model_path": os.path.join(CHECKPOINT_DIR_BASE, "phase5_final_90percent.pth"),
    "resume_from_best": False, # Set to True to try resuming from best_model_path if it exists
    "resume_checkpoint_path": None, # Or specify a path to a specific checkpoint to resume from

    # Dataset
    "data_root": DATASET_BASE_PATH,
    "train_split_ratio": 0.85,
    "num_classes": 7,
    "num_workers": 8,  # Reduced from 12 based on previous warning, adjust as needed
    "normalize_data": True,

    # Enhanced Model Architecture
    "model_name": "DiseaseAwareHVT_XL",
    "pretrained_checkpoint_path": os.path.join(PRETRAIN_CHECKPOINT_DIR_BASE, "hvt_xl_pretrain.pth"),
    "load_pretrained": True, # Set to False if you want to train from scratch (e.g. if checkpoint is missing)
    "use_gradient_checkpointing": True, # NEW: Enable/disable gradient checkpointing in the model

    # XL Architecture Parameters
    "hvt_patch_size": 14,
    "hvt_embed_dim_rgb": 128,
    "hvt_embed_dim_spectral": 128,
    "hvt_spectral_channels": 3,
    "hvt_depths": [3, 6, 18, 3],
    "hvt_num_heads": [4, 8, 16, 32],
    "hvt_mlp_ratio": 4.0,
    "hvt_qkv_bias": True,
    "hvt_model_drop_rate": 0.2,
    "hvt_attn_drop_rate": 0.1,
    "hvt_drop_path_rate": 0.3,
    "hvt_use_dfca": True,
    "hvt_dfca_heads": 16,

    # Training Params
    "img_size": (448, 448),  # MODIFIED: Reduced from (512, 512)
    "epochs": 200,
    "batch_size": 32,  # MODIFIED: Reduced from 64
    "accumulation_steps": 4, # MODIFIED: Increased from 2
    "amp_enabled": True,
    "clip_grad_norm": 3.0,
    "log_interval": 50,

    # Optimizer
    "optimizer": "LAMB",
    "learning_rate": 3e-4,
    "weight_decay": 0.02,
    "optimizer_params": {"betas": (0.9, 0.999)},
    "layer_decay": 0.75,

    # Advanced Training Strategies
    "freeze_backbone_epochs": 10,
    "head_lr_factor": 10.0,
    "use_llrd": True,
    "llrd_rate": 0.85,

    # Loss Function
    "loss_type": "focal",
    "focal_alpha": None,
    "focal_gamma": 2.0,
    "label_smoothing": 0.15,
    "use_weighted_sampler": False,
    "use_weighted_loss": True,

    # Augmentations
    "augmentations_enabled": True,
    "rand_augment_num_ops": 3,
    "rand_augment_magnitude": 12,
    "color_jitter_params": {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
    "random_erase_prob": 0.3,
    "mixup_alpha": 0.4,
    "cutmix_alpha": 0.4,
    "mixup_cutmix_prob": 0.7,
    "auto_augment_policy": None,
    # Multi-scale training params (used in Finetuner)
    "multi_scale_training_epoch_interval": 10,
    "multi_scale_training_factors": [0.8, 1.0, 1.1], # MODIFIED: Reduced max scale factor slightly
    "multi_scale_max_factor_after_unfreeze": 1.0, # NEW: Cap max scale after unfreeze

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