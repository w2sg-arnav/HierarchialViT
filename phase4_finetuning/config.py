# phase4_finetuning/config.py
import os
import logging
import torch

# --- Attempt to Import base configurations from Phase 3 ---
# This makes Phase 4 config inherit and potentially override Phase 3 settings.
# It assumes phase3_pretraining is in PYTHONPATH.
try:
    from phase3_pretraining.config import config as phase3_base_config
    print("Successfully imported base config from phase3_pretraining.config")
    # Extract necessary base values that might be reused or provide defaults
    RANDOM_SEED_base = phase3_base_config.get('seed', 42)
    DEVICE_base = phase3_base_config.get('device', "cuda" if torch.cuda.is_available() else "cpu")
    DATASET_BASE_PATH_base = phase3_base_config.get('data_root', "/path/to/dataset/placeholder") # Ensure phase3 has 'data_root'
    NUM_CLASSES_base = phase3_base_config.get('num_classes', 7)
    # Path to the Phase 3 HVT-XL pre-trained checkpoint
    # Use the final checkpoint name from phase3_config if available
    PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT = os.path.join(
        phase3_base_config.get('checkpoint_dir', "../pretrain_checkpoints_h100_xl"), # Default relative if phase3_config not fully loaded
        phase3_base_config.get('pretrain_final_checkpoint_name', 'hvt_xl_h100_prod_final.pth')
    )
    # HVT-XL architecture params from Phase 3
    HVT_XL_PATCH_SIZE = phase3_base_config.get('hvt_patch_size', 14)
    HVT_XL_EMBED_DIM_RGB = phase3_base_config.get('hvt_embed_dim_rgb', 192)
    HVT_XL_EMBED_DIM_SPECTRAL = phase3_base_config.get('hvt_embed_dim_spectral', 192)
    HVT_XL_SPECTRAL_CHANNELS = phase3_base_config.get('hvt_spectral_channels', 3)
    HVT_XL_DEPTHS = phase3_base_config.get('hvt_depths', [3, 6, 24, 3])
    HVT_XL_NUM_HEADS = phase3_base_config.get('hvt_num_heads', [6, 12, 24, 48])
    HVT_XL_MLP_RATIO = phase3_base_config.get('hvt_mlp_ratio', 4.0)
    HVT_XL_QKV_BIAS = phase3_base_config.get('hvt_qkv_bias', True)
    HVT_XL_DROP_PATH_RATE = phase3_base_config.get('hvt_drop_path_rate', 0.2)
    HVT_XL_USE_DFCA = phase3_base_config.get('hvt_use_dfca', True)
    HVT_XL_DFCA_HEADS = phase3_base_config.get('hvt_dfca_heads', 24) # Match the value from H100 XL config
    HVT_XL_DFCA_DROP_RATE = phase3_base_config.get('dfca_drop_rate', 0.1)
    HVT_XL_DFCA_USE_DISEASE_MASK = phase3_base_config.get('dfca_use_disease_mask', True)


except ImportError:
    print("Warning (phase4_config): Could not import config from phase3_pretraining. Using fallback defaults for Phase 4.")
    RANDOM_SEED_base = 42
    DEVICE_base = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANT: Update this path if fallback is used
    DATASET_BASE_PATH_base = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
    NUM_CLASSES_base = 7
    PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT = "path/to/your/hvt_xl_h100_prod_final.pth" # User must specify
    # Fallback HVT-XL architecture (should match Phase 3 if possible)
    HVT_XL_PATCH_SIZE = 14
    HVT_XL_EMBED_DIM_RGB = 192
    HVT_XL_EMBED_DIM_SPECTRAL = 192
    HVT_XL_SPECTRAL_CHANNELS = 3
    HVT_XL_DEPTHS = [3, 6, 24, 3]
    HVT_XL_NUM_HEADS = [6, 12, 24, 48]
    HVT_XL_MLP_RATIO = 4.0
    HVT_XL_QKV_BIAS = True
    HVT_XL_DROP_PATH_RATE = 0.2
    HVT_XL_USE_DFCA = True
    HVT_XL_DFCA_HEADS = 24
    HVT_XL_DFCA_DROP_RATE = 0.1
    HVT_XL_DFCA_USE_DISEASE_MASK = True

# --- Fine-tuning Specific Configurations ---
config = {
    # General
    "seed": RANDOM_SEED_base,
    "device": DEVICE_base,
    "log_dir": "logs_finetune_xl", # Separate logs for XL finetuning
    "log_file_finetune": "finetune_hvt_xl.log",
    "best_model_path": "best_finetuned_hvt_xl.pth",
    "final_model_path": "final_finetuned_hvt_xl.pth",

    # Data
    "data_root": DATASET_BASE_PATH_base,
    "original_dataset_name": default_config.get('ORIGINAL_DATASET_NAME', "Original Dataset") if 'default_config' in globals() else "Original Dataset",
    "augmented_dataset_name": default_config.get('AUGMENTED_DATASET_NAME', "Augmented Dataset") if 'default_config' in globals() else "Augmented Dataset",
    "img_size": (448, 448), # Fine-tune at the same or slightly smaller/larger res than pre-train
    "num_classes": NUM_CLASSES_base,
    "train_split_ratio": default_config.get('TRAIN_SPLIT_RATIO', 0.8) if 'default_config' in globals() else 0.8, # Use TRAIN_SPLIT_RATIO from imported config
    "normalize_data": True,
    "use_weighted_sampler": True,
    "num_workers": default_config.get('num_workers', 4) if 'default_config' in globals() else 4, # Inherit num_workers from phase3 if possible, or set for finetuning
    "prefetch_factor": default_config.get('prefetch_factor', 2) if 'default_config' in globals() and default_config.get('num_workers',4) > 0 else None,


    # Model - HVT-XL Specific
    "model_architecture": "DiseaseAwareHVT_XL", # To identify the model structure
    "pretrained_checkpoint_path": PRETRAINED_HVT_XL_CHECKPOINT_DEFAULT,
    "load_pretrained_backbone": True, # Load the backbone weights
    "freeze_backbone_epochs": 0,  # Number of initial epochs to keep backbone frozen (0 means unfreeze immediately)
    "unfreeze_backbone_lr_factor": 0.1, # Factor to reduce LR when unfreezing backbone (e.g., 0.1 * main_lr)

    # HVT-XL Architecture (SHOULD MATCH THE PRE-TRAINED MODEL)
    "hvt_patch_size": HVT_XL_PATCH_SIZE,
    "hvt_embed_dim_rgb": HVT_XL_EMBED_DIM_RGB,
    "hvt_embed_dim_spectral": HVT_XL_EMBED_DIM_SPECTRAL,
    "hvt_spectral_channels": HVT_XL_SPECTRAL_CHANNELS, # Important for model instantiation
    "hvt_depths": HVT_XL_DEPTHS,
    "hvt_num_heads": HVT_XL_NUM_HEADS,
    "hvt_mlp_ratio": HVT_XL_MLP_RATIO,
    "hvt_qkv_bias": HVT_XL_QKV_BIAS,
    # Dropout and DropPath can be adjusted for fine-tuning
    "hvt_model_drop_rate": 0.1, # Example: add some dropout
    "hvt_attn_drop_rate": 0.0,
    "hvt_drop_path_rate": HVT_XL_DROP_PATH_RATE, # Can keep same as pre-train or reduce
    "hvt_use_dfca": HVT_XL_USE_DFCA,
    "hvt_dfca_heads": HVT_XL_DFCA_HEADS,
    "dfca_drop_rate": HVT_XL_DFCA_DROP_RATE,
    "dfca_use_disease_mask": HVT_XL_DFCA_USE_DISEASE_MASK,
    "use_gradient_checkpointing": False, # Usually False for fine-tuning unless model is huge and batch size is constrained

    # --- PyTorch Performance Settings (can inherit or set for finetuning) ---
    "enable_torch_compile": default_config.get("enable_torch_compile", True) if 'default_config' in globals() else True,
    "torch_compile_mode": default_config.get("torch_compile_mode", "reduce-overhead") if 'default_config' in globals() else "reduce-overhead",
    "matmul_precision": default_config.get("matmul_precision", 'high') if 'default_config' in globals() else 'high',
    "cudnn_benchmark": default_config.get("cudnn_benchmark", True) if 'default_config' in globals() else True,


    # Training Loop
    "epochs": 50, # Number of fine-tuning epochs
    "batch_size": 32, # Adjust based on H100 memory for 448px (HVT-XL)
    "accumulation_steps": 1,
    "amp_enabled": True,
    "clip_grad_norm": 1.0,
    "log_interval": 20,

    # Optimizer
    "optimizer": "AdamW",
    "learning_rate": 5e-5, # Fine-tuning LR is typically smaller
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)},

    # Schedulers
    "scheduler": "WarmupCosine",
    "warmup_epochs": 5, # Warmup for fine-tuning
    # For WarmupCosine, T_max will be calculated based on total epochs - warmup_epochs
    "eta_min_lr": 1e-6, # Min LR for cosine scheduler

    # Loss Function
    "loss_label_smoothing": 0.1,

    # Augmentations
    "augmentations_enabled": True,
    # FinetuneAugmentation in utils/augmentations.py will be used

    # Evaluation & Early Stopping
    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 10,
    "metric_to_monitor_early_stopping": "f1_macro", # e.g. "accuracy" or "f1_macro"
}

# Provide NUM_CLASSES for baseline models if they are used and import from this config
NUM_CLASSES = config['num_classes']

# Logging after config dict is defined
logger = logging.getLogger(__name__)
logger.info(f"Phase 4 Fine-tuning Configuration Loaded. Model: {config.get('model_architecture', config.get('model_name'))}, Pretrained: {config.get('load_pretrained_backbone')}")