# phase4_finetuning/config.py
import os
import torch
import logging

# --- Logger for this config file ---
_config_module_logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# --- Project Path Configuration ---
try:
    PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_PATH = os.path.dirname(PACKAGE_ROOT)
except NameError:
    PROJECT_ROOT_PATH = "/teamspace/studios/this_studio/cvpr25" # MUST BE CORRECT
    PACKAGE_ROOT = os.path.join(PROJECT_ROOT_PATH, "phase4_finetuning")
    _config_module_logger.warning(f"Guessed PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH} and PACKAGE_ROOT: {PACKAGE_ROOT}")

# --- Default Values ---
DEFAULT_RANDOM_SEED = 42
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DATASET_BASE_PATH = os.path.join(PROJECT_ROOT_PATH, "SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection")
DEFAULT_NUM_CLASSES = 7
DEFAULT_ORIGINAL_DATASET_NAME = "Original Dataset"
DEFAULT_AUGMENTED_DATASET_NAME = "Augmented Dataset"
DEFAULT_TRAIN_SPLIT_RATIO_FINETUNE = 0.8

# --- Attempt to Import Phase 3 Config ---
phase3_config_imported = False
phase3_cfg = {}
try:
    from phase3_pretraining.config import config as phase3_base_config_dict_imported
    phase3_cfg = phase3_base_config_dict_imported.copy()
    phase3_config_imported = True
    _config_module_logger.info("Successfully imported base config dict from phase3_pretraining.config")
except ImportError as e:
    _config_module_logger.warning(f"Could not import config from phase3_pretraining: {e}. Phase 4 will use its own defaults.")

# --- Resolve Key Parameters ---
SEED = phase3_cfg.get('seed', DEFAULT_RANDOM_SEED)
DEVICE_RESOLVED = phase3_cfg.get('device', DEFAULT_DEVICE)
DATA_ROOT_RESOLVED = phase3_cfg.get('data_root', DEFAULT_DATASET_BASE_PATH)
NUM_CLASSES_RESOLVED = phase3_cfg.get('num_classes', DEFAULT_NUM_CLASSES)
ORIGINAL_DATASET_NAME_RESOLVED = phase3_cfg.get('original_dataset_name', DEFAULT_ORIGINAL_DATASET_NAME)
AUGMENTED_DATASET_NAME_RESOLVED = phase3_cfg.get('augmented_dataset_name', DEFAULT_AUGMENTED_DATASET_NAME)

# Path to the HVT SSL Pre-trained Checkpoint
default_phase3_ckpt_filename = f"{phase3_cfg.get('model_arch_name_for_ckpt', 'hvt_xl_simclr')}_best_probe.pth" # Or specific epoch
default_phase3_ckpt_path = None
if phase3_config_imported:
    phase3_pkg_root = phase3_cfg.get('PACKAGE_ROOT_PATH', os.path.join(PROJECT_ROOT_PATH, 'phase3_pretraining'))
    phase3_ckpt_dir_name = phase3_cfg.get('checkpoint_dir_name', 'pretrain_checkpoints_hvt_xl')
    # Constructing the path based on your previous successful load:
    default_phase3_ckpt_path = "/teamspace/studios/this_studio/cvpr25/phase3_pretraining/pretrain_checkpoints_hvt_xl/hvt_xl_simclr_t4_resumed_best_probe.pth" # From your log
    _config_module_logger.info(f"Using Phase 3 checkpoint path: {default_phase3_ckpt_path}")
else:
    default_phase3_ckpt_path = "/path/to/your/MANUALLY_SPECIFIED_phase3_best_probe.pth"
    _config_module_logger.warning(f"Phase 3 config import failed. PRETRAINED_HVT_CHECKPOINT_PATH set to: {default_phase3_ckpt_path}. VERIFY THIS.")

HVT_ARCH_PARAMS_FOR_FINETUNE = phase3_cfg.get('hvt_params_for_backbone', {})
if not HVT_ARCH_PARAMS_FOR_FINETUNE: # If empty or not found
    _config_module_logger.warning("Using FALLBACK HVT architecture params for fine-tuning. VERIFY!")
    HVT_ARCH_PARAMS_FOR_FINETUNE = {
        "patch_size": 14, "embed_dim_rgb": 192, "embed_dim_spectral": 192, "spectral_channels": 0,
        "depths": [3, 6, 24, 3], "num_heads": [6, 12, 24, 48], "mlp_ratio": 4.0, "qkv_bias": True,
        "model_drop_rate": 0.0, "attn_drop_rate": 0.0, "drop_path_rate": 0.2, "norm_layer_name": "LayerNorm",
        "use_dfca": False, "use_gradient_checkpointing": True,
        "ssl_enable_mae": False, "ssl_enable_contrastive": False, "enable_consistency_loss_heads": False,
    }
# Fine-tuning specific overrides for HVT architecture
HVT_ARCH_PARAMS_FOR_FINETUNE['model_drop_rate'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('model_drop_rate_finetune', 0.15) # Slightly increased dropout
HVT_ARCH_PARAMS_FOR_FINETUNE['drop_path_rate'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('drop_path_rate_finetune', 0.15) # Slightly increased
HVT_ARCH_PARAMS_FOR_FINETUNE['use_gradient_checkpointing'] = HVT_ARCH_PARAMS_FOR_FINETUNE.get('use_gradient_checkpointing_finetune', True) # KEEP TRUE for T4

# --- Fine-tuning Specific Configurations ---
config = {
    "seed": SEED, "device": DEVICE_RESOLVED,
    "log_dir": "logs_finetune_attempt2", # New log directory
    "log_file_finetune": "finetune_hvt_xl_attempt2.log",
    "best_model_filename": "best_finetuned_hvt_xl_attempt2.pth",
    "final_model_filename": "final_finetuned_hvt_xl_attempt2.pth",
    "checkpoint_save_dir_name": "checkpoints_attempt2",

    "data_root": DATA_ROOT_RESOLVED, "original_dataset_name": ORIGINAL_DATASET_NAME_RESOLVED,
    "augmented_dataset_name": None, # Typically don't use SSL-augmented data for supervised finetuning
    "img_size": (448, 448), "num_classes": NUM_CLASSES_RESOLVED,
    "train_split_ratio": 0.8, "normalize_data": True, "use_weighted_sampler": True,
    "num_workers": 4, # Keep as per T4
    "prefetch_factor": 2 if DEVICE_RESOLVED == 'cuda' and 4 > 0 else None,

    "model_architecture_name": "DiseaseAwareHVT_SSL_Finetuned_Attempt2",
    "pretrained_checkpoint_path": default_phase3_ckpt_path, # Resolved path to SSL model
    "load_pretrained_backbone": True,
    "freeze_backbone_epochs": 10,      # <<< INCREASED: Freeze backbone for more initial epochs
    "unfreeze_backbone_lr_factor": 0.1, # LR for backbone when unfrozen will be 0.1 * main_lr

    "hvt_params_for_model_init": HVT_ARCH_PARAMS_FOR_FINETUNE,

    "enable_torch_compile": False, # Keep False for T4 stability during tuning
    "torch_compile_mode": "reduce-overhead",
    "matmul_precision": 'high', "cudnn_benchmark": True,

    "epochs": 60,  # <<< INCREASED: Total fine-tuning epochs
    "batch_size": 16, # T4 constraint for HVT-XL 448px
    "accumulation_steps": 2, # Effective BS = 32
    "amp_enabled": True, "clip_grad_norm": 1.0, "log_interval": 20,

    "optimizer": "AdamW",
    "learning_rate": 2e-5, # <<< SLIGHTLY REDUCED initial LR for full model fine-tuning
                           # The head will initially train with head_lr_multiplier * this.
    "head_lr_multiplier": 5.0, # <<< INCREASED: Head trains with a higher LR (5 * 2e-5 = 1e-4) during freeze phase
    "weight_decay": 0.05,
    "optimizer_params": {"betas": (0.9, 0.999)},

    "scheduler": "WarmupCosine",
    "warmup_epochs": 5, # Warmup over 5 epochs (can be total, or per phase if optimizer is reset)
                        # If backbone is frozen, this warmup applies to the head.
                        # When backbone unfreezes, a new optimizer/scheduler for it might be better, or continue.
                        # The current trainer uses one optimizer and scheduler.
    "eta_min_lr": 1e-7,

    "loss_label_smoothing": 0.1,
    "augmentations_enabled": True,

    "evaluate_every_n_epochs": 1,
    "early_stopping_patience": 15, # Slightly more patience
    "metric_to_monitor_early_stopping": "f1_macro",
}

NUM_CLASSES = config['num_classes']
_config_module_logger.info(f"Phase 4 Config (Finetune Attempt 2 for T4) Loaded. Pretrained: {config.get('pretrained_checkpoint_path')}")
_config_module_logger.info(f"Finetune LR: {config['learning_rate']}, Head LR Mult: {config['head_lr_multiplier']}, Freeze Epochs: {config['freeze_backbone_epochs']}")
_config_module_logger.info(f"Total Fine-tune Epochs: {config['epochs']}, Eff. BS: {config['batch_size']*config['accumulation_steps']}")