# phase3_pretraining/config.py
import torch
import logging

# --- General Reproducibility & Setup ---
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FILE_PRETRAIN = "pretrain_main.log" # Log for the main pretrain script
LOG_FILE_TRAINER = "pretrain_trainer.log" # Log specifically for trainer class if needed

# --- Dataset Config ---
# Assuming SAR-CLD-2024 dataset path is consistent
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
ORIGINAL_DATASET_NAME = "Original Dataset"
AUGMENTED_DATASET_NAME = "Augmented Dataset"
TRAIN_SPLIT_RATIO = 0.8 # For splitting SAR-CLD into train/val for linear probing

# --- Model Configurations (Imported/adapted from Phase 2 HVT) ---
# HVT Base Config (from phase2_model/config.py, ensure consistency)
PATCH_SIZE = 16
EMBED_DIM_RGB = 96      # Initial embedding dim for HVT RGB stream
EMBED_DIM_SPECTRAL = 96 # Initial embedding dim for HVT Spectral stream
HVT_DEPTHS = [2, 2, 6, 2]
HVT_NUM_HEADS = [3, 6, 12, 24]
MLP_RATIO = 4.0
QKV_BIAS = True
HVT_MODEL_DROP_RATE = 0.0   # General dropout for HVT
HVT_ATTN_DROP_RATE = 0.0    # Attention dropout for HVT
HVT_DROP_PATH_RATE = 0.1    # Stochastic depth for HVT

# DFCA Config (from phase2_model/config.py)
# DFCA_EMBED_DIM will be the output dim of HVT stages
DFCA_NUM_HEADS = HVT_NUM_HEADS[-1] # Or a specific value, e.g., 8
DFCA_DROP_RATE = 0.1

# General Model Settings
NUM_CLASSES = 7  # From SAR-CLD-2024 dataset
SPECTRAL_CHANNELS = 1 # For NDVI, this is used by the HVT if spectral data is fed

# --- Pre-training Specific (SimCLR) ---
PRETRAIN_IMAGE_RESOLUTIONS = [(224, 224), (384, 384)] # Resolutions to use for pre-training stages
PRETRAIN_EPOCHS = 20
PRETRAIN_BATCH_SIZE = 16 # Adjust based on GPU memory
PRETRAIN_LR = 3e-4
PRETRAIN_WEIGHT_DECAY = 1e-4
ACCUMULATION_STEPS = 2 # Gradient accumulation
TEMPERATURE = 0.07 # For InfoNCE Loss
PROJECTION_DIM = 256 # Output dim of the SimCLR projection head
PROJECTION_HIDDEN_DIM = 512 # Hidden dim of the SimCLR projection head

# --- Linear Probing Evaluation during Pre-training ---
LINEAR_PROBE_EPOCHS = 10
LINEAR_PROBE_LR = 1e-3
EVALUATE_EVERY_N_EPOCHS = 5 # How often to run linear probe

# --- Checkpointing ---
PRETRAIN_CHECKPOINT_DIR = "pretrain_checkpoints"

# Configure Root Logger (once, can be in main script or here)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Individual loggers will be set up by setup_logging utility