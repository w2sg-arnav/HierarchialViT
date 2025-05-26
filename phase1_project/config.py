# config.py
import logging
import torch
import numpy as np
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# For full reproducibility, uncommenting these might be necessary,
# but they can impact performance.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# --- Core Paths ---
# IMPORTANT: Update this path to your actual dataset location
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"

ORIGINAL_DATASET_ROOT = os.path.join(DATASET_BASE_PATH, "Original Dataset")
AUGMENTED_DATASET_ROOT = os.path.join(DATASET_BASE_PATH, "Augmented Dataset") # If you use it

# Output directories
MODEL_SAVE_PATH = "saved_models"
VISUALIZATION_DIR = "visualizations"

# Ensure output directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# --- Dataset Constants ---
DEFAULT_STAGE_MAP = {
    # Example: map class index to stage name
    # This needs to be aligned with how your classes are defined.
    # If classes are 'diseaseA', 'diseaseB', 'healthy', then map those indices.
    # e.g., if class 'Alternaria Leaf Spot' (index 0) has stages:
    # 0: 'early', # for Alternaria Leaf Spot
    # If another disease (index 1) also has stages:
    # 1: 'mid', # for Bacterial Blight, etc.
    # This map is primarily for the progression simulator if `apply_progression=True`
    # and the dataset label is used to infer a default stage for simulation.
    0: 'early', 1: 'mid', 2: 'advanced',
    3: 'early', 4: 'mid', 5: 'advanced',
    6: 'unknown' # for healthy or classes without distinct progression stages in simulation
}
# Number of classes will be determined dynamically from the dataset in data_utils.py

# Image sizes
IMAGE_SIZE_RGB = (224, 224)  # Input size for RGB models
IMAGE_SIZE_SPECTRAL = (224, 224) # Target size for spectral data

# --- Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "resnet50"  # Options: "resnet50", "vit_base_patch16_224", etc. (see models.py)
PRETRAINED = True

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Start with a small number for testing, increase for real training
NUM_WORKERS = 2 # Adjust based on your system
WEIGHT_DECAY = 1e-4

# Data Splitting
VALIDATION_SPLIT_RATIO = 0.2 # 20% of data for validation
TEST_SPLIT_RATIO = 0.0 # Set to > 0 if you want a separate test set from the start

# Augmentation & Dataset options for training
APPLY_PROGRESSION_TRAIN = False # Whether to apply disease progression simulation during training
USE_SPECTRAL_TRAIN = False      # Whether to load/use spectral data during training (for baselines, typically False)

# --- Evaluation ---
METRICS_AVERAGE = 'macro' # 'macro', 'micro', or 'weighted' for precision, recall, f1