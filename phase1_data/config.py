# config.py
import logging
import torch
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Potentially set deterministic algorithms if desired, though it might impact performance
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# Dataset paths
# It's good practice to use environment variables or a more robust config system for paths
# For now, direct paths are used as per the original structure.
# Ensure this path is correct for your environment.
DATASET_BASE_PATH = "/teamspace/studios/this_studio/cvpr25/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection"
ORIGINAL_DATASET_ROOT = f"{DATASET_BASE_PATH}/Original Dataset"
AUGMENTED_DATASET_ROOT = f"{DATASET_BASE_PATH}/Augmented Dataset"

# Dataset constants
# IMPORTANT: DEFAULT_STAGE_MAP is critical.
DEFAULT_STAGE_MAP = {
    0: 'early', 1: 'mid', 2: 'advanced',
    3: 'early', 4: 'mid', 5: 'advanced',
    6: 'unknown' # Example: A 'healthy' class or a disease with no defined stages for progression
    # Ensure all your class labels are covered or have a default like 'unknown'
}


IMAGE_SIZE = (224, 224)  # Default input size for RGB models
SPECTRAL_SIZE = (224, 224)  # Target size for spectral data

# Model & Training Hyperparameters (Placeholders - to be expanded)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
NUM_WORKERS = 2 # Adjust based on your system's capabilities
# Set NUM_WORKERS = 0 for easier debugging if DataLoader issues persist