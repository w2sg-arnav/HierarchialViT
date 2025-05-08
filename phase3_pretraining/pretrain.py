# phase3_pretraining/pretrain.py
import torch
import os
import sys
import logging

# --- Path Setup ---
# Goal: Allow this script to be run directly, and also allow imports from sibling packages (like phase2_model)
# when the ultimate project root (cvpr25) is implicitly or explicitly in sys.path.

# Get the directory of the current script (phase3_pretraining/pretrain.py)
script_dir = os.path.dirname(os.path.abspath(__file__)) # .../phase3_pretraining

# Get the parent of the script_dir (this should be .../phase3_pretraining if script is pretrain.py)
# If script is pretrain.py, then script_dir is '.../phase3_pretraining'.
# The package name is 'phase3_pretraining'.
# If we add the parent of 'phase3_pretraining' (i.e., 'cvpr25') to sys.path,
# then we can do 'from phase3_pretraining.config ...'

package_root = script_dir # .../phase3_pretraining
project_root = os.path.dirname(package_root) # .../cvpr25 (parent of phase3_pretraining)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added project root to sys.path: {project_root}")

# Now, standard absolute imports from the project root should work.
# For example, 'from phase3_pretraining.config import ...'
# and 'from phase2_model.models.hvt import ...'

# --- Config and Util Imports ---
# These should now work because 'project_root' (e.g., /path/to/cvpr25) is in sys.path,
# allowing Python to find the 'phase3_pretraining' package within it.
from phase3_pretraining.config import (
    DEVICE, PRETRAIN_IMAGE_RESOLUTIONS, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE,
    DATASET_BASE_PATH, ORIGINAL_DATASET_NAME, AUGMENTED_DATASET_NAME, TRAIN_SPLIT_RATIO,
    TEMPERATURE, EVALUATE_EVERY_N_EPOCHS, LOG_FILE_PRETRAIN, RANDOM_SEED # Added RANDOM_SEED
)
from phase3_pretraining.utils.logging_setup import setup_logging
from phase3_pretraining.utils.augmentations import SimCLRAugmentation
from phase3_pretraining.utils.losses import InfoNCELoss
from phase3_pretraining.dataset import SARCLD2024Dataset # This uses RANDOM_SEED from config internally
from phase3_pretraining.models.hvt_wrapper import HVTForPretraining
from phase3_pretraining.pretrain.trainer import Pretrainer # Correct path to trainer

from torch.utils.data import DataLoader

# Setup main logger for this script
# Make sure RANDOM_SEED is available for SARCLD2024Dataset via its config import
# In phase3_pretraining/dataset.py, change:
# from .config import RANDOM_SEED as config_RANDOM_SEED
# to:
# from phase3_pretraining.config import RANDOM_SEED as config_RANDOM_SEED
# This is because dataset.py might be imported when phase3_pretraining is not yet fully a package in sys.path context.
# The sys.path modification at the top of *this* script (pretrain.py) makes it work when this script is run.

# Apply RANDOM_SEED for torch and numpy at the start of the main script
torch.manual_seed(RANDOM_SEED)
import numpy as np
np.random.seed(RANDOM_SEED)


setup_logging(log_file_name=LOG_FILE_PRETRAIN, logger_name=None) # None for root logger
logger = logging.getLogger(__name__)


def main_pretrain():
    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ: # Set only if not already set
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    img_size = PRETRAIN_IMAGE_RESOLUTIONS[0] 
    logger.info(f"Selected image size for pre-training: {img_size}")

    logger.info("Initializing datasets for pre-training and linear probing...")
    pretrain_dataset = SARCLD2024Dataset(
        root_dir=DATASET_BASE_PATH, 
        img_size=img_size, 
        split="train", 
        train_split_ratio=TRAIN_SPLIT_RATIO, 
        normalize_for_model=False, 
        original_dataset_name=ORIGINAL_DATASET_NAME,
        augmented_dataset_name=AUGMENTED_DATASET_NAME
    )
    probe_train_dataset = SARCLD2024Dataset(
        root_dir=DATASET_BASE_PATH, img_size=img_size, split="train", 
        train_split_ratio=TRAIN_SPLIT_RATIO, normalize_for_model=True
    ) 
    probe_val_dataset = SARCLD2024Dataset(
        root_dir=DATASET_BASE_PATH, img_size=img_size, split="val", 
        train_split_ratio=TRAIN_SPLIT_RATIO, normalize_for_model=True
    )

    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=(True if DEVICE=='cuda' and int(torch.__version__.split('.')[1]) >= 7 else False) # persistent_workers needs PyTorch 1.7+
    )
    probe_train_loader = DataLoader(
        probe_train_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    probe_val_loader = DataLoader(
        probe_val_dataset, batch_size=PRETRAIN_BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False
    )
    logger.info(f"Pretrain DataLoader: {len(pretrain_loader)} batches. "
                f"Probe Train: {len(probe_train_loader)} batches. Probe Val: {len(probe_val_loader)} batches.")

    logger.info("Initializing model, augmentations, and loss function...")
    model = HVTForPretraining(img_size=img_size) 
    
    augmentations = SimCLRAugmentation(img_size=img_size)
    loss_fn = InfoNCELoss(temperature=TEMPERATURE)

    logger.info("Initializing Pretrainer...")
    pretrainer_instance = Pretrainer(
        model=model,
        augmentations=augmentations,
        loss_fn=loss_fn,
        device=DEVICE,
        train_loader_for_probe=probe_train_loader,
        val_loader_for_probe=probe_val_loader
    )

    logger.info(f"Starting pre-training for {PRETRAIN_EPOCHS} epochs.")
    try:
        for epoch in range(1, PRETRAIN_EPOCHS + 1):
            avg_loss = pretrainer_instance.train_one_epoch(pretrain_loader, epoch, PRETRAIN_EPOCHS)
            
            if epoch % EVALUATE_EVERY_N_EPOCHS == 0 or epoch == PRETRAIN_EPOCHS:
                pretrainer_instance.evaluate_linear_probe(current_epoch=epoch)
                pretrainer_instance.save_checkpoint(epoch=epoch)
            
            if torch.cuda.is_available():
                logger.debug(f"CUDA Mem Summary after epoch {epoch}: "
                            f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB, "
                            f"Reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
            
    except KeyboardInterrupt:
        logger.warning("Pre-training interrupted by user.")
    except Exception as e:
        logger.exception(f"An error occurred during pre-training: {e}")
    finally:
        logger.info("Pre-training finished or was interrupted.")
        if 'pretrainer_instance' in locals(): # Ensure pretrainer_instance was initialized
            pretrainer_instance.save_checkpoint(epoch="final")

if __name__ == "__main__":
    main_pretrain()