# phase2_model/config.py
import logging
import torch
import numpy as np

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Reproducibility ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# For stricter reproducibility (may impact performance):
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# --- General Training & Model Config ---
NUM_CLASSES = 7  # Number of output classes for classification
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Image & Patch Configuration ---
INITIAL_IMAGE_SIZE = (256, 256) # Base image size for model instantiation (CHOOSE A COMPATIBLE ONE)
PATCH_SIZE = 16                # Patch size for HVT

# List of image resolutions for progressive training/testing
# For HVT with patch_size=16 and 4 stages (3 merging operations, so divisible by 2^3=8):
# (128/16=8, 8%8=0), (256/16=16, 16%8=0), (384/16=24, 24%8=0)
PROGRESSIVE_RESOLUTIONS_TEST = [(128, 128), (256, 256), (384, 384)] # ADJUSTED

# --- HVT (Hierarchical Vision Transformer) Model Parameters ---
# This dictionary will be imported by main.py and passed to the HVT factory.
HVT_MODEL_PARAMS = {
    "patch_size": PATCH_SIZE,
    "embed_dim_rgb": 96,
    "embed_dim_spectral": 96,
    "spectral_channels": 3,
    "depths": [2, 2, 6, 2],      # num_stages = 4
    "num_heads": [3, 6, 12, 24],
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "model_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.1,
    "norm_layer_name": "LayerNorm",

    "use_dfca": True,
    "dfca_embed_dim_match_rgb": True,
    "dfca_num_heads": 12,
    "dfca_drop_rate": 0.1,
    "dfca_use_disease_mask": True,

    "use_gradient_checkpointing": False,

    "ssl_enable_mae": True,
    "ssl_mae_mask_ratio": 0.75,
    "ssl_mae_decoder_dim": 128,
    "ssl_mae_norm_pix_loss": True,

    "ssl_enable_contrastive": True,
    "ssl_contrastive_projector_dim": 128,
    "ssl_contrastive_projector_depth": 2,

    "enable_consistency_loss_heads": True
}

# --- Log Key Configurations After Definition ---
logger.info(f"Phase 2 Model Config Loaded from: {__file__}")
logger.info(f"Device: {DEVICE}, Num Classes: {NUM_CLASSES}")
logger.info(f"Initial Img Size for HVT instantiation: {INITIAL_IMAGE_SIZE}, Patch Size: {PATCH_SIZE}")
logger.info(f"HVT RGB Embed Dim: {HVT_MODEL_PARAMS['embed_dim_rgb']}, Spectral Embed Dim: {HVT_MODEL_PARAMS['embed_dim_spectral']}, Spectral Channels: {HVT_MODEL_PARAMS['spectral_channels']}")
logger.info(f"HVT Depths: {HVT_MODEL_PARAMS['depths']}, HVT Num Heads per Stage: {HVT_MODEL_PARAMS['num_heads']}")

if HVT_MODEL_PARAMS['use_dfca'] and HVT_MODEL_PARAMS['spectral_channels'] > 0:
    logger.info(f"DFCA Fusion Enabled: DFCA Heads={HVT_MODEL_PARAMS['dfca_num_heads']}, DFCA Drop Rate={HVT_MODEL_PARAMS['dfca_drop_rate']}")
else:
    logger.info("DFCA Fusion: Not used or no spectral channels.")

if HVT_MODEL_PARAMS['ssl_enable_mae']:
    logger.info(f"HVT MAE Components: Enabled (Mask Ratio: {HVT_MODEL_PARAMS['ssl_mae_mask_ratio']}, Decoder Dim: {HVT_MODEL_PARAMS['ssl_mae_decoder_dim']})")
else:
    logger.info("HVT MAE Components: Disabled.")

if HVT_MODEL_PARAMS['ssl_enable_contrastive']:
    logger.info(f"HVT Contrastive Projector: Enabled (Output Dim: {HVT_MODEL_PARAMS['ssl_contrastive_projector_dim']}, Depth: {HVT_MODEL_PARAMS['ssl_contrastive_projector_depth']})")
else:
    logger.info("HVT Contrastive Projector: Disabled.")

if HVT_MODEL_PARAMS['enable_consistency_loss_heads']:
    logger.info("HVT Consistency Loss Auxiliary Heads: Enabled.")
else:
    logger.info("HVT Consistency Loss Auxiliary Heads: Disabled.")

logger.info(f"Gradient Checkpointing for HVT: {'Enabled' if HVT_MODEL_PARAMS['use_gradient_checkpointing'] else 'Disabled'}")
logger.info(f"Resolutions for HVT testing in main.py: {PROGRESSIVE_RESOLUTIONS_TEST}")