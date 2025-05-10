# phase2_model/config.py
import logging
import torch
import torch.nn as nn
import numpy as np

# Initialize logger at the very beginning of the file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Define logger instance

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Image & Patch Config ---
IMAGE_SIZE = (224, 224)
SPECTRAL_SIZE = (224, 224) # Should ideally match IMAGE_SIZE for current HVT patch embed
PATCH_SIZE = 16 # Corresponds to hvt_patch_size in HVT defaults

# --- HVT Backbone Config ---
HVT_EMBED_DIM_RGB = 96
HVT_EMBED_DIM_SPECTRAL = 96
HVT_SPECTRAL_CHANNELS = 3 # Number of input channels for spectral data (main.py imports this)
HVT_DEPTHS = [2, 2, 6, 2] # Number of transformer blocks in each stage
HVT_NUM_HEADS = [3, 6, 12, 24] # Number of attention heads in each stage
HVT_MLP_RATIO = 4.0
HVT_QKV_BIAS = True
HVT_MODEL_DROP_RATE = 0.0 # Dropout rate for embeddings and MLP layers
HVT_ATTN_DROP_RATE = 0.0 # Dropout rate for attention weights
HVT_DROP_PATH_RATE = 0.1 # Stochastic depth rate

# --- DFCA (Disease-Focused Cross-Attention) Config ---
HVT_USE_DFCA = True
DFCA_NUM_HEADS = HVT_NUM_HEADS[-1] # Or specify directly, e.g., 16
DFCA_DROP_RATE = 0.1
DFCA_USE_DISEASE_MASK = True # As per dfca.py

# --- Classifier Head ---
NUM_CLASSES = 7 # main.py imports this

# --- Training Strategy ---
PROGRESSIVE_RESOLUTIONS = [(224, 224), (384, 384)] # main.py imports this
USE_GRADIENT_CHECKPOINTING = False

# --- Self-Supervised Learning (SSL) Config ---
SSL_ENABLE_MAE = True # main.py imports this
SSL_MAE_MASK_RATIO = 0.75
SSL_MAE_DECODER_DIM = 64
SSL_MAE_RECONSTRUCTION_NORM_PIX_LOSS = True

SSL_ENABLE_CONTRASTIVE = True # main.py imports this
SSL_CONTRASTIVE_PROJECTOR_DIM = 128 # main.py imports this
SSL_CONTRASTIVE_PROJECTOR_DEPTH = 2

ENABLE_CONSISTENCY_LOSS_HEADS = True # main.py imports this

# --- Fallback defaults (These are for reference or for hvt.py's internal get_cfg_param if needed) ---
DEFAULT_HVT_PARAMS_REF = {
    "hvt_patch_size": PATCH_SIZE,
    "hvt_embed_dim_rgb": HVT_EMBED_DIM_RGB,
    "hvt_embed_dim_spectral": HVT_EMBED_DIM_SPECTRAL,
    "hvt_spectral_channels": HVT_SPECTRAL_CHANNELS,
    "hvt_depths": HVT_DEPTHS,
    "hvt_num_heads": HVT_NUM_HEADS,
    "hvt_mlp_ratio": HVT_MLP_RATIO,
    "hvt_qkv_bias": HVT_QKV_BIAS,
    "hvt_model_drop_rate": HVT_MODEL_DROP_RATE,
    "hvt_attn_drop_rate": HVT_ATTN_DROP_RATE,
    "hvt_drop_path_rate": HVT_DROP_PATH_RATE,
    "hvt_use_dfca": HVT_USE_DFCA,
    "num_classes": NUM_CLASSES,
    "hvt_dfca_heads": DFCA_NUM_HEADS,
    "dfca_drop_rate": DFCA_DROP_RATE,
    "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "ssl_enable_mae": SSL_ENABLE_MAE,
    "ssl_mae_mask_ratio": SSL_MAE_MASK_RATIO,
    "ssl_mae_decoder_dim": SSL_MAE_DECODER_DIM,
    "ssl_enable_contrastive": SSL_ENABLE_CONTRASTIVE,
    "ssl_contrastive_projector_dim": SSL_CONTRASTIVE_PROJECTOR_DIM,
    "enable_consistency_loss_heads": ENABLE_CONSISTENCY_LOSS_HEADS,
}

logger.info("Configuration loaded. (phase2_model/config.py)") # Clarify which config this is
logger.info(f"Image Size: {IMAGE_SIZE}, Patch Size: {PATCH_SIZE}, Num Classes: {NUM_CLASSES}")
logger.info(f"HVT Depths: {HVT_DEPTHS}, HVT Heads: {HVT_NUM_HEADS}")
logger.info(f"HVT Spectral Channels: {HVT_SPECTRAL_CHANNELS}")
if HVT_USE_DFCA:
    logger.info(f"DFCA Enabled: Heads={DFCA_NUM_HEADS}, Drop Rate={DFCA_DROP_RATE}")
if SSL_ENABLE_MAE:
    logger.info(f"MAE Enabled: Mask Ratio={SSL_MAE_MASK_RATIO}, Decoder Dim={SSL_MAE_DECODER_DIM}")
if SSL_ENABLE_CONTRASTIVE:
    logger.info(f"Contrastive Learning Enabled: Projector Dim={SSL_CONTRASTIVE_PROJECTOR_DIM}")
if ENABLE_CONSISTENCY_LOSS_HEADS:
    logger.info("Cross-Modal Consistency Loss Auxiliary Heads: Enabled")