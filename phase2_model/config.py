# config.py
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGE_SIZE = (224, 224) 
SPECTRAL_SIZE = (224, 224)

PATCH_SIZE = 16
EMBED_DIM_RGB = 96
EMBED_DIM_SPECTRAL = 96

HVT_DEPTHS = [2, 2, 6, 2]
HVT_NUM_HEADS = [3, 6, 12, 24]

HVT_WINDOW_SIZE = 7
MLP_RATIO = 4.0
QKV_BIAS = True
ATTN_DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
MODEL_DROP_RATE = 0.0

DFCA_NUM_HEADS = HVT_NUM_HEADS[-1] 
DFCA_DROP_RATE = 0.1

NUM_CLASSES = 7
SPECTRAL_CHANNELS = 1

PROGRESSIVE_RESOLUTIONS = [(224, 224), (384, 384)]