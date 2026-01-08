# tests/conftest.py
"""Pytest configuration and shared fixtures."""
import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_batch():
    """Create a small batch of random images."""
    return torch.randn(2, 3, 256, 256)


@pytest.fixture
def hvt_config():
    """Minimal HVT configuration for testing."""
    return {
        "patch_size": 16,
        "embed_dim_rgb": 96,
        "embed_dim_spectral": 96,
        "spectral_channels": 0,
        "depths": [2, 2, 2, 2],
        "num_heads": [3, 6, 12, 24],
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "model_drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_layer_name": "LayerNorm",
        "use_dfca": False,
        "use_gradient_checkpointing": False,
        "ssl_enable_mae": False,
        "ssl_enable_contrastive": False,
        "enable_consistency_loss_heads": False,
    }


@pytest.fixture
def num_classes():
    """Number of disease classes."""
    return 7


@pytest.fixture
def img_size():
    """Standard image size for testing."""
    return (256, 256)
