"""
Tests for the HierarchicalViT (HVT) model.

These tests verify the core functionality of the DiseaseAwareHVT model
including forward pass, different modes, and model components.
"""
import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvit.models import create_disease_aware_hvt, DiseaseAwareHVT


@pytest.fixture
def hvt_params():
    """Default HVT parameters for testing (small config for speed)."""
    return {
        "patch_size": 16,
        "embed_dim_rgb": 96,
        "embed_dim_spectral": 96,
        "spectral_channels": 0,  # RGB only for basic tests
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


# Image size must be compatible with patch_size and number of downsampling stages
# With patch_size=16 and 4 stages (3 downsamples), 256x256 -> 16x16 -> 8x8 -> 4x4 -> 2x2
TEST_IMG_SIZE = 256


@pytest.fixture
def model(hvt_params):
    """Create a test model instance."""
    return create_disease_aware_hvt(
        current_img_size=(TEST_IMG_SIZE, TEST_IMG_SIZE),
        num_classes=7,
        model_params_dict=hvt_params
    )


class TestModelCreation:
    """Tests for model instantiation."""

    def test_model_creation(self, model):
        """Test that model can be created successfully."""
        assert isinstance(model, DiseaseAwareHVT)

    def test_model_has_expected_attributes(self, model):
        """Test that model has required attributes."""
        assert hasattr(model, 'rgb_patch_embed')
        assert hasattr(model, 'rgb_stages')
        assert hasattr(model, 'classifier_head')

    def test_model_parameter_count(self, model):
        """Test that model has reasonable parameter count."""
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        # Small model should have < 50M params
        assert param_count < 50_000_000


class TestForwardPass:
    """Tests for model forward pass."""

    def test_classify_mode_output_shape(self, model):
        """Test classification mode produces correct output shape."""
        batch_size = 2
        x = torch.randn(batch_size, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)
        output = model(x, mode='classify')
        assert output.shape == (batch_size, 7)

    def test_classify_mode_with_different_batch_sizes(self, model):
        """Test classification works with various batch sizes."""
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)
            output = model(x, mode='classify')
            assert output.shape == (batch_size, 7)

    def test_get_embeddings_mode(self, model):
        """Test get_embeddings mode returns dictionary."""
        x = torch.randn(2, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)
        embeddings = model(x, mode='get_embeddings')
        assert isinstance(embeddings, dict)
        assert 'fused_pooled' in embeddings
        assert 'rgb_pooled' in embeddings


class TestGradients:
    """Tests for gradient computation."""

    def test_gradients_flow(self, model):
        """Test that gradients flow through the model."""
        x = torch.randn(2, 3, TEST_IMG_SIZE, TEST_IMG_SIZE, requires_grad=True)
        output = model(x, mode='classify')
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_model_is_trainable(self, model):
        """Test that model parameters receive gradients."""
        x = torch.randn(2, 3, TEST_IMG_SIZE, TEST_IMG_SIZE)
        output = model(x, mode='classify')
        loss = output.sum()
        loss.backward()
        
        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in model.parameters() if p.requires_grad)
        assert has_grad
