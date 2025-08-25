import pytest
import torch

from hvit.models.hvt import HierarchialViT

@pytest.fixture
def model_config():
    return {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "num_classes": 1000,
        "embed_dims": [64, 128, 256, 512],
        "num_heads": [1, 2, 4, 8],
        "mlp_ratios": [4, 4, 4, 4],
        "depths": [3, 4, 6, 3],
        "sr_ratios": [8, 4, 2, 1],
    }

@pytest.fixture
def model(model_config):
    return HierarchialViT(**model_config)

def test_model_output_shape(model):
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    assert output.shape == (batch_size, 1000)

def test_feature_pyramid(model):
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    features = model.forward_features(x)
    
    expected_shapes = [
        (batch_size, 64, 56, 56),
        (batch_size, 128, 28, 28),
        (batch_size, 256, 14, 14),
        (batch_size, 512, 7, 7),
    ]
    
    for feat, shape in zip(features, expected_shapes):
        assert feat.shape == shape

def test_attention_patterns(model):
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    attention_maps = model.get_attention_maps(x)
    
    for stage_idx, attn_map in enumerate(attention_maps):
        H = W = 224 // (16 * (2 ** stage_idx))
        num_heads = model.num_heads[stage_idx]
        assert attn_map.shape == (batch_size, num_heads, H * W, H * W)
