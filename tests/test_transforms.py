import pytest
import torch
import torch.nn as nn

from hvit.data.transforms import build_transform_pipeline

@pytest.fixture
def transform_config():
    return {
        "resize_size": 256,
        "crop_size": 224,
        "color_jitter": 0.4,
        "auto_augment": True,
        "random_erase": 0.25,
    }

def test_transform_pipeline(transform_config):
    transform = build_transform_pipeline(**transform_config)
    img = torch.randn(3, 256, 256)
    output = transform(img)
    assert output.shape == (3, 224, 224)

def test_augmentation_consistency():
    """Test that augmentations are applied consistently within a batch."""
    transform = build_transform_pipeline(
        resize_size=256,
        crop_size=224,
        color_jitter=0.4,
    )
    
    img = torch.randn(3, 256, 256)
    out1 = transform(img)
    out2 = transform(img)
    assert not torch.allclose(out1, out2)  # Transformations should be random

def test_normalization_values():
    """Test that normalization is applied correctly."""
    transform = build_transform_pipeline(
        resize_size=224,
        crop_size=224,
        normalize=True,
    )
    
    img = torch.ones(3, 224, 224)
    output = transform(img)
    
    # Check if values are normalized (assuming ImageNet normalization)
    expected_means = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    expected_stds = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    expected = (img - expected_means) / expected_stds
    
    assert torch.allclose(output, expected)
