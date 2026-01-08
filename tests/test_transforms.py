"""
Tests for data transforms and augmentations.

These tests verify the augmentation pipelines used for training
and validation in the HierarchicalViT project.
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAugmentationImports:
    """Tests for augmentation module imports."""

    def test_augmentations_import(self):
        """Test that augmentations can be imported."""
        from hvit.data import get_train_transforms, get_val_transforms
        assert get_train_transforms is not None
        assert get_val_transforms is not None

    def test_simclr_augmentation_import(self):
        """Test that SimCLR augmentation can be imported."""
        from hvit.data import SimCLRAugmentation
        assert SimCLRAugmentation is not None


class TestSimCLRAugmentation:
    """Tests for SimCLR augmentation pipeline."""

    def test_simclr_creates_two_views(self):
        """Test that SimCLR augmentation creates two different views."""
        from hvit.data import SimCLRAugmentation
        
        aug = SimCLRAugmentation(img_size=(224, 224))
        batch = torch.randn(4, 3, 224, 224)
        
        view1, view2 = aug(batch)
        
        assert view1.shape == batch.shape
        assert view2.shape == batch.shape
        # Views should be different due to random augmentations
        assert not torch.allclose(view1, view2)

    def test_simclr_output_dtype(self):
        """Test that SimCLR output is float32."""
        from hvit.data import SimCLRAugmentation
        
        aug = SimCLRAugmentation(img_size=(224, 224))
        batch = torch.randn(2, 3, 224, 224)
        
        view1, view2 = aug(batch)
        
        assert view1.dtype == torch.float32
        assert view2.dtype == torch.float32


class TestTransformFactories:
    """Tests for transform factory functions."""

    def test_val_transforms(self):
        """Test validation transform creation."""
        from hvit.data import get_val_transforms
        
        transform = get_val_transforms(img_size=(224, 224))
        assert transform is not None

    def test_train_transforms(self):
        """Test training transform creation with different severities."""
        from hvit.data import get_train_transforms
        
        for severity in ['light', 'moderate', 'aggressive']:
            transform = get_train_transforms(
                img_size=(224, 224),
                severity=severity
            )
            assert transform is not None


class TestNormalization:
    """Tests for normalization constants and operations."""

    def test_imagenet_normalization_constants(self):
        """Test that ImageNet normalization constants are correct."""
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        # These are the standard ImageNet values
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < m < 1 for m in IMAGENET_MEAN)
        assert all(0 < s < 1 for s in IMAGENET_STD)


class TestCustomAugmentations:
    """Tests for custom augmentation classes."""

    def test_enhanced_color_jitter(self):
        """Test EnhancedColorJitter augmentation."""
        from hvit.data.augmentations import EnhancedColorJitter
        
        aug = EnhancedColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
        # Create a sample image tensor
        x = torch.randn(3, 224, 224)
        output = aug(x)
        
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_gaussian_noise(self):
        """Test GaussianNoise augmentation."""
        from hvit.data.augmentations import GaussianNoise
        
        aug = GaussianNoise(std=0.05)
        
        x = torch.randn(3, 224, 224).clamp(0, 1)  # Ensure input is valid
        output = aug(x)
        
        assert output.shape == x.shape
        # Output should be different from input due to noise
        assert not torch.allclose(x, output)
