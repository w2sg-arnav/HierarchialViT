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

    def test_phase4_augmentations_import(self):
        """Test that phase4 augmentations can be imported."""
        from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
        assert create_cotton_leaf_augmentation is not None

    def test_phase3_augmentations_import(self):
        """Test that phase3 augmentations can be imported."""
        from phase3_pretraining.utils.augmentations import SimCLRAugmentation
        assert SimCLRAugmentation is not None


class TestSimCLRAugmentation:
    """Tests for SimCLR augmentation pipeline."""

    def test_simclr_creates_two_views(self):
        """Test that SimCLR augmentation creates two different views."""
        from phase3_pretraining.utils.augmentations import SimCLRAugmentation
        
        aug = SimCLRAugmentation(img_size=(224, 224))
        batch = torch.randn(4, 3, 224, 224)
        
        view1, view2 = aug(batch)
        
        assert view1.shape == batch.shape
        assert view2.shape == batch.shape
        # Views should be different due to random augmentations
        assert not torch.allclose(view1, view2)

    def test_simclr_output_dtype(self):
        """Test that SimCLR output is float32."""
        from phase3_pretraining.utils.augmentations import SimCLRAugmentation
        
        aug = SimCLRAugmentation(img_size=(224, 224))
        batch = torch.randn(2, 3, 224, 224)
        
        view1, view2 = aug(batch)
        
        assert view1.dtype == torch.float32
        assert view2.dtype == torch.float32


class TestCottonLeafAugmentation:
    """Tests for cotton leaf disease augmentation pipeline."""

    def test_minimal_strategy(self):
        """Test minimal augmentation strategy."""
        from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
        
        transform = create_cotton_leaf_augmentation(
            strategy='minimal',
            img_size=(224, 224)
        )
        
        assert transform is not None

    def test_cotton_disease_strategy(self):
        """Test cotton disease augmentation strategy."""
        from phase4_finetuning.utils.augmentations import create_cotton_leaf_augmentation
        
        transform = create_cotton_leaf_augmentation(
            strategy='cotton_disease',
            img_size=(224, 224),
            severity='moderate'
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

    def test_normalization_operation(self):
        """Test that normalization produces expected range."""
        import torchvision.transforms.v2 as T_v2
        
        normalize = T_v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Input in [0, 1] range
        img = torch.rand(3, 224, 224)
        normalized = normalize(img)
        
        # Normalized values should be roughly in [-2.5, 2.5] range
        assert normalized.min() >= -3.0
        assert normalized.max() <= 3.0


class TestTensorOperations:
    """Tests for tensor operations in augmentations."""

    def test_resize_maintains_channels(self):
        """Test that resize operations maintain channel count."""
        import torchvision.transforms.v2 as T_v2
        
        resize = T_v2.Resize((224, 224))
        img = torch.randn(3, 256, 256)
        resized = resize(img)
        
        assert resized.shape[0] == 3
        assert resized.shape[1] == 224
        assert resized.shape[2] == 224

    def test_batch_augmentation_shape(self):
        """Test that batch augmentation maintains shape."""
        import torchvision.transforms.v2 as T_v2
        
        transform = T_v2.Compose([
            T_v2.RandomHorizontalFlip(p=0.5),
            T_v2.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        batch = torch.randn(4, 3, 224, 224)
        transformed = transform(batch)
        
        assert transformed.shape == batch.shape
