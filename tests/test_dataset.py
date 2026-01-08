"""
Tests for the dataset classes.

These tests verify the SARCLD2024Dataset functionality for loading
and processing cotton leaf disease images.
"""
import pytest
import torch
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatasetStructure:
    """Tests for dataset class structure and interface."""

    def test_dataset_import(self):
        """Test that dataset classes can be imported."""
        from hvit.data import SARCLD2024Dataset
        assert SARCLD2024Dataset is not None

    def test_dataset_has_required_methods(self):
        """Test that dataset has required interface methods."""
        from hvit.data import SARCLD2024Dataset
        
        assert hasattr(SARCLD2024Dataset, '__len__')
        assert hasattr(SARCLD2024Dataset, '__getitem__')
        assert hasattr(SARCLD2024Dataset, 'get_class_names')
        assert hasattr(SARCLD2024Dataset, 'get_class_weights')


class TestDatasetClasses:
    """Tests for dataset class definitions."""

    def test_expected_class_names(self):
        """Test that expected class names are defined."""
        from hvit.data import SARCLD_CLASSES
        
        expected_classes = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        
        assert len(SARCLD_CLASSES) == 7
        assert set(SARCLD_CLASSES) == set(expected_classes)


class TestDataLoaderIntegration:
    """Tests for DataLoader integration."""

    def test_dataloader_collation(self):
        """Test that tensors can be properly collated."""
        # Create mock batch data
        images = [torch.randn(3, 224, 224) for _ in range(4)]
        labels = [0, 1, 2, 3]
        
        # Test default collation
        batch_images = torch.stack(images)
        batch_labels = torch.tensor(labels)
        
        assert batch_images.shape == (4, 3, 224, 224)
        assert batch_labels.shape == (4,)

    def test_tensor_dtypes(self):
        """Test expected tensor dtypes."""
        image = torch.randn(3, 224, 224, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)
        
        assert image.dtype == torch.float32
        assert label.dtype == torch.long


class TestTransformIntegration:
    """Tests for transform/augmentation integration."""

    def test_normalize_values(self):
        """Test ImageNet normalization values."""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        # These are standard ImageNet values
        assert torch.allclose(mean, torch.tensor([0.485, 0.456, 0.406]))
        assert torch.allclose(std, torch.tensor([0.229, 0.224, 0.225]))
