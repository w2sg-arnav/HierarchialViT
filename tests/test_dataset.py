import pytest
import torch
from torch.utils.data import DataLoader

from hvit.data.dataset import HViTDataset

@pytest.fixture
def dataset_config():
    return {
        "root": "data/imagenet",
        "split": "train",
        "transform": None,
    }

def test_dataset_loading(dataset_config):
    dataset = HViTDataset(**dataset_config)
    assert len(dataset) > 0
    
    item = dataset[0]
    assert isinstance(item[0], torch.Tensor)  # image
    assert isinstance(item[1], int)  # label

def test_dataloader():
    dataset = HViTDataset(
        root="data/imagenet",
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )
    
    batch = next(iter(loader))
    assert len(batch) == 2  # (images, labels)
    assert batch[0].shape == (32, 3, 224, 224)
    assert batch[1].shape == (32,)

def test_dataset_transforms():
    transform = torch.nn.Sequential(
        torch.nn.ConvertImageDtype(torch.float32),
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    )
    
    dataset = HViTDataset(
        root="data/imagenet",
        split="train",
        transform=transform,
    )
    
    img, _ = dataset[0]
    assert img.dtype == torch.float32
    assert img.min() >= -3 and img.max() <= 3  # Normalized values
