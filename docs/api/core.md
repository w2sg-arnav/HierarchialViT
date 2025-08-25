# HierarchialViT API Documentation

## Core Components

### Model Architecture

```python
class HierarchialViT(nn.Module):
    """Hierarchical Vision Transformer model.
    
    This model processes visual information through multiple hierarchical stages,
    with progressive dimension reduction and feature refinement.
    
    Args:
        img_size (int): Input image size (default: 224)
        patch_size (int): Size of each image patch (default: 16)
        in_chans (int): Number of input channels (default: 3)
        num_classes (int): Number of classes for classification (default: 1000)
        embed_dims (List[int]): Embedding dimensions for each stage (default: [64, 128, 256, 512])
        num_heads (List[int]): Number of attention heads for each stage (default: [1, 2, 4, 8])
        mlp_ratios (List[int]): MLP expansion ratio for each stage (default: [4, 4, 4, 4])
        depths (List[int]): Number of transformer blocks for each stage (default: [3, 4, 6, 3])
        sr_ratios (List[int]): Spatial reduction ratios for each stage (default: [8, 4, 2, 1])
        
    Attributes:
        patch_embed (PatchEmbed): Patch embedding layer
        pos_embed (nn.Parameter): Position embedding
        stages (nn.ModuleList): List of transformer stages
        norm (nn.LayerNorm): Final normalization layer
        head (nn.Linear): Classification head
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        pass
```

### Hierarchical Stage

```python
class HierarchicalStage(nn.Module):
    """A single stage in the hierarchical transformer.
    
    Each stage processes features at a specific scale, with its own attention
    and feed-forward blocks.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP expansion ratio
        depth (int): Number of transformer blocks
        sr_ratio (int): Spatial reduction ratio
        
    Attributes:
        blocks (nn.ModuleList): List of transformer blocks
        downsample (Optional[nn.Module]): Optional downsampling layer
    """
    pass
```

### Attention Mechanism

```python
class MultiScaleAttention(nn.Module):
    """Multi-scale self attention mechanism.
    
    Implements attention that can operate at different scales simultaneously.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projection
        sr_ratio (int): Spatial reduction ratio
        
    Attributes:
        scale (float): Attention scale factor
        qkv (nn.Linear): Combined QKV projection
        proj (nn.Linear): Output projection
    """
    pass
```

## Training Components

### Trainer

```python
class HViTTrainer:
    """Trainer class for HierarchialViT.
    
    Handles the training loop, validation, and model checkpointing.
    
    Args:
        model (HierarchialViT): Model instance
        optimizer (torch.optim.Optimizer): Optimizer instance
        criterion (nn.Module): Loss function
        device (torch.device): Device to train on
        config (Dict): Training configuration
        
    Methods:
        train_epoch(): Train for one epoch
        validate(): Run validation
        save_checkpoint(): Save model checkpoint
        load_checkpoint(): Load model checkpoint
    """
    pass
```

### Data Pipeline

```python
class HViTDataset(Dataset):
    """Dataset class for HierarchialViT.
    
    Handles data loading and augmentation.
    
    Args:
        root (str): Root directory of dataset
        transform (Optional[Callable]): Data transformation pipeline
        split (str): Dataset split ('train' or 'val')
        
    Methods:
        __getitem__(): Get a single sample
        __len__(): Get dataset size
    """
    pass
```

## Utility Functions

### Metrics

```python
def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calculate various performance metrics.
    
    Args:
        outputs (torch.Tensor): Model outputs
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    pass
```

### Visualization

```python
def visualize_attention(
    model: HierarchialViT,
    image: torch.Tensor,
    save_path: str
) -> None:
    """Visualize attention patterns.
    
    Args:
        model (HierarchialViT): Model instance
        image (torch.Tensor): Input image
        save_path (str): Path to save visualization
    """
    pass
```
