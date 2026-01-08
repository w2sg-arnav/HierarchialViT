# 🏗️ Model Architecture

This document provides a detailed technical description of the HierarchicalViT (HViT) architecture.

## Table of Contents

- [Overview](#overview)
- [Patch Embedding](#patch-embedding)
- [Hierarchical Stages](#hierarchical-stages)
- [Transformer Block](#transformer-block)
- [Patch Merging](#patch-merging)
- [Disease-Focused Cross-Attention (DFCA)](#disease-focused-cross-attention-dfca)
- [Classification Head](#classification-head)
- [Model Variants](#model-variants)

---

## Overview

HierarchicalViT is a hierarchical vision transformer that processes images through multiple stages with progressive spatial downsampling. The architecture is inspired by Swin Transformer but optimized for plant disease classification.

### Key Design Principles

1. **Hierarchical Feature Learning**: Multi-scale representations from fine to coarse
2. **Efficient Computation**: Linear complexity with respect to image size within each stage
3. **Strong Inductive Biases**: Locality preserved through windowed attention patterns
4. **Flexible Fusion**: Optional multi-modal fusion via DFCA module

### Architecture Diagram

```
Input Image (B, 3, H, W)
         │
         ▼
┌────────────────────────────────────────────────────┐
│              Patch Embedding (16×16)                │
│   Conv2D(3 → 96, kernel=16, stride=16) + Reshape   │
│              Output: (B, H/16 × W/16, 96)          │
└────────────────────────────────────────────────────┘
         │
         ▼ + Positional Embedding (learnable)
         │
┌────────────────────────────────────────────────────┐
│                   Stage 1                           │
│   depth=2, dim=96, heads=3, mlp_ratio=4.0          │
│   Resolution: H/16 × W/16                           │
└────────────────────────────────────────────────────┘
         │
         ▼ Patch Merging (spatial 2× downsample)
         │
┌────────────────────────────────────────────────────┐
│                   Stage 2                           │
│   depth=2, dim=192, heads=6, mlp_ratio=4.0         │
│   Resolution: H/32 × W/32                           │
└────────────────────────────────────────────────────┘
         │
         ▼ Patch Merging
         │
┌────────────────────────────────────────────────────┐
│                   Stage 3                           │
│   depth=6, dim=384, heads=12, mlp_ratio=4.0        │
│   Resolution: H/64 × W/64                           │
└────────────────────────────────────────────────────┘
         │
         ▼ Patch Merging
         │
┌────────────────────────────────────────────────────┐
│                   Stage 4                           │
│   depth=2, dim=768, heads=24, mlp_ratio=4.0        │
│   Resolution: H/128 × W/128                         │
└────────────────────────────────────────────────────┘
         │
         ▼ Global Average Pooling
         │
┌────────────────────────────────────────────────────┐
│         LayerNorm + Linear(768, num_classes)       │
└────────────────────────────────────────────────────┘
         │
         ▼
   Output Logits (B, num_classes)
```

---

## Patch Embedding

The patch embedding layer converts input images into a sequence of tokens.

### Implementation

```python
class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 96
    ):
        # Convolutional projection
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 16 | Size of each patch (16×16 pixels) |
| `in_chans` | 3 | Input channels (RGB) |
| `embed_dim` | 96 | Output embedding dimension |

### Computation

For a 256×256 input image:
- Number of patches: `(256/16) × (256/16) = 16 × 16 = 256`
- Output shape: `(B, 256, 96)`

---

## Hierarchical Stages

Each stage consists of multiple transformer blocks operating at the same spatial resolution.

### Stage Configuration (HViT-Small)

| Stage | Depth | Channels | Heads | Head Dim | MLP Hidden |
|:-----:|:-----:|:--------:|:-----:|:--------:|:----------:|
| 1 | 2 | 96 | 3 | 32 | 384 |
| 2 | 2 | 192 | 6 | 32 | 768 |
| 3 | 6 | 384 | 12 | 32 | 1536 |
| 4 | 2 | 768 | 24 | 32 | 3072 |

### Resolution Progression

| Stage | Input Resolution | After Patch Merge |
|:-----:|:----------------:|:-----------------:|
| 1 | 16 × 16 | 8 × 8 |
| 2 | 8 × 8 | 4 × 4 |
| 3 | 4 × 4 | 2 × 2 |
| 4 | 2 × 2 | - (final) |

---

## Transformer Block

Each transformer block follows the pre-norm design with Drop Path regularization.

### Block Structure

```
Input
  │
  ├──────────────────────────────┐
  │                              │
  ▼                              │
LayerNorm                        │
  │                              │
  ▼                              │
Multi-Head Self-Attention        │
  │                              │
  ▼                              │
DropPath (stochastic depth)      │
  │                              │
  ▼                              │
Add ◄────────────────────────────┘
  │
  ├──────────────────────────────┐
  │                              │
  ▼                              │
LayerNorm                        │
  │                              │
  ▼                              │
MLP (Linear → GELU → Linear)     │
  │                              │
  ▼                              │
DropPath                         │
  │                              │
  ▼                              │
Add ◄────────────────────────────┘
  │
  ▼
Output
```

### Multi-Head Self-Attention

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
```

### MLP Block

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features)
        self.drop = nn.Dropout(drop)
```

---

## Patch Merging

Patch merging reduces spatial resolution by 2× while doubling the channel dimension.

### Algorithm

1. **Reshape**: Arrange patches into 2×2 groups
2. **Concatenate**: Stack 4 patches along channel dimension (C → 4C)
3. **Project**: Linear layer reduces channels (4C → 2C)
4. **Normalize**: LayerNorm on output

### Implementation

```python
class PatchMerging(nn.Module):
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Pad if needed for even dimensions
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        # Split into 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        # Concatenate and project
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        
        return x
```

---

## Disease-Focused Cross-Attention (DFCA)

The DFCA module enables fusion between RGB and spectral features (optional).

### Architecture

```
RGB Features (B, N, C)      Spectral Features (B, N, C)
        │                           │
        ▼                           ▼
   Query (Wq)              Key (Wk), Value (Wv)
        │                           │
        └─────────┬─────────────────┘
                  │
                  ▼
        Cross-Attention
        Attention(Q_rgb, K_spec, V_spec)
                  │
                  ▼
            Projection
                  │
                  ▼
        LayerNorm + Residual
                  │
                  ▼
         Fused Features (B, N, C)
```

### Implementation

```python
class DiseaseFocusedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
```

### Usage

Enable DFCA in configuration:

```yaml
model:
  use_dfca: true
  spectral_channels: 4  # Additional spectral bands
```

---

## Classification Head

The classification head converts pooled features to class predictions.

### Architecture

```
Stage 4 Output (B, 4, 768)
         │
         ▼
Global Average Pooling
   (B, 768)
         │
         ▼
LayerNorm(768)
         │
         ▼
Linear(768, num_classes)
         │
         ▼
Logits (B, num_classes)
```

---

## Model Variants

### HViT-Tiny

| Parameter | Value |
|-----------|-------|
| Embed Dim | 64 |
| Depths | [2, 2, 4, 2] |
| Heads | [2, 4, 8, 16] |
| Parameters | ~12M |

### HViT-Small (Default)

| Parameter | Value |
|-----------|-------|
| Embed Dim | 96 |
| Depths | [2, 2, 6, 2] |
| Heads | [3, 6, 12, 24] |
| Parameters | ~28M |

### HViT-Base

| Parameter | Value |
|-----------|-------|
| Embed Dim | 128 |
| Depths | [2, 2, 18, 2] |
| Heads | [4, 8, 16, 32] |
| Parameters | ~88M |

---

## Creating Custom Models

```python
from hvit.models import create_disease_aware_hvt

# Custom configuration
model = create_disease_aware_hvt(
    current_img_size=(224, 224),
    num_classes=10,
    model_params_dict={
        "patch_size": 16,
        "embed_dim_rgb": 64,
        "depths": [2, 2, 4, 2],
        "num_heads": [2, 4, 8, 16],
        "mlp_ratio": 4.0,
        "drop_path_rate": 0.1,
        "use_dfca": False,
        "use_gradient_checkpointing": True,
    }
)
```

---

## References

- [Swin Transformer](https://arxiv.org/abs/2103.14030) - Liu et al., 2021
- [Vision Transformer](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020
- [DeiT](https://arxiv.org/abs/2012.12877) - Touvron et al., 2021

For implementation details, see [`hvit/models/hvt.py`](../hvit/models/hvt.py).
