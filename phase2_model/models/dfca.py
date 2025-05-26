# phase2_model/models/dfca.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DiseaseFocusedCrossAttention(nn.Module):
    """
    Disease-Focused Cross-Attention (DFCA) for multi-modal fusion of RGB and spectral features.
    Uses spectral features (potentially enhanced by a disease prior) to modulate RGB features.
    Expects inputs of shape: [seq_len, batch_size, embed_dim]
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout_rate: float = 0.1, use_disease_mask: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"DFCA Error: embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_disease_mask = use_disease_mask

        if self.use_disease_mask:
            self.disease_mask_param = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Additive bias
            # Consider nn.init.normal_(self.disease_mask_param, std=0.02) for non-zero start
            logger.info("DFCA: Initialized with learnable disease mask parameter (additive).")

        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        logger.info(f"DiseaseFocusedCrossAttention module initialized: embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout_rate}, use_mask={use_disease_mask}")

    def forward(self, rgb_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_features (torch.Tensor): RGB features [seq_len, batch_size, embed_dim].
            spectral_features (torch.Tensor): Spectral features [seq_len, batch_size, embed_dim].

        Returns:
            torch.Tensor: Fused features [seq_len, batch_size, embed_dim].
        """
        spectral_qkv = spectral_features
        if self.use_disease_mask:
            spectral_qkv = spectral_features + self.disease_mask_param # Apply additive bias

        # Cross-attention: rgb_features are queries, (masked) spectral_features are keys and values
        attn_output, _ = self.cross_attention(
            query=rgb_features,
            key=spectral_qkv,
            value=spectral_qkv
        )

        # First residual connection & normalization (Post-LN style)
        fused_features = rgb_features + self.dropout1(attn_output)
        fused_features = self.norm1(fused_features)

        # Feed-forward network
        ffn_output = self.ffn(fused_features)

        # Second residual connection & normalization
        fused_features = fused_features + self.dropout2(ffn_output)
        fused_features = self.norm2(fused_features)

        return fused_features