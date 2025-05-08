# models/dfca.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class DiseaseFocusedCrossAttention(nn.Module):
    """
    Disease-Focused Cross-Attention (DFCA) for multi-modal fusion of RGB and spectral features.
    Uses spectral features (potentially enhanced by a disease prior) to modulate RGB features.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 dropout_rate: float = 0.1, use_disease_mask: bool = True):
        """
        Args:
            embed_dim (int): Embedding dimension for both RGB and spectral features.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for attention and FFN.
            use_disease_mask (bool): Whether to use a learnable disease prior/mask.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_disease_mask = use_disease_mask

        if self.use_disease_mask:
            # Learnable disease prior (e.g., a global bias or scaling factor for spectral features)
            # Shape [1, 1, embed_dim] to be broadcastable across sequence length and batch.
            # This is a simple form; more complex spatial masks could be learned.
            self.disease_mask_param = nn.Parameter(torch.zeros(1, 1, embed_dim)) # Initialize to zeros
            # Consider nn.init.xavier_uniform_(self.disease_mask_param) or similar if starting non-zero
            logger.info("DFCA: Initialized with learnable disease mask parameter.")
        
        # Cross-attention module: RGB is query, Spectral is key/value
        # batch_first=False because HVT typically outputs (seq_len, batch, embed_dim) before this
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=False)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Feed-forward network (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # Common MLP ratio of 4
            nn.GELU(),
            nn.Dropout(dropout_rate), # Dropout after GELU and before second linear
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate) # Dropout after FFN output / before residual
        
        logger.info(f"DFCA initialized: embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout_rate}")

    def forward(self, rgb_features: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_features (torch.Tensor): RGB features [seq_len, batch_size, embed_dim].
            spectral_features (torch.Tensor): Spectral features [seq_len, batch_size, embed_dim].
        
        Returns:
            torch.Tensor: Fused features [seq_len, batch_size, embed_dim].
        """
        # logger.debug(f"DFCA input - RGB shape: {rgb_features.shape}, Spectral shape: {spectral_features.shape}")

        spectral_qkv = spectral_features
        if self.use_disease_mask:
            # Apply disease mask: additive bias to spectral features
            # This could make spectral features more "attentive" to disease patterns.
            spectral_qkv = spectral_features + self.disease_mask_param
            # Alternative: multiplicative self.disease_mask_param = nn.Parameter(torch.ones(1, 1, embed_dim))
            # spectral_qkv = spectral_features * self.disease_mask_param

        # Cross-attention: rgb_features are queries, (masked) spectral_features are keys and values
        # attn_output: [seq_len, batch_size, embed_dim]
        attn_output, attn_weights = self.cross_attention(
            query=rgb_features,
            key=spectral_qkv,
            value=spectral_qkv
        )
        # logger.debug(f"DFCA attention output shape: {attn_output.shape}")
        
        # First residual connection & normalization (Post-LN style)
        # x = query + dropout(attention_output)
        # x = norm(x)
        fused_features = rgb_features + self.dropout1(attn_output)
        fused_features = self.norm1(fused_features)
        
        # Feed-forward network
        ffn_output = self.ffn(fused_features)
        
        # Second residual connection & normalization
        fused_features = fused_features + self.dropout2(ffn_output)
        fused_features = self.norm2(fused_features)
        
        # logger.debug(f"DFCA output shape: {fused_features.shape}")
        return fused_features