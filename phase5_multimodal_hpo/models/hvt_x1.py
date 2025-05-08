import torch
import torch.nn as nn
from .dfca import DFCA

class DiseaseAwareHVT_XL(nn.Module):
    def __init__(self, img_size=512, patch_size=14, in_chans=3, spectral_chans=3,
                 num_classes=7, embed_dim_rgb=128, embed_dim_spectral=128,
                 depths=[3,6,18,3], num_heads=[4,8,16,32], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0.2, attn_drop_rate=0.1,
                 drop_path_rate=0.3, use_dfca=True, dfca_heads=16):
        
        super().__init__()
        
        # Enhanced Patch Embedding
        self.rgb_patch_embed = nn.Conv2d(in_chans, embed_dim_rgb, 
                                       kernel_size=patch_size, 
                                       stride=patch_size)
        self.spectral_patch_embed = nn.Conv2d(spectral_chans, embed_dim_spectral,
                                            kernel_size=patch_size,
                                            stride=patch_size)
        
        # Positional Embeddings
        num_patches = (img_size // patch_size) ** 2
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim_rgb))
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim_spectral))
        
        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.rgb_stages = nn.ModuleList()
        self.spectral_stages = nn.ModuleList()
        self.dfca_modules = nn.ModuleList()
        
        for i in range(len(depths)):
            rgb_blocks = []
            spectral_blocks = []
            for j in range(depths[i]):
                # RGB Branch
                rgb_blocks.append(
                    TransformerBlock(embed_dim_rgb, num_heads[i], mlp_ratio,
                                    qkv_bias, drop_rate, attn_drop_rate, dpr.pop(0))
                )
                # Spectral Branch
                spectral_blocks.append(
                    TransformerBlock(embed_dim_spectral, num_heads[i], mlp_ratio,
                                    qkv_bias, drop_rate, attn_drop_rate, dpr.pop(0))
                )
            
            self.rgb_stages.append(nn.Sequential(*rgb_blocks))
            self.spectral_stages.append(nn.Sequential(*spectral_blocks))
            
            if use_dfca and i < len(depths)-1:
                self.dfca_modules.append(
                    DFCA(embed_dim_rgb, embed_dim_spectral, num_heads=dfca_heads)
                )
        
        # Final Head
        self.head_norm = nn.LayerNorm(embed_dim_rgb + embed_dim_spectral)
        self.head = nn.Linear(embed_dim_rgb + embed_dim_spectral, num_classes)
        
        # Initialization
        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, rgb, spectral):
        # Feature extraction logic
        # ... [previous implementation] ...
        return fused_features
    
    def forward(self, rgb, spectral=None):
        x = self.forward_features(rgb, spectral)
        x = self.head_norm(x)
        return self.head(x.mean(dim=1))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_dim=int(dim*mlp_ratio), drop=drop)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x