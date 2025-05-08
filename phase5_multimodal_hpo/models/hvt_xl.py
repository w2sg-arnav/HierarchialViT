import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple

# --- Add torch.utils.checkpoint ---
from torch.utils.checkpoint import checkpoint as gradient_checkpoint # Renamed to avoid conflict


# --- Helper Modules (DropPath, Mlp, TransformerBlock, DFCA - UNCHANGED from previous version) ---
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DFCA(nn.Module):
    def __init__(self, embed_dim_rgb, embed_dim_spectral, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert embed_dim_rgb == embed_dim_spectral, "DFCA expects RGB and Spectral dims to be equal for this placeholder."
        self.embed_dim = embed_dim_rgb
        self.num_heads = num_heads
        self.norm_rgb = nn.LayerNorm(embed_dim_rgb)
        self.norm_spectral = nn.LayerNorm(embed_dim_spectral)
        self.rgb_to_spectral_attn = nn.MultiheadAttention(embed_dim_rgb, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.spectral_to_rgb_attn = nn.MultiheadAttention(embed_dim_spectral, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.proj_rgb = nn.Linear(embed_dim_rgb, embed_dim_rgb)
        self.proj_spectral = nn.Linear(embed_dim_spectral, embed_dim_spectral)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, rgb_features, spectral_features):
        rgb_norm = self.norm_rgb(rgb_features)
        spectral_norm = self.norm_spectral(spectral_features)
        rgb_enhanced, _ = self.rgb_to_spectral_attn(rgb_norm, spectral_norm, spectral_norm)
        rgb_fused = rgb_features + self.proj_drop(self.proj_rgb(rgb_enhanced))
        spectral_enhanced, _ = self.spectral_to_rgb_attn(spectral_norm, rgb_norm, rgb_norm)
        spectral_fused = spectral_features + self.proj_drop(self.proj_spectral(spectral_enhanced))
        return rgb_fused, spectral_fused
# --- END Helper Modules ---


class DiseaseAwareHVT_XL(nn.Module):
    def __init__(self, img_size=512, patch_size=14, in_chans=3, spectral_chans=3,
                 num_classes=7, embed_dim_rgb=128, embed_dim_spectral=128,
                 depths=[3,6,18,3], num_heads=[4,8,16,32], mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0.2, attn_drop_rate=0.1,
                 drop_path_rate=0.3, use_dfca=True, dfca_heads=16,
                 norm_layer=nn.LayerNorm,
                 use_gradient_checkpointing=False): # MODIFIED: Added gradient checkpointing flag

        super().__init__()
        self.num_classes = num_classes
        self.embed_dim_rgb = embed_dim_rgb
        self.embed_dim_spectral = embed_dim_spectral
        self.use_dfca = use_dfca
        self.num_stages = len(depths)
        self.use_gradient_checkpointing = use_gradient_checkpointing # MODIFIED: Store the flag

        # --- RGB Branch ---
        self.rgb_patch_embed = nn.Conv2d(in_chans, embed_dim_rgb,
                                         kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim_rgb))
        self.rgb_pos_drop = nn.Dropout(p=drop_rate)

        # --- Spectral Branch ---
        self.spectral_patch_embed = nn.Conv2d(spectral_chans, embed_dim_spectral,
                                              kernel_size=patch_size, stride=patch_size)
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim_spectral))
        self.spectral_pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.rgb_stages = nn.ModuleList()
        self.spectral_stages = nn.ModuleList()
        if self.use_dfca:
            self.dfca_modules = nn.ModuleList()

        current_dpr_idx = 0
        for i in range(self.num_stages):
            rgb_stage_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dim_rgb, num_heads=num_heads[i], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[current_dpr_idx + j], norm_layer=norm_layer)
                for j in range(depths[i])
            ])
            self.rgb_stages.append(rgb_stage_blocks)

            spectral_stage_blocks = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dim_spectral, num_heads=num_heads[i], mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path_rate=dpr[current_dpr_idx + j], norm_layer=norm_layer)
                for j in range(depths[i])
            ])
            self.spectral_stages.append(spectral_stage_blocks)
            
            current_dpr_idx += depths[i]

            if self.use_dfca and i < self.num_stages - 1 :
                self.dfca_modules.append(
                    DFCA(embed_dim_rgb, embed_dim_spectral, num_heads=dfca_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate)
                )
        
        self.norm_rgb_final = norm_layer(embed_dim_rgb)
        self.norm_spectral_final = norm_layer(embed_dim_spectral)
        
        self.head_norm = norm_layer(embed_dim_rgb + embed_dim_spectral)
        self.head = nn.Linear(embed_dim_rgb + embed_dim_spectral, num_classes)

        nn.init.trunc_normal_(self.rgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _interpolate_pos_embed(self, checkpoint_pos_embed, current_pos_embed, patch_size, img_size_tuple):
        # ... (this function remains the same as previous version)
        N_src = checkpoint_pos_embed.shape[1]
        N_tgt = current_pos_embed.shape[1]
        if N_src == N_tgt: return checkpoint_pos_embed
        gs_old = int(math.sqrt(N_src))
        gs_new_h = img_size_tuple[0] // patch_size
        gs_new_w = img_size_tuple[1] // patch_size
        checkpoint_pos_embed_2d = checkpoint_pos_embed.permute(0, 2, 1).reshape(1, self.embed_dim_rgb, gs_old, gs_old)
        interpolated_pos_embed = F.interpolate(checkpoint_pos_embed_2d, size=(gs_new_h, gs_new_w), mode='bicubic', align_corners=False)
        interpolated_pos_embed = interpolated_pos_embed.reshape(1, self.embed_dim_rgb, N_tgt).permute(0, 2, 1)
        return interpolated_pos_embed

    def forward_features(self, rgb, spectral):
        B, _, H, W = rgb.shape

        x_rgb = self.rgb_patch_embed(rgb).flatten(2).transpose(1, 2)
        x_spec = self.spectral_patch_embed(spectral).flatten(2).transpose(1, 2)
        
        # Simplified Positional Embedding Handling (assuming fixed size or pre-interpolation)
        # For dynamic resizing during multi-scale, more robust pos_embed interpolation in forward might be needed,
        # but it's often handled by ensuring model's pos_embed matches current feature map size.
        # The previous dynamic interpolation logic in forward_features was a bit complex;
        # for now, assume pos_embed size matches num_patches or is handled at weight loading.
        if x_rgb.shape[1] == self.rgb_pos_embed.shape[1]:
            x_rgb = x_rgb + self.rgb_pos_embed
        else: # Fallback if num_patches changes (e.g. during multi-scale training)
            # This is a simple resizing, might not be optimal. Better to handle in dataset/model init.
            num_patches_current = x_rgb.shape[1]
            gs_current_h = H // self.rgb_patch_embed.kernel_size[0]
            gs_current_w = W // self.rgb_patch_embed.kernel_size[0]
            
            gs_orig_pe = int(math.sqrt(self.rgb_pos_embed.shape[1])) # Assuming original PE was square
            
            orig_pe_reshaped = self.rgb_pos_embed.permute(0,2,1).reshape(1, self.embed_dim_rgb, gs_orig_pe, gs_orig_pe)
            resized_pe = F.interpolate(orig_pe_reshaped, size=(gs_current_h, gs_current_w), mode='bicubic', align_corners=False)
            x_rgb = x_rgb + resized_pe.reshape(1, self.embed_dim_rgb, num_patches_current).permute(0,2,1)


        if x_spec.shape[1] == self.spectral_pos_embed.shape[1]:
            x_spec = x_spec + self.spectral_pos_embed
        else:
            num_patches_current = x_spec.shape[1]
            gs_current_h = H // self.spectral_patch_embed.kernel_size[0]
            gs_current_w = W // self.spectral_patch_embed.kernel_size[0]
            gs_orig_pe = int(math.sqrt(self.spectral_pos_embed.shape[1]))
            orig_pe_reshaped = self.spectral_pos_embed.permute(0,2,1).reshape(1, self.embed_dim_spectral, gs_orig_pe, gs_orig_pe)
            resized_pe = F.interpolate(orig_pe_reshaped, size=(gs_current_h, gs_current_w), mode='bicubic', align_corners=False)
            x_spec = x_spec + resized_pe.reshape(1, self.embed_dim_spectral, num_patches_current).permute(0,2,1)


        x_rgb = self.rgb_pos_drop(x_rgb)
        x_spec = self.spectral_pos_drop(x_spec)

        for i in range(self.num_stages):
            # MODIFIED: Apply gradient checkpointing if enabled
            if self.training and self.use_gradient_checkpointing:
                # Checkpointing each block - can be memory saving but slower
                for blk_rgb in self.rgb_stages[i]:
                    x_rgb = gradient_checkpoint(blk_rgb, x_rgb, use_reentrant=False)
                for blk_spec in self.spectral_stages[i]:
                    x_spec = gradient_checkpoint(blk_spec, x_spec, use_reentrant=False)
                if self.use_dfca and i < self.num_stages - 1:
                    # Checkpointing DFCA (ensure DFCA's forward args match what checkpoint expects)
                    # For a module with multiple inputs like DFCA(rgb, spectral),
                    # you might need a wrapper if checkpoint doesn't handle it directly.
                    # For now, assume it works or apply checkpointing inside DFCA if needed.
                    # Or, don't checkpoint DFCA if it's less memory intensive.
                     x_rgb, x_spec = gradient_checkpoint(self.dfca_modules[i], x_rgb, x_spec, use_reentrant=False)
            else: # No checkpointing (validation or disabled)
                for blk_rgb in self.rgb_stages[i]:
                    x_rgb = blk_rgb(x_rgb)
                for blk_spec in self.spectral_stages[i]:
                    x_spec = blk_spec(x_spec)
                if self.use_dfca and i < self.num_stages - 1:
                    x_rgb, x_spec = self.dfca_modules[i](x_rgb, x_spec)
        
        x_rgb = self.norm_rgb_final(x_rgb)
        x_spec = self.norm_spectral_final(x_spec)
        fused_features = torch.cat((x_rgb, x_spec), dim=2)
        return fused_features

    def forward(self, rgb, spectral=None):
        if spectral is None and self.embed_dim_spectral > 0 and self.spectral_patch_embed.in_channels > 0:
            raise ValueError("Spectral input is required for this DiseaseAwareHVT_XL configuration.")
            
        fused_features = self.forward_features(rgb, spectral)
        x = self.head_norm(fused_features.mean(dim=1))
        return self.head(x)