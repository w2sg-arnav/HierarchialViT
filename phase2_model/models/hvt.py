# phase2_model/models/hvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import math 

# Assuming phase3_pretraining.config is the single source of truth now
# Need to make sure that when this file is imported, the sys.path allows finding phase3_pretraining
# The sys.path modification in phase3_pretraining/pretrain.py achieves this when that script is run.
try:
    # Import config values using the phase3 path and aliases
    from phase3_pretraining.config import ( 
        PATCH_SIZE as cfg_PATCH_SIZE,
        EMBED_DIM_RGB as cfg_EMBED_DIM_RGB, 
        EMBED_DIM_SPECTRAL as cfg_EMBED_DIM_SPECTRAL,
        HVT_DEPTHS as cfg_HVT_DEPTHS, 
        HVT_NUM_HEADS as cfg_HVT_NUM_HEADS,
        MLP_RATIO as cfg_MLP_RATIO, 
        QKV_BIAS as cfg_QKV_BIAS,
        HVT_ATTN_DROP_RATE as cfg_HVT_ATTN_DROP_RATE, 
        HVT_MODEL_DROP_RATE as cfg_HVT_MODEL_DROP_RATE, 
        HVT_DROP_PATH_RATE as cfg_HVT_DROP_PATH_RATE,
        DFCA_NUM_HEADS as cfg_DFCA_NUM_HEADS, 
        DFCA_DROP_RATE as cfg_DFCA_DROP_RATE,
        NUM_CLASSES as cfg_NUM_CLASSES, 
        SPECTRAL_CHANNELS as cfg_SPECTRAL_CHANNELS
    )
    # === CORRECTED IMPORT for DFCA ===
    # Import DFCA using relative import, assuming dfca.py is in the same directory (models/)
    from .dfca import DiseaseFocusedCrossAttention 
    # ================================

except ImportError as e:
    # Fallback or error if imports fail
    print(f"ERROR in phase2_model/models/hvt.py: Could not import config from phase3_pretraining or relative DFCA. Error: {e}")
    # Define dummy values or raise error to prevent proceeding with incorrect config
    cfg_PATCH_SIZE, cfg_EMBED_DIM_RGB, cfg_EMBED_DIM_SPECTRAL = 16, 96, 96
    cfg_HVT_DEPTHS, cfg_HVT_NUM_HEADS = [2,2,6,2], [3,6,12,24]
    cfg_MLP_RATIO, cfg_QKV_BIAS = 4.0, True
    cfg_HVT_ATTN_DROP_RATE, cfg_HVT_MODEL_DROP_RATE, cfg_HVT_DROP_PATH_RATE = 0.0, 0.0, 0.1
    cfg_DFCA_NUM_HEADS, cfg_DFCA_DROP_RATE = 8, 0.1 
    cfg_NUM_CLASSES, cfg_SPECTRAL_CHANNELS = 7, 1 
    # Define a dummy DFCA class if import failed
    class DiseaseFocusedCrossAttention(nn.Module): 
        def __init__(self, *args, **kwargs): super().__init__(); self.identity = nn.Identity()
        def forward(self, rgb, spec): return rgb # Passthrough

import logging

logger = logging.getLogger(__name__)

# --- Helper Modules (DropPath, PatchEmbed, Attention, Mlp, TransformerBlock, PatchMerging) ---
# Unchanged from previous version - keep as they were
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            logger.warning(f"PatchEmbed: Image dimensions {img_size} are not perfectly divisible by patch_size {patch_size}.")
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution_patches: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution_patches = input_resolution_patches 
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H_patch, W_patch = self.input_resolution_patches
        if L != H_patch * W_patch:
            if math.isqrt(L)**2 == L:
                H_patch = W_patch = math.isqrt(L)
            else:
                raise ValueError(f"PatchMerging: L={L} != H_patch*W_patch={H_patch*W_patch}, and L not perfect square.")
        
        x = x.view(B, H_patch, W_patch, C)
        H_slice = (H_patch // 2) * 2
        W_slice = (W_patch // 2) * 2
        x0 = x[:, 0:H_slice:2, 0:W_slice:2, :]
        x1 = x[:, 1:H_slice:2, 0:W_slice:2, :]
        x2 = x[:, 0:H_slice:2, 1:W_slice:2, :]
        x3 = x[:, 1:H_slice:2, 1:W_slice:2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution_patches={self.input_resolution_patches}, dim={self.dim}"

class HVTStage(nn.Module):
    def __init__(self, dim: int, input_resolution_patches: Tuple[int, int], depth: int, num_heads: int,
                 mlp_ratio: float, qkv_bias: bool, drop: float, attn_drop: float,
                 drop_path_prob: Union[List[float], float] = 0., norm_layer=nn.LayerNorm,
                 downsample_class: Optional[type[nn.Module]] = None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution_patches = input_resolution_patches
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                             norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample_layer = None
        if downsample_class is not None:
            self.downsample_layer = downsample_class(input_resolution_patches=input_resolution_patches, dim=dim, norm_layer=norm_layer)
            self.output_resolution_patches = (input_resolution_patches[0] // 2, input_resolution_patches[1] // 2)
        else:
            self.output_resolution_patches = input_resolution_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and not torch.jit.is_scripting() and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x


# --- DiseaseAwareHVT Class (Main Backbone) ---
class DiseaseAwareHVT(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), 
                 patch_size: int = cfg_PATCH_SIZE,
                 num_classes: int = cfg_NUM_CLASSES,
                 embed_dim_rgb: int = cfg_EMBED_DIM_RGB, 
                 embed_dim_spectral: int = cfg_EMBED_DIM_SPECTRAL,
                 spectral_channels: int = cfg_SPECTRAL_CHANNELS,
                 depths: List[int] = cfg_HVT_DEPTHS, 
                 num_heads: List[int] = cfg_HVT_NUM_HEADS,
                 mlp_ratio: float = cfg_MLP_RATIO, 
                 qkv_bias: bool = cfg_QKV_BIAS,
                 drop_rate: float = cfg_HVT_MODEL_DROP_RATE, 
                 attn_drop_rate: float = cfg_HVT_ATTN_DROP_RATE,
                 drop_path_rate: float = cfg_HVT_DROP_PATH_RATE,
                 norm_layer=nn.LayerNorm, 
                 dfca_num_heads: int = cfg_DFCA_NUM_HEADS, 
                 dfca_drop_rate: float = cfg_DFCA_DROP_RATE,
                 use_dfca: bool = True
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.patch_size = patch_size
        self.use_dfca = use_dfca
        self.img_size_at_init = img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        # --- RGB Stream ---
        self.rgb_patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size,
                                          in_chans=3, embed_dim=embed_dim_rgb,
                                          norm_layer=norm_layer)
        num_patches_at_init = self.rgb_patch_embed.num_patches
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_init, embed_dim_rgb))
        nn.init.trunc_normal_(self.rgb_pos_embed, std=.02)
        self.pos_drop_rgb = nn.Dropout(p=drop_rate)

        self.rgb_stages = nn.ModuleList()
        current_dim_rgb = embed_dim_rgb
        current_resolution_patches_rgb = self.rgb_patch_embed.grid_size
        for i_stage in range(self.num_stages):
            stage = HVTStage(dim=current_dim_rgb,
                             input_resolution_patches=current_resolution_patches_rgb,
                             depth=depths[i_stage],
                             num_heads=num_heads[i_stage],
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path_prob=dpr[sum(depths[:i_stage]):sum(depths[:i_stage+1])],
                             norm_layer=norm_layer,
                             downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                            )
            self.rgb_stages.append(stage)
            if i_stage < self.num_stages - 1:
                current_resolution_patches_rgb = stage.output_resolution_patches
                current_dim_rgb *= 2
        self.norm_rgb = norm_layer(current_dim_rgb)

        # --- Spectral Stream ---
        if spectral_channels > 0:
            self.spectral_patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size,
                                                   in_chans=spectral_channels, embed_dim=embed_dim_spectral,
                                                   norm_layer=norm_layer)
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_init, embed_dim_spectral))
            nn.init.trunc_normal_(self.spectral_pos_embed, std=.02)
            self.pos_drop_spectral = nn.Dropout(p=drop_rate)
            self.spectral_stages = nn.ModuleList()
            current_dim_spectral = embed_dim_spectral
            current_resolution_patches_spectral = self.spectral_patch_embed.grid_size
            for i_stage in range(self.num_stages):
                stage = HVTStage(dim=current_dim_spectral,
                                 input_resolution_patches=current_resolution_patches_spectral,
                                 depth=depths[i_stage],
                                 num_heads=num_heads[i_stage],
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path_prob=dpr[sum(depths[:i_stage]):sum(depths[:i_stage+1])],
                                 norm_layer=norm_layer,
                                 downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                                )
                self.spectral_stages.append(stage)
                if i_stage < self.num_stages - 1:
                    current_resolution_patches_spectral = stage.output_resolution_patches
                    current_dim_spectral *= 2
            self.norm_spectral = norm_layer(current_dim_spectral)
        else:
            self.spectral_patch_embed = None
            self.spectral_stages = None
            self.norm_spectral = None
            current_dim_spectral = 0

        # --- Fusion ---
        self.fusion_embed_dim = current_dim_rgb
        
        if self.use_dfca and self.spectral_patch_embed is not None:
            self.spectral_dfca_proj = nn.Identity()
            if current_dim_rgb != current_dim_spectral:
                logger.warning(f"DFCA Input: RGB dim ({current_dim_rgb}) != Spectral dim ({current_dim_spectral}). Spectral will be projected.")
                self.spectral_dfca_proj = nn.Linear(current_dim_spectral, current_dim_rgb)
            
            # Use the DFCA imported at the top (which should now be the correct one)
            self.dfca = DiseaseFocusedCrossAttention(embed_dim=self.fusion_embed_dim,
                                                     num_heads=dfca_num_heads, 
                                                     dropout_rate=dfca_drop_rate) 
            final_head_in_features = self.fusion_embed_dim
        elif self.spectral_patch_embed is not None: # Simple fusion if spectral exists but no DFCA
            self.simple_fusion_proj = nn.Linear(current_dim_rgb + current_dim_spectral, self.fusion_embed_dim)
            final_head_in_features = self.fusion_embed_dim
        else: # Only RGB stream
            final_head_in_features = current_dim_rgb

        self.head_norm = norm_layer(final_head_in_features)
        self.head = nn.Linear(final_head_in_features, num_classes) # Final classification head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _interpolate_pos_embed(self, pos_embed_param: nn.Parameter, x_patches: torch.Tensor, 
                               current_patch_grid_H: int, current_patch_grid_W: int):
        N_current = x_patches.shape[1]
        N_original = pos_embed_param.shape[1]
        if N_current == N_original:
            return x_patches + pos_embed_param

        dim = x_patches.shape[2]
        H0 = self.img_size_at_init[0] // self.patch_size
        W0 = self.img_size_at_init[1] // self.patch_size
        if N_original != H0 * W0:
             logger.error(f"Positional embedding N_original={N_original} != H0*W0 ({H0*W0}). Cannot interpolate.")
             return x_patches

        pos_embed_to_interp = pos_embed_param.reshape(1, H0, W0, dim).permute(0, 3, 1, 2)
        H_new, W_new = current_patch_grid_H, current_patch_grid_W
        pos_embed_interp = F.interpolate(pos_embed_to_interp, size=(H_new, W_new), mode='bicubic', align_corners=False)
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).flatten(1, 2)
        return x_patches + pos_embed_interp

    def forward_features(self, x_rgb: torch.Tensor, x_spectral: Optional[torch.Tensor]) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        B_rgb, _, H_img_rgb, W_img_rgb = x_rgb.shape
        H_patch_grid_rgb, W_patch_grid_rgb = H_img_rgb // self.patch_size, W_img_rgb // self.patch_size
        
        x_rgb = self.rgb_patch_embed(x_rgb)
        x_rgb = self._interpolate_pos_embed(self.rgb_pos_embed, x_rgb, H_patch_grid_rgb, W_patch_grid_rgb)
        x_rgb = self.pos_drop_rgb(x_rgb)
        
        current_res_patches_rgb = (H_patch_grid_rgb, W_patch_grid_rgb)
        for stage in self.rgb_stages:
            stage.input_resolution_patches = current_res_patches_rgb
            x_rgb = stage(x_rgb)
            current_res_patches_rgb = stage.output_resolution_patches
        x_rgb = self.norm_rgb(x_rgb)

        if x_spectral is not None and self.spectral_patch_embed is not None:
            B_spec, _, H_img_spec, W_img_spec = x_spectral.shape
            H_patch_grid_spec, W_patch_grid_spec = H_img_spec // self.patch_size, W_img_spec // self.patch_size
            x_spectral = self.spectral_patch_embed(x_spectral)
            x_spectral = self._interpolate_pos_embed(self.spectral_pos_embed, x_spectral, H_patch_grid_spec, W_patch_grid_spec)
            x_spectral = self.pos_drop_spectral(x_spectral)
            current_res_patches_spectral = (H_patch_grid_spec, W_patch_grid_spec)
            for stage in self.spectral_stages:
                stage.input_resolution_patches = current_res_patches_spectral
                x_spectral = stage(x_spectral)
                current_res_patches_spectral = stage.output_resolution_patches
            x_spectral = self.norm_spectral(x_spectral)
        else:
            x_spectral = None
        return x_rgb, x_spectral

    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_rgb, x_spectral = self.forward_features(rgb_img, spectral_img)
        
        if x_spectral is not None and self.spectral_patch_embed is not None:
            if self.use_dfca:
                projected_x_spectral = self.spectral_dfca_proj(x_spectral)
                x_rgb_dfca = x_rgb.transpose(0, 1)
                x_spectral_dfca = projected_x_spectral.transpose(0, 1)
                # Assuming DFCA MHA is batch_first=False (default)
                fused_features = self.dfca(x_rgb_dfca, x_spectral_dfca).transpose(0, 1) 
            else: 
                combined = torch.cat((x_rgb, x_spectral), dim=2) 
                fused_features = self.simple_fusion_proj(combined) 
        else: 
            fused_features = x_rgb
        
        pooled_features = fused_features.mean(dim=1) 
        x = self.head_norm(pooled_features)
        logits = self.head(x) # Final classification head
        return logits

# --- Helper function (Restored and Corrected) ---
def create_disease_aware_hvt_from_config(img_size_tuple):
    """Helper function to instantiate DiseaseAwareHVT using config values."""
    logger.info(f"Creating DiseaseAwareHVT backbone for img_size: {img_size_tuple} using imported config.")
    # Ensure imported config names (with cfg_ prefix) are used here
    return DiseaseAwareHVT(
        img_size=img_size_tuple,
        patch_size=cfg_PATCH_SIZE,
        num_classes=cfg_NUM_CLASSES,
        embed_dim_rgb=cfg_EMBED_DIM_RGB,
        embed_dim_spectral=cfg_EMBED_DIM_SPECTRAL,
        spectral_channels=cfg_SPECTRAL_CHANNELS,
        depths=cfg_HVT_DEPTHS,
        num_heads=cfg_HVT_NUM_HEADS,
        mlp_ratio=cfg_MLP_RATIO,
        qkv_bias=cfg_QKV_BIAS,
        drop_rate=cfg_HVT_MODEL_DROP_RATE,
        attn_drop_rate=cfg_HVT_ATTN_DROP_RATE,
        drop_path_rate=cfg_HVT_DROP_PATH_RATE,
        dfca_num_heads=cfg_DFCA_NUM_HEADS,
        dfca_drop_rate=cfg_DFCA_DROP_RATE
        # use_dfca defaults to True
    )