# phase2_model/models/hvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict
import math
import logging
from torch.utils.checkpoint import checkpoint as gradient_checkpoint # Renamed to avoid conflict


logger = logging.getLogger(__name__)

# --- Attempt to Import Config from phase3_pretraining and DFCA ---
CONFIG_LOADED_SUCCESSFULLY = False
cfg_module: Dict[str, any] = {} # This will store the parameters for HVT instantiation

# Define DEFAULT_HVT_PARAMS (Phase 2 HVT defaults) at the module level
# These are used if phase3_pretraining.config import fails OR if create_disease_aware_hvt_from_config
# is called and phase3_pretraining.config hasn't provided a specific value.
DEFAULT_HVT_PARAMS = {
    "hvt_patch_size": 16, "hvt_embed_dim_rgb": 96, "hvt_embed_dim_spectral": 96,
    "hvt_spectral_channels": 1, "hvt_depths": [2,2,6,2], "hvt_num_heads": [3,6,12,24],
    "hvt_mlp_ratio": 4.0, "hvt_qkv_bias": True, "hvt_model_drop_rate": 0.0,
    "hvt_attn_drop_rate": 0.0, "hvt_drop_path_rate": 0.1,
    "num_classes": 7,
    # DFCA related defaults
    "hvt_use_dfca": True, 'hvt_dfca_heads': 24, 'dfca_drop_rate': 0.1, 'dfca_use_disease_mask': True,
    # Other HVT related
    "use_gradient_checkpointing": False,
    # SSL related defaults (for DiseaseAwareHVT's own SSL mechanisms if used directly)
    "ssl_enable_mae": True, "ssl_mae_mask_ratio": 0.75, "ssl_mae_decoder_dim": 64,
    "ssl_mae_norm_pix_loss": True,
    "ssl_enable_contrastive": True, "ssl_contrastive_projector_dim": 128, "ssl_contrastive_projector_depth": 2,
    "enable_consistency_loss_heads": True,
}

try:
    # Import the 'config' dictionary from phase3_pretraining
    # This assumes phase3_pretraining is in sys.path when this module is imported.
    from phase3_pretraining.config import config as phase3_config_dict

    # Populate cfg_module using values from phase3_config_dict,
    # falling back to DEFAULT_HVT_PARAMS for any keys not present in phase3_config_dict.
    # This makes phase3_config_dict override Phase 2 defaults if a key is present.
    cfg_module = DEFAULT_HVT_PARAMS.copy() # Start with defaults
    # Update with values from phase3_config_dict if they exist
    for key in DEFAULT_HVT_PARAMS.keys(): # Iterate over known expected keys
        if key in phase3_config_dict:
            cfg_module[key] = phase3_config_dict[key]
        # If key related to hvt_* is not in phase3_config_dict, it retains default from DEFAULT_HVT_PARAMS

    # Special handling for dfca_drop_rate and dfca_use_disease_mask if not directly in phase3_config_dict under those names
    cfg_module['dfca_drop_rate'] = phase3_config_dict.get('dfca_drop_rate', DEFAULT_HVT_PARAMS['dfca_drop_rate'])
    cfg_module['dfca_use_disease_mask'] = phase3_config_dict.get('dfca_use_disease_mask', DEFAULT_HVT_PARAMS['dfca_use_disease_mask'])


    from .dfca import DiseaseFocusedCrossAttention # Relative import for DFCA
    CONFIG_LOADED_SUCCESSFULLY = True
    logger.info(f"HVT Config: Successfully merged DEFAULT_HVT_PARAMS with overrides from 'phase3_pretraining.config'. DFCA imported. Current cfg_module patch_size: {cfg_module.get('hvt_patch_size')}")

except ImportError as e:
    logger.error(f"HVT Config ERROR: Could not import 'config' dict from 'phase3_pretraining.config' or relative DFCA. Error: {e}", exc_info=True)
    logger.warning("HVT Config: Using strictly DEFAULT_HVT_PARAMS (Phase 2 defaults) due to import error.")
    cfg_module = DEFAULT_HVT_PARAMS.copy()
    # Define a dummy DFCA if the import fails, so the rest of the file can be parsed
    class DiseaseFocusedCrossAttention(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.identity = nn.Identity()
        def forward(self, rgb, spec): return rgb # Acts as passthrough
    CONFIG_LOADED_SUCCESSFULLY = False # Explicitly set to false

except KeyError as e:
     logger.error(f"HVT Config ERROR: Missing expected key when trying to override defaults with 'phase3_config_dict': {e}", exc_info=True)
     logger.warning("HVT Config: Using strictly DEFAULT_HVT_PARAMS (Phase 2 defaults) due to KeyError.")
     cfg_module = DEFAULT_HVT_PARAMS.copy()
     CONFIG_LOADED_SUCCESSFULLY = False


# Helper function to get config value from the globally populated cfg_module
# create_disease_aware_hvt_from_config will call this.
def get_cfg_param(key: str, default_value_from_caller: Optional[any] = None):
    """
    Gets a parameter from the globally populated cfg_module.
    If key is not in cfg_module, it uses the default_value_from_caller.
    If default_value_from_caller is also None, it tries DEFAULT_HVT_PARAMS.
    """
    if key in cfg_module and cfg_module[key] is not None: # Prioritize cfg_module if value is not None
        return cfg_module[key]
    elif default_value_from_caller is not None: # Then caller's default
        return default_value_from_caller
    else: # Fallback to this module's DEFAULT_HVT_PARAMS
        return DEFAULT_HVT_PARAMS.get(key)


# --- Helper Modules (DropPath, PatchEmbed, Attention, Mlp, TransformerBlock, PatchMerging) ---
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob; shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device); random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16, in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__(); self.img_size = img_size; self.patch_size = patch_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0: logger.warning(f"PatchEmbed: Img dims {img_size} not divisible by patch {patch_size}.")
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size); self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size); self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor: B, C, H, W = x.shape; x = self.proj(x); x = x.flatten(2); x = x.transpose(1, 2); return self.norm(x) # B, N, C

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__(); self.num_heads = num_heads
        if dim % num_heads != 0: raise ValueError(f"dim {dim} should be divisible by num_heads {num_heads}")
        head_dim = dim // num_heads; self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.attn_drop = nn.Dropout(attn_drop); self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        if C % self.num_heads != 0: raise ValueError(f"Input C {C} must be divisible by num_heads {self.num_heads}")
        head_dim_actual = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim_actual).permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x); return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__(); out_features = out_features or in_features; hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features); self.act = act_layer(); self.fc2 = nn.Linear(hidden_features, out_features); self.drop = nn.Dropout(drop)
    def forward(self, x): x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); return self.drop(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(); self.norm1 = norm_layer(dim); self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(); self.norm2 = norm_layer(dim); mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x): x = x + self.drop_path(self.attn(self.norm1(x))); x = x + self.drop_path(self.mlp(self.norm2(x))); return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution_patches: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__(); self.input_resolution_patches = input_resolution_patches; self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False); self.norm = norm_layer(4 * dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: B, L, C
        B, L, C = x.shape; H_patch, W_patch = self.input_resolution_patches
        if L != H_patch * W_patch:
            if math.isqrt(L)**2 == L: H_patch = W_patch = math.isqrt(L)
            else: raise ValueError(f"PatchMerging L={L} != H*W={H_patch*W_patch} for input_resolution_patches, and L is not a perfect square for dynamic sizing.")
        x = x.view(B, H_patch, W_patch, C)
        H_slice, W_slice = (H_patch // 2) * 2, (W_patch // 2) * 2
        x0 = x[:, 0:H_slice:2, 0:W_slice:2, :]; x1 = x[:, 1:H_slice:2, 0:W_slice:2, :]; x2 = x[:, 0:H_slice:2, 1:W_slice:2, :]; x3 = x[:, 1:H_slice:2, 1:W_slice:2, :]
        x = torch.cat([x0, x1, x2, x3], -1); x = x.view(B, -1, 4 * C); x = self.norm(x); return self.reduction(x)
    def extra_repr(self) -> str: return f"input_res={self.input_resolution_patches}, dim={self.dim}"

class HVTStage(nn.Module):
    def __init__(self, dim: int, input_resolution_patches: Tuple[int, int], depth: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop: float, attn_drop: float, drop_path_prob: Union[List[float], float] = 0., norm_layer=nn.LayerNorm, downsample_class: Optional[type[nn.Module]] = None, use_checkpoint=False):
        super().__init__(); self.dim = dim; self.input_resolution_patches = input_resolution_patches; self.depth = depth; self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob, norm_layer=norm_layer) for i in range(depth)])
        self.downsample_layer = None
        if downsample_class is not None: self.downsample_layer = downsample_class(input_resolution_patches=input_resolution_patches, dim=dim, norm_layer=norm_layer); self.output_resolution_patches = (input_resolution_patches[0] // 2, input_resolution_patches[1] // 2)
        else: self.output_resolution_patches = input_resolution_patches
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and not torch.jit.is_scripting() and self.training: x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else: x = blk(x)
        if self.downsample_layer is not None: x = self.downsample_layer(x)
        return x

class MAEPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, decoder_embed_dim: int, patch_size: int, out_chans: int):
        super().__init__()
        self.decoder_pred = nn.Sequential(
            nn.Linear(embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, patch_size * patch_size * out_chans)
        )
        self.patch_size = patch_size
        self.out_chans = out_chans
        logger.info(f"MAEPredictionHead initialized: embed_dim={embed_dim}, decoder_dim={decoder_embed_dim}, target_patch_pixels={patch_size*patch_size*out_chans}")
    def forward(self, x_encoded_patches: torch.Tensor) -> torch.Tensor:
        x = self.decoder_pred(x_encoded_patches)
        return x

# --- DiseaseAwareHVT Class (Main Backbone) ---
class DiseaseAwareHVT(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int],
                 patch_size: int,
                 num_classes: int,
                 embed_dim_rgb: int,
                 embed_dim_spectral: int,
                 spectral_channels: int,
                 depths: List[int],
                 num_heads: List[int],
                 mlp_ratio: float,
                 qkv_bias: bool,
                 drop_rate: float,
                 attn_drop_rate: float,
                 drop_path_rate: float,
                 norm_layer=nn.LayerNorm,
                 dfca_num_heads: Optional[int] = None,
                 dfca_drop_rate: Optional[float] = None,
                 dfca_use_disease_mask: bool = True,
                 use_dfca: bool = True,
                 use_gradient_checkpointing: bool = False,
                 ssl_enable_mae: bool = False,
                 ssl_mae_mask_ratio: float = 0.75,
                 ssl_mae_decoder_dim: int = 64,
                 ssl_mae_norm_pix_loss: bool = True,
                 ssl_enable_contrastive: bool = False,
                 ssl_contrastive_projector_dim: int = 128,
                 ssl_contrastive_projector_depth: int = 2,
                 enable_consistency_loss_heads: bool = False
                 ):
        super().__init__()
        self.config_params = {k: v for k, v in locals().items() if k != 'self' and k != 'norm_layer'}
        self.config_params['norm_layer_name'] = norm_layer.__name__ if norm_layer else 'None'


        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.patch_size = patch_size
        self.use_dfca = use_dfca
        self.img_size_at_init = img_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.norm_layer = norm_layer

        self.ssl_enable_mae = ssl_enable_mae
        self.ssl_mae_mask_ratio = ssl_mae_mask_ratio
        self.ssl_mae_norm_pix_loss = ssl_mae_norm_pix_loss
        self.ssl_enable_contrastive = ssl_enable_contrastive
        self.enable_consistency_loss_heads = enable_consistency_loss_heads

        _dfca_num_heads = dfca_num_heads if dfca_num_heads is not None else num_heads[-1]
        _dfca_drop_rate = dfca_drop_rate if dfca_drop_rate is not None else 0.1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.rgb_patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=3, embed_dim=embed_dim_rgb, norm_layer=norm_layer)
        num_patches_at_init = self.rgb_patch_embed.num_patches
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_init, embed_dim_rgb))
        nn.init.trunc_normal_(self.rgb_pos_embed, std=.02)
        self.pos_drop_rgb = nn.Dropout(p=drop_rate)
        self.rgb_stages = nn.ModuleList()
        current_dim_rgb = embed_dim_rgb
        self.final_encoded_dim_rgb = embed_dim_rgb * (2**(self.num_stages -1 if self.num_stages > 0 else 0))
        current_resolution_patches_rgb = self.rgb_patch_embed.grid_size
        for i_stage in range(self.num_stages):
            stage = HVTStage(dim=current_dim_rgb, input_resolution_patches=current_resolution_patches_rgb, depth=depths[i_stage], num_heads=num_heads[i_stage], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path_prob=dpr[sum(depths[:i_stage]):sum(depths[:i_stage+1])], norm_layer=norm_layer, downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None, use_checkpoint=self.use_gradient_checkpointing)
            self.rgb_stages.append(stage)
            if i_stage < self.num_stages - 1: current_resolution_patches_rgb = stage.output_resolution_patches; current_dim_rgb *= 2
        self.norm_rgb_final_encoder = norm_layer(current_dim_rgb)

        self.spectral_patch_embed = None; self.spectral_pos_embed = None; self.spectral_stages = None; self.norm_spectral_final_encoder = None
        current_dim_spectral = 0
        self.final_encoded_dim_spectral = 0
        if spectral_channels > 0:
            self.spectral_patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=spectral_channels, embed_dim=embed_dim_spectral, norm_layer=norm_layer)
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_init, embed_dim_spectral))
            nn.init.trunc_normal_(self.spectral_pos_embed, std=.02)
            self.pos_drop_spectral = nn.Dropout(p=drop_rate)
            self.spectral_stages = nn.ModuleList()
            current_dim_spectral = embed_dim_spectral
            self.final_encoded_dim_spectral = embed_dim_spectral * (2**(self.num_stages-1 if self.num_stages > 0 else 0))
            current_resolution_patches_spectral = self.spectral_patch_embed.grid_size
            for i_stage in range(self.num_stages):
                stage = HVTStage(dim=current_dim_spectral, input_resolution_patches=current_resolution_patches_spectral, depth=depths[i_stage], num_heads=num_heads[i_stage], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path_prob=dpr[sum(depths[:i_stage]):sum(depths[:i_stage+1])], norm_layer=norm_layer, downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None, use_checkpoint=self.use_gradient_checkpointing)
                self.spectral_stages.append(stage)
                if i_stage < self.num_stages - 1: current_resolution_patches_spectral = stage.output_resolution_patches; current_dim_spectral *= 2
            self.norm_spectral_final_encoder = norm_layer(current_dim_spectral)

        self.fusion_embed_dim = current_dim_rgb
        self.dfca = None; self.simple_fusion_proj = None; self.spectral_dfca_proj = None
        final_head_in_features = current_dim_rgb
        if self.use_dfca and self.spectral_patch_embed is not None:
            self.spectral_dfca_proj = nn.Identity()
            if current_dim_rgb != current_dim_spectral:
                logger.warning(f"DFCA Input: Dim mismatch RGB ({current_dim_rgb}) != Spectral ({current_dim_spectral}). Projecting Spectral.")
                self.spectral_dfca_proj = nn.Linear(current_dim_spectral, current_dim_rgb)
            # Ensure DiseaseFocusedCrossAttention is defined (it should be, due to fallback if import fails)
            self.dfca = DiseaseFocusedCrossAttention(embed_dim=self.fusion_embed_dim, num_heads=_dfca_num_heads, dropout_rate=_dfca_drop_rate, use_disease_mask=dfca_use_disease_mask)
            final_head_in_features = self.fusion_embed_dim
            logger.info("DFCA fusion enabled.")
        elif self.spectral_patch_embed is not None:
             if current_dim_rgb == current_dim_spectral:
                 self.simple_fusion_proj = nn.Linear(current_dim_rgb + current_dim_spectral, self.fusion_embed_dim)
                 final_head_in_features = self.fusion_embed_dim
                 logger.info("Simple concatenation fusion enabled.")
             else:
                 logger.warning(f"Simple fusion: Dim mismatch RGB ({current_dim_rgb}) != Spectral ({current_dim_spectral}). Using RGB only for head.")
                 final_head_in_features = current_dim_rgb
        else:
            logger.info("Running RGB stream only.")
            final_head_in_features = current_dim_rgb

        self.head_norm = norm_layer(final_head_in_features)
        self.head = nn.Linear(final_head_in_features, num_classes)

        if self.ssl_enable_mae:
            self.mae_decoder_rgb = MAEPredictionHead(self.final_encoded_dim_rgb, ssl_mae_decoder_dim, self.patch_size, 3)
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0 :
                self.mae_decoder_spectral = MAEPredictionHead(self.final_encoded_dim_spectral, ssl_mae_decoder_dim, self.patch_size, spectral_channels) # Use actual spectral_channels
            else:
                self.mae_decoder_spectral = None
            logger.info("MAE components initialized for DiseaseAwareHVT.")

        if self.ssl_enable_contrastive:
            contrast_layers = []
            current_contrast_dim = final_head_in_features
            if ssl_contrastive_projector_depth > 0 :
                for i in range(ssl_contrastive_projector_depth):
                    out_c_dim = ssl_contrastive_projector_dim if i == ssl_contrastive_projector_depth - 1 else ssl_contrastive_projector_dim # Or some hidden dim
                    contrast_layers.append(nn.Linear(current_contrast_dim, out_c_dim))
                    if i < ssl_contrastive_projector_depth - 1: # No GELU/BN on last layer for some contrastive setups
                        contrast_layers.append(nn.GELU()) # Or nn.BatchNorm1d(out_c_dim) then GELU
                    current_contrast_dim = out_c_dim
            self.contrastive_projector = nn.Sequential(*contrast_layers) if contrast_layers else nn.Identity()
            logger.info(f"Contrastive projector initialized for DiseaseAwareHVT with output dim {current_contrast_dim}.")


        if self.enable_consistency_loss_heads:
            self.aux_head_rgb = nn.Linear(self.final_encoded_dim_rgb, num_classes)
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0:
                self.aux_head_spectral = nn.Linear(self.final_encoded_dim_spectral, num_classes)
            else:
                self.aux_head_spectral = None
            logger.info("Auxiliary heads for consistency loss initialized for DiseaseAwareHVT.")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.trunc_normal_(m.weight, std=.02);
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu');
        if isinstance(m, nn.Conv2d) and m.bias is not None: nn.init.constant_(m.bias, 0)

    def _interpolate_pos_embed(self, pos_embed_param, x_patches, current_patch_grid_H, current_patch_grid_W):
        N_current = x_patches.shape[1]; N_original = pos_embed_param.shape[1]
        if N_current == N_original and (current_patch_grid_H * current_patch_grid_W == N_original):
            return x_patches + pos_embed_param
        dim = x_patches.shape[2]; H0 = self.img_size_at_init[0] // self.patch_size; W0 = self.img_size_at_init[1] // self.patch_size
        if N_original != H0 * W0:
            if math.isqrt(N_original)**2 == N_original: H0 = W0 = math.isqrt(N_original)
            else: logger.error(f"Pos embed N_orig {N_original} != H0*W0 {H0*W0}, not square. Using x_patches."); return x_patches
        pos_embed_to_interp = pos_embed_param.reshape(1, H0, W0, dim).permute(0, 3, 1, 2)
        H_new, W_new = current_patch_grid_H, current_patch_grid_W
        pos_embed_interp = F.interpolate(pos_embed_to_interp, size=(H_new, W_new), mode='bicubic', align_corners=False)
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).flatten(1, 2)
        return x_patches + pos_embed_interp

    def _forward_stream(self, x_img, patch_embed_layer, pos_embed_param, pos_drop_layer, stages_list, final_norm_layer):
        B, _, H_img, W_img = x_img.shape
        H_patch_grid, W_patch_grid = H_img // self.patch_size, W_img // self.patch_size
        x_patches = patch_embed_layer(x_img)
        x_patches = self._interpolate_pos_embed(pos_embed_param, x_patches, H_patch_grid, W_patch_grid)
        x_patches = pos_drop_layer(x_patches)
        current_res_patches = (H_patch_grid, W_patch_grid)
        # Store features at each original patch location *before* any merging for potential MAE use
        # This is a simplification; a true hierarchical MAE needs careful thought.
        # For now, `all_patch_features_unmerged` would be `x_patches` if MAE decoder needs unmerged features.
        # Or, if MAE decoder can handle hierarchically merged features, this isn't needed.
        for stage_idx, stage in enumerate(stages_list):
            stage.input_resolution_patches = current_res_patches
            x_patches = stage(x_patches)
            current_res_patches = stage.output_resolution_patches
        x_encoded = final_norm_layer(x_patches)
        return x_encoded, (H_patch_grid, W_patch_grid)

    def forward_features_encoded(self, rgb_img, spectral_img=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int,int], Optional[Tuple[int,int]]]:
        x_rgb_encoded, rgb_orig_patch_grid = self._forward_stream(
            rgb_img, self.rgb_patch_embed, self.rgb_pos_embed, self.pos_drop_rgb, self.rgb_stages, self.norm_rgb_final_encoder
        )
        x_spectral_encoded, spectral_orig_patch_grid = None, None
        if spectral_img is not None and self.spectral_patch_embed is not None:
            x_spectral_encoded, spectral_orig_patch_grid = self._forward_stream(
                spectral_img, self.spectral_patch_embed, self.spectral_pos_embed, self.pos_drop_spectral, self.spectral_stages, self.norm_spectral_final_encoder
            )
        return x_rgb_encoded, x_spectral_encoded, rgb_orig_patch_grid, spectral_orig_patch_grid

    def _mae_reconstruct(self, x_img_orig, x_encoded_modality_final_stage, orig_patch_grid_H, orig_patch_grid_W, mae_decoder_modality, mask_orig_patches_flat):
        # x_encoded_modality_final_stage: (B, N_encoded_final, C_encoded_final) - output of final encoder stage (potentially after patch merging)
        # mask_orig_patches_flat: (B, N_orig_patches) - boolean, True for MASKED original patches
        # Goal: Predict masked *original* patches using some representation from the encoder.

        # SimMIM-like: Use the encoded representation OF THE MASKED PATCHES from the final encoder stage.
        # This is tricky if N_encoded_final < N_orig_patches due to PatchMerging.
        # If PatchMerging happens, a single token in x_encoded_modality_final_stage corresponds to multiple original patches.

        # Strategy:
        # 1. Get target values for all original patches.
        # 2. For MAE, we need input to the decoder. If decoder expects one feature vector per masked original patch:
        #    a. If no PatchMerging (N_encoded_final == N_orig_patches), we can select the encoded features
        #       of the masked original patches directly from x_encoded_modality_final_stage.
        #    b. If PatchMerging, this direct selection is not possible. We would need to:
        #       i.  Upsample x_encoded_modality_final_stage to N_orig_patches resolution.
        #       ii. Or, use features from an earlier stage before merging.
        #       iii.Or, design a more complex MAE decoder that handles hierarchical inputs.

        # Current MAEPredictionHead expects (num_masked_patches_total, C_encoded_final).
        # We will proceed with assumption (a) for simplicity. If PatchMerging is used, this MAE path will be suboptimal or incorrect.
        # The `DiseaseAwareHVT` should ideally be configured without PatchMerging for this specific MAE implementation to work best.

        B, C_img, H_img, W_img = x_img_orig.shape
        P = self.patch_size
        N_orig_patches = orig_patch_grid_H * orig_patch_grid_W

        if x_encoded_modality_final_stage.shape[1] != N_orig_patches:
            logger.warning(
                f"MAE reconstruction: Mismatch between num_encoded_tokens ({x_encoded_modality_final_stage.shape[1]}) "
                f"and num_original_patches ({N_orig_patches}). This occurs if PatchMerging is active. "
                f"MAE predictions from final stage features might be misaligned or suboptimal. "
                f"Consider MAE on features before PatchMerging or a hierarchical MAE decoder."
            )
            # Option: return None, None if strict alignment is required and violated.
            # For now, we'll try to proceed but it's a known issue.
            # We cannot directly select features for original masked patches from the final merged stage.
            # Let's return None to indicate failure for this MAE variant under PatchMerging.
            return None, None


        target_patches_unfold = x_img_orig.unfold(2, P, P).unfold(3, P, P) # B, C, H_grid, W_grid, P, P
        target_patches = target_patches_unfold.permute(0, 2, 3, 1, 4, 5).reshape(B, N_orig_patches, C_img*P*P) # B, N_orig, Pixels

        # Ensure mask_orig_patches_flat is correctly shaped (B, N_orig_patches)
        if mask_orig_patches_flat.shape != (B, N_orig_patches):
            logger.error(f"MAE mask shape error. Expected ({B}, {N_orig_patches}), got {mask_orig_patches_flat.shape}")
            return None, None

        # Select encoded features corresponding to MASKED original patches
        # This assumes x_encoded_modality_final_stage has one token per original patch
        selected_encoded_features = x_encoded_modality_final_stage[mask_orig_patches_flat] # (TotalMasked, C_encoded_final)
        selected_target_patches = target_patches[mask_orig_patches_flat] # (TotalMasked, Pixels)

        if selected_encoded_features.numel() == 0: # No patches were masked
            return torch.empty(0, P*P*C_img, device=x_img_orig.device), torch.empty(0, P*P*C_img, device=x_img_orig.device)

        predictions = mae_decoder_modality(selected_encoded_features)
        return predictions, selected_target_patches


    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None,
                mode: str = 'classify',
                mae_mask_custom: Optional[torch.Tensor] = None
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:

        x_rgb_encoded_final, x_spectral_encoded_final, \
            (rgb_orig_H_grid, rgb_orig_W_grid), \
            spec_orig_grids_nullable = self.forward_features_encoded(rgb_img, spectral_img)

        spec_orig_H_grid, spec_orig_W_grid = (None, None)
        if spec_orig_grids_nullable is not None:
            spec_orig_H_grid, spec_orig_W_grid = spec_orig_grids_nullable


        if mode == 'mae':
            if not self.ssl_enable_mae: raise ValueError("MAE mode called but not enabled in HVT config.")
            B_rgb = rgb_img.shape[0]
            num_patches_rgb_orig = rgb_orig_H_grid * rgb_orig_W_grid
            mask_rgb_flat: torch.Tensor

            if mae_mask_custom is not None:
                mask_rgb_flat = mae_mask_custom.to(rgb_img.device).view(B_rgb, -1)
                if mask_rgb_flat.shape[1] != num_patches_rgb_orig:
                    raise ValueError(f"Custom RGB MAE mask shape mismatch. Expected Bx{num_patches_rgb_orig}, got {mask_rgb_flat.shape}")
            else:
                noise = torch.rand(B_rgb, num_patches_rgb_orig, device=rgb_img.device)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                len_keep_rgb = int(num_patches_rgb_orig * (1 - self.ssl_mae_mask_ratio))
                mask_rgb_flat_sorted = torch.ones(B_rgb, num_patches_rgb_orig, dtype=torch.bool, device=rgb_img.device)
                mask_rgb_flat_sorted[:, :len_keep_rgb] = False # True means MASKED
                mask_rgb_flat = torch.gather(mask_rgb_flat_sorted, dim=1, index=ids_restore)

            pred_rgb, target_rgb = self._mae_reconstruct(rgb_img, x_rgb_encoded_final, rgb_orig_H_grid, rgb_orig_W_grid, self.mae_decoder_rgb, mask_rgb_flat)
            mae_outputs = {'pred_rgb': pred_rgb, 'target_rgb': target_rgb, 'mask_rgb': mask_rgb_flat.view(B_rgb, rgb_orig_H_grid, rgb_orig_W_grid)}

            if spectral_img is not None and self.spectral_patch_embed and self.mae_decoder_spectral and spec_orig_H_grid is not None:
                B_spec = spectral_img.shape[0]
                num_patches_spec_orig = spec_orig_H_grid * spec_orig_W_grid
                mask_spec_flat: torch.Tensor
                if mae_mask_custom is not None and num_patches_spec_orig == num_patches_rgb_orig :
                     mask_spec_flat = mae_mask_custom.to(spectral_img.device).view(B_spec, -1)
                else:
                    noise_s = torch.rand(B_spec, num_patches_spec_orig, device=spectral_img.device)
                    ids_shuffle_s = torch.argsort(noise_s, dim=1); ids_restore_s = torch.argsort(ids_shuffle_s, dim=1)
                    len_keep_s = int(num_patches_spec_orig * (1 - self.ssl_mae_mask_ratio))
                    mask_spec_flat_sorted = torch.ones(B_spec, num_patches_spec_orig, dtype=torch.bool, device=spectral_img.device)
                    mask_spec_flat_sorted[:, :len_keep_s] = False
                    mask_spec_flat = torch.gather(mask_spec_flat_sorted, dim=1, index=ids_restore_s)

                pred_spec, target_spec = self._mae_reconstruct(spectral_img, x_spectral_encoded_final, spec_orig_H_grid, spec_orig_W_grid, self.mae_decoder_spectral, mask_spec_flat)
                mae_outputs.update({'pred_spectral': pred_spec, 'target_spectral': target_spec, 'mask_spectral': mask_spec_flat.view(B_spec, spec_orig_H_grid, spec_orig_W_grid) if spec_orig_H_grid else None})
            return mae_outputs

        fused_features_encoded = x_rgb_encoded_final # Default to RGB if no fusion
        if x_spectral_encoded_final is not None and self.spectral_patch_embed is not None:
            if self.use_dfca and self.dfca is not None:
                projected_x_spectral = self.spectral_dfca_proj(x_spectral_encoded_final)
                x_rgb_dfca_in = x_rgb_encoded_final.transpose(0, 1)
                x_spectral_dfca_in = projected_x_spectral.transpose(0, 1)
                fused_features_encoded = self.dfca(x_rgb_dfca_in, x_spectral_dfca_in).transpose(0, 1)
            elif self.simple_fusion_proj is not None:
                if x_rgb_encoded_final.shape[1] == x_spectral_encoded_final.shape[1]:
                    combined = torch.cat((x_rgb_encoded_final, x_spectral_encoded_final), dim=2)
                    fused_features_encoded = self.simple_fusion_proj(combined)
                else: fused_features_encoded = x_rgb_encoded_final # Fallback
            else: fused_features_encoded = x_rgb_encoded_final # Fallback
        pooled_features = fused_features_encoded.mean(dim=1)

        if mode == 'get_embeddings':
            embeddings = {'fused': pooled_features, 'rgb': x_rgb_encoded_final.mean(dim=1)}
            if x_spectral_encoded_final is not None: embeddings['spectral'] = x_spectral_encoded_final.mean(dim=1)
            return embeddings
        if mode == 'contrastive':
            if not self.ssl_enable_contrastive: raise ValueError("Contrastive mode called but HVT.ssl_enable_contrastive is False.")
            return self.contrastive_projector(pooled_features)
        if mode == 'classify':
            x_for_head = self.head_norm(pooled_features)
            main_logits = self.head(x_for_head)
            if self.enable_consistency_loss_heads:
                aux_outputs = {}
                aux_outputs['logits_rgb'] = self.aux_head_rgb(x_rgb_encoded_final.mean(dim=1))
                if x_spectral_encoded_final is not None and self.spectral_patch_embed and self.aux_head_spectral:
                    aux_outputs['logits_spectral'] = self.aux_head_spectral(x_spectral_encoded_final.mean(dim=1))
                return main_logits, aux_outputs
            else: return main_logits
        raise ValueError(f"Unknown forward mode: {mode}")

# --- Factory Function ---
def create_disease_aware_hvt_from_config(img_size_tuple: Tuple[int, int]):
    """ Instantiates DiseaseAwareHVT using parameters from the globally populated cfg_module,
        with fallbacks to DEFAULT_HVT_PARAMS if a parameter is not found in cfg_module.
    """
    logger.info(f"Creating DiseaseAwareHVT for img_size: {img_size_tuple}. CFG_LOAD_SUCCESS: {CONFIG_LOADED_SUCCESSFULLY}. Patch_size from cfg_module: {cfg_module.get('hvt_patch_size')}")

    # If CONFIG_LOADED_SUCCESSFULLY is false, cfg_module is already DEFAULT_HVT_PARAMS.
    # get_cfg_param will fetch from cfg_module, which is appropriately set.

    # Ensure all required args for DiseaseAwareHVT.__init__ are resolvable
    # Defaults provided here are only used if get_cfg_param itself returns None (which it shouldn't if DEFAULT_HVT_PARAMS is comprehensive)
    init_args = {
        "img_size": img_size_tuple,
        "patch_size": get_cfg_param('hvt_patch_size', DEFAULT_HVT_PARAMS['hvt_patch_size']),
        "num_classes": get_cfg_param('num_classes', DEFAULT_HVT_PARAMS['num_classes']),
        "embed_dim_rgb": get_cfg_param('hvt_embed_dim_rgb', DEFAULT_HVT_PARAMS['hvt_embed_dim_rgb']),
        "embed_dim_spectral": get_cfg_param('hvt_embed_dim_spectral', DEFAULT_HVT_PARAMS['hvt_embed_dim_spectral']),
        "spectral_channels": get_cfg_param('hvt_spectral_channels', DEFAULT_HVT_PARAMS['hvt_spectral_channels']),
        "depths": get_cfg_param('hvt_depths', DEFAULT_HVT_PARAMS['hvt_depths']),
        "num_heads": get_cfg_param('hvt_num_heads', DEFAULT_HVT_PARAMS['hvt_num_heads']),
        "mlp_ratio": get_cfg_param('hvt_mlp_ratio', DEFAULT_HVT_PARAMS['hvt_mlp_ratio']),
        "qkv_bias": get_cfg_param('hvt_qkv_bias', DEFAULT_HVT_PARAMS['hvt_qkv_bias']),
        "drop_rate": get_cfg_param('hvt_model_drop_rate', DEFAULT_HVT_PARAMS['hvt_model_drop_rate']),
        "attn_drop_rate": get_cfg_param('hvt_attn_drop_rate', DEFAULT_HVT_PARAMS['hvt_attn_drop_rate']),
        "drop_path_rate": get_cfg_param('hvt_drop_path_rate', DEFAULT_HVT_PARAMS['hvt_drop_path_rate']),
        "use_dfca": get_cfg_param('hvt_use_dfca', DEFAULT_HVT_PARAMS['hvt_use_dfca']),
        "dfca_num_heads": get_cfg_param('hvt_dfca_heads', DEFAULT_HVT_PARAMS['hvt_dfca_heads']),
        "dfca_drop_rate": get_cfg_param('dfca_drop_rate', DEFAULT_HVT_PARAMS['dfca_drop_rate']),
        "dfca_use_disease_mask": get_cfg_param('dfca_use_disease_mask', DEFAULT_HVT_PARAMS['dfca_use_disease_mask']),
        "use_gradient_checkpointing": get_cfg_param('use_gradient_checkpointing', DEFAULT_HVT_PARAMS['use_gradient_checkpointing']),
        "ssl_enable_mae": get_cfg_param('ssl_enable_mae', DEFAULT_HVT_PARAMS['ssl_enable_mae']),
        "ssl_mae_mask_ratio": get_cfg_param('ssl_mae_mask_ratio', DEFAULT_HVT_PARAMS['ssl_mae_mask_ratio']),
        "ssl_mae_decoder_dim": get_cfg_param('ssl_mae_decoder_dim', DEFAULT_HVT_PARAMS['ssl_mae_decoder_dim']),
        "ssl_mae_norm_pix_loss": get_cfg_param('ssl_mae_norm_pix_loss', DEFAULT_HVT_PARAMS['ssl_mae_norm_pix_loss']),
        "ssl_enable_contrastive": get_cfg_param('ssl_enable_contrastive', DEFAULT_HVT_PARAMS['ssl_enable_contrastive']),
        "ssl_contrastive_projector_dim": get_cfg_param('ssl_contrastive_projector_dim', DEFAULT_HVT_PARAMS['ssl_contrastive_projector_dim']),
        "ssl_contrastive_projector_depth": get_cfg_param('ssl_contrastive_projector_depth', DEFAULT_HVT_PARAMS['ssl_contrastive_projector_depth']),
        "enable_consistency_loss_heads": get_cfg_param('enable_consistency_loss_heads', DEFAULT_HVT_PARAMS['enable_consistency_loss_heads']),
        "norm_layer": nn.LayerNorm # Default norm_layer, can be made configurable if needed
    }
    # Log the actual parameters being used for instantiation
    logger.info(f"Instantiating DiseaseAwareHVT with effective parameters: patch_size={init_args['patch_size']}, embed_dim_rgb={init_args['embed_dim_rgb']}, depths={init_args['depths']}, ssl_enable_mae={init_args['ssl_enable_mae']}")

    return DiseaseAwareHVT(**init_args)