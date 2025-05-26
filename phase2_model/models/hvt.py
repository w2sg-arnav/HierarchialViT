# phase2_model/models/hvt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict, Any
import math
import logging

# Import DFCA relative to this file's location
from .dfca import DiseaseFocusedCrossAttention

logger = logging.getLogger(__name__)

# --- Helper Modules (DropPath, PatchEmbed, Attention, Mlp, TransformerBlock, PatchMerging) ---
# These are standard building blocks for ViT-like architectures.
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_() # binarize
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = img_size; self.patch_size = patch_size
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            logger.warning(f"PatchEmbed: Img dims {img_size} not perfectly divisible by patch_size {patch_size}.")
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # B, C_embed, H_grid, W_grid
        x = x.flatten(2) # B, C_embed, N_patches_flat
        x = x.transpose(1, 2) # B, N_patches_flat, C_embed
        return self.norm(x)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(f"Attention Error: dim ({dim}) must be divisible by num_heads ({num_heads}).")
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Dynamically get head_dim if C is different from init 'dim' (e.g. in DFCA context if dimensions don't match perfectly)
        # However, for standard self-attention in HVT stages, C should equal self.dim (which is init_dim)
        actual_head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, actual_head_dim).permute(2, 0, 3, 1, 4) # 3, B, num_heads, N, head_dim
        q, k, v = qkv.unbind(0) # Makes q, k, v of shape [B, num_heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x); x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution_patches: Tuple[int, int], dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution_patches = input_resolution_patches # H_patch_in, W_patch_in
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.output_resolution_patches = (input_resolution_patches[0] // 2, input_resolution_patches[1] // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: B, L, C
        B, L, C = x.shape
        H_patch, W_patch = self.input_resolution_patches
        if L != H_patch * W_patch:
            # Attempt to infer H_patch, W_patch if L is a perfect square, assuming square patch grid
            if math.isqrt(L)**2 == L: H_patch = W_patch = math.isqrt(L)
            else: raise ValueError(f"PatchMerging: L ({L}) != H_patch*W_patch ({H_patch*W_patch}), and L is not a perfect square.")
        if H_patch % 2 != 0 or W_patch % 2 != 0:
            raise ValueError(f"PatchMerging: Input patch resolution H_patch={H_patch}, W_patch={W_patch} must be even for 2x2 merging.")

        x = x.view(B, H_patch, W_patch, C)
        # Select 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
        x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
        x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C) # B, (H/2)*(W/2), 4*C
        x = self.norm(x)
        return self.reduction(x) # B, (H/2)*(W/2), 2*C
    def extra_repr(self) -> str: return f"input_resolution_patches={self.input_resolution_patches}, dim={self.dim}"

class HVTStage(nn.Module):
    def __init__(self, dim: int, current_input_resolution_patches: Tuple[int, int], depth: int,
                 num_heads: int, mlp_ratio: float, qkv_bias: bool, drop: float, attn_drop: float,
                 drop_path_prob: Union[List[float], float] = 0., norm_layer=nn.LayerNorm,
                 downsample_class: Optional[type[PatchMerging]] = None, use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution_patches = current_input_resolution_patches # Store for reference, PatchMerging uses it
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             drop=drop, attn_drop=attn_drop,
                             drop_path=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                             norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample_layer = None
        self.output_resolution_patches = current_input_resolution_patches # Default if no downsampling
        if downsample_class is not None:
            self.downsample_layer = downsample_class(input_resolution_patches=current_input_resolution_patches, dim=dim, norm_layer=norm_layer)
            self.output_resolution_patches = self.downsample_layer.output_resolution_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training and not torch.jit.is_scripting():
                # use_reentrant=False is recommended for PyTorch >= 1.11 for better efficiency
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x

class MAEPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, decoder_embed_dim: int, patch_size: int, out_chans: int):
        super().__init__()
        self.decoder_pred = nn.Sequential(
            nn.Linear(embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, patch_size * patch_size * out_chans)
        )
        self.patch_size = patch_size; self.out_chans = out_chans
        logger.info(f"MAEPredictionHead: in_dim={embed_dim}, decoder_hidden_dim={decoder_embed_dim}, target_pixels_per_patch={patch_size*patch_size*out_chans}")

    def forward(self, x_encoded_patches: torch.Tensor) -> torch.Tensor:
        return self.decoder_pred(x_encoded_patches)

# --- DiseaseAwareHVT Class (Main Backbone) ---
class DiseaseAwareHVT(nn.Module):
    def __init__(self,
                 img_size: Tuple[int, int], # Current image size (can change during progressive training)
                 num_classes: int,
                 hvt_params: Dict[str, Any] # Dictionary containing all HVT-specific parameters
                 ):
        super().__init__()
        self.current_img_size = img_size # Store current image size for _interpolate_pos_embed
        self.num_classes = num_classes
        self.hvt_params = hvt_params # Store for reference

        # Extract parameters from hvt_params dict, providing defaults if missing (though factory should ensure they exist)
        _p = lambda key, default: self.hvt_params.get(key, default)
        self.patch_size = _p('patch_size', 16)
        embed_dim_rgb = _p('embed_dim_rgb', 96)
        embed_dim_spectral = _p('embed_dim_spectral', 96)
        spectral_channels = _p('spectral_channels', 0)
        depths = _p('depths', [2,2,6,2])
        num_heads_list = _p('num_heads', [3,6,12,24]) # Renamed to avoid conflict with self.num_heads
        mlp_ratio = _p('mlp_ratio', 4.0)
        qkv_bias = _p('qkv_bias', True)
        model_drop_rate = _p('model_drop_rate', 0.0)
        attn_drop_rate = _p('attn_drop_rate', 0.0)
        drop_path_rate = _p('drop_path_rate', 0.1)
        norm_layer_str = _p('norm_layer_name', 'LayerNorm')
        self.norm_layer = getattr(nn, norm_layer_str) if hasattr(nn, norm_layer_str) else nn.LayerNorm

        self.use_dfca = _p('use_dfca', True) and spectral_channels > 0
        dfca_embed_dim_match_rgb = _p('dfca_embed_dim_match_rgb', True)
        dfca_num_heads = _p('dfca_num_heads', 12)
        dfca_drop_rate = _p('dfca_drop_rate', 0.1)
        dfca_use_disease_mask = _p('dfca_use_disease_mask', True)

        self.use_gradient_checkpointing = _p('use_gradient_checkpointing', False)

        self.ssl_enable_mae = _p('ssl_enable_mae', False)
        self.ssl_mae_mask_ratio = _p('ssl_mae_mask_ratio', 0.75)
        ssl_mae_decoder_dim = _p('ssl_mae_decoder_dim', 64)
        # self.ssl_mae_norm_pix_loss = _p('ssl_mae_norm_pix_loss', True) # Used in loss calculation, not model

        self.ssl_enable_contrastive = _p('ssl_enable_contrastive', False)
        ssl_contrastive_projector_dim = _p('ssl_contrastive_projector_dim', 128)
        ssl_contrastive_projector_depth = _p('ssl_contrastive_projector_depth', 2)

        self.enable_consistency_loss_heads = _p('enable_consistency_loss_heads', False)
        # --- End Parameter Extraction ---

        self.num_stages = len(depths)
        dpr_per_block = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # Stochastic depth decay rule

        # RGB Stream
        self.rgb_patch_embed = PatchEmbed(img_size=self.current_img_size, patch_size=self.patch_size, in_chans=3,
                                          embed_dim=embed_dim_rgb, norm_layer=self.norm_layer)
        num_patches_at_current_res = self.rgb_patch_embed.num_patches
        # Positional embedding is defined for the *initial* image size passed to the factory.
        # It will be interpolated if current_img_size changes.
        # For robust interpolation, it's better to define pos_embed for a canonical size if known,
        # or ensure the first img_size passed to __init__ is that canonical size.
        # Here, we use num_patches_at_current_res, implying it adapts if HVT is re-instantiated.
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_current_res, embed_dim_rgb))
        nn.init.trunc_normal_(self.rgb_pos_embed, std=.02)
        self.pos_drop_rgb = nn.Dropout(p=model_drop_rate)

        self.rgb_stages = nn.ModuleList()
        current_dim_rgb = embed_dim_rgb
        current_res_patches_rgb = self.rgb_patch_embed.grid_size
        for i_stage in range(self.num_stages):
            stage = HVTStage(dim=current_dim_rgb, current_input_resolution_patches=current_res_patches_rgb,
                             depth=depths[i_stage], num_heads=num_heads_list[i_stage], mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias, drop=model_drop_rate, attn_drop=attn_drop_rate,
                             drop_path_prob=dpr_per_block[sum(depths[:i_stage]):sum(depths[:i_stage+1])],
                             norm_layer=self.norm_layer,
                             downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                             use_checkpoint=self.use_gradient_checkpointing)
            self.rgb_stages.append(stage)
            current_res_patches_rgb = stage.output_resolution_patches
            if i_stage < self.num_stages - 1: current_dim_rgb *= 2
        self.final_encoded_dim_rgb = current_dim_rgb
        self.norm_rgb_final_encoder = self.norm_layer(self.final_encoded_dim_rgb)

        # Spectral Stream (optional)
        self.spectral_patch_embed = None; self.spectral_pos_embed = None; self.spectral_stages = None; self.norm_spectral_final_encoder = None
        self.final_encoded_dim_spectral = 0
        if spectral_channels > 0:
            self.spectral_patch_embed = PatchEmbed(img_size=self.current_img_size, patch_size=self.patch_size,
                                                   in_chans=spectral_channels, embed_dim=embed_dim_spectral,
                                                   norm_layer=self.norm_layer)
            num_patches_spectral = self.spectral_patch_embed.num_patches
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches_spectral, embed_dim_spectral))
            nn.init.trunc_normal_(self.spectral_pos_embed, std=.02)
            self.pos_drop_spectral = nn.Dropout(p=model_drop_rate)

            self.spectral_stages = nn.ModuleList()
            current_dim_spectral = embed_dim_spectral
            current_res_patches_spectral = self.spectral_patch_embed.grid_size
            for i_stage in range(self.num_stages):
                stage = HVTStage(dim=current_dim_spectral, current_input_resolution_patches=current_res_patches_spectral,
                                 depth=depths[i_stage], num_heads=num_heads_list[i_stage], mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, drop=model_drop_rate, attn_drop=attn_drop_rate,
                                 drop_path_prob=dpr_per_block[sum(depths[:i_stage]):sum(depths[:i_stage+1])],
                                 norm_layer=self.norm_layer,
                                 downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                                 use_checkpoint=self.use_gradient_checkpointing)
                self.spectral_stages.append(stage)
                current_res_patches_spectral = stage.output_resolution_patches
                if i_stage < self.num_stages - 1: current_dim_spectral *= 2
            self.final_encoded_dim_spectral = current_dim_spectral
            self.norm_spectral_final_encoder = self.norm_layer(self.final_encoded_dim_spectral)
        else: # Ensure spectral_channels is 0 if no spectral stream
            self.hvt_params['spectral_channels'] = 0


        # Fusion and Classifier Head
        self.dfca_module = None; self.simple_fusion_projector = None
        final_classifier_input_dim = self.final_encoded_dim_rgb

        if self.use_dfca and self.spectral_patch_embed is not None:
            dfca_actual_embed_dim = self.final_encoded_dim_rgb if dfca_embed_dim_match_rgb else self.final_encoded_dim_spectral # Or a fixed value
            self.spectral_to_dfca_proj = nn.Identity()
            if self.final_encoded_dim_spectral != dfca_actual_embed_dim:
                logger.info(f"DFCA: Projecting spectral features from {self.final_encoded_dim_spectral} to {dfca_actual_embed_dim} for DFCA input.")
                self.spectral_to_dfca_proj = nn.Linear(self.final_encoded_dim_spectral, dfca_actual_embed_dim)
            
            self.rgb_to_dfca_proj = nn.Identity()
            if self.final_encoded_dim_rgb != dfca_actual_embed_dim:
                 logger.info(f"DFCA: Projecting RGB features from {self.final_encoded_dim_rgb} to {dfca_actual_embed_dim} for DFCA input.")
                 self.rgb_to_dfca_proj = nn.Linear(self.final_encoded_dim_rgb, dfca_actual_embed_dim)

            self.dfca_module = DiseaseFocusedCrossAttention(embed_dim=dfca_actual_embed_dim, num_heads=dfca_num_heads,
                                                            dropout_rate=dfca_drop_rate, use_disease_mask=dfca_use_disease_mask)
            final_classifier_input_dim = dfca_actual_embed_dim
            logger.info(f"DFCA fusion enabled. DFCA embed_dim: {dfca_actual_embed_dim}")
        elif self.spectral_patch_embed is not None: # No DFCA, but spectral exists -> try simple concat
            if self.final_encoded_dim_rgb == self.final_encoded_dim_spectral: # Only concat if dims match
                logger.info(f"Simple concatenation fusion enabled. RGB dim: {self.final_encoded_dim_rgb}, Spectral dim: {self.final_encoded_dim_spectral}")
                self.simple_fusion_projector = nn.Linear(self.final_encoded_dim_rgb + self.final_encoded_dim_spectral, self.final_encoded_dim_rgb)
                final_classifier_input_dim = self.final_encoded_dim_rgb # Project back to RGB dim
            else:
                logger.warning("Simple fusion skipped due to dim mismatch between RGB and Spectral encoded features. Using RGB only for head.")
                # final_classifier_input_dim remains self.final_encoded_dim_rgb
        else:
            logger.info("HVT: Running RGB stream only. No fusion.")
            # final_classifier_input_dim remains self.final_encoded_dim_rgb

        self.classifier_head_norm = self.norm_layer(final_classifier_input_dim)
        self.classifier_head = nn.Linear(final_classifier_input_dim, num_classes)

        # SSL Components Initialization
        if self.ssl_enable_mae:
            self.mae_decoder_rgb = MAEPredictionHead(self.final_encoded_dim_rgb, ssl_mae_decoder_dim, self.patch_size, 3)
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0:
                self.mae_decoder_spectral = MAEPredictionHead(self.final_encoded_dim_spectral, ssl_mae_decoder_dim, self.patch_size, _p('spectral_channels',1))
            else: self.mae_decoder_spectral = None
            logger.info("HVT MAE decoders initialized.")

        if self.ssl_enable_contrastive:
            contrast_layers = []
            current_contrast_dim = final_classifier_input_dim # Use features before final classification
            if ssl_contrastive_projector_depth > 0:
                for i in range(ssl_contrastive_projector_depth):
                    out_c_dim = ssl_contrastive_projector_dim if i == ssl_contrastive_projector_depth - 1 else ssl_contrastive_projector_dim
                    contrast_layers.append(nn.Linear(current_contrast_dim, out_c_dim))
                    if i < ssl_contrastive_projector_depth - 1: contrast_layers.append(nn.GELU())
                    current_contrast_dim = out_c_dim
            self.contrastive_projector = nn.Sequential(*contrast_layers) if contrast_layers else nn.Identity()
            logger.info(f"HVT Contrastive projector initialized (output dim: {current_contrast_dim}).")

        if self.enable_consistency_loss_heads:
            self.aux_head_rgb = nn.Linear(self.final_encoded_dim_rgb, num_classes)
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0:
                self.aux_head_spectral = nn.Linear(self.final_encoded_dim_spectral, num_classes)
            else: self.aux_head_spectral = None
            logger.info("HVT Auxiliary heads for consistency loss initialized.")

        self.apply(self._init_weights)
        logger.info(f"DiseaseAwareHVT initialized for image size {self.current_img_size} and {self.num_classes} classes.")


    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv2d) and m.bias is not None: nn.init.constant_(m.bias, 0)

    def _interpolate_pos_embed(self, pos_embed_param: nn.Parameter,
                               current_patch_grid_H: int, current_patch_grid_W: int) -> torch.Tensor:
        # pos_embed_param: [1, N_original_patches, C_embed]
        # N_original_patches is num_patches for which pos_embed_param was defined (at init).
        N_original = pos_embed_param.shape[1]
        N_current = current_patch_grid_H * current_patch_grid_W

        if N_current == N_original: # No interpolation needed if grid size hasn't changed from param definition
            return pos_embed_param

        dim = pos_embed_param.shape[2]
        # Infer H0, W0 from N_original assuming it was square or matches initial rgb_patch_embed.grid_size
        # This assumes self.rgb_patch_embed.grid_size (at init) correctly reflects N_original
        # A more robust way would be to store H0, W0 at init.
        # For now, try to infer:
        H0 = W0 = 0
        if math.isqrt(N_original)**2 == N_original:
            H0 = W0 = math.isqrt(N_original)
        else: # Non-square, try to use initial patch embed grid if available (complex if HVT is re-used)
              # For simplicity, error if not square and N_current != N_original
            logger.error(f"Positional embedding interpolation error: N_original ({N_original}) is not a perfect square, "
                         f"and N_current ({N_current}) is different. Interpolation might be incorrect. "
                         "Positional embeddings should ideally be defined for a canonical grid size.")
            # Fallback: return un-interpolated if shapes don't allow simple square assumption
            return pos_embed_param

        pos_embed_to_interp = pos_embed_param.reshape(1, H0, W0, dim).permute(0, 3, 1, 2) # 1, C, H0, W0
        pos_embed_interp = F.interpolate(pos_embed_to_interp, size=(current_patch_grid_H, current_patch_grid_W),
                                         mode='bicubic', align_corners=False)
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).flatten(1, 2) # 1, N_current, C
        return pos_embed_interp

    def _forward_stream(self, x_img: torch.Tensor,
                        patch_embed_layer: PatchEmbed, pos_embed_param: nn.Parameter,
                        pos_drop_layer: nn.Dropout, stages_list: nn.ModuleList,
                        final_norm_layer: nn.Module) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, _, H_img, W_img = x_img.shape
        # Current patch grid based on current image size and fixed patch_size
        current_H_patch_grid, current_W_patch_grid = H_img // self.patch_size, W_img // self.patch_size

        x_patches = patch_embed_layer(x_img) # B, N_current, C_embed
        
        # Interpolate positional embeddings for the current patch grid size
        interpolated_pos_embed = self._interpolate_pos_embed(pos_embed_param, current_H_patch_grid, current_W_patch_grid)
        x_patches = x_patches + interpolated_pos_embed
        x_patches = pos_drop_layer(x_patches)

        # Dynamically update input_resolution_patches for each stage before forwarding
        # This is important if PatchMerging changes the number of patches between stages
        current_res_patches_for_stage = (current_H_patch_grid, current_W_patch_grid)
        for stage_idx, stage_module in enumerate(stages_list):
            # Update stage's internal idea of input resolution if it has PatchMerging
            # This is tricky; PatchMerging needs to know its input H, W in patches.
            # HVTStage constructor already sets current_input_resolution_patches,
            # and PatchMerging uses its own input_resolution_patches.
            # We need to ensure these are consistent if image size changes.
            # For now, assume HVTStage and PatchMerging are robustly handling this.
            # The key is that PatchMerging calculates its output_resolution_patches.
            if hasattr(stage_module.downsample_layer, 'input_resolution_patches'):
                stage_module.downsample_layer.input_resolution_patches = current_res_patches_for_stage

            x_patches = stage_module(x_patches)
            current_res_patches_for_stage = stage_module.output_resolution_patches # Update for next stage or final norm

        x_encoded = final_norm_layer(x_patches)
        # Return encoded features and the *original* patch grid dimensions for this stream
        return x_encoded, (current_H_patch_grid, current_W_patch_grid)


    def forward_features_encoded(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int,int], Optional[Tuple[int,int]]]:
        # Update current_img_size for this forward pass, for PatchEmbed and pos_embed interpolation
        self.rgb_patch_embed.img_size = (rgb_img.shape[2], rgb_img.shape[3])
        self.rgb_patch_embed.grid_size = (rgb_img.shape[2] // self.patch_size, rgb_img.shape[3] // self.patch_size)
        self.rgb_patch_embed.num_patches = self.rgb_patch_embed.grid_size[0] * self.rgb_patch_embed.grid_size[1]

        x_rgb_encoded, rgb_orig_patch_grid = self._forward_stream(
            rgb_img, self.rgb_patch_embed, self.rgb_pos_embed, self.pos_drop_rgb,
            self.rgb_stages, self.norm_rgb_final_encoder
        )
        x_spectral_encoded, spectral_orig_patch_grid = None, None
        if spectral_img is not None and self.spectral_patch_embed is not None:
            self.spectral_patch_embed.img_size = (spectral_img.shape[2], spectral_img.shape[3])
            self.spectral_patch_embed.grid_size = (spectral_img.shape[2] // self.patch_size, spectral_img.shape[3] // self.patch_size)
            self.spectral_patch_embed.num_patches = self.spectral_patch_embed.grid_size[0] * self.spectral_patch_embed.grid_size[1]

            x_spectral_encoded, spectral_orig_patch_grid = self._forward_stream(
                spectral_img, self.spectral_patch_embed, self.spectral_pos_embed, self.pos_drop_spectral,
                self.spectral_stages, self.norm_spectral_final_encoder
            )
        return x_rgb_encoded, x_spectral_encoded, rgb_orig_patch_grid, spectral_orig_patch_grid

    def _mae_reconstruct(self, x_img_orig: torch.Tensor, x_encoded_modality_final_stage: torch.Tensor,
                         orig_patch_grid_H: int, orig_patch_grid_W: int,
                         mae_decoder_modality: MAEPredictionHead,
                         mask_orig_patches_flat: torch.Tensor):
        B, C_img, _, _ = x_img_orig.shape # Use actual C_img for target
        P = self.patch_size
        N_orig_patches = orig_patch_grid_H * orig_patch_grid_W

        # Critical: MAE reconstruction assumes that x_encoded_modality_final_stage
        # has one token per *original* patch if no patch merging, or that the MAE decoder
        # can handle features from a stage *before* patch merging if merging is active.
        # The current MAEPredictionHead is simple and expects features corresponding to original patches.
        if x_encoded_modality_final_stage.shape[1] != N_orig_patches:
            logger.warning(
                f"MAE reconstruction: Mismatch between num_encoded_tokens ({x_encoded_modality_final_stage.shape[1]}) "
                f"and num_original_patches ({N_orig_patches}). This indicates PatchMerging is active. "
                f"The current simple MAE decoder may not work correctly with features from the final merged stage. "
                f"Consider using features from before merging or a hierarchical MAE decoder. "
                f"MAE for this modality will be skipped/return empty."
            )
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device) # Return empty

        # Target patches (pixels)
        target_patches_unfold = x_img_orig.unfold(2, P, P).unfold(3, P, P) # B, C_img, H_grid, W_grid, P, P
        target_patches = target_patches_unfold.permute(0, 2, 3, 1, 4, 5).reshape(B, N_orig_patches, C_img*P*P) # B, N_orig, Pixels

        if mask_orig_patches_flat.shape != (B, N_orig_patches): # Should be B, N_orig_patches
            logger.error(f"MAE mask shape error. Expected ({B}, {N_orig_patches}), got {mask_orig_patches_flat.shape}")
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device)

        selected_encoded_features = x_encoded_modality_final_stage[mask_orig_patches_flat] # (TotalMasked, C_encoded_final)
        selected_target_patches = target_patches[mask_orig_patches_flat] # (TotalMasked, Pixels)

        if selected_encoded_features.numel() == 0: # No patches were masked
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device)

        predictions = mae_decoder_modality(selected_encoded_features) # (TotalMasked, Pixels)
        return predictions, selected_target_patches


    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None,
                mode: str = 'classify',
                mae_mask_custom: Optional[torch.Tensor] = None # B, N_patches_orig
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str,torch.Tensor]], Dict[str, torch.Tensor]]:

        x_rgb_encoded, x_spectral_encoded, \
            (rgb_orig_H_grid, rgb_orig_W_grid), \
            spec_orig_grids_nullable = self.forward_features_encoded(rgb_img, spectral_img)

        spec_orig_H_grid, spec_orig_W_grid = (None, None)
        if spec_orig_grids_nullable is not None:
            spec_orig_H_grid, spec_orig_W_grid = spec_orig_grids_nullable

        # --- MAE Mode ---
        if mode == 'mae':
            if not self.ssl_enable_mae: raise ValueError("MAE mode called but not enabled in HVT config.")
            mae_outputs: Dict[str, Optional[torch.Tensor]] = {}
            
            # RGB MAE
            B_rgb = rgb_img.shape[0]
            num_patches_rgb_orig = rgb_orig_H_grid * rgb_orig_W_grid
            if mae_mask_custom is not None:
                mask_rgb_flat = mae_mask_custom.to(rgb_img.device).view(B_rgb, -1)
                if mask_rgb_flat.shape[1] != num_patches_rgb_orig:
                    raise ValueError(f"Custom RGB MAE mask shape error. Expected Bx{num_patches_rgb_orig}, got {mask_rgb_flat.shape}")
            else: # Random mask
                noise = torch.rand(B_rgb, num_patches_rgb_orig, device=rgb_img.device)
                ids_shuffle = torch.argsort(noise, dim=1); ids_restore = torch.argsort(ids_shuffle, dim=1)
                len_keep_rgb = int(num_patches_rgb_orig * (1 - self.ssl_mae_mask_ratio))
                mask_rgb_flat_sorted = torch.ones(B_rgb, num_patches_rgb_orig, dtype=torch.bool, device=rgb_img.device); mask_rgb_flat_sorted[:, :len_keep_rgb] = False
                mask_rgb_flat = torch.gather(mask_rgb_flat_sorted, dim=1, index=ids_restore)

            pred_rgb, target_rgb = self._mae_reconstruct(rgb_img, x_rgb_encoded, rgb_orig_H_grid, rgb_orig_W_grid, self.mae_decoder_rgb, mask_rgb_flat)
            mae_outputs.update({'pred_rgb': pred_rgb, 'target_rgb': target_rgb, 'mask_rgb': mask_rgb_flat.view(B_rgb, rgb_orig_H_grid, rgb_orig_W_grid)})

            # Spectral MAE (if applicable)
            if spectral_img is not None and x_spectral_encoded is not None and self.mae_decoder_spectral and \
               spec_orig_H_grid is not None and spec_orig_W_grid is not None:
                B_spec = spectral_img.shape[0]; num_patches_spec_orig = spec_orig_H_grid * spec_orig_W_grid
                if mae_mask_custom is not None and num_patches_spec_orig == num_patches_rgb_orig: mask_spec_flat = mae_mask_custom.to(spectral_img.device).view(B_spec, -1)
                else:
                    noise_s = torch.rand(B_spec, num_patches_spec_orig, device=spectral_img.device); ids_shuffle_s = torch.argsort(noise_s, dim=1); ids_restore_s = torch.argsort(ids_shuffle_s, dim=1)
                    len_keep_s = int(num_patches_spec_orig * (1 - self.ssl_mae_mask_ratio)); mask_spec_flat_sorted_s = torch.ones(B_spec, num_patches_spec_orig, dtype=torch.bool, device=spectral_img.device); mask_spec_flat_sorted_s[:, :len_keep_s] = False
                    mask_spec_flat = torch.gather(mask_spec_flat_sorted_s, dim=1, index=ids_restore_s)
                pred_spec, target_spec = self._mae_reconstruct(spectral_img, x_spectral_encoded, spec_orig_H_grid, spec_orig_W_grid, self.mae_decoder_spectral, mask_spec_flat)
                mae_outputs.update({'pred_spectral': pred_spec, 'target_spectral': target_spec, 'mask_spectral': mask_spec_flat.view(B_spec, spec_orig_H_grid, spec_orig_W_grid)})
            return mae_outputs

        # --- Fusion Logic for Classification/Embedding Modes ---
        fused_features = x_rgb_encoded # Default to RGB if no fusion
        if x_spectral_encoded is not None and self.spectral_patch_embed is not None:
            if self.use_dfca and self.dfca_module is not None:
                # Ensure dimensions match for DFCA (transpose to Seq, Batch, Dim for MultiheadAttention)
                projected_x_rgb_for_dfca = self.rgb_to_dfca_proj(x_rgb_encoded).transpose(0,1)
                projected_x_spectral_for_dfca = self.spectral_to_dfca_proj(x_spectral_encoded).transpose(0,1)
                fused_features_dfca = self.dfca_module(projected_x_rgb_for_dfca, projected_x_spectral_for_dfca).transpose(0,1)
                fused_features = fused_features_dfca # DFCA output is the new fused_features
            elif self.simple_fusion_projector is not None: # Use simple concat + projection
                if x_rgb_encoded.shape[1] == x_spectral_encoded.shape[1]: # Ensure patch numbers match
                    combined = torch.cat((x_rgb_encoded, x_spectral_encoded), dim=2)
                    fused_features = self.simple_fusion_projector(combined)
                # else: fused_features remains x_rgb_encoded (fallback handled by init)
            # else: fused_features remains x_rgb_encoded (no fusion method defined)

        pooled_features = fused_features.mean(dim=1) # Global Average Pooling over patch tokens

        # --- Other Modes ---
        if mode == 'get_embeddings':
            embeddings = {'fused_pooled': pooled_features, 'rgb_pooled': x_rgb_encoded.mean(dim=1)}
            if x_spectral_encoded is not None: embeddings['spectral_pooled'] = x_spectral_encoded.mean(dim=1)
            return embeddings
        if mode == 'contrastive':
            if not self.ssl_enable_contrastive: raise ValueError("Contrastive mode called but HVT.ssl_enable_contrastive is False.")
            return self.contrastive_projector(pooled_features)
        if mode == 'classify':
            x_for_head = self.classifier_head_norm(pooled_features)
            main_logits = self.classifier_head(x_for_head)
            if self.enable_consistency_loss_heads:
                aux_outputs = {}
                aux_outputs['logits_rgb'] = self.aux_head_rgb(x_rgb_encoded.mean(dim=1))
                if x_spectral_encoded is not None and self.spectral_patch_embed and self.aux_head_spectral:
                    aux_outputs['logits_spectral'] = self.aux_head_spectral(x_spectral_encoded.mean(dim=1))
                return main_logits, aux_outputs
            else: return main_logits
        raise ValueError(f"Unknown HVT forward mode: {mode}")


# --- Factory Function ---
def create_disease_aware_hvt(
        current_img_size: Tuple[int, int],
        num_classes: int,
        model_params_dict: Dict[str, Any]
    ) -> DiseaseAwareHVT:
    """
    Instantiates DiseaseAwareHVT using parameters from the provided model_params_dict.
    Args:
        current_img_size (Tuple[int,int]): The current image resolution (H, W).
        num_classes (int): Number of output classes for the main classifier.
        model_params_dict (Dict[str, Any]): Dictionary containing all parameters for HVT.
                                             Expected to match keys in HVT_MODEL_PARAMS from config.py.
    """
    logger.info(f"Factory: Creating DiseaseAwareHVT for img_size: {current_img_size}, num_classes: {num_classes}")
    # logger.debug(f"Factory: Received model_params_dict: {model_params_dict}")

    # The DiseaseAwareHVT constructor will extract necessary params from model_params_dict
    hvt_model = DiseaseAwareHVT(
        img_size=current_img_size,
        num_classes=num_classes,
        hvt_params=model_params_dict # Pass the whole dict
    )
    return hvt_model