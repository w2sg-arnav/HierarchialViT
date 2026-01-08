# hvit/models/hvt.py
"""
Disease-Aware Hierarchical Vision Transformer (HViT).

This module implements the core HViT architecture with:
- Multi-scale feature extraction through hierarchical stages
- Optional spectral data fusion via DFCA
- SSL support (MAE and contrastive learning)
- Progressive resolution training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict, Any
import math
import logging

from hvit.models.dfca import DiseaseFocusedCrossAttention

logger = logging.getLogger(__name__)

# --- Constants ---
INIT_STD = 0.02  # Standard deviation for weight initialization (following ViT convention)


# --- Helper Modules ---

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample.
    
    Args:
        x: Input tensor of shape (B, ...).
        drop_prob: Probability of dropping a path.
        training: Whether the model is in training mode.
    
    Returns:
        Tensor with dropped paths during training.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    
    Args:
        drop_prob: Probability of dropping a path. Default: None (0.0).
    """
    
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob or 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    
    Divides an image into patches and projects them to an embedding dimension.
    
    Args:
        img_size: Input image size as (H, W). Default: (224, 224).
        patch_size: Size of each patch. Default: 16.
        in_chans: Number of input channels. Default: 3.
        embed_dim: Embedding dimension. Default: 96.
        norm_layer: Normalization layer. Default: None.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        if img_size[0] % patch_size != 0 or img_size[1] % patch_size != 0:
            logger.warning(
                f"PatchEmbed: Img dims {img_size} not perfectly divisible by patch_size {patch_size}."
            )
        
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, C_embed, H_grid, W_grid
        x = x.flatten(2)  # B, C_embed, N_patches_flat
        x = x.transpose(1, 2)  # B, N_patches_flat, C_embed
        return self.norm(x)


class Attention(nn.Module):
    """Multi-head Self-Attention module.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to QKV projection.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(
                f"Attention Error: dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        actual_head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, actual_head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP module with GELU activation.
    
    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension.
        act_layer: Activation layer.
        drop: Dropout rate.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        qkv_bias: Whether to add bias to QKV.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Drop path rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging layer for hierarchical feature extraction.
    
    Reduces spatial resolution by 2x and doubles channel dimension.
    
    Args:
        input_resolution_patches: Input resolution in patches (H, W).
        dim: Input dimension.
        norm_layer: Normalization layer.
    """
    
    def __init__(
        self,
        input_resolution_patches: Tuple[int, int],
        dim: int,
        norm_layer: type = nn.LayerNorm
    ) -> None:
        super().__init__()
        self.input_resolution_patches = input_resolution_patches
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.output_resolution_patches = (
            input_resolution_patches[0] // 2,
            input_resolution_patches[1] // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H_patch, W_patch = self.input_resolution_patches
        
        if L != H_patch * W_patch:
            if math.isqrt(L) ** 2 == L:
                H_patch = W_patch = math.isqrt(L)
            else:
                raise ValueError(
                    f"PatchMerging: L ({L}) != H_patch*W_patch ({H_patch*W_patch}), "
                    "and L is not a perfect square."
                )
        
        if H_patch % 2 != 0 or W_patch % 2 != 0:
            raise ValueError(
                f"PatchMerging: Input patch resolution H_patch={H_patch}, W_patch={W_patch} "
                "must be even for 2x2 merging."
            )

        x = x.view(B, H_patch, W_patch, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        return self.reduction(x)

    def extra_repr(self) -> str:
        return f"input_resolution_patches={self.input_resolution_patches}, dim={self.dim}"


class HVTStage(nn.Module):
    """Single stage of the Hierarchical Vision Transformer.
    
    Contains multiple transformer blocks and optional downsampling.
    
    Args:
        dim: Input dimension.
        current_input_resolution_patches: Current resolution in patches.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        qkv_bias: Whether to add bias to QKV.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path_prob: Drop path probability.
        norm_layer: Normalization layer.
        downsample_class: Downsampling layer class.
        use_checkpoint: Whether to use gradient checkpointing.
    """
    
    def __init__(
        self,
        dim: int,
        current_input_resolution_patches: Tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path_prob: Union[List[float], float] = 0.0,
        norm_layer: type = nn.LayerNorm,
        downsample_class: Optional[type] = None,
        use_checkpoint: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution_patches = current_input_resolution_patches
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample_layer = None
        self.output_resolution_patches = current_input_resolution_patches
        if downsample_class is not None:
            self.downsample_layer = downsample_class(
                input_resolution_patches=current_input_resolution_patches,
                dim=dim,
                norm_layer=norm_layer
            )
            self.output_resolution_patches = self.downsample_layer.output_resolution_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint and self.training and not torch.jit.is_scripting():
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x


class MAEPredictionHead(nn.Module):
    """MAE prediction head for masked autoencoding.
    
    Args:
        embed_dim: Input embedding dimension.
        decoder_embed_dim: Decoder hidden dimension.
        patch_size: Patch size.
        out_chans: Number of output channels.
    """
    
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int,
        patch_size: int,
        out_chans: int
    ) -> None:
        super().__init__()
        self.decoder_pred = nn.Sequential(
            nn.Linear(embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, patch_size * patch_size * out_chans)
        )
        self.patch_size = patch_size
        self.out_chans = out_chans
        logger.info(
            f"MAEPredictionHead: in_dim={embed_dim}, decoder_hidden_dim={decoder_embed_dim}, "
            f"target_pixels_per_patch={patch_size * patch_size * out_chans}"
        )

    def forward(self, x_encoded_patches: torch.Tensor) -> torch.Tensor:
        return self.decoder_pred(x_encoded_patches)


# --- DiseaseAwareHVT Class (Main Backbone) ---
class DiseaseAwareHVT(nn.Module):
    """Disease-Aware Hierarchical Vision Transformer.
    
    A hierarchical vision transformer designed for plant disease classification
    with optional spectral data fusion via Disease-Focused Cross-Attention (DFCA).
    
    Args:
        img_size: Input image size as (H, W).
        num_classes: Number of output classes.
        hvt_params: Dictionary containing all HVT configuration parameters.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int],
        num_classes: int,
        hvt_params: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.current_img_size = img_size
        self.num_classes = num_classes
        self.hvt_params = hvt_params

        # Extract parameters from hvt_params dict
        def _p(key: str, default: Any) -> Any:
            return self.hvt_params.get(key, default)

        self.patch_size = _p('patch_size', 16)
        embed_dim_rgb = _p('embed_dim_rgb', 96)
        embed_dim_spectral = _p('embed_dim_spectral', 96)
        spectral_channels = _p('spectral_channels', 0)
        depths = _p('depths', [2, 2, 6, 2])
        num_heads_list = _p('num_heads', [3, 6, 12, 24])
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

        self.ssl_enable_contrastive = _p('ssl_enable_contrastive', False)
        ssl_contrastive_projector_dim = _p('ssl_contrastive_projector_dim', 128)
        ssl_contrastive_projector_depth = _p('ssl_contrastive_projector_depth', 2)

        self.enable_consistency_loss_heads = _p('enable_consistency_loss_heads', False)

        self.num_stages = len(depths)
        dpr_per_block = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # RGB Stream
        self.rgb_patch_embed = PatchEmbed(
            img_size=self.current_img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=embed_dim_rgb,
            norm_layer=self.norm_layer
        )
        num_patches_at_current_res = self.rgb_patch_embed.num_patches
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, num_patches_at_current_res, embed_dim_rgb))
        nn.init.trunc_normal_(self.rgb_pos_embed, std=INIT_STD)
        self.pos_drop_rgb = nn.Dropout(p=model_drop_rate)

        self.rgb_stages = nn.ModuleList()
        current_dim_rgb = embed_dim_rgb
        current_res_patches_rgb = self.rgb_patch_embed.grid_size
        for i_stage in range(self.num_stages):
            stage = HVTStage(
                dim=current_dim_rgb,
                current_input_resolution_patches=current_res_patches_rgb,
                depth=depths[i_stage],
                num_heads=num_heads_list[i_stage],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=model_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_prob=dpr_per_block[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                norm_layer=self.norm_layer,
                downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                use_checkpoint=self.use_gradient_checkpointing
            )
            self.rgb_stages.append(stage)
            current_res_patches_rgb = stage.output_resolution_patches
            if i_stage < self.num_stages - 1:
                current_dim_rgb *= 2
        self.final_encoded_dim_rgb = current_dim_rgb
        self.norm_rgb_final_encoder = self.norm_layer(self.final_encoded_dim_rgb)

        # Spectral Stream (optional)
        self.spectral_patch_embed = None
        self.spectral_pos_embed = None
        self.spectral_stages = None
        self.norm_spectral_final_encoder = None
        self.final_encoded_dim_spectral = 0

        if spectral_channels > 0:
            self.spectral_patch_embed = PatchEmbed(
                img_size=self.current_img_size,
                patch_size=self.patch_size,
                in_chans=spectral_channels,
                embed_dim=embed_dim_spectral,
                norm_layer=self.norm_layer
            )
            num_patches_spectral = self.spectral_patch_embed.num_patches
            self.spectral_pos_embed = nn.Parameter(torch.zeros(1, num_patches_spectral, embed_dim_spectral))
            nn.init.trunc_normal_(self.spectral_pos_embed, std=INIT_STD)
            self.pos_drop_spectral = nn.Dropout(p=model_drop_rate)

            self.spectral_stages = nn.ModuleList()
            current_dim_spectral = embed_dim_spectral
            current_res_patches_spectral = self.spectral_patch_embed.grid_size
            for i_stage in range(self.num_stages):
                stage = HVTStage(
                    dim=current_dim_spectral,
                    current_input_resolution_patches=current_res_patches_spectral,
                    depth=depths[i_stage],
                    num_heads=num_heads_list[i_stage],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=model_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr_per_block[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                    norm_layer=self.norm_layer,
                    downsample_class=PatchMerging if (i_stage < self.num_stages - 1) else None,
                    use_checkpoint=self.use_gradient_checkpointing
                )
                self.spectral_stages.append(stage)
                current_res_patches_spectral = stage.output_resolution_patches
                if i_stage < self.num_stages - 1:
                    current_dim_spectral *= 2
            self.final_encoded_dim_spectral = current_dim_spectral
            self.norm_spectral_final_encoder = self.norm_layer(self.final_encoded_dim_spectral)
        else:
            self.hvt_params['spectral_channels'] = 0

        # Fusion and Classifier Head
        self.dfca_module = None
        self.simple_fusion_projector = None
        final_classifier_input_dim = self.final_encoded_dim_rgb

        if self.use_dfca and self.spectral_patch_embed is not None:
            dfca_actual_embed_dim = (
                self.final_encoded_dim_rgb if dfca_embed_dim_match_rgb else self.final_encoded_dim_spectral
            )
            self.spectral_to_dfca_proj = nn.Identity()
            if self.final_encoded_dim_spectral != dfca_actual_embed_dim:
                logger.info(
                    f"DFCA: Projecting spectral features from {self.final_encoded_dim_spectral} "
                    f"to {dfca_actual_embed_dim} for DFCA input."
                )
                self.spectral_to_dfca_proj = nn.Linear(self.final_encoded_dim_spectral, dfca_actual_embed_dim)

            self.rgb_to_dfca_proj = nn.Identity()
            if self.final_encoded_dim_rgb != dfca_actual_embed_dim:
                logger.info(
                    f"DFCA: Projecting RGB features from {self.final_encoded_dim_rgb} "
                    f"to {dfca_actual_embed_dim} for DFCA input."
                )
                self.rgb_to_dfca_proj = nn.Linear(self.final_encoded_dim_rgb, dfca_actual_embed_dim)

            self.dfca_module = DiseaseFocusedCrossAttention(
                embed_dim=dfca_actual_embed_dim,
                num_heads=dfca_num_heads,
                dropout_rate=dfca_drop_rate,
                use_disease_mask=dfca_use_disease_mask
            )
            final_classifier_input_dim = dfca_actual_embed_dim
            logger.info(f"DFCA fusion enabled. DFCA embed_dim: {dfca_actual_embed_dim}")
        elif self.spectral_patch_embed is not None:
            if self.final_encoded_dim_rgb == self.final_encoded_dim_spectral:
                logger.info(
                    f"Simple concatenation fusion enabled. RGB dim: {self.final_encoded_dim_rgb}, "
                    f"Spectral dim: {self.final_encoded_dim_spectral}"
                )
                self.simple_fusion_projector = nn.Linear(
                    self.final_encoded_dim_rgb + self.final_encoded_dim_spectral,
                    self.final_encoded_dim_rgb
                )
                final_classifier_input_dim = self.final_encoded_dim_rgb
            else:
                logger.warning(
                    "Simple fusion skipped due to dim mismatch between RGB and Spectral encoded features. "
                    "Using RGB only for head."
                )
        else:
            logger.info("HVT: Running RGB stream only. No fusion.")

        self.classifier_head_norm = self.norm_layer(final_classifier_input_dim)
        self.classifier_head = nn.Linear(final_classifier_input_dim, num_classes)

        # SSL Components Initialization
        if self.ssl_enable_mae:
            self.mae_decoder_rgb = MAEPredictionHead(
                self.final_encoded_dim_rgb,
                ssl_mae_decoder_dim,
                self.patch_size,
                3
            )
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0:
                self.mae_decoder_spectral = MAEPredictionHead(
                    self.final_encoded_dim_spectral,
                    ssl_mae_decoder_dim,
                    self.patch_size,
                    _p('spectral_channels', 1)
                )
            else:
                self.mae_decoder_spectral = None
            logger.info("HVT MAE decoders initialized.")

        if self.ssl_enable_contrastive:
            contrast_layers = []
            current_contrast_dim = final_classifier_input_dim
            if ssl_contrastive_projector_depth > 0:
                for i in range(ssl_contrastive_projector_depth):
                    out_c_dim = ssl_contrastive_projector_dim
                    contrast_layers.append(nn.Linear(current_contrast_dim, out_c_dim))
                    if i < ssl_contrastive_projector_depth - 1:
                        contrast_layers.append(nn.GELU())
                    current_contrast_dim = out_c_dim
            self.contrastive_projector = nn.Sequential(*contrast_layers) if contrast_layers else nn.Identity()
            logger.info(f"HVT Contrastive projector initialized (output dim: {current_contrast_dim}).")

        if self.enable_consistency_loss_heads:
            self.aux_head_rgb = nn.Linear(self.final_encoded_dim_rgb, num_classes)
            if self.spectral_patch_embed and self.final_encoded_dim_spectral > 0:
                self.aux_head_spectral = nn.Linear(self.final_encoded_dim_spectral, num_classes)
            else:
                self.aux_head_spectral = None
            logger.info("HVT Auxiliary heads for consistency loss initialized.")

        self.apply(self._init_weights)
        logger.info(f"DiseaseAwareHVT initialized for image size {self.current_img_size} and {self.num_classes} classes.")

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights following ViT conventions."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=INIT_STD)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _interpolate_pos_embed(
        self,
        pos_embed_param: nn.Parameter,
        current_patch_grid_H: int,
        current_patch_grid_W: int
    ) -> torch.Tensor:
        """Interpolate positional embeddings for different resolutions."""
        N_original = pos_embed_param.shape[1]
        N_current = current_patch_grid_H * current_patch_grid_W

        if N_current == N_original:
            return pos_embed_param

        dim = pos_embed_param.shape[2]
        H0 = W0 = 0
        if math.isqrt(N_original) ** 2 == N_original:
            H0 = W0 = math.isqrt(N_original)
        else:
            logger.error(
                f"Positional embedding interpolation error: N_original ({N_original}) is not a perfect square."
            )
            return pos_embed_param

        pos_embed_to_interp = pos_embed_param.reshape(1, H0, W0, dim).permute(0, 3, 1, 2)
        pos_embed_interp = F.interpolate(
            pos_embed_to_interp,
            size=(current_patch_grid_H, current_patch_grid_W),
            mode='bicubic',
            align_corners=False
        )
        pos_embed_interp = pos_embed_interp.permute(0, 2, 3, 1).flatten(1, 2)
        return pos_embed_interp

    def _forward_stream(
        self,
        x_img: torch.Tensor,
        patch_embed_layer: PatchEmbed,
        pos_embed_param: nn.Parameter,
        pos_drop_layer: nn.Dropout,
        stages_list: nn.ModuleList,
        final_norm_layer: nn.Module
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Forward pass through a single stream (RGB or spectral)."""
        B, _, H_img, W_img = x_img.shape
        current_H_patch_grid = H_img // self.patch_size
        current_W_patch_grid = W_img // self.patch_size

        x_patches = patch_embed_layer(x_img)
        interpolated_pos_embed = self._interpolate_pos_embed(
            pos_embed_param,
            current_H_patch_grid,
            current_W_patch_grid
        )
        x_patches = x_patches + interpolated_pos_embed
        x_patches = pos_drop_layer(x_patches)

        current_res_patches_for_stage = (current_H_patch_grid, current_W_patch_grid)
        for stage_module in stages_list:
            if hasattr(stage_module.downsample_layer, 'input_resolution_patches'):
                stage_module.downsample_layer.input_resolution_patches = current_res_patches_for_stage
            x_patches = stage_module(x_patches)
            current_res_patches_for_stage = stage_module.output_resolution_patches

        x_encoded = final_norm_layer(x_patches)
        return x_encoded, (current_H_patch_grid, current_W_patch_grid)

    def forward_features_encoded(
        self,
        rgb_img: torch.Tensor,
        spectral_img: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int], Optional[Tuple[int, int]]]:
        """Extract encoded features from both streams."""
        self.rgb_patch_embed.img_size = (rgb_img.shape[2], rgb_img.shape[3])
        self.rgb_patch_embed.grid_size = (
            rgb_img.shape[2] // self.patch_size,
            rgb_img.shape[3] // self.patch_size
        )
        self.rgb_patch_embed.num_patches = (
            self.rgb_patch_embed.grid_size[0] * self.rgb_patch_embed.grid_size[1]
        )

        x_rgb_encoded, rgb_orig_patch_grid = self._forward_stream(
            rgb_img,
            self.rgb_patch_embed,
            self.rgb_pos_embed,
            self.pos_drop_rgb,
            self.rgb_stages,
            self.norm_rgb_final_encoder
        )

        x_spectral_encoded, spectral_orig_patch_grid = None, None
        if spectral_img is not None and self.spectral_patch_embed is not None:
            self.spectral_patch_embed.img_size = (spectral_img.shape[2], spectral_img.shape[3])
            self.spectral_patch_embed.grid_size = (
                spectral_img.shape[2] // self.patch_size,
                spectral_img.shape[3] // self.patch_size
            )
            self.spectral_patch_embed.num_patches = (
                self.spectral_patch_embed.grid_size[0] * self.spectral_patch_embed.grid_size[1]
            )

            x_spectral_encoded, spectral_orig_patch_grid = self._forward_stream(
                spectral_img,
                self.spectral_patch_embed,
                self.spectral_pos_embed,
                self.pos_drop_spectral,
                self.spectral_stages,
                self.norm_spectral_final_encoder
            )

        return x_rgb_encoded, x_spectral_encoded, rgb_orig_patch_grid, spectral_orig_patch_grid

    def _mae_reconstruct(
        self,
        x_img_orig: torch.Tensor,
        x_encoded_modality_final_stage: torch.Tensor,
        orig_patch_grid_H: int,
        orig_patch_grid_W: int,
        mae_decoder_modality: MAEPredictionHead,
        mask_orig_patches_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct masked patches for MAE."""
        B, C_img, _, _ = x_img_orig.shape
        P = self.patch_size
        N_orig_patches = orig_patch_grid_H * orig_patch_grid_W

        if x_encoded_modality_final_stage.shape[1] != N_orig_patches:
            logger.warning(
                f"MAE reconstruction: Mismatch between num_encoded_tokens "
                f"({x_encoded_modality_final_stage.shape[1]}) and num_original_patches ({N_orig_patches})."
            )
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device)

        target_patches_unfold = x_img_orig.unfold(2, P, P).unfold(3, P, P)
        target_patches = target_patches_unfold.permute(0, 2, 3, 1, 4, 5).reshape(B, N_orig_patches, C_img * P * P)

        if mask_orig_patches_flat.shape != (B, N_orig_patches):
            logger.error(
                f"MAE mask shape error. Expected ({B}, {N_orig_patches}), got {mask_orig_patches_flat.shape}"
            )
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device)

        selected_encoded_features = x_encoded_modality_final_stage[mask_orig_patches_flat]
        selected_target_patches = target_patches[mask_orig_patches_flat]

        if selected_encoded_features.numel() == 0:
            return torch.empty(0, device=x_img_orig.device), torch.empty(0, device=x_img_orig.device)

        predictions = mae_decoder_modality(selected_encoded_features)
        return predictions, selected_target_patches

    def forward(
        self,
        rgb_img: torch.Tensor,
        spectral_img: Optional[torch.Tensor] = None,
        mode: str = 'classify',
        mae_mask_custom: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Forward pass with multiple mode support.
        
        Args:
            rgb_img: RGB input tensor of shape (B, 3, H, W).
            spectral_img: Optional spectral input tensor.
            mode: One of 'classify', 'mae', 'contrastive', 'get_embeddings'.
            mae_mask_custom: Optional custom mask for MAE mode.
        
        Returns:
            Depends on mode:
            - 'classify': Classification logits (or tuple with aux outputs).
            - 'mae': Dictionary with predictions and targets.
            - 'contrastive': Projected features for contrastive loss.
            - 'get_embeddings': Dictionary of pooled embeddings.
        """
        x_rgb_encoded, x_spectral_encoded, \
            (rgb_orig_H_grid, rgb_orig_W_grid), \
            spec_orig_grids_nullable = self.forward_features_encoded(rgb_img, spectral_img)

        spec_orig_H_grid, spec_orig_W_grid = (None, None)
        if spec_orig_grids_nullable is not None:
            spec_orig_H_grid, spec_orig_W_grid = spec_orig_grids_nullable

        # --- MAE Mode ---
        if mode == 'mae':
            if not self.ssl_enable_mae:
                raise ValueError("MAE mode called but not enabled in HVT config.")
            mae_outputs: Dict[str, Optional[torch.Tensor]] = {}

            B_rgb = rgb_img.shape[0]
            num_patches_rgb_orig = rgb_orig_H_grid * rgb_orig_W_grid
            if mae_mask_custom is not None:
                mask_rgb_flat = mae_mask_custom.to(rgb_img.device).view(B_rgb, -1)
                if mask_rgb_flat.shape[1] != num_patches_rgb_orig:
                    raise ValueError(
                        f"Custom RGB MAE mask shape error. Expected Bx{num_patches_rgb_orig}, got {mask_rgb_flat.shape}"
                    )
            else:
                noise = torch.rand(B_rgb, num_patches_rgb_orig, device=rgb_img.device)
                ids_shuffle = torch.argsort(noise, dim=1)
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                len_keep_rgb = int(num_patches_rgb_orig * (1 - self.ssl_mae_mask_ratio))
                mask_rgb_flat_sorted = torch.ones(
                    B_rgb, num_patches_rgb_orig, dtype=torch.bool, device=rgb_img.device
                )
                mask_rgb_flat_sorted[:, :len_keep_rgb] = False
                mask_rgb_flat = torch.gather(mask_rgb_flat_sorted, dim=1, index=ids_restore)

            pred_rgb, target_rgb = self._mae_reconstruct(
                rgb_img, x_rgb_encoded, rgb_orig_H_grid, rgb_orig_W_grid,
                self.mae_decoder_rgb, mask_rgb_flat
            )
            mae_outputs.update({
                'pred_rgb': pred_rgb,
                'target_rgb': target_rgb,
                'mask_rgb': mask_rgb_flat.view(B_rgb, rgb_orig_H_grid, rgb_orig_W_grid)
            })

            if (spectral_img is not None and x_spectral_encoded is not None and
                    self.mae_decoder_spectral and spec_orig_H_grid is not None and spec_orig_W_grid is not None):
                B_spec = spectral_img.shape[0]
                num_patches_spec_orig = spec_orig_H_grid * spec_orig_W_grid
                if mae_mask_custom is not None and num_patches_spec_orig == num_patches_rgb_orig:
                    mask_spec_flat = mae_mask_custom.to(spectral_img.device).view(B_spec, -1)
                else:
                    noise_s = torch.rand(B_spec, num_patches_spec_orig, device=spectral_img.device)
                    ids_shuffle_s = torch.argsort(noise_s, dim=1)
                    ids_restore_s = torch.argsort(ids_shuffle_s, dim=1)
                    len_keep_s = int(num_patches_spec_orig * (1 - self.ssl_mae_mask_ratio))
                    mask_spec_flat_sorted_s = torch.ones(
                        B_spec, num_patches_spec_orig, dtype=torch.bool, device=spectral_img.device
                    )
                    mask_spec_flat_sorted_s[:, :len_keep_s] = False
                    mask_spec_flat = torch.gather(mask_spec_flat_sorted_s, dim=1, index=ids_restore_s)
                pred_spec, target_spec = self._mae_reconstruct(
                    spectral_img, x_spectral_encoded, spec_orig_H_grid, spec_orig_W_grid,
                    self.mae_decoder_spectral, mask_spec_flat
                )
                mae_outputs.update({
                    'pred_spectral': pred_spec,
                    'target_spectral': target_spec,
                    'mask_spectral': mask_spec_flat.view(B_spec, spec_orig_H_grid, spec_orig_W_grid)
                })
            return mae_outputs

        # --- Fusion Logic ---
        fused_features = x_rgb_encoded
        if x_spectral_encoded is not None and self.spectral_patch_embed is not None:
            if self.use_dfca and self.dfca_module is not None:
                projected_x_rgb_for_dfca = self.rgb_to_dfca_proj(x_rgb_encoded).transpose(0, 1)
                projected_x_spectral_for_dfca = self.spectral_to_dfca_proj(x_spectral_encoded).transpose(0, 1)
                fused_features_dfca = self.dfca_module(
                    projected_x_rgb_for_dfca, projected_x_spectral_for_dfca
                ).transpose(0, 1)
                fused_features = fused_features_dfca
            elif self.simple_fusion_projector is not None:
                if x_rgb_encoded.shape[1] == x_spectral_encoded.shape[1]:
                    combined = torch.cat((x_rgb_encoded, x_spectral_encoded), dim=2)
                    fused_features = self.simple_fusion_projector(combined)

        pooled_features = fused_features.mean(dim=1)

        # --- Other Modes ---
        if mode == 'get_embeddings':
            embeddings = {
                'fused_pooled': pooled_features,
                'rgb_pooled': x_rgb_encoded.mean(dim=1)
            }
            if x_spectral_encoded is not None:
                embeddings['spectral_pooled'] = x_spectral_encoded.mean(dim=1)
            return embeddings

        if mode == 'contrastive':
            if not self.ssl_enable_contrastive:
                raise ValueError("Contrastive mode called but HVT.ssl_enable_contrastive is False.")
            return self.contrastive_projector(pooled_features)

        if mode == 'classify':
            x_for_head = self.classifier_head_norm(pooled_features)
            main_logits = self.classifier_head(x_for_head)
            if self.enable_consistency_loss_heads:
                aux_outputs = {'logits_rgb': self.aux_head_rgb(x_rgb_encoded.mean(dim=1))}
                if x_spectral_encoded is not None and self.spectral_patch_embed and self.aux_head_spectral:
                    aux_outputs['logits_spectral'] = self.aux_head_spectral(x_spectral_encoded.mean(dim=1))
                return main_logits, aux_outputs
            else:
                return main_logits

        raise ValueError(f"Unknown HVT forward mode: {mode}")


# --- Factory Function ---
def create_disease_aware_hvt(
    current_img_size: Tuple[int, int],
    num_classes: int,
    model_params_dict: Dict[str, Any]
) -> DiseaseAwareHVT:
    """Factory function to create DiseaseAwareHVT instances.
    
    Args:
        current_img_size: Image resolution (H, W).
        num_classes: Number of output classes.
        model_params_dict: Dictionary containing all HVT parameters.
    
    Returns:
        Configured DiseaseAwareHVT instance.
    """
    logger.info(f"Factory: Creating DiseaseAwareHVT for img_size: {current_img_size}, num_classes: {num_classes}")
    hvt_model = DiseaseAwareHVT(
        img_size=current_img_size,
        num_classes=num_classes,
        hvt_params=model_params_dict
    )
    return hvt_model
