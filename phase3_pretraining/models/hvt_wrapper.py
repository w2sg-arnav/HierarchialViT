# phase3_pretraining/models/hvt_wrapper.py
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Import the canonical HVT model from Phase 2
# This assumes phase2_model is in PYTHONPATH or accessible
# For cleaner project structure, phase2_model might be a sub-package or its contents moved.
# For now, using the sys.path append from main pretrain script.
from phase2_model.models.hvt import DiseaseAwareHVT as HVTBackbone 
from .projection_head import ProjectionHead 
from ..config import ( # Use relative import for config within phase3_pretraining
    PROJECTION_DIM, PROJECTION_HIDDEN_DIM,
    # HVT specific configs needed for backbone instantiation if not passed directly
    PATCH_SIZE, EMBED_DIM_RGB, EMBED_DIM_SPECTRAL, HVT_DEPTHS, HVT_NUM_HEADS,
    MLP_RATIO, QKV_BIAS, HVT_MODEL_DROP_RATE, HVT_ATTN_DROP_RATE, HVT_DROP_PATH_RATE,
    DFCA_NUM_HEADS, DFCA_DROP_RATE, SPECTRAL_CHANNELS
)
import logging

logger = logging.getLogger(__name__)

class HVTForPretraining(nn.Module):
    """
    Wrapper around the DiseaseAwareHVT backbone for self-supervised pre-training.
    It replaces the final classification head with a projection head for contrastive loss.
    """
    def __init__(self, img_size: Tuple[int, int],
                 # Backbone HVT parameters (can be taken from config or passed)
                 hvt_patch_size: int = PATCH_SIZE,
                 hvt_embed_dim_rgb: int = EMBED_DIM_RGB,
                 hvt_embed_dim_spectral: int = EMBED_DIM_SPECTRAL,
                 hvt_spectral_channels: int = SPECTRAL_CHANNELS,
                 hvt_depths: list = HVT_DEPTHS,
                 hvt_num_heads: list = HVT_NUM_HEADS,
                 hvt_mlp_ratio: float = MLP_RATIO,
                 hvt_qkv_bias: bool = QKV_BIAS,
                 hvt_drop_rate: float = HVT_MODEL_DROP_RATE,
                 hvt_attn_drop_rate: float = HVT_ATTN_DROP_RATE,
                 hvt_drop_path_rate: float = HVT_DROP_PATH_RATE,
                 hvt_use_dfca: bool = True, # Backbone's DFCA
                 # Projection head parameters
                 projection_in_dim: Optional[int] = None, # If None, inferred from backbone
                 projection_hidden_dim: int = PROJECTION_HIDDEN_DIM,
                 projection_out_dim: int = PROJECTION_DIM
                 ):
        super().__init__()
        
        # Instantiate the backbone HVT (from Phase 2)
        # We set num_classes to a dummy value as its head will be replaced/ignored.
        self.backbone = HVTBackbone(
            img_size=img_size,
            patch_size=hvt_patch_size,
            num_classes=10, # Dummy, will be ignored
            embed_dim_rgb=hvt_embed_dim_rgb,
            embed_dim_spectral=hvt_embed_dim_spectral,
            spectral_channels=hvt_spectral_channels,
            depths=hvt_depths,
            num_heads=hvt_num_heads,
            mlp_ratio=hvt_mlp_ratio,
            qkv_bias=hvt_qkv_bias,
            drop_rate=hvt_drop_rate,
            attn_drop_rate=hvt_attn_drop_rate,
            drop_path_rate=hvt_drop_path_rate,
            use_dfca=hvt_use_dfca # Control if DFCA is part of backbone feature extraction
        )
        
        # Determine the input dimension for the projection head
        # This is the output dimension of the HVT backbone after pooling
        if projection_in_dim is None:
            # The HVT backbone's `fusion_embed_dim` or `current_dim_rgb` after stages
            # It's self.backbone.fusion_embed_dim (if DFCA) or self.backbone.norm_rgb.normalized_shape[0]
            # Simpler: use the 'in_features' of the original head if accessible,
            # or the known output dimension.
            # From HVT: current_dim_rgb * (2**(num_stages-1)) if PatchMerging used correctly
            # For our HVT, it's `self.backbone.head_norm.normalized_shape[0]`
            _projection_in_dim = self.backbone.head_norm.normalized_shape[0]
            logger.info(f"Inferred projection_in_dim from backbone: {_projection_in_dim}")
        else:
            _projection_in_dim = projection_in_dim

        self.projection_head = ProjectionHead(
            in_dim=_projection_in_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim
        )
        
        # For pre-training, we usually don't want the backbone's classification head.
        # We can either set it to nn.Identity() or just not use its output.
        # self.backbone.head = nn.Identity() # Option 1: Modify backbone
        # Option 2: Just don't call backbone's head in forward_pretrain, which is cleaner.

        logger.info("HVTForPretraining wrapper initialized.")

    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None, 
                pretrain_mode: bool = True, extract_features_for_probe: bool = False):
        """
        Args:
            rgb_img: RGB input.
            spectral_img: Optional spectral input.
            pretrain_mode: If True, pass features through projection_head.
            extract_features_for_probe: If True, return backbone features before projection head.
        """
        # Get features from the backbone
        # The backbone.forward method already does pooling and norm before its own head
        # We need features *before* the backbone's classification head but *after* pooling and norm.
        
        # Access backbone's feature extraction part
        x_rgb_feat, x_spec_feat = self.backbone.forward_features(rgb_img, spectral_img)

        if x_spec_feat is not None and self.backbone.spectral_patch_embed is not None:
            if self.backbone.use_dfca:
                projected_x_spec = self.backbone.spectral_dfca_proj(x_spec_feat)
                x_rgb_dfca = x_rgb_feat.transpose(0, 1)
                x_spec_dfca = projected_x_spec.transpose(0, 1)
                fused_features_seq = self.backbone.dfca(x_rgb_dfca, x_spec_dfca).transpose(0, 1)
            else:
                combined = torch.cat((x_rgb_feat, x_spec_feat), dim=2)
                fused_features_seq = self.backbone.simple_fusion_proj(combined)
        else:
            fused_features_seq = x_rgb_feat
            
        # Global average pooling over patch tokens (as done in backbone before its head)
        pooled_features = fused_features_seq.mean(dim=1)
        
        # Pass through the backbone's final normalization layer (head_norm)
        backbone_features = self.backbone.head_norm(pooled_features) # These are features for linear probe

        if extract_features_for_probe:
            return backbone_features # Return features before projection head

        if pretrain_mode:
            projected_features = self.projection_head(backbone_features)
            return projected_features
        else:
            # If not pretrain_mode and not for probe, this implies fine-tuning using original head
            # For that, one would call self.backbone(rgb_img, spectral_img) directly.
            # This wrapper's forward is mainly for pre-training or extracting probe features.
            # To use for classification, call self.backbone.head(backbone_features)
            return self.backbone.head(backbone_features) # Uses original classification head