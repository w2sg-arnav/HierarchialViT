# phase3_pretraining/models/hvt_wrapper.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import logging

# Import the canonical HVT model from Phase 2
# This assumes phase2_model is in PYTHONPATH or installed
try:
    from phase2_model.models.hvt import DiseaseAwareHVT as HVTBackbone
    from phase2_model.models.hvt import create_disease_aware_hvt_from_config as create_hvt_from_phase2_config_struct
except ImportError as e:
    logger.error(f"CRITICAL: Could not import HVTBackbone from phase2_model. Ensure it's in PYTHONPATH. Error: {e}")
    # Fallback to a dummy if import fails to allow other parts to load.
    class HVTBackbone(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.dummy = nn.Identity()
        def forward_features(self, rgb, spectral=None): return torch.randn(rgb.shape[0], 10, 768), (torch.randn(spectral.shape[0],10,768) if spectral is not None else None)
        def head_norm(self, x): return x
        def head(self,x): return x
    def create_hvt_from_phase2_config_struct(img_size_tuple): return HVTBackbone()


from .projection_head import ProjectionHead
# Import config dictionary from within this package
from ..config import config as phase3_cfg # Rename to avoid confusion with backbone's internal cfg

logger = logging.getLogger(__name__)

class HVTForPretraining(nn.Module):
    """
    Wrapper around the DiseaseAwareHVT backbone for self-supervised pre-training.
    Uses HVT-XL parameters from phase3_cfg.
    """
    def __init__(self, img_size: Tuple[int, int]):
        super().__init__()

        # Instantiate the backbone HVT using parameters from phase3_cfg
        # The HVTBackbone now uses its own config loader or defaults if phase2 config fails.
        # We just need to ensure we pass the img_size correctly.
        # The create_disease_aware_hvt_from_config in phase2 takes img_size and then uses
        # its *own* cfg_module (derived from phase3_pretraining.config or its defaults).

        # So, if phase2_model.models.hvt.py correctly imports from phase3_pretraining.config,
        # then create_disease_aware_hvt_from_config will use the HVT-XL parameters from phase3_cfg.
        logger.info(f"HVTWrapper: Attempting to create HVTBackbone using params from phase3_pretraining.config via phase2_model's config loading mechanism.")
        self.backbone: HVTBackbone = create_hvt_from_phase2_config_struct(img_size_tuple=img_size)

        # Verify if backbone was loaded with expected SSL capabilities (though not used by this wrapper directly)
        if hasattr(self.backbone, 'ssl_enable_mae') and self.backbone.ssl_enable_mae:
            logger.info("HVTWrapper: Backbone has MAE capabilities enabled (though not used by SimCLR wrapper).")
        if hasattr(self.backbone, 'ssl_enable_contrastive') and self.backbone.ssl_enable_contrastive:
             logger.info("HVTWrapper: Backbone has its own contrastive projector (though SimCLR wrapper uses its own).")


        logger.info(f"Initialized HVT backbone ({phase3_cfg['model_name']}) for pre-training with img_size {img_size}.")
        logger.info(f"Backbone final RGB encoded dim: {self.backbone.final_encoded_dim_rgb if hasattr(self.backbone, 'final_encoded_dim_rgb') else 'N/A'}")
        if self.backbone.spectral_patch_embed:
            logger.info(f"Backbone final Spectral encoded dim: {self.backbone.final_encoded_dim_spectral if hasattr(self.backbone, 'final_encoded_dim_spectral') else 'N/A'}")


        # Determine projection head input dimension
        # This should come from the features *before* the backbone's original classification head
        # It's the dimension of the pooled, normed features from the desired stream (RGB for SimCLR).
        # The HVTBackbone's `final_encoded_dim_rgb` attribute should give this.
        if hasattr(self.backbone, 'final_encoded_dim_rgb') and self.backbone.final_encoded_dim_rgb > 0:
            _projection_in_dim = self.backbone.final_encoded_dim_rgb
        else: # Fallback if attribute not found
            _projection_in_dim = phase3_cfg['hvt_embed_dim_rgb'] * (2**(len(phase3_cfg['hvt_depths']) - 1))
            logger.warning(f"HVTWrapper: Could not find 'final_encoded_dim_rgb' on backbone. Inferred proj_in_dim={_projection_in_dim} from config. This might be incorrect.")

        logger.info(f"Projection head input dim determined as: {_projection_in_dim}")

        # Create projection head using dimensions from config
        self.projection_head = ProjectionHead(
            in_dim=_projection_in_dim,
            hidden_dim=phase3_cfg['projection_hidden_dim'], # Corrected key
            out_dim=phase3_cfg['projection_dim'],           # Corrected key
            use_batch_norm=True
        )

        # Store config params used to initialize backbone (if available from backbone)
        if hasattr(self.backbone, 'config_params'):
            self.backbone_init_config = self.backbone.config_params
        else:
            # Fallback: reconstruct from phase3_cfg if backbone doesn't store it
            self.backbone_init_config = {
                "img_size": img_size, "patch_size": phase3_cfg['hvt_patch_size'], "num_classes": phase3_cfg['num_classes'],
                "embed_dim_rgb": phase3_cfg['hvt_embed_dim_rgb'], "embed_dim_spectral": phase3_cfg['hvt_embed_dim_spectral'],
                "spectral_channels": phase3_cfg['hvt_spectral_channels'], "depths": phase3_cfg['hvt_depths'],
                "num_heads": phase3_cfg['hvt_num_heads'], "mlp_ratio": phase3_cfg['hvt_mlp_ratio'],
                "qkv_bias": phase3_cfg['hvt_qkv_bias'], "drop_rate": phase3_cfg['hvt_model_drop_rate'],
                "attn_drop_rate": phase3_cfg['hvt_attn_drop_rate'], "drop_path_rate": phase3_cfg['hvt_drop_path_rate'],
                "use_dfca": phase3_cfg['hvt_use_dfca'], "dfca_num_heads": phase3_cfg.get('hvt_dfca_heads', phase3_cfg['hvt_num_heads'][-1]),
                # Add SSL params from phase3_cfg as they are part of HVTBackbone's __init__
                "ssl_enable_mae": phase3_cfg.get('ssl_enable_mae', False),
                "ssl_mae_mask_ratio": phase3_cfg.get('ssl_mae_mask_ratio', 0.75),
                "ssl_mae_decoder_dim": phase3_cfg.get('ssl_mae_decoder_dim', 64),
                "ssl_mae_norm_pix_loss": phase3_cfg.get('ssl_mae_norm_pix_loss', True),
                "ssl_enable_contrastive": phase3_cfg.get('ssl_enable_contrastive', False),
                "ssl_contrastive_projector_dim": phase3_cfg.get('ssl_contrastive_projector_dim', 128),
                "ssl_contrastive_projector_depth": phase3_cfg.get('ssl_contrastive_projector_depth', 2),
                "enable_consistency_loss_heads": phase3_cfg.get('enable_consistency_loss_heads', False),
                "use_gradient_checkpointing": phase3_cfg.get('use_gradient_checkpointing', False)
            }
            logger.info("HVTWrapper: Stored reconstructed backbone config based on phase3_cfg.")


        logger.info("HVTForPretraining wrapper initialized.")

    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None,
                mode: str = 'pretrain' # 'pretrain', 'probe_extract', 'finetune_classify'
               ):
        """
        Forward pass for pre-training wrapper.
        - 'pretrain': Returns projected features for SimCLR.
        - 'probe_extract': Returns backbone features before projection for linear probe.
        - 'finetune_classify': Returns classification logits from backbone's original head (if wrapper is used for direct finetuning).
        """

        # --- Feature Extraction from Backbone ---
        # The HVTBackbone's forward_features_encoded returns (x_rgb_encoded, x_spectral_encoded, ...)
        # For SimCLR, we are interested in x_rgb_encoded.
        # The spectral_img argument to forward_features_encoded can be None.
        x_rgb_encoded, x_spectral_encoded, _, _ = self.backbone.forward_features_encoded(rgb_img, spectral_img if self.backbone.spectral_patch_embed else None)
        # x_rgb_encoded is (B, N_rgb_enc, C_rgb_enc)

        # Global average pool the RGB encoded features
        # This gives (B, C_rgb_enc)
        pooled_rgb_features = x_rgb_encoded.mean(dim=1)

        # Apply the backbone's final norm for the RGB stream if it exists,
        # otherwise, use the pooled features directly. This norm is usually part of the encoder.
        # The HVTBackbone applies norm_rgb_final_encoder *inside* _forward_stream, so x_rgb_encoded is already normed.
        backbone_output_features = pooled_rgb_features # Already normed by norm_rgb_final_encoder


        # --- Output Selection based on mode ---
        if mode == 'probe_extract':
            # For linear probing, we need features *before* the SSL projection head.
            # These are the features that the linear classifier will take as input.
            return backbone_output_features

        elif mode == 'pretrain':
            # For SSL pre-training (SimCLR), pass through the projection head.
            if not hasattr(self, 'projection_head'):
                raise AttributeError("Projection head is not defined, cannot run in 'pretrain' mode.")
            projected_features = self.projection_head(backbone_output_features)
            return projected_features

        elif mode == 'finetune_classify':
            # For fine-tuning or inference using the backbone's original classification head.
            # The HVTBackbone's main forward pass handles fusion and its own head.
            # This mode implies the wrapper is used for downstream task directly.
            # We need to decide if we pass fused features or just RGB features to the backbone's head.
            # For consistency with SimCLR pre-training on RGB, let's assume we'd fine-tune on RGB features passed to the backbone's head.
            # However, the HVT's main head expects potentially fused features after its own head_norm.

            # Option 1: Use HVT's full forward pass (more complex if spectral is involved)
            #   return self.backbone(rgb_img, spectral_img, mode='classify') # This would re-run feature extraction.
            # Option 2: Mimic HVT's head path using already extracted RGB features
            if self.backbone.use_dfca or self.backbone.simple_fusion_proj:
                logger.warning("HVTWrapper 'finetune_classify' mode: Backbone expects fused features for its head, but wrapper currently only processes RGB stream features for this mode. Results may differ from direct backbone usage with fusion.")
            # Use the HVT's head_norm and head on the (already normed) pooled RGB features
            normed_for_head = self.backbone.head_norm(backbone_output_features) # HVT's head_norm
            logits = self.backbone.head(normed_for_head) # HVT's classification head
            return logits
        else:
            raise ValueError(f"Unknown mode for HVTForPretraining: {mode}")