# phase3_pretraining/models/hvt_wrapper.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import logging

# Import the canonical HVT model and its factory from Phase 2
try:
    # Assuming phase2_model is in PYTHONPATH or accessible
    # The factory function name was 'create_disease_aware_hvt'
    from phase2_model.models.hvt import DiseaseAwareHVT as HVTBackbone
    from phase2_model.models.hvt import create_disease_aware_hvt as hvt_factory_from_phase2
except ImportError as e:
    # Fallback for environments where phase2_model might not be directly importable this way
    # This usually indicates a PYTHONPATH issue.
    print(f"CRITICAL IMPORT ERROR (hvt_wrapper.py): Cannot import HVTBackbone/factory from phase2_model.models.hvt. Error: {e}")
    print("Ensure 'phase2_model' is in your PYTHONPATH or the project structure allows this import.")
    # Define dummy classes to allow script to load for debugging import paths, but it won't work.
    class HVTBackbone(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.final_encoded_dim_rgb = 768; self.spectral_patch_embed=None
        def forward_features_encoded(self, rgb, spectral=None): return torch.randn(rgb.shape[0],10,self.final_encoded_dim_rgb), None, (10,10),None
        def head_norm(self,x):return x;
        def head(self,x):return x;
        def __getattr__(self, name): return lambda *args, **kwargs: None # Mock methods
    def hvt_factory_from_phase2(*args, **kwargs): return HVTBackbone()
# --- End Fallback ---

from .projection_head import ProjectionHead # Relative import for projection_head
from ..config import config as phase3_run_config # Relative import for this run's config

logger = logging.getLogger(__name__)

class HVTForPretraining(nn.Module):
    def __init__(self, img_size: Tuple[int, int], num_classes_for_probe: int):
        super().__init__()
        logger.info(f"Initializing HVTForPretraining wrapper for img_size: {img_size}")

        # Get HVT backbone parameters from the *Phase 3 run configuration*
        hvt_backbone_params_from_phase3_cfg = phase3_run_config.get('hvt_params_for_backbone')
        if not isinstance(hvt_backbone_params_from_phase3_cfg, dict):
            logger.error("HVT backbone parameters ('hvt_params_for_backbone') not found or not a dict in Phase 3 config.")
            raise ValueError("Missing or invalid HVT backbone parameters in Phase 3 config.")

        logger.info(f"Instantiating HVTBackbone using parameters defined in Phase 3 config (hvt_params_for_backbone).")
        # The Phase 2 HVT factory takes current_img_size, num_classes (for its own head, can be dummy), and model_params_dict
        # For pre-training, num_classes for the backbone's internal head is less critical if we don't use it.
        # However, the HVT code might expect it. Let's pass the probe's num_classes.
        self.backbone: HVTBackbone = hvt_factory_from_phase2(
            current_img_size=img_size,
            num_classes=num_classes_for_probe, # Or a dummy value if HVT's head isn't used
            model_params_dict=hvt_backbone_params_from_phase3_cfg
        )
        # Store the actual config used to initialize the backbone (for saving checkpoints)
        # The backbone itself should store its init_params in self.hvt_params
        self.backbone_init_config = self.backbone.hvt_params if hasattr(self.backbone, 'hvt_params') else hvt_backbone_params_from_phase3_cfg


        # Determine projection head input dimension
        # This should come from the features *before* the backbone's original classification head.
        # The HVTBackbone's `final_encoded_dim_rgb` attribute should give this.
        if not hasattr(self.backbone, 'final_encoded_dim_rgb') or self.backbone.final_encoded_dim_rgb <= 0:
            logger.error("HVT backbone does not have 'final_encoded_dim_rgb' attribute or it's invalid. Cannot determine projection head input size.")
            # Fallback attempt (less reliable)
            _fallback_dim = hvt_backbone_params_from_phase3_cfg.get('embed_dim_rgb', 96) * \
                            (2**(len(hvt_backbone_params_from_phase3_cfg.get('depths', [0,0,0,0])) - 1))
            logger.warning(f"Using fallback projection input dim: {_fallback_dim}. This may be incorrect.")
            projection_in_dim = _fallback_dim
        else:
            projection_in_dim = self.backbone.final_encoded_dim_rgb
        logger.info(f"Projection head input dimension set to: {projection_in_dim}")


        self.projection_head = ProjectionHead(
            in_dim=projection_in_dim,
            hidden_dim=phase3_run_config['projection_hidden_dim'],
            out_dim=phase3_run_config['projection_dim'],
            use_batch_norm=True # Standard for SimCLR projection heads
        )
        logger.info("HVTForPretraining wrapper initialized successfully.")


    def forward(self, rgb_img: torch.Tensor, spectral_img: Optional[torch.Tensor] = None, mode: str = 'pretrain'):
        # For SimCLR pre-training, we primarily use the RGB stream.
        # Ensure spectral_img is only passed if the backbone is configured for it.
        use_spectral_in_backbone = self.backbone.hvt_params.get('spectral_channels', 0) > 0 and \
                                   self.backbone.spectral_patch_embed is not None

        # --- Feature Extraction from Backbone ---
        x_rgb_encoded, _, _, _ = self.backbone.forward_features_encoded(
            rgb_img,
            spectral_img if use_spectral_in_backbone and spectral_img is not None else None
        ) # Returns: x_rgb_encoded, x_spectral_encoded, rgb_orig_patch_grid, spectral_orig_patch_grid
          # We only need x_rgb_encoded for RGB-based SimCLR.

        # Global average pool the RGB encoded features [B, N_tokens, C_encoded] -> [B, C_encoded]
        pooled_rgb_features = x_rgb_encoded.mean(dim=1)
        # The HVT's _forward_stream already applies final_norm_layer, so features are normed.
        backbone_output_features = pooled_rgb_features


        if mode == 'probe_extract':
            return backbone_output_features # Features for linear probe
        elif mode == 'pretrain':
            return self.projection_head(backbone_output_features) # Projected features for SimCLR loss
        elif mode == 'finetune_classify':
            # Use the HVT backbone's original classification mechanism.
            # This re-runs the full forward with mode='classify' on the backbone.
            # Note: This assumes that if spectral_img is provided, it should be used for classification.
            return self.backbone(rgb_img, spectral_img if use_spectral_in_backbone else None, mode='classify')
        else:
            raise ValueError(f"Unknown mode for HVTForPretraining: {mode}")