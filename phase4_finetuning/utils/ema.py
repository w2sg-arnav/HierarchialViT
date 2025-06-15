# phase4_finetuning/utils/ema.py

import torch
import torch.nn as nn
import logging
import timm
from copy import deepcopy

class EMA:
    """
    Exponential Moving Average of model weights.
    A robust implementation that handles custom models and timm models,
    including Vision Transformers with variable image sizes.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay must be between 0 and 1.")

        self.logger = logging.getLogger(__name__)
        self.model = model
        self.decay = decay
        
        # Create the shadow model before any potential model compilation (e.g., torch.compile)
        self.shadow_model = self._create_shadow_copy()
        
        self.logger.info(f"EMA helper initialized with decay={self.decay}. Shadow model created.")

    def _create_shadow_copy(self) -> nn.Module:
        """
        Creates a deep copy of the model for the shadow weights.
        It intelligently re-instantiates the model class to ensure compatibility,
        especially for models sensitive to input size like Vision Transformers.
        """
        # Case 1: Our custom HVT model, which has a specific constructor signature.
        if hasattr(self.model, 'hvt_params') and hasattr(self.model, 'current_img_size') and hasattr(self.model, 'num_classes'):
            self.logger.info("Creating EMA shadow model by re-instantiating custom HVT class.")
            # Ensure we are using the un-compiled model's class if torch.compile was used
            model_class = type(self.model._orig_mod) if hasattr(self.model, '_orig_mod') else type(self.model)
            
            shadow_model = model_class(
                img_size=self.model.current_img_size,
                num_classes=self.model.num_classes,
                hvt_params=self.model.hvt_params
            )
            shadow_model.load_state_dict(self.model.state_dict())
        
        # Case 2: Standard timm model, which might be a ViT.
        elif hasattr(self.model, 'default_cfg') and 'architecture' in self.model.default_cfg:
            self.logger.info("Creating EMA shadow model by re-instantiating timm model.")
            model_name = self.model.default_cfg['architecture']
            num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else self.model.get_classifier().out_features
            
            # --- START OF CRITICAL FIX for ViT Positional Embeddings ---
            # For ViT-like models, we MUST pass the correct img_size during creation
            # to ensure the positional embeddings are initialized with the correct shape,
            # preventing the "size mismatch" error.
            timm_kwargs = {'pretrained': False, 'num_classes': num_classes}
            
            # Check if the live model has patch embedding information, typical for ViTs.
            if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'img_size'):
                # Get the actual image size the live model was configured for.
                live_img_size = self.model.patch_embed.img_size
                if isinstance(live_img_size, (tuple, list)):
                    self.logger.info(f"Explicitly passing img_size={live_img_size} to timm.create_model for shadow copy.")
                    timm_kwargs['img_size'] = live_img_size
            # --- END OF CRITICAL FIX ---
            
            shadow_model = timm.create_model(model_name, **timm_kwargs)
            shadow_model.load_state_dict(self.model.state_dict())
            
        # Case 3: Fallback for any other model type.
        else:
            self.logger.warning("Model type not recognized as HVT or timm. Falling back to deepcopy for EMA shadow model. This might be less robust.")
            shadow_model = deepcopy(self.model)
        
        # Ensure the shadow model is on the correct device and in evaluation mode.
        device = next(self.model.parameters()).device
        shadow_model.to(device)
        shadow_model.eval()
        
        for p in shadow_model.parameters():
            p.requires_grad = False
            
        return shadow_model
    
    @torch.no_grad()
    def update(self):
        """
        Update shadow model weights with the live model's weights using exponential decay.
        This should be called after each optimizer step.
        """
        if not self.model.training:
            self.logger.warning("EMA.update() called when model is in eval mode. This is unusual and may not be intended.")
        
        # If the model is compiled, we need to access the parameters of the original model.
        model_to_get_params_from = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        
        model_params = model_to_get_params_from.parameters()
        shadow_params = self.shadow_model.parameters()
        
        for model_p, shadow_p in zip(model_params, shadow_params):
            if shadow_p.is_floating_point():
                shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            else:
                shadow_p.data.copy_(model_p.data)

    def state_dict(self) -> dict:
        """ Returns the state dictionary of the shadow model. """
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        """ Loads a state dictionary into the shadow model. """
        try:
            self.shadow_model.load_state_dict(state_dict)
            self.logger.info("EMA state (shadow model weights) loaded successfully from checkpoint.")
        except Exception as e:
            self.logger.error(f"Failed to load EMA state_dict into shadow model: {e}", exc_info=True)