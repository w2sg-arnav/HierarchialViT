# phase4_finetuning/utils/ema.py

import torch
import torch.nn as nn
import logging
import timm
from copy import deepcopy

class EMA:
    """
    Exponential Moving Average of model weights.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay must be between 0 and 1.")

        self.logger = logging.getLogger(__name__)
        self.model = model
        self.decay = decay
        
        self.shadow_model = self._create_shadow_copy()
        
        self.logger.info(f"EMA helper initialized with decay={self.decay}. Shadow model created.")

    def _create_shadow_copy(self) -> nn.Module:
        """
        Creates a deep copy of the model for the shadow weights.
        Re-instantiates the model class if possible, otherwise falls back to deepcopy.
        """
        # Case 1: Custom HVT model
        if hasattr(self.model, 'hvt_params') and hasattr(self.model, 'current_img_size') and hasattr(self.model, 'num_classes'):
            self.logger.info("Creating EMA shadow model by re-instantiating custom HVT class.")
            model_class = type(self.model)
            shadow_model = model_class(
                img_size=self.model.current_img_size,
                num_classes=self.model.num_classes,
                hvt_params=self.model.hvt_params
            )
            shadow_model.load_state_dict(self.model.state_dict())
        
        # Case 2: Standard timm model
        elif hasattr(self.model, 'default_cfg') and 'architecture' in self.model.default_cfg:
            self.logger.info("Creating EMA shadow model by re-instantiating timm model.")
            model_name = self.model.default_cfg['architecture']
            # We assume the `num_classes` attribute exists on the timm model instance.
            num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else self.model.get_classifier().out_features
            shadow_model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            shadow_model.load_state_dict(self.model.state_dict())
            
        # Case 3: Fallback for any other model type
        else:
            self.logger.warning("Model type not recognized as HVT or timm. Falling back to deepcopy for EMA shadow model. This might be less robust.")
            shadow_model = deepcopy(self.model)
        
        # Ensure the shadow model is on the correct device and in eval mode
        device = next(self.model.parameters()).device
        shadow_model.to(device)
        shadow_model.eval()
        
        for p in shadow_model.parameters():
            p.requires_grad = False
            
        return shadow_model
    
    @torch.no_grad()
    def update(self):
        """
        Update shadow model weights with the live model's weights.
        """
        if not self.model.training:
            self.logger.warning("EMA update called when model is in eval mode.")
        
        model_params = self.model.parameters()
        shadow_params = self.shadow_model.parameters()
        
        for model_p, shadow_p in zip(model_params, shadow_params):
            if shadow_p.is_floating_point():
                shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            else:
                shadow_p.data.copy_(model_p.data)

    def state_dict(self) -> dict:
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        try:
            self.shadow_model.load_state_dict(state_dict)
            self.logger.info("EMA state (shadow model weights) loaded successfully from checkpoint.")
        except Exception as e:
            self.logger.error(f"Failed to load EMA state_dict into shadow model: {e}", exc_info=True)