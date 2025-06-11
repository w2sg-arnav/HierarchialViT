# phase4_finetuning/utils/ema.py

import torch
import torch.nn as nn
import logging
from copy import deepcopy

class EMA:
    """
    Exponential Moving Average of model weights.

    This class maintains a full 'shadow' model whose weights are a moving average
    of the live model's weights. Using the shadow model for validation can lead
    to more stable and better-performing results.

    Usage:
        ema = EMA(model, decay=0.999)
        ...
        # After each optimizer step in the training loop:
        ema.update()
        ...
        # For validation:
        model_for_validation = ema.shadow_model
        model_for_validation.eval()
        # ... perform validation ...
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
        """
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay must be between 0 and 1.")

        self.logger = logging.getLogger(__name__)
        self.model = model
        self.decay = decay
        
        # Create a shadow model that is a completely separate instance
        self.shadow_model = self._create_shadow_copy()
        
        self.logger.info(f"EMA helper initialized with decay={self.decay}. Shadow model created.")

    def _create_shadow_copy(self) -> nn.Module:
        """
        Creates a deep copy of the model for the shadow weights.
        The most robust method is to re-instantiate the model class if its
        configuration is known, as is the case with our HVT model.
        """
        # This robustly re-creates the HVT model using its own stored configuration
        # which it gets during its own __init__ call.
        if hasattr(self.model, 'hvt_params') and hasattr(self.model, 'current_img_size') and hasattr(self.model, 'num_classes'):
            self.logger.info("Creating EMA shadow model by re-instantiating HVT class.")
            model_class = type(self.model)
            # The HVT model factory from Phase 2 could also be used here if imported
            shadow_model = model_class(
                img_size=self.model.current_img_size,
                num_classes=self.model.num_classes,
                hvt_params=self.model.hvt_params
            )
            shadow_model.load_state_dict(self.model.state_dict())
        else:
            # A general fallback for other model types
            self.logger.warning("Model does not have HVT parameters. Falling back to deepcopy for EMA shadow model. This might be less robust.")
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
        This should be called after each optimizer step.
        """
        if not self.model.training:
            self.logger.warning("EMA update called when model is in eval mode.")
        
        # Iterate over parameters of both models
        model_params = self.model.parameters()
        shadow_params = self.shadow_model.parameters()
        
        for model_p, shadow_p in zip(model_params, shadow_params):
            # Update shadow parameter in-place: shadow_p = decay * shadow_p + (1 - decay) * model_p
            if shadow_p.is_floating_point():
                shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            else:
                shadow_p.data.copy_(model_p.data)

    def state_dict(self) -> dict:
        """
        Returns the state of the EMA helper, which is the state of the shadow model.
        """
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        """
        Loads the shadow model's state from a checkpoint.
        """
        try:
            self.shadow_model.load_state_dict(state_dict)
            self.logger.info("EMA state (shadow model weights) loaded successfully from checkpoint.")
        except Exception as e:
            self.logger.error(f"Failed to load EMA state_dict into shadow model: {e}", exc_info=True)