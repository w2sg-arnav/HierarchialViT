# hvit/utils/ema.py
"""Exponential Moving Average (EMA) for model weights.

This module provides an EMA helper class that maintains a shadow copy
of model weights for improved generalization.
"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)

# Conditionally import timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class EMA:
    """Exponential Moving Average of model weights.
    
    Maintains a shadow model that tracks the exponential moving average
    of model parameters. This typically improves generalization and is
    used during validation/inference.
    
    Args:
        model: The model to track.
        decay: EMA decay factor (higher = slower updates).
    
    Example:
        >>> ema = EMA(model, decay=0.9999)
        >>> # In training loop:
        >>> optimizer.step()
        >>> ema.update()
        >>> # For validation:
        >>> predictions = ema.shadow_model(inputs)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay must be between 0 and 1.")

        self.model = model
        self.decay = decay
        
        # Create shadow model before any potential torch.compile
        self.shadow_model = self._create_shadow_copy()
        
        logger.info(f"EMA initialized with decay={self.decay}")

    def _create_shadow_copy(self) -> nn.Module:
        """Create a deep copy of the model for shadow weights.
        
        Intelligently handles custom HVT models, timm models, and generic models.
        """
        # Case 1: Custom HVT model
        if (hasattr(self.model, 'hvt_params') and 
            hasattr(self.model, 'current_img_size') and 
            hasattr(self.model, 'num_classes')):
            logger.info("Creating EMA shadow model by re-instantiating custom HVT class.")
            
            # Handle compiled models
            model_class = (
                type(self.model._orig_mod) if hasattr(self.model, '_orig_mod') 
                else type(self.model)
            )
            
            shadow_model = model_class(
                img_size=self.model.current_img_size,
                num_classes=self.model.num_classes,
                hvt_params=self.model.hvt_params
            )
            shadow_model.load_state_dict(self.model.state_dict())

        # Case 2: timm model (ViT-compatible)
        elif TIMM_AVAILABLE and hasattr(self.model, 'default_cfg') and 'architecture' in self.model.default_cfg:
            logger.info("Creating EMA shadow model by re-instantiating timm model.")
            model_name = self.model.default_cfg['architecture']
            
            num_classes = (
                self.model.num_classes if hasattr(self.model, 'num_classes')
                else self.model.get_classifier().out_features
            )
            
            timm_kwargs = {'pretrained': False, 'num_classes': num_classes}
            
            # Handle ViT positional embeddings correctly
            if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'img_size'):
                live_img_size = self.model.patch_embed.img_size
                if isinstance(live_img_size, (tuple, list)):
                    logger.info(f"Using img_size={live_img_size} for shadow model.")
                    timm_kwargs['img_size'] = live_img_size
            
            shadow_model = timm.create_model(model_name, **timm_kwargs)
            shadow_model.load_state_dict(self.model.state_dict())

        # Case 3: Fallback to deepcopy
        else:
            logger.warning(
                "Model type not recognized. Falling back to deepcopy for EMA shadow model."
            )
            shadow_model = deepcopy(self.model)

        # Setup shadow model
        device = next(self.model.parameters()).device
        shadow_model.to(device)
        shadow_model.eval()
        
        for p in shadow_model.parameters():
            p.requires_grad = False
            
        return shadow_model

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow model with EMA of current model weights.
        
        Should be called after each optimizer step during training.
        """
        if not self.model.training:
            logger.warning("EMA.update() called in eval mode. This is unusual.")

        # Handle compiled models
        model_to_use = (
            self.model._orig_mod if hasattr(self.model, '_orig_mod') 
            else self.model
        )

        for model_p, shadow_p in zip(model_to_use.parameters(), self.shadow_model.parameters()):
            if shadow_p.is_floating_point():
                shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            else:
                shadow_p.data.copy_(model_p.data)

    def state_dict(self) -> dict:
        """Get state dictionary of shadow model."""
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dictionary into shadow model."""
        try:
            self.shadow_model.load_state_dict(state_dict)
            logger.info("EMA state loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load EMA state: {e}", exc_info=True)


__all__ = ["EMA"]
