# phase4_finetuning/utils/ema.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EMA:
    """
    Exponential Moving Average of model weights.
    Maintains a "shadow" copy of the model's weights, which are updated
    slowly over time. This can lead to better generalization and more
    stable performance during validation.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.register()
        logger.info(f"EMA initialized with decay={decay}.")

    def register(self):
        """ Creates the shadow copy of the weights. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """ Updates the shadow weights with the current model weights. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """ Swaps the model's current weights with the shadow weights for inference. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """ Restores the original model weights after inference. """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict.get('decay', self.decay)
        self.shadow = state_dict.get('shadow', {})
        self.logger.info("EMA state loaded from checkpoint.")