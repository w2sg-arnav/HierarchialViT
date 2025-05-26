# phase2_model/models/baseline.py
import torch
import torch.nn as nn
import torchvision.models as tv_models # Renamed to avoid conflict with local 'models'
from typing import Optional, Union, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes: int, model_params: Dict[str, Any], pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        # Get spectral_channels from model_params, default to 0 if not present
        self.spectral_channels = model_params.get('spectral_channels', 0)
        self.img_input_size = model_params.get('img_size', (299,299)) # InceptionV3 expects 299x299

        if pretrained:
            try: # Try modern weights API
                weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1
                logger.info("Loading Inception V3 with IMAGENET1K_V1 (modern API).")
            except AttributeError: # Fallback for older torchvision
                weights = 'IMAGENET1K_V1' # String for older pretrained=True mechanism
                logger.info("Loading Inception V3 with IMAGENET1K_V1 (older API string).")
        else:
            weights = None
            logger.info("Initializing Inception V3 from scratch.")

        self.inception_model = tv_models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        # transform_input=False is important if you handle normalization/resizing outside

        original_conv1 = self.inception_model.Conv2d_1a_3x3.conv
        self.rgb_conv1 = nn.Conv2d(3, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                                   stride=original_conv1.stride, padding=original_conv1.padding,
                                   bias=(original_conv1.bias is not None))
        if self.spectral_channels > 0:
            self.spectral_conv1 = nn.Conv2d(self.spectral_channels, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size, stride=original_conv1.stride,
                                            padding=original_conv1.padding, bias=(original_conv1.bias is not None))
        else:
            self.spectral_conv1 = None

        if pretrained:
            with torch.no_grad():
                self.rgb_conv1.weight.copy_(original_conv1.weight)
                if original_conv1.bias is not None: self.rgb_conv1.bias.copy_(original_conv1.bias)
                if self.spectral_conv1:
                    mean_rgb_weights = original_conv1.weight.data.mean(dim=1, keepdim=True)
                    self.spectral_conv1.weight.data.copy_(mean_rgb_weights.repeat(1, self.spectral_channels, 1, 1))
                    if original_conv1.bias is not None: self.spectral_conv1.bias.data.copy_(original_conv1.bias)
        
        self.inception_model.Conv2d_1a_3x3.conv = nn.Identity()
        self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, self.num_classes)
        if hasattr(self.inception_model, 'AuxLogits') and self.inception_model.AuxLogits is not None:
            self.inception_model.AuxLogits.fc = nn.Linear(self.inception_model.AuxLogits.fc.in_features, self.num_classes)
        logger.info(f"InceptionV3Baseline: Adapted for {self.num_classes} classes, spectral_channels={self.spectral_channels}.")

    def forward(self, rgb_input: torch.Tensor, spectral_input: Optional[torch.Tensor] = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        # InceptionV3 expects 299x299. If input is different, it might error or give poor results.
        # Consider adding a resize check or operation here if not handled by transforms.
        if rgb_input.shape[2:] != self.img_input_size:
             logger.warning(f"InceptionV3 input RGB size {rgb_input.shape[2:]} != expected {self.img_input_size}. Performance may suffer.")
        
        x_rgb = self.rgb_conv1(rgb_input)
        if spectral_input is not None and self.spectral_conv1 is not None:
            if spectral_input.shape[2:] != self.img_input_size:
                 logger.warning(f"InceptionV3 input Spectral size {spectral_input.shape[2:]} != expected {self.img_input_size}.")
            x_spectral = self.spectral_conv1(spectral_input)
            x = x_rgb + x_spectral
        else:
            x = x_rgb
            
        if self.training and self.inception_model.aux_logits:
            return self.inception_model(x) # Returns (main_output, aux_output)
        else:
            return self.inception_model(x) # Returns main_output