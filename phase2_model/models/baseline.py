# models/baseline.py
import torch
from typing import Optional, Union, Tuple
import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, SPECTRAL_CHANNELS # Assuming these are defined in your main config
import logging

logger = logging.getLogger(__name__)

class InceptionV3Baseline(nn.Module):
    """
    Inception V3 baseline for comparison, adapted for multi-modal input (RGB + Spectral).
    It processes RGB and Spectral streams with separate initial convolutions,
    then sums their features before feeding into the main Inception V3 body.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, 
                 spectral_channels: int = SPECTRAL_CHANNELS,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.spectral_channels = spectral_channels

        if pretrained:
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            logger.info("Loading Inception V3 with IMAGENET1K_V1 pretrained weights.")
        else:
            weights = None
            logger.info("Initializing Inception V3 from scratch (no pretrained weights).")

        # Load Inception V3
        self.inception_model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        # InceptionV3 expects input of 299x299. If using other sizes, ensure transform_input=False
        # and potentially adapt first layer or resize input.
        # For now, we assume input images will be resized to 299x299 by transforms if using this model.

        # Adapt the first convolutional layer for multi-modal input
        original_conv1 = self.inception_model.Conv2d_1a_3x3.conv
        
        # RGB stream conv: keep original weights for 3 input channels
        self.rgb_conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=original_conv1.out_channels,
                                   kernel_size=original_conv1.kernel_size,
                                   stride=original_conv1.stride,
                                   padding=original_conv1.padding,
                                   bias=original_conv1.bias is not None)

        # Spectral stream conv
        self.spectral_conv1 = nn.Conv2d(in_channels=self.spectral_channels,
                                        out_channels=original_conv1.out_channels,
                                        kernel_size=original_conv1.kernel_size,
                                        stride=original_conv1.stride,
                                        padding=original_conv1.padding,
                                        bias=original_conv1.bias is not None)

        if pretrained:
            with torch.no_grad():
                # Copy weights for RGB conv
                self.rgb_conv1.weight.copy_(original_conv1.weight)
                if original_conv1.bias is not None:
                    self.rgb_conv1.bias.copy_(original_conv1.bias)

                # Initialize spectral conv weights (e.g., by averaging RGB weights across input channels)
                # This is a common heuristic.
                mean_rgb_weights = original_conv1.weight.data.mean(dim=1, keepdim=True)
                self.spectral_conv1.weight.data.copy_(mean_rgb_weights.repeat(1, self.spectral_channels, 1, 1))
                if original_conv1.bias is not None:
                    self.spectral_conv1.bias.data.copy_(original_conv1.bias)
        
        # Replace the original first conv layer with a pass-through, as we handle it outside
        self.inception_model.Conv2d_1a_3x3.conv = nn.Identity() 

        # Modify the final fully connected layer for the new number of classes
        in_features_fc = self.inception_model.fc.in_features
        self.inception_model.fc = nn.Linear(in_features_fc, self.num_classes)

        # Modify the auxiliary classifier's FC layer as well, if it exists
        if hasattr(self.inception_model, 'AuxLogits') and self.inception_model.AuxLogits is not None:
            in_features_aux = self.inception_model.AuxLogits.fc.in_features
            self.inception_model.AuxLogits.fc = nn.Linear(in_features_aux, self.num_classes)
            logger.info(f"Adapted InceptionV3 AuxLogits FC layer for {self.num_classes} classes.")

        logger.info(f"Adapted InceptionV3 main FC layer for {self.num_classes} classes.")
        logger.info(f"Input RGB conv: {self.rgb_conv1}")
        logger.info(f"Input Spectral conv: {self.spectral_conv1}")


    def forward(self, rgb_input: torch.Tensor, spectral_input: Optional[torch.Tensor] = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the InceptionV3 baseline.

        Args:
            rgb_input (torch.Tensor): RGB image tensor [batch, 3, H, W].
                                      Expected H, W to be 299x299 for InceptionV3.
            spectral_input (Optional[torch.Tensor]): Spectral image tensor 
                                                     [batch, spectral_channels, H, W].
                                                     If None, only RGB stream is used.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                Logits [batch, num_classes]. If training and aux_logits are enabled,
                returns a tuple (main_output, aux_output). Otherwise, just main_output.
        """
        # Debug log for input shapes (use sparingly in production)
        # logger.debug(f"InceptionV3 - RGB input shape: {rgb_input.shape}")
        # if spectral_input is not None:
        #     logger.debug(f"InceptionV3 - Spectral input shape: {spectral_input.shape}")

        # Process RGB stream
        x_rgb = self.rgb_conv1(rgb_input) # Output shape: [batch, 32, H/2, W/2] (approx)

        if spectral_input is not None:
            x_spectral = self.spectral_conv1(spectral_input)
            # Element-wise sum for early fusion
            x = x_rgb + x_spectral 
        else:
            # If no spectral input, proceed with RGB features only
            # Potentially scale rgb_features if spectral usually contributes significantly
            x = x_rgb 

        # Pass the combined (or RGB-only) features through the rest of the InceptionV3 model
        # The model.Conv2d_1a_3x3.conv is now an Identity, so 'x' is directly fed to its BN and ReLU
        # then to Conv2d_2a_3x3, etc.
        
        # The InceptionV3 model handles aux_logits internally based on self.training
        if self.training and self.inception_model.aux_logits:
            main_output, aux_output = self.inception_model(x)
            return main_output, aux_output
        else:
            main_output = self.inception_model(x)
            return main_output

# Need to import Optional and Union for type hints
from typing import Optional, Union, Tuple