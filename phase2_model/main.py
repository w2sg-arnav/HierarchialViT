# main.py (for model testing)
import torch
import logging
from typing import Tuple, Optional # Ensure Optional is imported

# Assuming config.py is in the parent directory or accessible in PYTHONPATH
# If config.py is in the same directory as this main.py (e.g. phase2_model/config.py)
from config import PROGRESSIVE_RESOLUTIONS, NUM_CLASSES, SPECTRAL_CHANNELS, IMAGE_SIZE # Added IMAGE_SIZE for Inception

# Assuming models are in a 'models' subdirectory relative to this main.py
from models.hvt import DiseaseAwareHVT, create_disease_aware_hvt_from_config
from models.baseline import InceptionV3Baseline

logger = logging.getLogger(__name__)


def test_model_forward(model: torch.nn.Module, 
                       img_size: Tuple[int, int], 
                       batch_size: int = 2, # Smaller batch for faster test
                       spectral_channels_test: int = SPECTRAL_CHANNELS,
                       use_spectral_input: bool = True):
    """Test the model's forward pass with dummy inputs."""
    logger.info(f"--- Testing Model: {model.__class__.__name__} with img_size: {img_size} ---")
    
    rgb_dummy = torch.randn(batch_size, 3, img_size[0], img_size[1])
    
    if use_spectral_input:
        spectral_dummy = torch.randn(batch_size, spectral_channels_test, img_size[0], img_size[1])
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral shape {spectral_dummy.shape}")
    else:
        spectral_dummy = None
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral: None")

    model.eval() # Set model to evaluation mode
    try:
        with torch.no_grad(): # No gradient calculation needed for forward pass test
            if isinstance(model, InceptionV3Baseline):
                # InceptionV3 might return tuple if self.training is True and aux_logits active
                # but model.eval() should make it return single tensor
                output = model(rgb_dummy, spectral_dummy)
                if isinstance(output, tuple): # Should not happen in eval mode usually
                    logits = output[0]
                else:
                    logits = output
            elif isinstance(model, DiseaseAwareHVT):
                logits = model(rgb_dummy, spectral_dummy)
            else:
                logger.error(f"Unsupported model type for testing: {model.__class__.__name__}")
                return

        logger.info(f"Output logits shape: {logits.shape}")
        expected_shape = (batch_size, NUM_CLASSES)
        assert logits.shape == expected_shape, \
            f"Output shape mismatch! Expected {expected_shape}, got {logits.shape}"
        logger.info(f"Model {model.__class__.__name__} forward pass successful for img_size {img_size}.")

    except Exception as e:
        logger.error(f"Error during forward pass for {model.__class__.__name__} with img_size {img_size}: {e}", exc_info=True)
        raise # Re-raise the exception to fail the test explicitly

def run_hvt_tests():
    logger.info("--- Starting DiseaseAwareHVT Tests ---")
    for res_idx, img_size_tuple in enumerate(PROGRESSIVE_RESOLUTIONS):
        logger.info(f"Testing HVT with resolution: {img_size_tuple} ({res_idx+1}/{len(PROGRESSIVE_RESOLUTIONS)})")
        # Create a new model instance for each resolution to ensure correct PatchEmbed init
        hvt_model = create_disease_aware_hvt_from_config(img_size_tuple)
        
        # Test with spectral input
        test_model_forward(hvt_model, img_size=img_size_tuple, use_spectral_input=True)
        # Test without spectral input (ablation)
        test_model_forward(hvt_model, img_size=img_size_tuple, use_spectral_input=False)

def run_baseline_tests():
    logger.info("--- Starting InceptionV3Baseline Tests ---")
    # InceptionV3 expects 299x299. We test with a configured IMAGE_SIZE, assuming transforms handle resizing.
    # Or, for testing, we can pass 299x299 directly.
    # Let's use the largest progressive resolution as a stand-in, but note Inception's true requirement.
    
    # Test with a resolution from config (e.g., largest progressive one for consistency)
    # Note: For actual training with InceptionV3, images should be resized to 299x299.
    # test_img_size_inception = PROGRESSIVE_RESOLUTIONS[-1] 
    test_img_size_inception = (299,299) # More canonical for InceptionV3
    logger.info(f"Testing InceptionV3 with resolution: {test_img_size_inception} (Note: InceptionV3 prefers 299x299)")

    inception_model_pretrained = InceptionV3Baseline(num_classes=NUM_CLASSES, spectral_channels=SPECTRAL_CHANNELS, pretrained=True)
    # Test with spectral input
    test_model_forward(inception_model_pretrained, img_size=test_img_size_inception, use_spectral_input=True)
    # Test without spectral input
    test_model_forward(inception_model_pretrained, img_size=test_img_size_inception, use_spectral_input=False)

    inception_model_scratch = InceptionV3Baseline(num_classes=NUM_CLASSES, spectral_channels=SPECTRAL_CHANNELS, pretrained=False)
    test_model_forward(inception_model_scratch, img_size=test_img_size_inception, use_spectral_input=True)


def main_model_tests():
    # Setup basic logging for the main script execution
    # This might conflict if config.py also does basicConfig.
    # It's safer if config.py gets a logger instance rather than basicConfig.
    # For now, this is okay for a simple script.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("======== Running Model Sanity Checks ========")
    run_hvt_tests()
    run_baseline_tests()
    logger.info("======== Model Sanity Checks Finished ========")

if __name__ == "__main__":
    # The test_concatenation function was specific to an older InceptionV3 design.
    # The current InceptionV3 handles optional spectral input differently.
    # You can add more specific unit tests for model components if needed.
    # For now, focusing on end-to-end forward pass tests.
    main_model_tests()