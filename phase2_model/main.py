# phase2_model/main.py
import torch
import logging # Keep logging import at the top
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union

# --- Path Setup (Good practice, though less critical if main.py only uses local modules) ---
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../phase2_model
project_root_from_phase2_main = os.path.dirname(current_script_dir) # .../cvpr25

if project_root_from_phase2_main not in sys.path:
    sys.path.insert(0, project_root_from_phase2_main)
    # Use print for this early debug message as logger might not be set up
    print(f"DEBUG (phase2_model/main.py): Added project root to sys.path: {project_root_from_phase2_main}")
# --- End Path Setup ---

# --- Configuration Import from local phase2_model/config.py ---
# This script will exclusively use constants defined in its own 'config.py'.
# The names imported here MUST match the top-level constant names in 'phase2_model/config.py'.
try:
    from config import (
        PROGRESSIVE_RESOLUTIONS,
        NUM_CLASSES,
        HVT_SPECTRAL_CHANNELS,  # Import the exact name defined in phase2_model/config.py
        IMAGE_SIZE,
        PATCH_SIZE,
        SSL_ENABLE_MAE,
        SSL_ENABLE_CONTRASTIVE,
        ENABLE_CONSISTENCY_LOSS_HEADS,
        SSL_CONTRASTIVE_PROJECTOR_DIM
        # Add any other constants from phase2_model/config.py that this main.py needs
    )
    # Alias HVT_SPECTRAL_CHANNELS for local use if the rest of the script uses the shorter name
    SPECTRAL_CHANNELS = HVT_SPECTRAL_CHANNELS

    # Initialize logger for this file AFTER successful config import (as config.py sets up basicConfig)
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported constants from local phase2_model/config.py")

except ImportError as e:
    # Fallback logging if config.py (which sets up logging) fails to import
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__) # Define logger here for the error message
    logger.error(f"CRITICAL ERROR in phase2_model/main.py: Failed to import required constants from 'phase2_model/config.py'.")
    logger.error(f"Import Error: {e}")
    logger.error("Please ensure 'phase2_model/config.py' exists and defines all necessary top-level constants such as: \n"
                 "PROGRESSIVE_RESOLUTIONS, NUM_CLASSES, HVT_SPECTRAL_CHANNELS, IMAGE_SIZE, PATCH_SIZE, \n"
                 "SSL_ENABLE_MAE, SSL_ENABLE_CONTRASTIVE, ENABLE_CONSISTENCY_LOSS_HEADS, SSL_CONTRASTIVE_PROJECTOR_DIM.")
    logger.error("Exiting due to missing or incorrect configuration in 'phase2_model/config.py'.")
    sys.exit(1) # Exit if essential config is missing
# --- End Configuration Import ---


# --- Model Imports (from local 'models' subdirectory) ---
try:
    from models.hvt import DiseaseAwareHVT, create_disease_aware_hvt_from_config
except ImportError as e:
    logger.error(f"Failed to import HVT models from 'phase2_model/models/hvt.py': {e}")
    logger.error("Ensure 'phase2_model/models/hvt.py' exists and is free of import errors itself.")
    sys.exit(1)
# from models.baseline import InceptionV3Baseline # Uncomment if baseline tests are needed


# --- Main Test Functions ---
def test_hvt_model(model: DiseaseAwareHVT,
                   img_size: Tuple[int, int],
                   batch_size: int = 2,
                   spectral_channels_test: int = SPECTRAL_CHANNELS, # Uses the (potentially aliased) SPECTRAL_CHANNELS
                   use_spectral_input: bool = True):
    """Test the DiseaseAwareHVT model's forward pass in various modes."""
    logger.info(f"--- Testing HVT Model: {model.__class__.__name__} with img_size: {img_size} ---")

    rgb_dummy = torch.randn(batch_size, 3, img_size[0], img_size[1])
    spectral_dummy = None
    if use_spectral_input and spectral_channels_test > 0:
        spectral_dummy = torch.randn(batch_size, spectral_channels_test, img_size[0], img_size[1])
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral shape {spectral_dummy.shape}")
    else:
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral: None (channels={spectral_channels_test}, use_spectral={use_spectral_input})")

    model.eval()

    # Test 'classify' mode
    try:
        with torch.no_grad():
            output = model(rgb_dummy, spectral_dummy, mode='classify')
            if hasattr(model, 'enable_consistency_loss_heads') and model.enable_consistency_loss_heads:
                main_logits, aux_outputs = output
                assert isinstance(aux_outputs, dict), "Aux outputs should be a dict"
                logger.info(f"Classify mode: Main logits shape {main_logits.shape}, Aux keys: {list(aux_outputs.keys())}")
                if 'logits_rgb' in aux_outputs: assert aux_outputs['logits_rgb'].shape == (batch_size, NUM_CLASSES)
                if spectral_dummy is not None and model.spectral_patch_embed and \
                   'logits_spectral' in aux_outputs and aux_outputs['logits_spectral'] is not None:
                    assert aux_outputs['logits_spectral'].shape == (batch_size, NUM_CLASSES)
            else:
                main_logits = output
                logger.info(f"Classify mode: Main logits shape {main_logits.shape}")

            expected_shape = (batch_size, NUM_CLASSES)
            assert main_logits.shape == expected_shape, \
                f"Classify output shape mismatch! Expected {expected_shape}, got {main_logits.shape}"
            logger.info(f"HVT 'classify' mode successful for img_size {img_size}.")
    except Exception as e:
        logger.error(f"Error during 'classify' mode for HVT with img_size {img_size}: {e}", exc_info=True); raise

    # Test 'get_embeddings' mode
    try:
        with torch.no_grad():
            embeddings: Dict[str, torch.Tensor] = model(rgb_dummy, spectral_dummy, mode='get_embeddings')
            assert isinstance(embeddings, dict), "Embeddings should be a dict"
            assert 'fused' in embeddings, "Fused embeddings missing"
            logger.info(f"Get_embeddings mode: Fused shape {embeddings['fused'].shape}, Keys: {list(embeddings.keys())}")
            logger.info(f"HVT 'get_embeddings' mode successful for img_size {img_size}.")
    except Exception as e:
        logger.error(f"Error during 'get_embeddings' mode for HVT with img_size {img_size}: {e}", exc_info=True); raise

    # Test 'contrastive' mode (if enabled in model's config)
    if hasattr(model, 'ssl_enable_contrastive') and model.ssl_enable_contrastive:
        try:
            with torch.no_grad():
                projected_features = model(rgb_dummy, spectral_dummy, mode='contrastive')
                expected_contrast_dim = SSL_CONTRASTIVE_PROJECTOR_DIM # From local config
                if hasattr(model, 'contrastive_projector') and isinstance(model.contrastive_projector, nn.Sequential) and len(model.contrastive_projector) > 0:
                    last_layer = model.contrastive_projector[-1]
                    if hasattr(last_layer, 'out_features'):
                        expected_contrast_dim = last_layer.out_features
                expected_contrast_shape = (batch_size, expected_contrast_dim)
                assert projected_features.shape == expected_contrast_shape, \
                    f"Contrastive output shape mismatch! Expected {expected_contrast_shape}, got {projected_features.shape}"
                logger.info(f"HVT 'contrastive' mode successful: Output shape {projected_features.shape}.")
        except Exception as e:
            logger.error(f"Error during 'contrastive' mode for HVT with img_size {img_size}: {e}", exc_info=True); raise
    else:
        logger.info("HVT 'contrastive' mode skipped (model.ssl_enable_contrastive is False or attribute missing).")

    # Test 'mae' mode (if enabled in model's config)
    if hasattr(model, 'ssl_enable_mae') and model.ssl_enable_mae:
        try:
            model.train() # MAE might have specific train-time behavior (though test uses no_grad)
            with torch.no_grad():
                mae_output: Dict[str, Optional[torch.Tensor]] = model(rgb_dummy, spectral_dummy, mode='mae')
                assert isinstance(mae_output, dict), "MAE output should be a dict"
                assert 'mask_rgb' in mae_output # Mask should always be there

                if mae_output.get('pred_rgb') is not None and mae_output.get('target_rgb') is not None:
                    num_masked_rgb = mae_output['mask_rgb'].sum().item()
                    expected_pixels_per_patch = 3 * PATCH_SIZE * PATCH_SIZE
                    if num_masked_rgb > 0:
                        assert mae_output['pred_rgb'].shape == (num_masked_rgb, expected_pixels_per_patch)
                        assert mae_output['target_rgb'].shape == (num_masked_rgb, expected_pixels_per_patch)
                    else: # Handle case where no patches are masked
                        assert mae_output['pred_rgb'].shape == (0, expected_pixels_per_patch)
                        assert mae_output['target_rgb'].shape == (0, expected_pixels_per_patch)
                    logger.info(f"HVT 'mae' mode (RGB): pred shape {mae_output['pred_rgb'].shape}, target shape {mae_output['target_rgb'].shape}, num_masked {num_masked_rgb}")
                else:
                    logger.info(f"HVT 'mae' mode (RGB): pred_rgb or target_rgb is None. Mask sum: {mae_output['mask_rgb'].sum().item()}")


                if spectral_dummy is not None and model.spectral_patch_embed and \
                   mae_output.get('pred_spectral') is not None and \
                   mae_output.get('target_spectral') is not None and \
                   mae_output.get('mask_spectral') is not None:
                    num_masked_spec = mae_output['mask_spectral'].sum().item()
                    expected_pixels_per_patch_spec = SPECTRAL_CHANNELS * PATCH_SIZE * PATCH_SIZE
                    if num_masked_spec > 0:
                        assert mae_output['pred_spectral'].shape == (num_masked_spec, expected_pixels_per_patch_spec)
                        assert mae_output['target_spectral'].shape == (num_masked_spec, expected_pixels_per_patch_spec)
                    else:
                        assert mae_output['pred_spectral'].shape == (0, expected_pixels_per_patch_spec)
                        assert mae_output['target_spectral'].shape == (0, expected_pixels_per_patch_spec)
                    logger.info(f"HVT 'mae' mode (Spectral): pred shape {mae_output['pred_spectral'].shape}, target shape {mae_output['target_spectral'].shape}, num_masked {num_masked_spec}")
                logger.info(f"HVT 'mae' mode test successful for img_size {img_size}.")
            model.eval()
        except Exception as e:
            logger.error(f"Error during 'mae' mode for HVT with img_size {img_size}: {e}", exc_info=True); model.eval(); raise
    else:
        logger.info("HVT 'mae' mode skipped (model.ssl_enable_mae is False or attribute missing).")


def run_hvt_tests():
    logger.info("--- Starting DiseaseAwareHVT Tests (phase2_model/main.py) ---")
    for res_idx, img_size_tuple in enumerate(PROGRESSIVE_RESOLUTIONS):
        logger.info(f"Testing HVT with resolution: {img_size_tuple} ({res_idx+1}/{len(PROGRESSIVE_RESOLUTIONS)})")
        # create_disease_aware_hvt_from_config will use its own internal config loading mechanism
        # which should pick up values from phase2_model/config.py
        hvt_model = create_disease_aware_hvt_from_config(img_size_tuple)

        if SPECTRAL_CHANNELS > 0:
            test_hvt_model(hvt_model, img_size=img_size_tuple, use_spectral_input=True, spectral_channels_test=SPECTRAL_CHANNELS)
        else:
            logger.info("Skipping HVT test with spectral input as SPECTRAL_CHANNELS is 0 or less.")

        test_hvt_model(hvt_model, img_size=img_size_tuple, use_spectral_input=False, spectral_channels_test=0)


def main_model_tests():
    # Configure logging if not already done by importing config.py
    # config.py in phase2_model should call logging.basicConfig()
    # If logger has no handlers, it means config.py might not have been imported or failed early.
    if not logging.getLogger().hasHandlers() and not logging.getLogger(__name__).hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        logger.info("Basic logging configured in main_model_tests as fallback.")

    logger.info("======== Running Model Sanity Checks (phase2_model/main.py) ========")
    run_hvt_tests()
    logger.info("======== Model Sanity Checks Finished (phase2_model/main.py) ========")

if __name__ == "__main__":
    main_model_tests()