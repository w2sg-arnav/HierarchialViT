# phase2_model/main.py
import torch
import torch.nn as nn # For type hints and potential direct use
import logging
from typing import Tuple, Dict, Any, Optional # Added Any for model_params_dict

# --- Path Setup ---
# Ensures that if this script is run directly, it can find sibling modules (like config)
# and potentially modules from the project root if needed for other phases.
import sys
import os
_current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../phase2_model
_project_root_candidate = os.path.dirname(_current_script_dir)   # .../ (e.g. phase2_project)

# Add project root to sys.path if not already present.
# This is more for inter-phase imports later, less critical for self-contained phase2_model tests.
if _project_root_candidate not in sys.path:
    sys.path.insert(0, _project_root_candidate)
# --- End Path Setup ---

# --- Configuration Import from local phase2_model/config.py ---
try:
    # Import specific configuration variables needed by this main.py
    from .config import (
        NUM_CLASSES,
        HVT_MODEL_PARAMS,         # The main dictionary for HVT parameters
        DEVICE,
        INITIAL_IMAGE_SIZE,       # Used to instantiate the model initially
        PROGRESSIVE_RESOLUTIONS_TEST # List of resolutions to test
    )
    logger = logging.getLogger(__name__) # Initialize logger after config successfully imports
    logger.info("Successfully imported constants from local phase2_model/config.py for main.py execution.")
except ImportError as e_cfg:
    # Fallback basic logging if config.py (which should set up logging) fails
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger = logging.getLogger(__name__) # Define logger for error message
    logger.error(f"CRITICAL ERROR in phase2_model/main.py: Failed to import from './config.py'. Error: {e_cfg}", exc_info=True)
    logger.error("Ensure 'phase2_model/config.py' exists and defines: NUM_CLASSES, HVT_MODEL_PARAMS, DEVICE, INITIAL_IMAGE_SIZE, PROGRESSIVE_RESOLUTIONS_TEST.")
    sys.exit(1)
# --- End Configuration Import ---


# --- Model Imports (from phase2_model.models subpackage) ---
try:
    # Using relative import from .models, assuming main.py is part of phase2_model package
    from .models import DiseaseAwareHVT, create_disease_aware_hvt # Use the renamed factory
    # from .models import InceptionV3Baseline # Uncomment if InceptionV3 tests are also needed here
    logger.info("HVT models imported successfully into phase2_model/main.py.")
except ImportError as e_model:
    logger.error(f"CRITICAL ERROR in phase2_model/main.py: Failed to import models from './models'. Error: {e_model}", exc_info=True)
    logger.error("Ensure 'phase2_model/models/__init__.py' and model files (hvt.py, etc.) exist and are correct.")
    sys.exit(1)
# --- End Model Imports ---


def test_hvt_model_at_resolution(
        model: DiseaseAwareHVT,
        current_img_size: Tuple[int, int],
        num_classes_test: int,
        hvt_config_params: Dict[str, Any], # Pass the config for SSL flags etc.
        batch_size: int = 2,
        device_test: str = DEVICE
    ):
    """Test the DiseaseAwareHVT model's forward pass at a specific resolution."""
    logger.info(f"--- Testing HVT @ {current_img_size} (Device: {device_test}) ---")
    model.to(device_test)
    model.eval() # Set to evaluation mode for testing

    # Extract spectral_channels from the passed HVT config for dummy data creation
    spectral_channels_test = hvt_config_params.get('spectral_channels', 0)

    rgb_dummy = torch.randn(batch_size, 3, current_img_size[0], current_img_size[1], device=device_test)
    spectral_dummy = None
    if spectral_channels_test > 0:
        spectral_dummy = torch.randn(batch_size, spectral_channels_test, current_img_size[0], current_img_size[1], device=device_test)
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral shape {spectral_dummy.shape if spectral_dummy is not None else 'None'}")
    else:
        logger.info(f"Input: RGB shape {rgb_dummy.shape}, Spectral: None (spectral_channels_test={spectral_channels_test})")

    # Test 'classify' mode
    try:
        with torch.no_grad():
            output = model(rgb_dummy, spectral_dummy, mode='classify')
            main_logits: torch.Tensor
            if hvt_config_params.get('enable_consistency_loss_heads', False):
                main_logits, aux_outputs = output
                assert isinstance(aux_outputs, dict), "Aux outputs should be a dict"
                logger.info(f"Classify: Main logits {main_logits.shape}, Aux keys: {list(aux_outputs.keys())}")
            else:
                main_logits = output
                logger.info(f"Classify: Main logits shape {main_logits.shape}")
            assert main_logits.shape == (batch_size, num_classes_test)
    except Exception as e: logger.error(f"Error HVT 'classify' @{current_img_size}: {e}", exc_info=True); raise

    # Test 'get_embeddings' mode
    try:
        with torch.no_grad():
            embeddings: Dict[str, torch.Tensor] = model(rgb_dummy, spectral_dummy, mode='get_embeddings')
            assert 'fused_pooled' in embeddings
            logger.info(f"Embeddings: Fused shape {embeddings['fused_pooled'].shape}, Keys: {list(embeddings.keys())}")
    except Exception as e: logger.error(f"Error HVT 'get_embeddings' @{current_img_size}: {e}", exc_info=True); raise

    # Test 'contrastive' mode (if enabled in HVT config)
    if hvt_config_params.get('ssl_enable_contrastive', False):
        try:
            with torch.no_grad():
                proj_feat = model(rgb_dummy, spectral_dummy, mode='contrastive')
                expected_dim = hvt_config_params.get('ssl_contrastive_projector_dim', 128)
                assert proj_feat.shape == (batch_size, expected_dim)
                logger.info(f"Contrastive: Output shape {proj_feat.shape}")
        except Exception as e: logger.error(f"Error HVT 'contrastive' @{current_img_size}: {e}", exc_info=True); raise
    else: logger.info(f"Contrastive mode skipped (ssl_enable_contrastive: False for HVT @{current_img_size}).")

    # Test 'mae' mode (if enabled in HVT config)
    # MAE mode involves masking, so it's more complex to assert shapes without knowing mask.
    # We'll check if it runs and if outputs seem reasonable (e.g., not None if expected).
    if hvt_config_params.get('ssl_enable_mae', False):
        model.train() # MAE might have specific train-time behavior like dropout in decoder.
                      # No gradient calculation here, so it's fine for a quick test.
        try:
            with torch.no_grad(): # Still no grad for testing inference path
                # Create a dummy mask (masking first 25% of patches)
                num_patches_total = (current_img_size[0] // hvt_config_params['patch_size']) * \
                                    (current_img_size[1] // hvt_config_params['patch_size'])
                num_masked = num_patches_total // 4
                dummy_mae_mask = torch.zeros(batch_size, num_patches_total, dtype=torch.bool, device=device_test)
                if num_masked > 0 : dummy_mae_mask[:, :num_masked] = True


                mae_output: Dict[str, Optional[torch.Tensor]] = model(rgb_dummy, spectral_dummy, mode='mae', mae_mask_custom=dummy_mae_mask)
                assert isinstance(mae_output, dict)
                logger.info(f"MAE Output Keys @{current_img_size}: {list(mae_output.keys())}")
                if mae_output.get('pred_rgb') is not None: logger.info(f"MAE pred_rgb shape: {mae_output['pred_rgb'].shape}")
                if mae_output.get('pred_spectral') is not None: logger.info(f"MAE pred_spectral shape: {mae_output['pred_spectral'].shape}")
        except Exception as e: logger.error(f"Error HVT 'mae' @{current_img_size}: {e}", exc_info=True); raise
        finally: model.eval() # Ensure model is back in eval mode
    else: logger.info(f"MAE mode skipped (ssl_enable_mae: False for HVT @{current_img_size}).")
    logger.info(f"--- HVT Tests @ {current_img_size} PASSED ---")


def run_all_hvt_tests():
    logger.info(f"\n======== Starting DiseaseAwareHVT Architecture Tests (Device: {DEVICE}) ========")

    # Instantiate the HVT model ONCE using the INITIAL_IMAGE_SIZE.
    # The model's internal state (like PatchEmbed.img_size) will be updated
    # by its forward_features_encoded method before processing each resolution.
    # The positional embeddings will be interpolated.
    logger.info(f"Instantiating HVT with initial_img_size: {INITIAL_IMAGE_SIZE}")
    hvt_model = create_disease_aware_hvt(
        current_img_size=INITIAL_IMAGE_SIZE, # This is the size for which pos_embed is initially defined
        num_classes=NUM_CLASSES,
        model_params_dict=HVT_MODEL_PARAMS # Pass the whole config dict
    )

    # Test the *same* model instance at different resolutions
    for res_idx, img_size_tuple_test in enumerate(PROGRESSIVE_RESOLUTIONS_TEST):
        logger.info(f"\n>>> Testing HVT at Resolution: {img_size_tuple_test} ({res_idx+1}/{len(PROGRESSIVE_RESOLUTIONS_TEST)}) <<<")
        test_hvt_model_at_resolution(
            model=hvt_model,
            current_img_size=img_size_tuple_test,
            num_classes_test=NUM_CLASSES,
            hvt_config_params=HVT_MODEL_PARAMS # Pass HVT params for SSL flags etc.
        )

    logger.info("======== DiseaseAwareHVT Architecture Tests Finished Successfully ========")


if __name__ == "__main__":
    # This check ensures that if main.py is the entry point, logging is set up.
    # (config.py also tries to set it up, so this is a safeguard)
    if not logging.getLogger().hasHandlers() and not logger.hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
         logger.info("Basic logging configured in phase2_model/main.py __main__ block.")

    run_all_hvt_tests()
    