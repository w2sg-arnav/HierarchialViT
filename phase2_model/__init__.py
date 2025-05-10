# phase2_model/__init__.py
import logging

logger = logging.getLogger(__name__)

# Define __all__ outside the try block to ensure it's always defined
__all__ = []

try:
    # Relative imports from the 'models' subdirectory within 'phase2_model'
    from .models.baseline import InceptionV3Baseline
    from .models.dfca import DiseaseFocusedCrossAttention
    from .models.hvt import DiseaseAwareHVT 
    # from .models.hvt import create_disease_aware_hvt_from_config # Expose if used by other modules

    # Populate __all__ only if imports succeed
    __all__ = [
        "InceptionV3Baseline",
        "DiseaseFocusedCrossAttention",
        "DiseaseAwareHVT",
        # "create_disease_aware_hvt_from_config",
    ]
except ImportError as e:
    logger.warning(f"Warning in phase2_model/__init__.py: Could not perform standard model imports "
                   f"from submodules (e.g., .models.baseline). Check file structure and imports. Error: {e}")
except Exception as e: # Catch any other potential errors during import
    logger.error(f"Unexpected error in phase2_model/__init__.py: {e}", exc_info=True)