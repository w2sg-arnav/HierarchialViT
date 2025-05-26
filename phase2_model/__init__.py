# phase2_model/__init__.py
import logging
logger = logging.getLogger(__name__)

# This file makes models available directly under 'phase2_model'
# e.g., from phase2_model import DiseaseAwareHVT

# Default __all__ to empty list
__all__ = []

try:
    from .models import ( # Import from the .models subpackage
        InceptionV3Baseline,
        DiseaseFocusedCrossAttention,
        DiseaseAwareHVT,
        create_disease_aware_hvt, # Use the renamed factory
    )
    # If imports are successful, populate __all__
    __all__ = [
        "InceptionV3Baseline",
        "DiseaseFocusedCrossAttention",
        "DiseaseAwareHVT",
        "create_disease_aware_hvt",
    ]
    logger.info("Models re-exported successfully by phase2_model/__init__.py.")

except ImportError as e:
    logger.error(f"Error re-exporting models in phase2_model/__init__.py: {e}", exc_info=True)
    # __all__ remains empty or partially populated if some imports failed.
    # This helps avoid NameErrors if this package is imported but sub-modules failed.