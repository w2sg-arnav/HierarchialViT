# phase2_model/models/__init__.py
import logging
logger = logging.getLogger(__name__)

try:
    from .baseline import InceptionV3Baseline
    from .dfca import DiseaseFocusedCrossAttention
    from .hvt import DiseaseAwareHVT, create_disease_aware_hvt # Changed factory name
    # For clarity, let's rename create_disease_aware_hvt_from_config to create_disease_aware_hvt

    __all__ = [
        "InceptionV3Baseline",
        "DiseaseFocusedCrossAttention",
        "DiseaseAwareHVT",
        "create_disease_aware_hvt",
    ]
    logger.info("Models (InceptionV3Baseline, DFCA, DiseaseAwareHVT, factory) imported successfully into phase2_model.models package.")

except ImportError as e:
    logger.error(f"Error importing models in phase2_model/models/__init__.py: {e}", exc_info=True)
    # Define __all__ as empty or with whatever imported successfully to prevent further NameErrors
    __all__ = [name for name in ["InceptionV3Baseline", "DiseaseFocusedCrossAttention", "DiseaseAwareHVT", "create_disease_aware_hvt"] if name in locals()]