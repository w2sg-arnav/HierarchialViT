# phase2_model/__init__.py

# This file makes 'phase2_model' a Python package.
# We expose the core model classes using relative imports from the 'models' sub-directory.
# REMOVED direct imports from 'config'. Config values should be passed during model instantiation.

import logging # Import logging for the warning message

logger = logging.getLogger(__name__)

try:
    # Assuming baseline.py, hvt.py etc. are in phase2_model/models/
    from .models.baseline import InceptionV3Baseline
    from .models.hvt import DiseaseAwareHVT 
    # Keep the helper create function if needed/used elsewhere
    # from .models.hvt import create_disease_aware_hvt_from_config 
    from .models.dfca import DiseaseFocusedCrossAttention

    __all__ = [
        "InceptionV3Baseline",
        "DiseaseAwareHVT",
        # "create_disease_aware_hvt_from_config", # Expose if needed
        "DiseaseFocusedCrossAttention"
    ]
except ImportError as e:
    # This might happen if the structure is different or due to circular imports.
    logger.warning(f"Warning: Could not perform standard imports in phase2_model/__init__.py. Structure might be unexpected. Error: {e}")
    __all__ = []