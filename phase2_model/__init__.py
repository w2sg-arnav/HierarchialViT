# phase2_model/__init__.py

# This file makes 'phase2_model' a Python package.
# Use absolute imports starting from the package name 'phase2_model'

# Option 1: If baseline.py is inside phase2_model/models/
# from phase2_model.models.baseline import InceptionV3Baseline 
# from phase2_model.models.hvt import DiseaseAwareHVT, create_disease_aware_hvt_from_config
# from phase2_model.models.dfca import DiseaseFocusedCrossAttention

# Option 2: If baseline.py etc are directly inside phase2_model/ (seems unlikely given your phase 2 structure)
# from phase2_model.baseline import InceptionV3Baseline
# ...

# Let's assume the structure from Phase 2 was phase2_model/models/baseline.py etc.
# The import 'from .baseline import ...' suggests it expected baseline.py 
# to be in the *same* directory as this __init__.py.
# If baseline.py is actually in phase2_model/models/, the import should be:
# from .models.baseline import ... 
# Let's try making the import relative to the `models` subfolder:

try:
    # Assuming baseline.py, hvt.py etc. are in phase2_model/models/
    from .models.baseline import InceptionV3Baseline
    from .models.hvt import DiseaseAwareHVT, create_disease_aware_hvt_from_config
    from .models.dfca import DiseaseFocusedCrossAttention

    __all__ = [
        "InceptionV3Baseline",
        "DiseaseAwareHVT",
        "create_disease_aware_hvt_from_config",
        "DiseaseFocusedCrossAttention"
    ]
except ImportError as e:
    # This might happen if the files are directly in phase2_model/ instead of phase2_model/models/
    # Or if there's a circular import issue.
    print(f"Warning: Could not perform standard imports in phase2_model/__init__.py. Structure might be unexpected. Error: {e}")
    __all__ = []


# Alternative (if models are directly under phase2_model):
# from .baseline import InceptionV3Baseline
# from .hvt import DiseaseAwareHVT, create_disease_aware_hvt_from_config
# from .dfca import DiseaseFocusedCrossAttention