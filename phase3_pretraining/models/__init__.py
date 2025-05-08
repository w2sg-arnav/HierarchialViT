# phase3_pretraining/models/__init__.py

# This makes 'models' a Python package.
# We will primarily use models from phase2_model, but might define wrappers here.

from .projection_head import ProjectionHead
from .hvt_wrapper import HVTForPretraining # This will be the new wrapper

__all__ = [
    "ProjectionHead",
    "HVTForPretraining"
]