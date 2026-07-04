from mortality.models.classical import CLASSICAL_MODELS
from mortality.models.hybrid import HYBRID_MODELS
from mortality.models.neural import NEURAL_MODELS

ALL_MODELS = {**CLASSICAL_MODELS, **NEURAL_MODELS, **HYBRID_MODELS}

__all__ = ["CLASSICAL_MODELS", "NEURAL_MODELS", "HYBRID_MODELS", "ALL_MODELS"]
