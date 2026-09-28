"""Query-conditioned latent predictor. Separate from ``src.model.VL_JEPA``."""

from src.conditional.config import FORBIDDEN_CONFIG_KEYS, PredictorConfig, config_from_mapping
from src.conditional.model import ConditionalLatentPredictor, build_predictor

__all__ = [
    "FORBIDDEN_CONFIG_KEYS",
    "PredictorConfig",
    "ConditionalLatentPredictor",
    "build_predictor",
    "config_from_mapping",
]
