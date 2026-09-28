"""Configuration for ``conditional_latent_predictor``.

Unknown keys and the historical trainer's inactive switches are rejected.
The predictor temperature defaults to 1. CLIP's ``log(1/0.07)`` scale is not
read and is not a valid config key here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "memory_bank_size",
        "wise_ft_alpha",
        "use_model_ema",
        "model_ema_decay",
        "momentum_tau",
        "momentum_tau_end",
        "predictor_lr_multiplier",
        "predictor_lr_scale",
        "jepa_alpha",
        "use_teacher",
        "mask_ratio",
        "logit_scale",
    }
)


@dataclass(frozen=True)
class PredictorConfig:
    family: str = "conditional_latent_predictor"
    vision_model: str = "ViT-B-16"
    vision_pretrained: str = "openai"
    image_size: int = 224
    keep_cls: bool = True
    vision_dim: int = 768
    query_dim: int = 512
    target_dim: int = 512
    predictor_layers: int = 4
    predictor_width: int = 384
    predictor_heads: int = 6
    dropout: float = 0.0
    temperature: float = 1.0
    objective: str = "infonce"
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    microbatch: int = 128
    seed: int = 0
    query_encoder_id: str = "clip_vitb16_text"
    context_length: int = 77
    pooling: str = "query_token_mean"
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.family != "conditional_latent_predictor":
            raise ValueError(f"unexpected model family {self.family!r}")
        if self.dropout != 0.0:
            raise ValueError("dropout must be 0 so a resumed run can match bitwise")
        if self.predictor_layers < 1:
            raise ValueError("predictor_layers must be positive")
        if self.predictor_width % self.predictor_heads != 0:
            raise ValueError("predictor_width must be divisible by predictor_heads")
        if self.objective not in {"infonce", "cosine"}:
            raise ValueError(f"objective must be infonce or cosine, got {self.objective!r}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not self.keep_cls:
            raise ValueError(
                "the initial predictor keeps the CLIP CLS token and attends to it; "
                "dropping it is a different model"
            )
        if self.context_length != 77 and self.query_encoder_id == "clip_vitb16_text":
            raise ValueError("CLIP ViT-B/16 text uses context length 77; do not slice it")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_mapping(data: Mapping[str, Any] | None = None) -> PredictorConfig:
    payload = dict(data or {})
    forbidden = sorted(set(payload) & FORBIDDEN_CONFIG_KEYS)
    if forbidden:
        raise KeyError(
            "forbidden config keys for conditional_latent_predictor: "
            + ", ".join(forbidden)
        )
    known = {item.name for item in fields(PredictorConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise KeyError("unknown config keys: " + ", ".join(unknown))
    return PredictorConfig(**payload)
