"""Predictor contract: frozen query/vision interface, no historical machinery."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.conditional.config import FORBIDDEN_CONFIG_KEYS, PredictorConfig, config_from_mapping
from src.conditional.model import build_predictor
from src.conditional.trainer import ConditionalTrainer, seed_everything


def _tiny(**overrides) -> PredictorConfig:
    payload = {
        "vision_dim": 8,
        "query_dim": 8,
        "target_dim": 8,
        "predictor_layers": 1,
        "predictor_width": 32,
        "predictor_heads": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
    }
    payload.update(overrides)
    return config_from_mapping(payload)


def test_forbidden_and_unknown_keys_are_rejected() -> None:
    for key in (
        "memory_bank_size",
        "wise_ft_alpha",
        "use_model_ema",
        "model_ema_decay",
        "momentum_tau",
        "predictor_lr_multiplier",
        "jepa_alpha",
        "use_teacher",
        "mask_ratio",
        "logit_scale",
    ):
        assert key in FORBIDDEN_CONFIG_KEYS
        try:
            config_from_mapping({key: 1})
        except KeyError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(key)
    try:
        config_from_mapping({"not_a_real_option": 1})
    except KeyError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown keys must be rejected")
    try:
        config_from_mapping({"dropout": 0.1})
    except ValueError:
        pass
    else:
        raise AssertionError("dropout must stay 0")


def test_default_temperature_is_one_and_there_is_one_learning_rate() -> None:
    config = PredictorConfig()
    assert config.temperature == 1.0
    assert config.family == "conditional_latent_predictor"
    assert config.predictor_layers == 4
    assert config.predictor_width == 384
    assert config.predictor_heads == 6
    model = build_predictor(_tiny())
    trainer = ConditionalTrainer(model, model.config, total_steps=4)
    assert len(trainer.optimizer.param_groups) == 1
    assert trainer.optimizer.param_groups[0]["lr"] == pytest_lr(model)
    assert not hasattr(model, "target_encoder")
    assert not hasattr(model, "memory_bank")
    assert not any("teacher" in name or "ema" in name for name in model.state_dict())


def pytest_lr(model) -> float:
    return model.config.lr


class _Recorder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mask = "unset"

    def forward(self, images, mask=None):
        self.mask = mask
        return torch.zeros(images.shape[0], 3, self.dim)


def test_vision_path_does_not_receive_a_mask_and_cls_is_kept() -> None:
    config = _tiny()
    encoder = _Recorder(config.vision_dim)
    model = build_predictor(config, vision_encoder=encoder)
    images = torch.randn(2, 3, 8, 8)
    tokens = model.encode_image(images)
    assert encoder.mask is None
    assert tokens.shape == (2, 3, config.vision_dim)
    assert "mask" not in model.encode_image.__code__.co_varnames
    query = torch.randn(2, config.query_dim)
    visual = torch.randn(2, 4, config.vision_dim)
    first = model.predict_from_tokens(visual, query)
    changed = visual.clone()
    changed[:, 0, :] += 2.0
    second = model.predict_from_tokens(changed, query)
    assert not torch.allclose(first, second, atol=1e-5)
    assert torch.allclose(first.norm(dim=-1), torch.ones(2), atol=1e-5)
    flat = torch.randn(2, config.query_dim)
    sequenced = model.predict_from_tokens(visual, flat.unsqueeze(1))
    unsqueezed = model.predict_from_tokens(visual, flat)
    assert torch.allclose(sequenced, unsqueezed, atol=1e-6)


def test_constructor_does_not_reseed() -> None:
    config = _tiny()
    seed_everything(0)
    build_predictor(config)
    after_build = torch.randn(3)
    seed_everything(0)
    model = build_predictor(config)
    ConditionalTrainer(model, config, total_steps=3)
    after_trainer = torch.randn(3)
    assert torch.equal(after_build, after_trainer)


def test_query_change_changes_the_prediction() -> None:
    config = _tiny()
    model = build_predictor(config)
    visual = torch.randn(2, 3, config.vision_dim)
    query = torch.randn(2, config.query_dim)
    first = model.predict_from_tokens(visual, query)
    other = query.clone()
    other[0] += 3
    second = model.predict_from_tokens(visual, other)
    assert not torch.allclose(first[0], second[0], atol=1e-5)
