"""alpha==0 / gamma==0 must not let unused NaN terms poison a finite contrastive loss."""

from __future__ import annotations

import math

import torch

from src.model import VL_JEPA, compute_jepa_loss
from src.trainer import VL_JEPA_Trainer


def _tiny(**kwargs):
    defaults = dict(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
    )
    defaults.update(kwargs)
    return VL_JEPA(**defaults)


def test_alpha_zero_with_nan_predicted_patches_is_finite() -> None:
    model = _tiny()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    outputs = model(images, input_ids, compute_jepa=True)
    outputs["predicted_patches"] = outputs["predicted_patches"].new_full(
        outputs["predicted_patches"].shape, float("nan"),
    )
    outputs["target_patches"] = outputs["target_patches"].new_full(
        outputs["target_patches"].shape, float("nan"),
    )

    loss_dict = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0)
    assert torch.isfinite(loss_dict["total_loss"])
    assert torch.isfinite(loss_dict["nce_loss"])
    assert float(loss_dict["mse_loss"].item()) == 0.0


def test_gamma_zero_with_nan_raw_proj_is_finite() -> None:
    model = _tiny()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    outputs = model(images, input_ids)
    outputs["vision_proj_raw"] = outputs["vision_proj_raw"].new_full(
        outputs["vision_proj_raw"].shape, float("nan"),
    )
    outputs["language_proj_raw"] = outputs["language_proj_raw"].new_full(
        outputs["language_proj_raw"].shape, float("nan"),
    )
    loss_dict = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0)
    assert torch.isfinite(loss_dict["total_loss"])
    assert float(loss_dict["var_loss"].item()) == 0.0


def test_compute_jepa_false_skips_predictor_output_shape() -> None:
    model = _tiny()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    outputs = model(images, input_ids, compute_jepa=False)
    assert outputs["predicted_patches"].shape[1] == 1
    assert torch.isfinite(outputs["predicted_patches"]).all()
    loss_dict = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0)
    assert torch.isfinite(loss_dict["total_loss"])


def test_alpha_zero_train_step_does_not_run_predictor() -> None:
    model = _tiny()

    def _boom(*_args, **_kwargs):
        raise AssertionError("predictor should not run when alpha is 0")

    model.predictor.forward = _boom  # type: ignore[method-assign]
    trainer = VL_JEPA_Trainer(
        model, torch.device("cpu"), learning_rate=1e-3, warmup_steps=0, max_steps=20,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=0,
    )
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    metrics = trainer.train_step(images, input_ids)
    assert metrics.get("skipped") is False
    assert math.isfinite(metrics["total_loss"])
    assert math.isfinite(metrics["nce_loss"])
