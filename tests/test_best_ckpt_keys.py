"""Checkpoint saves must keep config, live weights, and eval weights."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer, wise_ft_interpolate


def test_save_checkpoint_writes_config_and_eval_state(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = VL_JEPA(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
    )
    trainer = VL_JEPA_Trainer(
        model, device, warmup_steps=0, max_steps=10, memory_bank_size=0,
        use_model_ema=True, model_ema_decay=0.9, wise_ft_alpha=0.5,
    )
    cfg = {
        "model": {
            "hidden_dim": 96,
            "projection_type": "mlp",
            "text_pool": "mean",
            "text_backbone": "custom",
        },
        "training": {"epochs": 1},
        "loss": {"alpha": 0.0, "beta": 1.0},
    }
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    trainer.train_step(images, input_ids)

    best = tmp_path / "checkpoint_best.pt"
    trainer.save_checkpoint(
        str(best),
        {"epoch": 3, "config": cfg, "val_loss": 1.2, "retrieval_score": 0.4},
    )
    periodic = tmp_path / "checkpoint_epoch3.pt"
    trainer.save_checkpoint(str(periodic), {"epoch": 3, "config": cfg})

    for path in (best, periodic):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert "config" in ckpt, f"{path.name} missing config"
        assert ckpt["config"]["model"]["hidden_dim"] == 96
        assert ckpt["config"]["model"]["projection_type"] == "mlp"
        assert ckpt["config"]["model"]["text_pool"] == "mean"
        assert "model_state_dict" in ckpt
        assert "model_eval_state" in ckpt
        live = ckpt["model_state_dict"]
        ev = ckpt["model_eval_state"]
        assert set(live.keys()) == set(ev.keys())
        assert any(
            isinstance(live[k], torch.Tensor)
            and live[k].dtype.is_floating_point
            and not torch.equal(live[k].cpu(), ev[k].cpu())
            for k in live
        ), f"{path.name}: live and eval states should differ under EMA/WiSE-FT"


def _tiny_trainer(**kwargs):
    device = torch.device("cpu")
    model = VL_JEPA(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
    )
    defaults = dict(
        warmup_steps=0, max_steps=10, memory_bank_size=0,
        use_model_ema=True, model_ema_decay=0.9, wise_ft_alpha=0.5,
    )
    defaults.update(kwargs)
    trainer = VL_JEPA_Trainer(model, device, **defaults)
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    trainer.train_step(images, input_ids)
    return trainer


def _float_key(state):
    return next(
        k for k, v in state.items()
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point and v.numel() > 1
    )


def test_best_ckpt_inside_eval_weights_keeps_live_student(tmp_path: Path) -> None:
    """Production best-save happens inside eval_weights(); live student must remain."""
    trainer = _tiny_trainer(use_model_ema=True, model_ema_decay=0.9, wise_ft_alpha=0.5)
    live_snap = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    eval_snap = trainer.build_eval_state_dict()
    key = _float_key(live_snap)
    assert not torch.equal(live_snap[key], eval_snap[key])

    best = tmp_path / "checkpoint_best.pt"
    with trainer.eval_weights():
        trainer.save_checkpoint(
            str(best),
            {"epoch": 3, "retrieval_score": 0.4},
            live_state_dict=live_snap,
        )

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert torch.equal(ckpt["model_state_dict"][key].cpu(), live_snap[key])
    assert torch.allclose(ckpt["model_eval_state"][key].cpu(), eval_snap[key], atol=1e-6)
    restored = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
    assert torch.equal(restored[key], live_snap[key])


def test_best_ckpt_no_ema_wise_ft_is_not_double_interpolated(tmp_path: Path) -> None:
    trainer = _tiny_trainer(use_model_ema=False, wise_ft_alpha=0.5)
    live_snap = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    expected_eval = trainer.build_eval_state_dict()
    key = _float_key(live_snap)
    double = wise_ft_interpolate(trainer._zeroshot_state, expected_eval, trainer.wise_ft_alpha)
    assert not torch.allclose(expected_eval[key], double[key], atol=1e-5)

    best = tmp_path / "checkpoint_best.pt"
    with trainer.eval_weights():
        trainer.save_checkpoint(str(best), {"epoch": 1}, live_state_dict=live_snap)

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert torch.equal(ckpt["model_state_dict"][key].cpu(), live_snap[key])
    assert torch.allclose(ckpt["model_eval_state"][key].cpu(), expected_eval[key], atol=1e-6)
    assert not torch.allclose(ckpt["model_eval_state"][key].cpu(), double[key], atol=1e-5)
