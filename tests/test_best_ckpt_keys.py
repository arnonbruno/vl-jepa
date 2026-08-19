"""Checkpoint saves must keep config, live weights, and eval weights."""

from __future__ import annotations

from pathlib import Path

import torch

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


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
