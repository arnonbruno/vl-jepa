"""A skipped non-finite batch must reset accumulation and not enqueue the bank."""

from __future__ import annotations

import torch

import pytest

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


def _tiny():
    return VL_JEPA(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
    )


def _trainable_snapshot(model):
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def test_nan_mid_window_resets_counter_and_grads() -> None:
    device = torch.device("cpu")
    model = _tiny()
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=0,
    )
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    accum = 4

    trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    assert trainer._accum_counter == 2
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    before = _trainable_snapshot(model)
    bad = images.clone()
    bad[0, 0, 0, 0] = float("nan")
    metrics = trainer.train_step(
        bad, input_ids, accumulation_steps=accum, step_optimizer=False,
    )

    assert metrics.get("skipped") is True
    assert trainer._accum_counter == 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert torch.equal(p.detach(), before[name])
        assert p.grad is None


def test_nan_on_flush_batch_does_not_step() -> None:
    device = torch.device("cpu")
    model = _tiny()
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=0,
    )
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    accum = 4

    trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    before = _trainable_snapshot(model)
    step_before = trainer._step

    bad = images.clone()
    bad[0, 0, 0, 0] = float("nan")
    metrics = trainer.train_step(
        bad, input_ids, accumulation_steps=accum, step_optimizer=True,
    )
    assert metrics.get("skipped") is True
    assert trainer._accum_counter == 0
    assert trainer._step == step_before
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p.detach(), before[name])
            assert p.grad is None


def test_next_window_after_skip_starts_clean() -> None:
    device = torch.device("cpu")
    model = _tiny()
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=0,
    )
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))
    accum = 4

    trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    bad = images.clone()
    bad[0, 0, 0, 0] = float("nan")
    trainer.train_step(
        bad, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    assert trainer._accum_counter == 0
    assert all(p.grad is None for p in model.parameters() if p.requires_grad)

    metrics = trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=False,
    )
    assert metrics.get("skipped") is False
    assert trainer._accum_counter == 1
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    metrics = trainer.train_step(
        images, input_ids, accumulation_steps=accum, step_optimizer=True,
    )
    assert metrics.get("skipped") is False
    assert trainer._accum_counter == 0
    assert trainer._step == 1


def test_skipped_batch_does_not_enqueue_memory_bank() -> None:
    device = torch.device("cpu")
    model = _tiny()
    trainer = VL_JEPA_Trainer(
        model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=64,
    )
    assert trainer.memory_bank is not None
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))

    bad = images.clone()
    bad[0, 0, 0, 0] = float("nan")
    metrics = trainer.train_step(bad, input_ids)
    assert metrics.get("skipped") is True
    assert trainer.memory_bank.num_filled == 0

    metrics = trainer.train_step(images, input_ids)
    assert metrics.get("skipped") is False
    assert trainer.memory_bank.num_filled == 2


def test_successful_short_final_window_matches_full_mean_scale() -> None:
    """A flushed last window with len % accum != 0 must not under-weight grads.

    Production epochs call ``step_optimizer=True`` on the last micro-batch even
    when the window is shorter than ``accumulation_steps``. Scaling every
    micro-batch by ``1/accum`` would shrink that update; the trainer must
    restore mean-gradient scale so 2-of-4 equals a 2-of-2 window.
    Adam's first-step update size is weakly sensitive to grad scale, so this
    asserts ``grad_norm`` (the clipped accumulated gradient) rather than Δw.
    """
    device = torch.device("cpu")
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 8))

    def _run(accum: int, n_micro: int) -> tuple[float, dict]:
        torch.manual_seed(0)
        model = _tiny()
        trainer = VL_JEPA_Trainer(
            model, device, learning_rate=1e-3, warmup_steps=0, max_steps=100,
            alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=0,
        )
        before = _trainable_snapshot(model)
        last = None
        for i in range(n_micro):
            last = trainer.train_step(
                images, input_ids,
                accumulation_steps=accum,
                step_optimizer=(i == n_micro - 1),
            )
        assert last is not None
        assert last.get("skipped") is False
        assert trainer._accum_counter == 0
        assert trainer._step == 1
        delta = {
            name: model.get_parameter(name).detach().clone() - before[name]
            for name in before
        }
        assert any(d.abs().sum() > 0 for d in delta.values())
        return float(last["grad_norm"]), delta

    gn_full, _ = _run(accum=2, n_micro=2)
    gn_short, _ = _run(accum=4, n_micro=2)
    assert gn_full > 0.0 and gn_short > 0.0
    assert gn_short == pytest.approx(gn_full, rel=1e-4, abs=1e-5)
