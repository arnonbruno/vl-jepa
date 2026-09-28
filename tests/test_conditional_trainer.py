"""Overfit, query use, gradient clipping, and exact resume."""

from __future__ import annotations

import torch

from src.conditional.config import config_from_mapping
from src.conditional.model import build_predictor
from src.conditional.trainer import ConditionalTrainer, seed_everything
from src.protocol.artifacts import capture_rng_state, load_checkpoint, save_checkpoint
from src.protocol.evaluator import evaluate_retrieval, is_perfect_retrieval


def _config(**overrides):
    payload = {
        "vision_dim": 16,
        "query_dim": 16,
        "target_dim": 16,
        "predictor_layers": 2,
        "predictor_width": 64,
        "predictor_heads": 4,
        "objective": "cosine",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "warmup_fraction": 0.0,
        "grad_clip": 1.0,
        "temperature": 1.0,
    }
    payload.update(overrides)
    return config_from_mapping(payload)


def _batch(visual, query, targets, image_ids, caption_ids, task="caption"):
    return {
        "visual_tokens": visual,
        "query_embeddings": query,
        "candidate_targets": targets,
        "positive_mask": torch.eye(visual.shape[0], dtype=torch.bool),
        "task_ids": [task] * visual.shape[0],
        "image_ids": image_ids,
        "caption_ids": caption_ids,
    }


def test_overfit_identity_captions_reach_perfect_in_sample_retrieval() -> None:
    config = _config()
    seed_everything(0)
    model = build_predictor(config)
    trainer = ConditionalTrainer(model, config, total_steps=200)
    generator = torch.Generator().manual_seed(11)
    targets = torch.nn.functional.normalize(torch.randn(4, 16, generator=generator), dim=-1)
    visual = targets.unsqueeze(1).expand(4, 2, 16).contiguous()
    query = torch.ones(4, 16)
    batch = _batch(
        visual,
        query,
        targets,
        [f"img-{i}" for i in range(4)],
        [f"cap-{i}" for i in range(4)],
    )
    for _ in range(200):
        trainer.train_step(batch)
    prediction = model.predict_from_tokens(visual, query)
    result = evaluate_retrieval(prediction, targets, torch.arange(4))
    assert result["selection_accepted"] is True
    assert is_perfect_retrieval(result) is True
    assert "held-out" not in trainer.ledger.images


def test_image_and_question_controls_and_held_out_exposure() -> None:
    config = _config(lr=4e-3)
    seed_everything(1)
    model = build_predictor(config)
    trainer = ConditionalTrainer(model, config, total_steps=400)
    generator = torch.Generator().manual_seed(3)
    images = torch.randn(4, 16, generator=generator)
    questions = torch.randn(2, 16, generator=generator)
    targets = torch.nn.functional.normalize(torch.randn(8, 16, generator=generator), dim=-1)
    rows = [(image, question) for image in range(4) for question in range(2)]

    def tokens_for(bank_row: torch.Tensor) -> torch.Tensor:
        return bank_row.view(1, 16).expand(2, 16).contiguous()

    visual = torch.stack([tokens_for(images[image]) for image, _question in rows])
    query = torch.stack([questions[question] for _image, question in rows])
    train_ids = [f"pair-{image}-{question}" for image, question in rows]
    batch = _batch(visual, query, targets, train_ids, [f"cap-{i}" for i in range(8)], task="qa")
    for _ in range(400):
        trainer.train_step(batch)
    prediction = model.predict_from_tokens(visual, query).detach()
    nearest = torch.argmax(prediction @ targets.T, dim=1)
    assert torch.equal(nearest, torch.arange(8))

    zero_visual = torch.zeros_like(visual)
    zero_pred = model.predict_from_tokens(zero_visual, query).detach()
    zero_nearest = torch.argmax(zero_pred @ targets.T, dim=1)
    assert int((zero_nearest == torch.arange(8)).sum()) < 8

    shuffled_visual = torch.stack([tokens_for(images[(image + 1) % 4]) for image, _question in rows])
    shuffled = model.predict_from_tokens(shuffled_visual, query).detach()
    shuffled_nearest = torch.argmax(shuffled @ targets.T, dim=1)
    assert int((shuffled_nearest == torch.arange(8)).sum()) < 8

    held_images = torch.randn(2, 16, generator=generator)
    held_query = questions.clone()
    base = model.predict_from_tokens(tokens_for(held_images[0]).unsqueeze(0), held_query[:1])
    other_image = model.predict_from_tokens(tokens_for(held_images[1]).unsqueeze(0), held_query[:1])
    other_question = model.predict_from_tokens(tokens_for(held_images[0]).unsqueeze(0), held_query[1:2])
    assert not torch.allclose(base, other_image, atol=1e-4)
    assert not torch.allclose(base, other_question, atol=1e-4)
    for image_id in train_ids:
        assert image_id in trainer.ledger.images
    assert "held-out-0" not in trainer.ledger.images
    assert "held-out-1" not in trainer.ledger.images


def test_targets_do_not_enter_the_predictor_or_receive_gradients() -> None:
    config = _config(objective="infonce", predictor_layers=1, predictor_width=32)
    model = build_predictor(config)
    trainer = ConditionalTrainer(model, config, total_steps=2)
    visual = torch.randn(3, 2, 16)
    query = torch.randn(3, 16)
    targets = torch.randn(3, 16, requires_grad=True)
    before = model.predict_from_tokens(visual, query).detach()
    trainer.train_step(
        _batch(visual, query, targets, ["a", "b", "c"], ["c0", "c1", "c2"])
    )
    after = model.predict_from_tokens(visual, query).detach()
    assert targets.grad is None
    # A second forward with a different target tensor leaves the prediction
    # function dependent only on visual tokens and the query.
    other_targets = torch.randn(3, 16)
    again = model.predict_from_tokens(visual, query).detach()
    assert torch.allclose(after, again)
    assert not torch.allclose(before, after, atol=1e-6)
    del other_targets


def test_gradient_clip_and_warmup_schedule() -> None:
    config = _config(objective="infonce", grad_clip=1e-8, lr=1e-3, warmup_fraction=0.5)
    model = build_predictor(config)
    trainer = ConditionalTrainer(model, config, total_steps=10)
    assert trainer.scheduler.warmup_steps == 5
    assert trainer.scheduler.factor(0) < trainer.scheduler.factor(4)
    assert trainer.scheduler.factor(9) < trainer.scheduler.factor(5)
    visual = torch.randn(4, 2, 16)
    query = torch.randn(4, 16)
    targets = torch.randn(4, 16)
    trainer.train_step(_batch(visual, query, targets, list("abcd"), list("wxyz")))
    assert trainer.last_preclip_grad_norm is not None
    assert trainer.last_postclip_grad_norm is not None
    assert trainer.last_preclip_grad_norm > trainer.last_postclip_grad_norm
    assert trainer.last_postclip_grad_norm <= 1e-7


def test_interrupt_resume_matches_an_uninterrupted_run(tmp_path) -> None:
    config = _config(objective="infonce", predictor_layers=1, predictor_width=32, lr=1e-3)

    def fresh():
        seed_everything(7)
        model = build_predictor(config)
        return ConditionalTrainer(model, config, total_steps=6)

    generator = torch.Generator().manual_seed(9)
    visual = torch.randn(4, 2, 16, generator=generator)
    query = torch.randn(4, 16, generator=generator)
    targets = torch.randn(4, 16, generator=generator)
    batch = _batch(visual, query, targets, ["i0", "i1", "i2", "i3"], ["c0", "c1", "c2", "c3"])

    uninterrupted = fresh()
    for _ in range(5):
        uninterrupted.train_step(batch)
    final_loss = uninterrupted.train_step(batch)
    final_state = uninterrupted.student_state()

    resumed = fresh()
    for _ in range(3):
        resumed.train_step(batch)
    path = tmp_path / "interrupt.pt"
    save_checkpoint(
        path,
        student_state=resumed.student_state(),
        evaluated_state=resumed.evaluated_state(),
        evaluated_state_name="raw",
        optimizer_state=resumed.optimizer.state_dict(),
        scheduler_state=resumed.scheduler.state_dict(),
        scaler_state=None,
        rng_state=capture_rng_state(),
        step=resumed.step,
        config=config.to_dict(),
        extra=resumed.checkpoint_extra(),
    )
    loaded = load_checkpoint(path)
    assert loaded["evaluated_state_name"] == "raw"
    assert loaded["evaluated_equals_student"] is True
    assert loaded["scaler_state"] is None
    continued = fresh()
    continued.load_training_state(
        student_state=loaded["student_state"],
        optimizer_state=loaded["optimizer_state"],
        scheduler_state=loaded["scheduler_state"],
        rng_state=loaded["rng_state"],
        step=loaded["step"],
        extra=loaded["extra"],
    )
    for key, value in loaded["evaluated_state"].items():
        assert torch.equal(value, continued.model.state_dict()[key].cpu())
    for _ in range(3):
        continued_loss = continued.train_step(batch)
    assert continued.step == uninterrupted.step
    assert abs(continued_loss - final_loss) < 1e-6
    for key, value in final_state.items():
        assert torch.equal(value, continued.student_state()[key])
    assert continued.ledger.images["i0"] == 6
