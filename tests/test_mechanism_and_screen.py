"""Mechanism definitions, adapter identity, and the screen gate."""

from __future__ import annotations

import torch

from src.conditional.diagnostics import (
    effective_rank,
    paraphrase_margin,
    separation_sufficient,
    small_nonlinear_probe_accuracy,
    linear_probe_accuracy,
)
from src.conditional.mechanism import (
    MECHANISM_CONDITIONS,
    ResidualTargetAdapter,
    adapter_margin_loss,
)
from src.conditional.screen import (
    ScreenRefused,
    benchmark_rows,
    planned_screen_arms,
    run_synthetic_plumbing,
    run_target_screen,
)
from src.conditional.targets import (
    EMBEDDING_GEMMA_300M,
    QWEN3_EMBEDDING_0_6B,
    format_target_text,
    mrl_truncate,
)
from src.protocol.preregistration import PREREGISTRATION
from src.protocol.tasks import (
    answer_ranking,
    classification_top1,
    paired_composition_prefers_matched,
)


def test_screen_refuses_without_a_sentinel_and_plumbing_is_not_a_benchmark() -> None:
    try:
        run_target_screen()
    except ScreenRefused:
        pass
    else:
        raise AssertionError("screen must refuse by default")
    opened = run_target_screen(
        allow_execution=True,
        frozen_shared_budget=True,
        manifests_clean=True,
        visual_cache_sentinel_present=True,
    )
    assert opened["runs_executed"] == 0
    rows = run_synthetic_plumbing(steps=2, seed=0)
    assert len(rows) == 6
    assert len({row["init_sha256"] for row in rows}) == 1
    assert {row["scope"] for row in rows} == {"synthetic_harness_not_a_result"}
    assert benchmark_rows(rows) == []
    arms = planned_screen_arms()
    assert len(arms) == 6
    assert len({arm.query_encoder_id for arm in arms}) == 1
    assert {arm.status for arm in arms} == {"not_launched"}
    assert {arm.max_steps for arm in arms} == {None}


def test_mechanism_conditions_share_exposure_and_adapter_is_identity() -> None:
    assert [item.condition_id for item in MECHANISM_CONDITIONS] == [
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]
    assert len({item.exposure_ledger_key for item in MECHANISM_CONDITIONS}) == 1
    assert {item.status for item in MECHANISM_CONDITIONS} == {"not_trained"}
    adapter = ResidualTargetAdapter()
    inputs = torch.randn(6, 512)
    assert torch.equal(adapter(inputs), inputs)
    loss = adapter_margin_loss(inputs, inputs, inputs, torch.randn_like(inputs))
    assert torch.isfinite(loss)
    adapter.freeze()
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_target_prompts_and_mrl() -> None:
    native = torch.tensor([[3.0, 0.0, 0.0, 4.0]])
    truncated = mrl_truncate(native, dim=2, native_dim=4)
    assert truncated.shape == (1, 2)
    assert torch.allclose(truncated.norm(dim=-1), torch.ones(1), atol=1e-5)
    try:
        mrl_truncate(native, dim=8, native_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("MRL must refuse a wider request than the native vector")
    assert format_target_text(EMBEDDING_GEMMA_300M, "a dog", "document").startswith(
        "title: none | text: "
    )
    qwen_query = format_target_text(QWEN3_EMBEDDING_0_6B, "a dog", "query", task="search")
    assert qwen_query.startswith("Instruct: search\nQuery: ")
    assert format_target_text(QWEN3_EMBEDDING_0_6B, "a dog", "document") == "a dog"


def test_diagnostics_and_task_evaluators() -> None:
    rank1 = torch.ones(5, 4)
    assert effective_rank(rank1) == 0.0 or effective_rank(rank1) < 1e-5
    spread = torch.eye(4)
    assert effective_rank(spread) > 2.0
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[0.0, 1.0]])
    assert paraphrase_margin(anchor, positive, negative) > 0.5
    assert separation_sufficient(positive, positive, negative) is True
    assert separation_sufficient(positive, positive, positive) is False
    features = torch.tensor([[0.0], [0.1], [1.0], [1.1], [0.05], [0.9]])
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    train = torch.tensor([0, 1, 2, 3])
    test = torch.tensor([4, 5])
    assert linear_probe_accuracy(features, labels, train, test) == 1.0
    assert small_nonlinear_probe_accuracy(features, labels, train, test, steps=40) == 1.0
    names = answer_ranking(
        torch.tensor([1.0, 0.0]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ["no", "yes"],
    )
    assert names[0] == "yes"
    assert classification_top1(
        torch.tensor([0.0, 1.0]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ["cat", "dog"],
    ) == "dog"
    assert paired_composition_prefers_matched(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert PREREGISTRATION["status"] == "preregistered_not_unblinded"
    assert PREREGISTRATION["confirmation_runs_executed"] == 0
    assert PREREGISTRATION["primary_compositional_min_gain_points"] == 2.0
    assert PREREGISTRATION["max_retrieval_r1_degradation_points"] == 1.0
    assert PREREGISTRATION["seeds"] == [0, 1, 2]
