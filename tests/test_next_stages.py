"""Shared-budget screen, mechanism exposure, and the confirmation rule."""

from __future__ import annotations

import torch

from src.conditional.confirmation import decide_confirmation, run_synthetic_confirmation
from src.conditional.mechanism import MECHANISM_CONDITIONS, run_synthetic_mechanism
from src.conditional.pooling import last_token_pool, mean_pool, pool_hidden
from src.conditional.screen import (
    ScreenRefused,
    benchmark_rows,
    execute_shared_screen,
    frozen_screen_arms,
    interpret_screen,
)
from src.protocol.preregistration import PREREGISTRATION


def _protected(**overrides: float) -> dict[str, float]:
    values = {name: 0.0 for name in PREREGISTRATION["protected_r1_endpoints"]}
    values.update(overrides)
    return values


def test_pooling_matches_the_target_contracts() -> None:
    hidden = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
            [[3.0, 0.0], [0.0, 4.0], [9.0, 9.0], [8.0, 8.0]],
        ]
    )
    left_pad = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    assert torch.equal(last_token_pool(hidden, left_pad)[0], torch.tensor([0.0, 4.0]))
    assert torch.equal(last_token_pool(hidden, left_pad)[1], torch.tensor([0.0, 4.0]))
    pooled = mean_pool(hidden, left_pad)
    assert torch.allclose(pooled[0], torch.tensor([1.5, 2.0]))
    assert torch.allclose(pooled[1], torch.tensor([1.5, 2.0]))
    assert torch.equal(pool_hidden("last_token", hidden, left_pad), last_token_pool(hidden, left_pad))
    try:
        pool_hidden("eot", hidden, left_pad)
    except ValueError as exc:
        assert "EOT" in str(exc)
    else:
        raise AssertionError("CLIP EOT must not use the mask pooler")


def test_shared_screen_keeps_init_cache_and_exposure() -> None:
    arms = frozen_screen_arms(max_steps=2, seed=0)
    assert len(arms) == 6
    generator = torch.Generator().manual_seed(4)
    visual = torch.randn(4, 2, 8, generator=generator)
    query = torch.randn(4, 8, generator=generator)
    batches = {}
    for index, target_id in enumerate(
        ("clip_vitb16_text", "embeddinggemma_300m_mrl512", "qwen3_embedding_0.6b_mrl512")
    ):
        target = torch.randn(4, 8, generator=generator) + index
        batches[target_id] = {
            "visual_tokens": visual,
            "query_embeddings": query,
            "candidate_targets": target,
            "positive_mask": torch.eye(4, dtype=torch.bool),
            "task_ids": ["caption"] * 4,
            "image_ids": ["a", "b", "c", "d"],
            "caption_ids": ["c0", "c1", "c2", "c3"],
        }
    rows = execute_shared_screen(
        arms,
        batches,
        scope="synthetic_screen_not_a_result",
        architecture={
            "predictor_layers": 1,
            "predictor_width": 32,
            "predictor_heads": 4,
            "lr": 1e-3,
        },
    )
    assert len(rows) == 6
    assert len({row["init_sha256"] for row in rows}) == 1
    assert len({tuple(sorted(row["image_counts"].items())) for row in rows}) == 1
    assert rows[0]["image_counts"] == {"a": 2, "b": 2, "c": 2, "d": 2}
    assert {row["objective"] for row in rows} == {"infonce", "cosine"}
    assert benchmark_rows(rows) == []
    changed = dict(batches)
    changed["clip_vitb16_text"] = dict(changed["clip_vitb16_text"])
    changed["clip_vitb16_text"]["query_embeddings"] = query + 1
    try:
        execute_shared_screen(
            arms,
            changed,
            scope="synthetic_screen_not_a_result",
            architecture={
                "predictor_layers": 1,
                "predictor_width": 32,
                "predictor_heads": 4,
                "lr": 1e-3,
            },
        )
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("a changed query encoder must be refused")
    try:
        execute_shared_screen(
            arms,
            batches,
            scope="scientific",
            architecture={
                "predictor_layers": 1,
                "predictor_width": 32,
                "predictor_heads": 4,
                "lr": 1e-3,
            },
        )
    except ScreenRefused:
        pass
    else:
        raise AssertionError("scientific scope must be refused")


def test_screen_interpretation_uses_caller_cuts() -> None:
    assert interpret_screen(
        text_margin=0.1, visual_probe=0.9, prediction=0.9, text_cut=0.5, probe_cut=0.5, prediction_cut=0.5
    ) == "change_target"
    assert interpret_screen(
        text_margin=0.9, visual_probe=0.9, prediction=0.1, text_cut=0.5, probe_cut=0.5, prediction_cut=0.5
    ) == "change_supervision"
    assert interpret_screen(
        text_margin=0.1, visual_probe=0.1, prediction=0.1, text_cut=0.5, probe_cut=0.5, prediction_cut=0.5
    ) == "change_visual"
    assert interpret_screen(
        text_margin=0.9, visual_probe=0.9, prediction=0.9, text_cut=0.5, probe_cut=0.5, prediction_cut=0.5
    ) == "stop"


def test_mechanism_screen_shares_exposure_and_freezes_the_adapter() -> None:
    rows = run_synthetic_mechanism(steps=2, adapter_steps=30, seed=0)
    assert [row["condition_id"] for row in rows] == [item.condition_id for item in MECHANISM_CONDITIONS]
    assert len({row["init_sha256"] for row in rows}) == 1
    assert len({row["exposure_ledger_key"] for row in rows}) == 1
    counts = [row["image_counts"] for row in rows]
    assert all(count == counts[0] for count in counts)
    assert counts[0] == {f"img-{index}": 2 for index in range(4)}
    by_id = {row["condition_id"]: row for row in rows}
    assert by_id["M5"]["answer_ids"] != by_id["M2"]["answer_ids"]
    assert by_id["M5"]["image_counts"] == by_id["M2"]["image_counts"]
    assert by_id["M1"]["uses_adapter"] is True
    assert by_id["M0"]["uses_adapter"] is False
    assert benchmark_rows(rows) == []
    assert {item.status for item in MECHANISM_CONDITIONS} == {"not_trained"}


def test_confirmation_rule_and_synthetic_rehearsal() -> None:
    protected = _protected()
    assert decide_confirmation(gain_points=2.5, paired_ci_low=0.4, paired_ci_high=4.0, r1_degradation=protected) == "proceed"
    assert decide_confirmation(gain_points=2.5, paired_ci_low=-0.1, paired_ci_high=4.0, r1_degradation=protected) == "stop"
    assert decide_confirmation(gain_points=1.5, paired_ci_low=0.4, paired_ci_high=4.0, r1_degradation=protected) == "stop"
    worse = _protected(coco_karpathy_test_i2t_r1=1.2)
    assert decide_confirmation(gain_points=3.0, paired_ci_low=0.2, paired_ci_high=4.0, r1_degradation=worse) == "stop"
    boundary = _protected(flickr30k_karpathy_test_t2i_r1=1.0)
    assert decide_confirmation(
        gain_points=2.0, paired_ci_low=0.1, paired_ci_high=3.0, r1_degradation=boundary
    ) == "proceed"
    try:
        decide_confirmation(gain_points=3.0, paired_ci_low=0.2, paired_ci_high=1.0, r1_degradation={"coco_karpathy_test_i2t_r1": 0.0})
    except KeyError:
        pass
    else:
        raise AssertionError("missing protected endpoints must raise")
    rows = run_synthetic_confirmation(steps=2)
    assert len(rows) == 9
    assert {row["seed"] for row in rows} == {0, 1, 2}
    assert {row["config"] for row in rows} == {"candidate", "matched_baseline", "explanatory_ablation"}
    assert {row["status"] for row in rows} == {"preregistered_not_unblinded"}
    assert {row["confirmation_runs_executed"] for row in rows} == {0}
    assert benchmark_rows(rows) == []
