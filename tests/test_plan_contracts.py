"""Contracts that stay executable without a COCO cache or a scientific run."""

from __future__ import annotations

import torch

from src.conditional.config import config_from_mapping
from src.conditional.confirmation import decide_confirmation, paired_bootstrap_ci
from src.conditional.controls import prediction_controls
from src.conditional.diagnostics import (
    collision_action,
    exact_collision_pairs,
    neighborhood_mean_cosine,
)
from src.conditional.ledger import ComputeLedger, LaunchRefused, assert_gqa_screening_cap
from src.conditional.mechanism import (
    ResidualTargetAdapter,
    answer_prior_l1,
    mechanism_claims,
)
from src.conditional.model import build_predictor
from src.conditional.screen import (
    ScreenRefused,
    SharedSchedule,
    benchmark_rows,
    execute_scheduled_screen,
    frozen_screen_arms,
)
from src.conditional.trainer import seed_everything
from src.protocol.preregistration import PREREGISTRATION


def _phases(generator: torch.Generator) -> tuple[dict, dict]:
    visual = torch.randn(4, 2, 8, generator=generator)
    image_ids = ["a", "b", "c", "d"]
    caption = {
        "visual_tokens": visual,
        "query_embeddings": torch.randn(4, 8, generator=generator),
        "candidate_targets": torch.randn(4, 8, generator=generator),
        "positive_mask": torch.eye(4, dtype=torch.bool),
        "task_ids": ["caption"] * 4,
        "image_ids": image_ids,
        "caption_ids": ["c0", "c1", "c2", "c3"],
    }
    qa = {
        "visual_tokens": visual,
        "query_embeddings": torch.randn(4, 8, generator=generator),
        "candidate_targets": torch.randn(4, 8, generator=generator),
        "positive_mask": torch.eye(4, dtype=torch.bool),
        "task_ids": ["qa"] * 4,
        "image_ids": list(image_ids),
        "caption_ids": ["q0", "q1", "q2", "q3"],
        "answer_ids": ["dog", "man", "dog", "cat"],
    }
    return caption, qa


def _screen_batches() -> tuple[dict, dict]:
    generator = torch.Generator().manual_seed(7)
    caption, qa = _phases(generator)
    caption_batches = {}
    qa_batches = {}
    for index, target_id in enumerate(
        ("clip_vitb16_text", "embeddinggemma_300m_mrl512", "qwen3_embedding_0.6b_mrl512")
    ):
        caption_arm = dict(caption)
        qa_arm = dict(qa)
        caption_arm["candidate_targets"] = caption["candidate_targets"] + index
        qa_arm["candidate_targets"] = qa["candidate_targets"] + index
        caption_batches[target_id] = caption_arm
        qa_batches[target_id] = qa_arm
    return caption_batches, qa_batches


def test_scheduled_screen_shares_phases_exposure_and_gradients() -> None:
    schedule = SharedSchedule(caption_steps=2, qa_steps=1)
    arms = frozen_screen_arms(schedule.total_steps, seed=0)
    caption_batches, qa_batches = _screen_batches()
    architecture = {
        "predictor_layers": 1,
        "predictor_width": 32,
        "predictor_heads": 4,
        "lr": 1e-3,
    }
    rows = execute_scheduled_screen(
        arms,
        caption_batches,
        qa_batches,
        scope="synthetic_screen_not_a_result",
        architecture=architecture,
        schedule=schedule,
    )
    assert len(rows) == 6
    assert len({row["init_sha256"] for row in rows}) == 1
    assert len({tuple(sorted(row["image_counts"].items())) for row in rows}) == 1
    assert rows[0]["image_counts"] == {"a": 3, "b": 3, "c": 3, "d": 3}
    assert len(rows[0]["grad_norms"]) == 3
    assert len(rows[0]["caption_losses"]) == 2
    assert len(rows[0]["qa_losses"]) == 1
    assert {row["schedule"]["caption_steps"] for row in rows} == {2}
    assert benchmark_rows(rows) == []
    broken = {key: dict(value) for key, value in qa_batches.items()}
    broken["clip_vitb16_text"]["visual_tokens"] = broken["clip_vitb16_text"]["visual_tokens"] + 1
    try:
        execute_scheduled_screen(
            arms,
            caption_batches,
            broken,
            scope="synthetic_screen_not_a_result",
            architecture=architecture,
            schedule=schedule,
        )
    except ValueError as exc:
        assert "visual" in str(exc)
    else:
        raise AssertionError("caption and QA must share visual tokens")
    try:
        execute_scheduled_screen(
            arms,
            caption_batches,
            qa_batches,
            scope="scientific",
            architecture=architecture,
            schedule=schedule,
        )
    except ScreenRefused:
        pass
    else:
        raise AssertionError("scientific scope must be refused")
    try:
        SharedSchedule(caption_steps=0, qa_steps=2)
    except ValueError:
        pass
    else:
        raise AssertionError("a caption phase is required")


def test_controls_move_when_visual_or_query_changes() -> None:
    seed_everything(0)
    config = config_from_mapping(
        {
            "vision_dim": 8,
            "query_dim": 8,
            "target_dim": 8,
            "predictor_layers": 1,
            "predictor_width": 32,
            "predictor_heads": 4,
            "objective": "cosine",
            "seed": 0,
        }
    )
    model = build_predictor(config)
    generator = torch.Generator().manual_seed(3)
    visual = torch.randn(4, 2, 8, generator=generator)
    query = torch.randn(4, 8, generator=generator)
    target = torch.nn.functional.normalize(torch.randn(4, 8, generator=generator), dim=-1)
    report = prediction_controls(model, visual, query, target)
    assert report["visual_changes_prediction"] is True
    assert report["query_changes_prediction"] is True
    assert report["shuffle_changes_visual"] is True
    assert isinstance(report["matched"], float)


def test_collisions_neighborhood_and_priors() -> None:
    distinct = torch.eye(4)
    copied = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert exact_collision_pairs(distinct) == 0
    assert exact_collision_pairs(copied) == 1
    assert collision_action(copied) == "switch_encoder_or_prepooling_tokens"
    assert collision_action(distinct) == "adapter_may_reshape_near_collisions"
    assert neighborhood_mean_cosine(distinct, k=1) == 0.0
    adapter = ResidualTargetAdapter(dim=2, hidden=4)
    outputs = adapter(copied)
    assert torch.equal(outputs[0], outputs[1])
    assert answer_prior_l1(["dog", "man"], ["man", "dog"]) == 0.0
    assert answer_prior_l1(["dog", "dog"], ["dog", "man"]) > 0.0


def test_mechanism_claims_follow_the_matched_baselines() -> None:
    synergy = mechanism_claims({"M0": 0, "M1": 1, "M2": 1, "M3": 5, "M4": 1, "M5": 99})
    assert synergy["m3_survives_strongest_matched_baseline"] is True
    assert synergy["claim_role_supervision_synergy"] is True
    assert synergy["claim_query_conditioning_essential"] is True
    assert synergy["claim_target_calibration_alone"] is False
    assert synergy["strongest_matched_baseline"] != "M5"
    calibration = mechanism_claims({"M0": 0, "M1": 6, "M2": 1, "M3": 4, "M4": 1, "M5": 0})
    assert calibration["claim_target_calibration_alone"] is True
    assert calibration["claim_role_supervision_synergy"] is False
    matched = mechanism_claims(
        {"M0": 0, "M1": 1, "M2": 1, "M3": 5.0, "M4": 5.0, "M5": 0},
        match_atol=0.1,
    )
    assert matched["claim_query_conditioning_essential"] is False
    assert matched["m3_survives_strongest_matched_baseline"] is False
    try:
        mechanism_claims({"M0": 1, "M3": 2})
    except KeyError:
        pass
    else:
        raise AssertionError("a partial mechanism screen must raise")


def test_bootstrap_interval_feeds_the_confirmation_rule() -> None:
    same = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tied = paired_bootstrap_ci(same, same, n_resamples=40, seed=0)
    assert tied["mean_difference"] == 0.0
    assert tied["ci_low"] <= 0.0 <= tied["ci_high"]
    lifted = paired_bootstrap_ci(same + 3.0, same, n_resamples=40, seed=1)
    assert lifted["mean_difference"] == 3.0
    assert lifted["ci_low"] > 0.0
    protected = {name: 0.0 for name in PREREGISTRATION["protected_r1_endpoints"]}
    assert (
        decide_confirmation(
            gain_points=lifted["mean_difference"],
            paired_ci_low=lifted["ci_low"],
            paired_ci_high=lifted["ci_high"],
            r1_degradation=protected,
        )
        == "proceed"
    )
    assert (
        decide_confirmation(
            gain_points=3.0,
            paired_ci_low=tied["ci_low"],
            paired_ci_high=tied["ci_high"],
            r1_degradation=protected,
        )
        == "stop"
    )


def test_launch_ledger_refuses_the_full_program() -> None:
    ledger = ComputeLedger()
    screen = ledger.authorize("target_screen", 6)
    assert screen["runs_executed"] == 0
    assert screen["authorized_total"] == 6
    try:
        ledger.authorize("target_screen", 1)
    except LaunchRefused:
        pass
    else:
        raise AssertionError("the target screen cap is 6")
    try:
        ledger.authorize("confirmation", 9)
    except LaunchRefused as exc:
        assert "development" in str(exc)
    else:
        raise AssertionError("confirmation waits for a development signal")
    ledger.note_development_signal(True)
    assert ledger.authorize("confirmation", 9)["cap"] == 9
    try:
        ledger.authorize("second_backbone", 6)
    except LaunchRefused:
        pass
    else:
        raise AssertionError("a second backbone waits for proceed")
    ledger.note_confirmation_decision("stop")
    try:
        ledger.authorize("second_backbone", 6)
    except LaunchRefused:
        pass
    else:
        raise AssertionError("stop does not open a second backbone")
    ledger.note_confirmation_decision("proceed")
    assert ledger.authorize("second_backbone", 6)["authorized_total"] == 6
    try:
        ledger.authorize_initial_program()
    except LaunchRefused as exc:
        assert "27" in str(exc)
    else:
        raise AssertionError("the 27 runs stay unauthorized as a block")
    assert_gqa_screening_cap(images=5000, pairs=20000)
    try:
        assert_gqa_screening_cap(images=5001, pairs=10)
    except ValueError:
        pass
    else:
        raise AssertionError("the GQA image cap is 5000")
    try:
        assert_gqa_screening_cap(images=10, pairs=20001)
    except ValueError:
        pass
    else:
        raise AssertionError("the GQA pair cap is 20000")
