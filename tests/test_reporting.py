"""Handoff text stays labeled. Historical rows are copied, not relabeled."""

from __future__ import annotations

import json
from pathlib import Path

from src.protocol.reporting import (
    render_blinded_analysis,
    render_integrity_report,
    render_mechanism_screen,
    render_target_diagnostics,
    write_final_results,
)

ROOT = Path(__file__).resolve().parents[1]


def test_report_language_and_thresholds() -> None:
    summary = {
        "counts": {
            "clean_train": 113287,
            "karpathy_coco_train": 82783,
            "karpathy_coco_restval": 30504,
            "clean_dev": 5000,
            "karpathy_coco_test": 5000,
            "removed_from_train": 0,
            "flickr30k_val": 1014,
            "flickr30k_test": 1000,
        },
        "overlap": {
            "karpathy_test_in_train2017": 4407,
            "train2017_equals_all_karpathy_ids": False,
            "train2017_equals_karpathy_minus_val2017": True,
            "train2017_val2017_overlap": 0,
            "karpathy_ids_outside_coco2017": 0,
        },
        "coco_sentence_histogram": {5: 122959, 6: 324, 7: 4},
        "flickr_sentence_histogram": {5: 31014},
    }
    reference = {
        "run_id": "phase0-reference-seed0",
        "scope": "phase0_reference_not_a_benchmark",
        "seed": 0,
        "git_commit": "dfeaaf6",
        "dirty": True,
        "trainable_params": 10,
        "checkpoint_sha256": "ab",
        "versions": {"torch": "recorded-at-runtime"},
    }
    text = render_integrity_report(
        summary=summary,
        reference=reference,
        historical_ok=True,
        vitl_backends={
            "experiments/exp_vljepa_vitl14_robust.json": "experiments/exp_jepa_1024d_20ep/checkpoint_best.pt",
            "experiments/exp_vljepa_vitl14_siglip.json": "experiments/exp_jepa_1024d_20ep/checkpoint_best.pt",
        },
    )
    assert "Karpathy-derived, benchmark-disjoint training set" in text
    assert "restval is included" in text
    assert "not a clean Karpathy-test model" in text
    assert "selection_accepted" in text
    assert "unresolved" in text
    assert "not claimed to be disjoint" in text
    assert "preregistered_not_unblinded" in text
    blinded = render_blinded_analysis()
    assert "preregistered_not_unblinded" in blinded
    assert "2.0" in blinded or "2 points" in blinded or "2.0 points" in blinded or "least 2" in blinded
    assert "has been trained" in render_target_diagnostics()
    mechanism = render_mechanism_screen()
    assert "M5" in mechanism
    assert "not_trained" in mechanism


def test_final_results_keep_historical_protocol(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    for name in (
        "exp_baseline_vitb16_zeroshot.json",
        "exp_vljepa_vitb16_robust.json",
    ):
        source = ROOT / "experiments" / name
        (experiments / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    reference = {
        "run_id": "phase0-reference-seed0",
        "scope": "phase0_reference_not_a_benchmark",
        "seed": 0,
        "losses": [1.0, 0.5],
        "seconds": 0.25,
    }
    write_final_results(tmp_path, reference)
    text = (tmp_path / "final_results.csv").read_text(encoding="utf-8")
    assert "not_recorded" in text
    assert "historical_protocol_b_coco_val2017" in text
    assert "coco_val2017_5k_not_karpathy_test" in text
    assert "not_launched" in text
    assert "synthetic_harness_not_a_result" not in text
    robust = json.loads((experiments / "exp_vljepa_vitb16_robust.json").read_text(encoding="utf-8"))
    assert str(robust["results"]["5k"]["rsum"]) in text or repr(robust["results"]["5k"]["rsum"]) in text
