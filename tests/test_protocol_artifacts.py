"""Run directories, checkpoints, and the historical hash pins."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from src.protocol.artifacts import (
    append_registry,
    capture_rng_state,
    create_run_dir,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from src.protocol.historical import unresolved_checkpoint_paths, verify_historical_hashes

ROOT = Path(__file__).resolve().parents[1]


def test_historical_hashes_and_unresolved_vitl_backend() -> None:
    found = verify_historical_hashes(ROOT)
    assert found["experiments/exp_baseline_vitb16_zeroshot.json"].startswith("15aa4c29")
    assert found["experiments/exp_vljepa_vitb16_robust.json"].startswith("58962f8e")
    backends = unresolved_checkpoint_paths(ROOT)
    assert len(set(backends.values())) == 1
    assert "exp_jepa_1024d_20ep/checkpoint_best.pt" in next(iter(backends.values()))


def test_run_directory_and_registry_reject_duplicates(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path, "phase0-reference-seed0")
    assert first.is_dir()
    try:
        create_run_dir(tmp_path, "phase0-reference-seed0")
    except FileExistsError:
        pass
    else:
        raise AssertionError("run directory was overwritten")
    row = {
        "run_id": "phase0-reference-seed0",
        "git_commit": "abc",
        "dirty": True,
        "dirty_sha256": "0" * 64,
        "seed": 0,
        "config_sha256": "1" * 64,
        "scope": "phase0_reference_not_a_benchmark",
        "evaluated_state": "raw",
        "checkpoint_sha256": "2" * 64,
        "trainable_params": 1,
        "total_params": 1,
        "image_exposures": 0,
        "caption_exposures": 0,
        "microbatch": 4,
        "optimizer_batch": 4,
        "candidate_pool": 4,
        "positive_rule": "identity_synthetic_captions",
        "precision": "fp32",
        "context_length": 77,
        "target_dim": 512,
        "pooling": "query_token_mean",
    }
    csv_path = tmp_path / "experiment_registry.csv"
    append_registry(csv_path, row)
    try:
        append_registry(csv_path, row)
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate run id was appended")


def test_checkpoint_keeps_student_and_evaluated_states_distinct(tmp_path: Path) -> None:
    student = {"w": torch.tensor([1.0, 2.0])}
    evaluated = {"w": torch.tensor([3.0, 4.0])}
    path = tmp_path / "ckpt.pt"
    digest = save_checkpoint(
        path,
        student_state=student,
        evaluated_state=evaluated,
        evaluated_state_name="raw",
        optimizer_state={"step": 1},
        scheduler_state={"last_step": 1},
        scaler_state=None,
        rng_state=capture_rng_state(),
        step=4,
        config={"family": "conditional_latent_predictor"},
    )
    loaded = load_checkpoint(path)
    assert loaded["evaluated_equals_student"] is False
    assert torch.equal(loaded["student_state"]["w"], student["w"])
    assert torch.equal(loaded["evaluated_state"]["w"], evaluated["w"])
    assert "ema" not in loaded
    assert len(digest) == 64


def test_rng_roundtrip() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    state = capture_rng_state()
    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(2)
    restore_rng_state(state)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    assert random.random() != 0  # the restored stream is the one after the initial seed
    restore_rng_state(state)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second
