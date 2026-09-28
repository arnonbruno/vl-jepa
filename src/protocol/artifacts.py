"""Non-overwritable run directories and the experiment registry.

A run id maps to one directory. Creating it twice raises. The registry CSV
rejects a repeated run id. Student weights and the evaluated weight mixture
are stored as separate checkpoint entries even when they are equal, which is
the case for this model family: there is no EMA and no WiSE-FT.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.protocol.hashing import sha256_bytes, sha256_file, sha256_json

REGISTRY_FIELDS = [
    "run_id",
    "git_commit",
    "dirty",
    "dirty_sha256",
    "seed",
    "config_sha256",
    "scope",
    "evaluated_state",
    "checkpoint_sha256",
    "trainable_params",
    "total_params",
    "image_exposures",
    "caption_exposures",
    "microbatch",
    "optimizer_batch",
    "candidate_pool",
    "positive_rule",
    "precision",
    "context_length",
    "target_dim",
    "pooling",
]


def git_state(cwd: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> bytes:
        return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL)

    commit = _run(["git", "rev-parse", "HEAD"]).decode().strip()
    status = _run(["git", "status", "--porcelain"])
    diff = _run(["git", "diff", "HEAD"])
    dirty = bool(status.strip())
    return {
        "commit": commit,
        "dirty": dirty,
        "status_porcelain": status.decode("utf-8", errors="replace"),
        "dirty_sha256": sha256_bytes(diff + b"\n" + status),
    }


def create_run_dir(root: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"invalid run id: {run_id!r}")
    path = Path(root) / run_id
    if path.exists():
        raise FileExistsError(f"run directory already exists and will not be overwritten: {path}")
    path.mkdir(parents=True)
    return path


def append_registry(csv_path: Path, row: Mapping[str, Any]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                existing.add(record["run_id"])
    run_id = str(row["run_id"])
    if run_id in existing:
        raise FileExistsError(f"run id {run_id} is already in {csv_path}")
    missing = [field for field in REGISTRY_FIELDS if field not in row]
    if missing:
        raise KeyError(f"registry row missing {missing}")
    write_header = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in REGISTRY_FIELDS})


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    student_state: dict[str, torch.Tensor],
    evaluated_state: dict[str, torch.Tensor],
    evaluated_state_name: str,
    optimizer_state: dict,
    scheduler_state: dict | None,
    scaler_state: dict | None,
    rng_state: Mapping[str, Any],
    step: int,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Write student and evaluated weights as distinct entries. Return sha256."""
    if evaluated_state_name not in {"raw", "ema", "wise_ft"}:
        raise ValueError(
            f"evaluated_state_name must be raw, ema, or wise_ft, got {evaluated_state_name!r}"
        )
    payload: dict[str, Any] = {
        "student_state": student_state,
        "evaluated_state": evaluated_state,
        "evaluated_state_name": evaluated_state_name,
        "evaluated_equals_student": _states_equal(student_state, evaluated_state),
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "rng_state": rng_state,
        "step": int(step),
        "config": dict(config),
    }
    if extra:
        overlap = set(extra) & set(payload)
        if overlap:
            raise KeyError(f"extra checkpoint keys collide with reserved names: {sorted(overlap)}")
        payload["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256_file(path)


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a = left[key]
        b = right[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            if a != b:
                return False
            continue
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            return False
    return True


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    for key in ("student_state", "evaluated_state", "evaluated_state_name", "optimizer_state", "rng_state", "config"):
        if key not in payload:
            raise KeyError(f"checkpoint {path} missing {key}")
    return payload


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_json(dict(config))
