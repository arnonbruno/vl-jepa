"""Confirmation locks. Synthetic output may be visible. Real benchmarks stay closed.

Three records freeze at different times. A dummy hash or an unresolved
placeholder is not a freeze. The chosen candidate is not invented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.protocol.preregistration import PREREGISTRATION

PLACEHOLDER_HASHES = frozenset({"", "unresolved", "placeholder", "dummy", "todo", "none"})
SYNTHETIC_OUTPUT_STATE = "visible"
REAL_BENCHMARK_STATE = "unopened"


class FreezeRefused(ValueError):
    pass


def _real_hash(value: str, label: str) -> str:
    text = (value or "").strip().lower()
    if text in PLACEHOLDER_HASHES or len(text) != 64:
        raise FreezeRefused(f"{label} is not a real sha256")
    return text


@dataclass
class EvaluationProtocolRecord:
    """Immutable before the relevant final results are accessed."""

    metrics: tuple[str, ...]
    splits: tuple[str, ...]
    selection_rule: str
    status: str = "open"
    thresholds: dict = field(default_factory=lambda: dict(PREREGISTRATION))

    def freeze(self) -> "EvaluationProtocolRecord":
        if self.status == "frozen":
            return self
        if not self.metrics or not self.splits or not self.selection_rule:
            raise FreezeRefused("the evaluation protocol is incomplete")
        if self.thresholds.get("status") != "preregistered_not_unblinded":
            raise FreezeRefused("real confirmation benchmarks stay unopened")
        return EvaluationProtocolRecord(
            metrics=tuple(self.metrics),
            splits=tuple(self.splits),
            selection_rule=self.selection_rule,
            status="frozen",
            thresholds=dict(self.thresholds),
        )


@dataclass
class ConfirmationExperimentConfig:
    """Immutable after development selection and before confirmation training."""

    methods: tuple[str, ...]
    target_artifact_sha256: str
    adapter_artifact_sha256: str
    hyperparameters_sha256: str
    seeds: tuple[int, ...]
    status: str = "open"

    def freeze(self) -> "ConfirmationExperimentConfig":
        if set(self.methods) != {"candidate", "matched_baseline", "explanatory_ablation"}:
            raise FreezeRefused("confirmation needs the candidate, matched baseline, and ablation")
        if list(self.seeds) != [0, 1, 2]:
            raise FreezeRefused("confirmation seeds are 0, 1, and 2")
        return ConfirmationExperimentConfig(
            methods=tuple(self.methods),
            target_artifact_sha256=_real_hash(self.target_artifact_sha256, "target artifact"),
            adapter_artifact_sha256=_real_hash(self.adapter_artifact_sha256, "adapter artifact"),
            hyperparameters_sha256=_real_hash(self.hyperparameters_sha256, "hyperparameters"),
            seeds=(0, 1, 2),
            status="frozen",
        )


@dataclass
class EvaluationCheckpointManifest:
    """Immutable before final benchmark evaluation."""

    checkpoint_sha256_by_run: Mapping[str, str]
    status: str = "open"

    def freeze(self) -> "EvaluationCheckpointManifest":
        if len(self.checkpoint_sha256_by_run) != 9:
            raise FreezeRefused("the checkpoint manifest needs one hash for each of the nine runs")
        cleaned = {
            run_id: _real_hash(digest, run_id)
            for run_id, digest in self.checkpoint_sha256_by_run.items()
        }
        return EvaluationCheckpointManifest(checkpoint_sha256_by_run=cleaned, status="frozen")


def rehearsal_visibility() -> dict[str, str]:
    """Synthetic rows can be inspected. Real confirmation benchmarks cannot."""
    return {
        "synthetic_outputs": SYNTHETIC_OUTPUT_STATE,
        "real_confirmation_benchmarks": REAL_BENCHMARK_STATE,
        "real_status": str(PREREGISTRATION["status"]),
        "confirmation_runs_executed": str(PREREGISTRATION["confirmation_runs_executed"]),
    }
