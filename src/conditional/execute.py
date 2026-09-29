"""Dry-run target-screen path.

Real COCO or GQA training is not invoked by this module's default. A run starts
only when execution is authorized, dry-run is off, and preflight has no
failures. Synthetic invocations stay rehearsal artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from src.conditional.batches import (
    MissingExclusionManifest,
    assert_shared_arm_contract,
    assert_screen_gqa_cap,
    interleaved_task_schedule,
    load_exclusion_manifest,
)
from src.conditional.encoders import WeightsUnavailable, assert_query_pin, require_screen_weights
from src.conditional.stores import authorizes_training

TrainerFn = Callable[..., Any]


def real_integration_status(*, weight_test_ran: bool) -> str:
    """A skipped weight-backed test is unverified, not a pass."""
    if weight_test_ran:
        return "checked"
    return "unverified"


def openai_clip_weights_present() -> bool:
    cache = Path.home() / ".cache" / "clip"
    if not cache.is_dir():
        return False
    return any(path.is_file() and path.stat().st_size > 1_000_000 for path in cache.iterdir())


def _preflight(assets: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    scope = assets.get("scope")
    if scope not in {"synthetic_screen_not_a_result", "real_target_screen"}:
        reasons.append("unknown_scope")
    if assets.get("dry_run", True):
        reasons.append("dry_run")
    if not assets.get("authorize_execution", False):
        reasons.append("execution_not_authorized")
    try:
        require_screen_weights(assets.get("weights_available") or {})
    except WeightsUnavailable:
        reasons.append("weights_unavailable")
    try:
        assert_query_pin(
            str(assets.get("query_checkpoint_sha256", "")),
            str(assets.get("query_preprocessing", "")),
        )
    except ValueError:
        reasons.append("query_encoder_unpinned")
    try:
        assert_screen_gqa_cap(
            images=int(assets.get("gqa_images", -1)),
            pairs=int(assets.get("gqa_pairs", -1)),
        )
    except (TypeError, ValueError):
        reasons.append("gqa_cap")
    try:
        expected = interleaved_task_schedule(int(assets["caption_steps"]), int(assets["qa_steps"]))
    except (KeyError, TypeError, ValueError):
        reasons.append("schedule_not_interleaved")
    else:
        if list(assets.get("schedule", [])) != expected:
            reasons.append("schedule_not_interleaved")
    try:
        assert_shared_arm_contract(list(assets.get("arms") or []))
    except (TypeError, ValueError):
        reasons.append("arms_not_shared")
    if scope == "real_target_screen":
        try:
            load_exclusion_manifest(assets.get("exclusion_manifest"))
        except MissingExclusionManifest:
            reasons.append("missing_exclusion_manifest")
        sentinel = assets.get("sentinel")
        if sentinel is None:
            reasons.append("sentinel_rejected")
        else:
            allowed, _message = authorizes_training(Path(sentinel))
            if not allowed:
                reasons.append("sentinel_rejected")
    return reasons


def run_target_screen_adapter(
    assets: Mapping[str, Any],
    trainer_fn: TrainerFn | None = None,
) -> dict[str, Any]:
    """Walk manifests, caches, batches, and the trainer. Default is a dry run."""
    reasons = _preflight(assets)
    blocked = {
        "runs_executed": 0,
        "research_runs_executed": 0,
        "trainer_invocations": 0,
        "reasons": reasons,
        "wrote_final_results": False,
        "development_score": None,
        "checkpoint_sha256": None,
    }
    if reasons:
        status = "dry_run" if reasons == ["dry_run"] else "blocked"
        return {"status": status, **blocked}
    if trainer_fn is None:
        raise RuntimeError("authorized execution has no trainer")
    scope = str(assets["scope"])
    schedule = list(assets["schedule"])
    completed = 0
    development_score = None
    checkpoint_sha256 = None
    error = None
    for arm in assets["arms"]:
        try:
            outcome = trainer_fn(
                arm,
                schedule,
                assets.get("caption_batch"),
                assets.get("qa_batch"),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        completed += 1
        if isinstance(outcome, dict):
            if "development_score" in outcome:
                development_score = outcome["development_score"]
            if outcome.get("checkpoint_sha256"):
                checkpoint_sha256 = outcome["checkpoint_sha256"]
    research_runs = 0 if scope == "synthetic_screen_not_a_result" else completed
    status = "rehearsal" if scope == "synthetic_screen_not_a_result" and error is None else "executed"
    if error is not None:
        status = "failed"
    record = {
        "status": status,
        "scope": scope,
        "runs_executed": research_runs,
        "research_runs_executed": research_runs,
        "trainer_invocations": completed,
        "reasons": [] if error is None else [error],
        "wrote_final_results": False,
        "development_score": development_score,
        "checkpoint_sha256": checkpoint_sha256,
        "schedule": schedule,
    }
    run_dir = assets.get("run_dir")
    if run_dir is not None:
        import json

        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        final_results = destination / "final_results.csv"
        if final_results.exists():
            raise RuntimeError("the screen adapter must not write final_results.csv")
    return record
