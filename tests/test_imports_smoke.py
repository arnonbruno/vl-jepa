"""Dependencies the new protocol imports, and gitignore exceptions."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_clean_imports() -> None:
    import huggingface_hub  # noqa: F401
    import open_clip  # noqa: F401
    import src.conditional
    import src.conditional.cache
    import src.conditional.clip_parity
    import src.conditional.diagnostics
    import src.conditional.losses
    import src.conditional.mechanism
    import src.conditional.model
    import src.conditional.reference
    import src.conditional.screen
    import src.conditional.targets
    import src.conditional.trainer
    import src.protocol.artifacts
    import src.protocol.environment
    import src.protocol.evaluator
    import src.protocol.historical
    import src.protocol.manifests
    import src.protocol.reporting
    import src.protocol.splits
    import src.protocol.tasks

    assert src.conditional.__doc__


def _ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def test_protocol_artifacts_are_not_gitignored() -> None:
    assert not _ignored("manifests/clean_adaptation_ids.csv")
    assert not _ignored("manifests/overlap_report.json")
    assert not _ignored("manifests/licenses.md")
    assert not _ignored("experiment_registry.csv")
    assert not _ignored("final_results.csv")
    assert not _ignored("cache_manifest.json")
    assert not _ignored("runs/phase0-reference-seed0/record.json")
    assert _ignored("runs/phase0-reference-seed0/checkpoint.pt")
    assert _ignored("data/external/dataset_coco.json")
