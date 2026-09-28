"""SHA-256 pins for result JSON committed at ``dfeaaf6``.

The two ViT-L files name one checkpoint path and store different scores.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.protocol.hashing import sha256_file

# The one baseline pair kept for regression. Protocol B COCO val2017 numbers.
# Not a Karpathy-test result. See the overlap audit.
BASELINE_FILES = {
    "experiments/exp_baseline_vitb16_zeroshot.json":
        "15aa4c2973a683b115709739a59071c3413c9d8e50d2fe81006ffaec7561adc9",
    "experiments/exp_vljepa_vitb16_robust.json":
        "58962f8e512319ee0e02f3802c2a0da13d7b4c3a6f7721810a320cb20aa69e2c",
}

# Same backend path, different scores. Do not pick a winner.
UNRESOLVED_VITL_FILES = {
    "experiments/exp_vljepa_vitl14_robust.json":
        "f5d63100f86156614315d40f81d7ff4566313b50211ca6b4f125162fc8f5bdf9",
    "experiments/exp_vljepa_vitl14_siglip.json":
        "dadcdd7d54e7a426906fccb8e336d74cc752539e5c88a339c7775384d719286b",
    "experiments/exp_baseline_vitl14_zeroshot.json":
        "8c0899028aaf6d172a7ed1ee1a31db93aba6723ce7ffb51b88c310b3c107b78b",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_historical_hashes(root: Path | None = None) -> dict[str, str]:
    """Return sha256 for every pinned file. Raise if a pin does not match."""
    root = repo_root() if root is None else Path(root)
    found: dict[str, str] = {}
    pins = {**BASELINE_FILES, **UNRESOLVED_VITL_FILES}
    for relative, expected in pins.items():
        path = root / relative
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"{relative} sha256 {digest} does not match the pinned {expected}. "
                "Do not relabel historical result files in place."
            )
        found[relative] = digest
    return found


def unresolved_checkpoint_paths(root: Path | None = None) -> dict[str, str]:
    """Backend strings recorded in the conflicting ViT-L JSON files."""
    root = repo_root() if root is None else Path(root)
    out: dict[str, str] = {}
    for relative in (
        "experiments/exp_vljepa_vitl14_robust.json",
        "experiments/exp_vljepa_vitl14_siglip.json",
    ):
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        out[relative] = str(payload.get("backend", ""))
    return out
