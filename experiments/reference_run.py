"""Seed, step, and record the synthetic phase-0 reference run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conditional.reference import gpu_throughput_probe, run_phase0_reference


def main() -> None:
    record = run_phase0_reference(ROOT)
    print(
        record["run_id"],
        "params",
        record["trainable_params"],
        "seconds",
        round(record["seconds"], 3),
        "sha256",
        record["checkpoint_sha256"],
    )
    probe = gpu_throughput_probe(ROOT, record["run_id"])
    print("throughput", probe["status"], probe.get("seconds_per_step"))


if __name__ == "__main__":
    main()
