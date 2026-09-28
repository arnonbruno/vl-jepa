"""Write manifests/ from the local Karpathy and COCO annotation files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocol.manifests import write_manifests


def main() -> None:
    summary = write_manifests(ROOT)
    counts = summary["counts"]
    print(
        "clean_train",
        counts["clean_train"],
        "clean_dev",
        counts["clean_dev"],
        "karpathy_test_in_train2017",
        summary["overlap"]["karpathy_test_in_train2017"],
    )


if __name__ == "__main__":
    main()
