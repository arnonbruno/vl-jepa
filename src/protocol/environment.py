"""Record the Python packages this project imports."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
from pathlib import Path

TRACKED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "numpy",
    "open_clip_torch",
    "transformers",
    "huggingface_hub",
    "timm",
    "pycocotools",
    "pillow",
    "pyyaml",
)


def distribution_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def write_environment_lock(path: Path) -> dict[str, str | None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    versions = distribution_versions()
    lines = [
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        "",
    ]
    for name, version in versions.items():
        lines.append(f"{name}={version}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return versions
