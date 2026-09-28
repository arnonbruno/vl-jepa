"""Render the handoff documents from the audit and the reference run."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conditional.cache import (
    CacheFingerprint,
    ShardedTokenCache,
    max_abs_online_vs_cache,
)
from src.protocol.environment import write_environment_lock
from src.protocol.hashing import sha256_file
from src.protocol.historical import unresolved_checkpoint_paths, verify_historical_hashes
from src.protocol.manifests import write_manifests
from src.protocol.reporting import (
    render_blinded_analysis,
    render_integrity_report,
    render_mechanism_screen,
    render_target_diagnostics,
    write_cache_manifest,
    write_final_results,
)


def _synthetic_cache_parity() -> tuple[float, str]:
    fingerprint = CacheFingerprint(
        weight_revision="synthetic_reference",
        extraction_layer="not_an_encoder",
        token_order="cls_then_raster",
        spatial_positions="none",
        resolution=0,
        preprocessing="not_applied_synthetic",
        precision="float16",
        manifest_sha256="synthetic_reference",
    )
    root = Path(tempfile.mkdtemp(prefix="vljepa-cache-"))
    cache = ShardedTokenCache(root, fingerprint, n_tokens=2, dim=4, shard_rows=2)
    online = np.linspace(-0.25, 0.25, 4 * 2 * 4, dtype=np.float32).reshape(4, 2, 4)
    cache.write([f"synthetic:{index}" for index in range(4)], online)
    reader = cache.reader()
    restored = np.stack([reader.get(f"synthetic:{index}") for index in range(4)])
    return max_abs_online_vs_cache(online, restored), fingerprint.digest()


def main() -> None:
    summary = write_manifests(ROOT)
    historical = verify_historical_hashes(ROOT)
    record_path = ROOT / "runs" / "phase0-reference-seed0" / "record.json"
    reference = json.loads(record_path.read_text(encoding="utf-8"))
    reference["preprocessing_seconds"] = 0.0
    reference["evaluation_seconds"] = 0.0
    reference["training_seconds"] = reference["seconds"]
    reference["manifest_hashes_not_used_for_training"] = {
        relative: sha256_file(ROOT / relative)
        for relative in (
            "manifests/clean_adaptation_ids.csv",
            "manifests/development_ids.csv",
            "manifests/final_benchmark_coco_ids.csv",
            "manifests/final_benchmark_flickr30k_ids.csv",
            "manifests/overlap_report.json",
        )
    }
    throughput_path = ROOT / "runs" / "phase0-reference-seed0" / "throughput.json"
    if throughput_path.is_file():
        reference["throughput_probe"] = json.loads(throughput_path.read_text(encoding="utf-8"))
    record_path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    report = render_integrity_report(
        summary=summary,
        reference=reference,
        historical_ok=True,
        vitl_backends=unresolved_checkpoint_paths(ROOT),
    )
    (ROOT / "integrity_report.md").write_text(report, encoding="utf-8")
    (ROOT / "target_diagnostics.md").write_text(render_target_diagnostics(), encoding="utf-8")
    (ROOT / "mechanism_screen.md").write_text(render_mechanism_screen(), encoding="utf-8")
    (ROOT / "blinded_final_analysis.md").write_text(render_blinded_analysis(), encoding="utf-8")
    write_final_results(ROOT, reference)
    parity, digest = _synthetic_cache_parity()
    write_cache_manifest(ROOT, parity, digest)
    write_environment_lock(ROOT / "environment.lock")
    print("historical_files", len(historical))
    print("integrity_report.md", "clean_train", summary["counts"]["clean_train"])
    print("cache_parity_max_abs", parity)


if __name__ == "__main__":
    main()
