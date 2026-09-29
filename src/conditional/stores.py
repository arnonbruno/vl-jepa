"""Fingerprinted caches and a sentinel that is valid only by its contents.

The production sentinel is not created here. A file that merely exists, or a
file whose fingerprint, ids, shard checksums, or live comparison are wrong,
does not authorize training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.conditional.cache import (
    CacheFingerprint,
    ShardedTokenCache,
    max_abs_online_vs_cache,
)
from src.conditional.vision_contract import CLIP_VITB16_SPATIAL, extract_spatial_tokens
from src.protocol.hashing import sha256_file, sha256_json

PRODUCTION_SENTINEL_NAME = "visual_cache_sentinel.json"
CACHE_KINDS = ("visual", "captions", "answers", "queries")
LIVE_PARITY_ATOL = 5e-2


class SentinelRejected(RuntimeError):
    pass


def production_sentinel_path(root: Path) -> Path:
    return Path(root) / PRODUCTION_SENTINEL_NAME


def visual_fingerprint(*, checkpoint_sha256: str, manifest_sha256: str) -> CacheFingerprint:
    contract = CLIP_VITB16_SPATIAL
    return CacheFingerprint(
        weight_revision=checkpoint_sha256,
        extraction_layer=str(contract["extraction_layer"]),
        token_order=str(contract["token_order"]),
        spatial_positions=str(contract["positional_treatment"]),
        resolution=int(contract["image_size"]),
        preprocessing=str(contract["preprocessing"]),
        precision="float16",
        manifest_sha256=manifest_sha256,
    )


def text_fingerprint(
    *,
    kind: str,
    checkpoint_sha256: str,
    tokenizer: str,
    prompt: str,
    pooling: str,
    projection: str,
    truncation: str,
    normalization: str,
    manifest_sha256: str,
) -> CacheFingerprint:
    if kind not in {"captions", "answers", "queries"}:
        raise ValueError(f"text cache kind must be captions, answers, or queries, got {kind!r}")
    recipe = {
        "tokenizer": tokenizer,
        "prompt": prompt,
        "truncation": truncation,
        "normalization": normalization,
        "inference_precision": "float32_or_bfloat16",
    }
    return CacheFingerprint(
        weight_revision=checkpoint_sha256,
        extraction_layer=kind,
        token_order=pooling,
        spatial_positions=projection,
        resolution=0,
        preprocessing=sha256_json(recipe),
        precision="float16",
        manifest_sha256=manifest_sha256,
    )


def build_visual_cache_from_encoder(
    encoder: torch.nn.Module,
    images: torch.Tensor,
    ids: list[str],
    root: Path,
    fingerprint: CacheFingerprint,
    *,
    sources: set[str],
) -> dict:
    """Extract real spatial tokens and store them. Does not write a sentinel."""
    if "coco" not in sources or "gqa" not in sources:
        raise ValueError("the visual cache needs both a COCO manifest and a GQA training manifest")
    if any(not image_id.startswith(("coco:", "gqa:")) for image_id in ids):
        raise ValueError("visual cache ids must be source-qualified coco: or gqa: ids")
    if not any(image_id.startswith("coco:") for image_id in ids) or not any(
        image_id.startswith("gqa:") for image_id in ids
    ):
        raise ValueError("the visual cache must contain both COCO and GQA rows")
    tokens = extract_spatial_tokens(encoder, images)
    array = tokens.detach().cpu().numpy()
    cache = ShardedTokenCache(
        root,
        fingerprint,
        n_tokens=int(CLIP_VITB16_SPATIAL["n_tokens"]),
        dim=int(CLIP_VITB16_SPATIAL["dim"]),
        shard_rows=2,
    )
    cache.write(ids, array)
    live = extract_spatial_tokens(encoder, images[:1]).detach().cpu().numpy()
    cached = cache.reader().get(ids[0])[None, ...]
    if not np.array_equal(live.astype(np.float16), cached.astype(np.float16)):
        raise RuntimeError("cached visual tokens differ from a second live forward")
    delta = max_abs_online_vs_cache(live, cached)
    if delta > LIVE_PARITY_ATOL:
        raise RuntimeError(f"cached visual tokens differ from a live forward by {delta}")
    return {"root": str(root), "digest": fingerprint.digest(), "ids": list(ids), "cached_vs_live_max_abs": delta}


def write_vector_cache(
    root: Path,
    fingerprint: CacheFingerprint,
    ids: list[str],
    vectors: np.ndarray,
) -> dict:
    if vectors.ndim != 2:
        raise ValueError("vector cache rows are (N, D)")
    tokens = vectors[:, None, :]
    cache = ShardedTokenCache(root, fingerprint, n_tokens=1, dim=int(vectors.shape[1]), shard_rows=8)
    cache.write(ids, tokens)
    return {"root": str(root), "digest": fingerprint.digest(), "ids": list(ids)}


def shard_checksums(root: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(Path(root).glob("shard_*.f16")):
        checksums[path.name] = sha256_file(path)
    if not checksums:
        raise SentinelRejected(f"no cache shards under {root}")
    return checksums


def _load_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SentinelRejected(f"empty sentinel {path}")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SentinelRejected(f"sentinel {path} is not json") from exc
    if not isinstance(payload, dict):
        raise SentinelRejected("sentinel must be a JSON object")
    return payload


def validate_sentinel(path: Path, *, live_atol: float = LIVE_PARITY_ATOL) -> dict:
    """Authorize only a complete, matching, checksummed sentinel."""
    path = Path(path)
    if not path.is_file():
        raise SentinelRejected(f"sentinel is absent at {path}")
    payload = _load_json(path)
    if payload.get("role") != "visual_cache_sentinel":
        raise SentinelRejected("sentinel role is not visual_cache_sentinel")
    caches = payload.get("caches")
    if not isinstance(caches, dict) or set(caches) != set(CACHE_KINDS):
        raise SentinelRejected("sentinel must name visual, captions, answers, and queries caches")
    for kind, entry in caches.items():
        root = Path(entry["root"])
        meta_path = root / "cache_fingerprint.json"
        if not meta_path.is_file():
            raise SentinelRejected(f"{kind} cache is missing its fingerprint")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("digest") != entry.get("digest"):
            raise SentinelRejected(f"{kind} cache fingerprint is stale")
        ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
        if list(ids) != list(entry.get("ids", [])):
            raise SentinelRejected(f"{kind} cache ids do not match the sentinel")
        recorded = entry.get("shard_sha256")
        if shard_checksums(root) != recorded:
            raise SentinelRejected(f"{kind} shard checksums do not match the sentinel")
    if float(payload.get("cached_vs_live_max_abs", 1.0)) > live_atol:
        raise SentinelRejected("cached-versus-live comparison is missing or above tolerance")
    return payload


def authorizes_training(path: Path) -> tuple[bool, str]:
    try:
        validate_sentinel(path)
    except SentinelRejected as exc:
        return False, str(exc)
    return True, "sentinel matches fingerprint, ids, shards, and the live comparison"


def write_test_sentinel(path: Path, caches: dict[str, dict], *, cached_vs_live_max_abs: float) -> None:
    """Write a sentinel for a temporary directory. Not the production file."""
    payload_caches = {}
    for kind, entry in caches.items():
        root = Path(entry["root"])
        payload_caches[kind] = {
            "root": str(root),
            "digest": entry["digest"],
            "ids": list(entry["ids"]),
            "shard_sha256": shard_checksums(root),
        }
    payload = {
        "role": "visual_cache_sentinel",
        "caches": payload_caches,
        "cached_vs_live_max_abs": cached_vs_live_max_abs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def assert_production_sentinel_absent(root: Path) -> None:
    path = production_sentinel_path(root)
    if path.exists():
        raise RuntimeError(f"production sentinel must stay absent during this build: {path}")
