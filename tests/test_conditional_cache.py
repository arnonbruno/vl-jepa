"""Float16 shard cache, fingerprint refusal, and the storage estimate."""

from __future__ import annotations

import numpy as np
import pytest

from src.conditional.cache import (
    FEATURE_DROPOUT_MASKS_BEFORE_ENCODER,
    CacheFingerprint,
    CacheFingerprintMismatch,
    ShardedTokenCache,
    gibibytes,
    max_abs_online_vs_cache,
    vision_cache_bytes,
    caption_target_bytes,
)


def _fingerprint(**overrides) -> CacheFingerprint:
    payload = dict(
        weight_revision="synthetic-encoder",
        extraction_layer="tokens",
        token_order="cls_then_raster",
        spatial_positions="2",
        resolution=224,
        preprocessing="not_applied_synthetic",
        precision="float16",
        manifest_sha256="abc",
    )
    payload.update(overrides)
    return CacheFingerprint(**payload)


def test_storage_estimate_matches_the_plan() -> None:
    assert 31.8 < gibibytes(vision_cache_bytes()) < 32.0
    assert 0.53 < gibibytes(caption_target_bytes()) < 0.55
    assert FEATURE_DROPOUT_MASKS_BEFORE_ENCODER is False


def test_cache_roundtrip_is_read_only_and_close_to_fp32(tmp_path) -> None:
    fingerprint = _fingerprint()
    cache = ShardedTokenCache(tmp_path, fingerprint, n_tokens=2, dim=4, shard_rows=2)
    online = np.linspace(-0.5, 0.5, 3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    cache.write(["a", "b", "c"], online)
    reader = cache.reader()
    assert reader.modes == ["r", "r"]
    restored = np.stack([reader.get("a"), reader.get("b"), reader.get("c")])
    assert max_abs_online_vs_cache(online, restored) < 1e-3
    with pytest.raises(ValueError):
        reader._mmaps[0][0, 0, 0] = np.float16(0)


def test_fingerprint_mismatch_raises(tmp_path) -> None:
    cache = ShardedTokenCache(tmp_path, _fingerprint(), n_tokens=1, dim=2, shard_rows=4)
    cache.write(["a"], np.zeros((1, 1, 2), dtype=np.float32))
    with pytest.raises(CacheFingerprintMismatch):
        ShardedTokenCache(tmp_path, _fingerprint(weight_revision="other"), n_tokens=1, dim=2).reader()
