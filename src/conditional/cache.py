"""Sharded float16 memmap cache for frozen encoder outputs.

Masking tokens after they have been written is feature dropout. It is not
evidence that the encoder never saw the corresponding patches. The constant
``FEATURE_DROPOUT_MASKS_BEFORE_ENCODER`` records that fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.protocol.hashing import sha256_json

FEATURE_DROPOUT_MASKS_BEFORE_ENCODER = False

# 113,287 Karpathy train+restval images, CLIP ViT-B/16 token grid, fp16.
STORAGE_IMAGES = 113_287
STORAGE_TOKENS = 197
STORAGE_VISION_DIM = 768
STORAGE_CAPTIONS_PER_IMAGE = 5
STORAGE_TARGET_DIM = 512


def vision_cache_bytes(
    n_images: int = STORAGE_IMAGES,
    n_tokens: int = STORAGE_TOKENS,
    dim: int = STORAGE_VISION_DIM,
    bytes_per_value: int = 2,
) -> int:
    return int(n_images) * int(n_tokens) * int(dim) * int(bytes_per_value)


def caption_target_bytes(
    n_images: int = STORAGE_IMAGES,
    captions_per_image: int = STORAGE_CAPTIONS_PER_IMAGE,
    dim: int = STORAGE_TARGET_DIM,
    bytes_per_value: int = 2,
) -> int:
    return int(n_images) * int(captions_per_image) * int(dim) * int(bytes_per_value)


def gibibytes(n_bytes: int) -> float:
    return n_bytes / float(1024**3)


@dataclass(frozen=True)
class CacheFingerprint:
    weight_revision: str
    extraction_layer: str
    token_order: str
    spatial_positions: str
    resolution: int
    preprocessing: str
    precision: str
    manifest_sha256: str

    def digest(self) -> str:
        return sha256_json(asdict(self))


class CacheFingerprintMismatch(ValueError):
    pass


class ShardedTokenCache:
    """Write float16 shards, then reopen them read-only."""

    def __init__(
        self,
        root: Path,
        fingerprint: CacheFingerprint,
        *,
        n_tokens: int,
        dim: int,
        shard_rows: int = 1024,
    ) -> None:
        if fingerprint.precision != "float16":
            raise ValueError("token cache stores float16")
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.n_tokens = int(n_tokens)
        self.dim = int(dim)
        self.shard_rows = int(shard_rows)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ids: list[str] = []

    def _meta_path(self) -> Path:
        return self.root / "cache_fingerprint.json"

    def _ids_path(self) -> Path:
        return self.root / "ids.json"

    def write(self, ids: list[str], tokens: np.ndarray) -> None:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.n_tokens, self.dim):
            raise ValueError(
                f"tokens must be (N, {self.n_tokens}, {self.dim}), got {tokens.shape}"
            )
        if len(ids) != tokens.shape[0]:
            raise ValueError("one id per token row")
        if len(set(ids)) != len(ids):
            raise ValueError("cache ids must be unique")
        stored = np.asarray(tokens, dtype=np.float32).astype(np.float16)
        self._ids = list(ids)
        for shard_index, start in enumerate(range(0, len(ids), self.shard_rows)):
            end = min(start + self.shard_rows, len(ids))
            path = self.root / f"shard_{shard_index:04d}.f16"
            mmap = np.memmap(
                path,
                dtype=np.float16,
                mode="w+",
                shape=(end - start, self.n_tokens, self.dim),
            )
            mmap[:] = stored[start:end]
            mmap.flush()
            del mmap
        meta = asdict(self.fingerprint)
        meta["digest"] = self.fingerprint.digest()
        meta["n_rows"] = len(ids)
        meta["n_tokens"] = self.n_tokens
        meta["dim"] = self.dim
        meta["shard_rows"] = self.shard_rows
        meta["feature_dropout_masks_before_encoder"] = FEATURE_DROPOUT_MASKS_BEFORE_ENCODER
        self._meta_path().write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        self._ids_path().write_text(json.dumps(self._ids), encoding="utf-8")

    def reader(self) -> "CacheReader":
        return CacheReader(self.root, self.fingerprint)


class CacheReader:
    def __init__(self, root: Path, expected: CacheFingerprint) -> None:
        self.root = Path(root)
        meta_path = self.root / "cache_fingerprint.json"
        if not meta_path.is_file():
            raise CacheFingerprintMismatch(f"missing fingerprint at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = CacheFingerprint(
            weight_revision=meta["weight_revision"],
            extraction_layer=meta["extraction_layer"],
            token_order=meta["token_order"],
            spatial_positions=meta["spatial_positions"],
            resolution=int(meta["resolution"]),
            preprocessing=meta["preprocessing"],
            precision=meta["precision"],
            manifest_sha256=meta["manifest_sha256"],
        )
        if recorded.digest() != meta["digest"]:
            raise CacheFingerprintMismatch("fingerprint file does not match its own digest")
        if recorded != expected:
            raise CacheFingerprintMismatch(
                "cache fingerprint does not match the requested encoder, "
                "preprocessing, or manifest"
            )
        self.fingerprint = recorded
        self.ids: list[str] = json.loads((self.root / "ids.json").read_text(encoding="utf-8"))
        self.n_tokens = int(meta["n_tokens"])
        self.dim = int(meta["dim"])
        self.shard_rows = int(meta["shard_rows"])
        self._mmaps: list[np.memmap] = []
        remaining = len(self.ids)
        shard_index = 0
        while remaining:
            take = min(self.shard_rows, remaining)
            path = self.root / f"shard_{shard_index:04d}.f16"
            mmap = np.memmap(
                path,
                dtype=np.float16,
                mode="r",
                shape=(take, self.n_tokens, self.dim),
            )
            if getattr(mmap, "mode", None) != "r":
                raise RuntimeError(f"cache shard {path} was not opened read-only")
            self._mmaps.append(mmap)
            remaining -= take
            shard_index += 1
        self.index = {image_id: row for row, image_id in enumerate(self.ids)}

    def get(self, image_id: str) -> np.ndarray:
        row = self.index[image_id]
        shard = row // self.shard_rows
        local = row % self.shard_rows
        return np.array(self._mmaps[shard][local], copy=True)

    @property
    def modes(self) -> list[str]:
        return [mmap.mode for mmap in self._mmaps]


def max_abs_online_vs_cache(online: np.ndarray, cached: np.ndarray) -> float:
    left = np.asarray(online, dtype=np.float32)
    right = np.asarray(cached, dtype=np.float32)
    return float(np.max(np.abs(left - right)))
