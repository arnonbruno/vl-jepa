"""Datasets and loaders for precomputed frozen-encoder embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import build_dataloader_kwargs, resolve_num_workers


class CachedEmbeddingDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """Load precomputed local/global vision and language embeddings per sample."""

    def __init__(self, cache_path: Union[str, Path]) -> None:
        path = Path(cache_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Embedding cache not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict in {path}, got {type(payload).__name__}")

        required = ("local_patch_emb", "global_patch_emb", "language_emb", "attention_mask")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Cache {path} missing keys: {missing}")

        self.local_patch_emb = payload["local_patch_emb"]
        self.global_patch_emb = payload["global_patch_emb"]
        self.language_emb = payload["language_emb"]
        self.attention_mask = payload["attention_mask"]
        self.metadata: Dict[str, Any] = payload.get("metadata", {})

        n = self.local_patch_emb.size(0)
        if (
            self.global_patch_emb.size(0) != n
            or self.language_emb.size(0) != n
            or self.attention_mask.size(0) != n
        ):
            raise ValueError(f"Inconsistent cache lengths in {path}")

    def __len__(self) -> int:
        return self.local_patch_emb.size(0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.local_patch_emb[index],
            self.global_patch_emb[index],
            self.language_emb[index],
            self.attention_mask[index],
        )


def create_cached_dataloaders(
    cache_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val loaders from ``<cache_dir>/train.pt`` and ``val.pt``."""
    root = Path(cache_dir).expanduser().resolve()
    train_path = root / "train.pt"
    val_path = root / "val.pt"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            f"Expected train.pt and val.pt under {root}. "
            "Run experiments/precompute_embeddings.py first.",
        )

    workers = resolve_num_workers(num_workers)
    loader_kwargs = build_dataloader_kwargs(
        workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    train_ds = CachedEmbeddingDataset(train_path)
    val_ds = CachedEmbeddingDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
