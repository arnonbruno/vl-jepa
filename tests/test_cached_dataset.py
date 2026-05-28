"""Tests for precomputed embedding cache datasets."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from src.cached_dataset import CachedEmbeddingDataset, create_cached_dataloaders
from src.model import VL_JEPA, apply_patch_mask_to_embeddings


def _write_synthetic_cache(path: Path, n: int = 8) -> None:
    hidden, seq = 32, 16
    local_n, global_n = 10, 50
    payload = {
        "local_patch_emb": torch.randn(n, local_n, hidden, dtype=torch.float16),
        "global_patch_emb": torch.randn(n, global_n, hidden, dtype=torch.float16),
        "language_emb": torch.randn(n, seq, hidden, dtype=torch.float16),
        "attention_mask": torch.ones(n, seq, dtype=torch.long),
        "metadata": {"split": path.stem},
    }
    torch.save(payload, path)


def test_cached_dataset_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_synthetic_cache(root / "train.pt")
        _write_synthetic_cache(root / "val.pt")
        train_loader, val_loader = create_cached_dataloaders(
            root, batch_size=4, num_workers=0,
        )
        local, global_emb, language_emb, attn = next(iter(train_loader))
        assert local.shape[0] == global_emb.shape[0] == language_emb.shape[0] == attn.shape[0]
        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 8


def test_forward_from_cache_matches_shapes() -> None:
    hidden = 96
    model = VL_JEPA(
        hidden_dim=hidden,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=16,
    )
    B, local_n, global_n, seq = 2, 10, 50, 16
    local = torch.randn(B, local_n, hidden)
    global_emb = torch.randn(B, global_n, hidden)
    language_emb = torch.randn(B, seq, hidden)
    attn = torch.ones(B, seq, dtype=torch.long)
    patch_mask = torch.zeros(B, local_n - 1, dtype=torch.bool)
    patch_mask[:, :5] = True

    masked = apply_patch_mask_to_embeddings(
        local, patch_mask, model.context_encoder.mask_token,
    )
    assert masked.shape == local.shape

    out = model.forward_from_cache(local, global_emb, language_emb, attn)
    assert out["vision_proj"].shape == (B, 16)
    assert out["predicted_patches"].shape == (B, local_n, hidden)
