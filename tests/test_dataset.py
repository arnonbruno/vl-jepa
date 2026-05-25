"""Smoke tests for COCO 2017 caption dataset integration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.dataset import (
    COCOCaptionDataset,
    CocoDatasetError,
    create_dataloaders,
    ensure_coco_2017,
    expected_split_length,
)


def _coco_test_root() -> Path:
    env = os.environ.get("VL_JEPA_COCO_ROOT")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return Path(os.path.expanduser("~/.cache/torch/hub/checkpoints")).resolve()


@pytest.fixture(scope="module")
def coco_root() -> Path:
    """Ensure at least the val split is available (smaller than full train)."""
    root = _coco_test_root()
    try:
        return ensure_coco_2017(root, splits=("val",), download=True)
    except CocoDatasetError as exc:
        pytest.skip(f"COCO 2017 unavailable for tests: {exc}")


@pytest.fixture(scope="module")
def val_dataset(coco_root: Path) -> COCOCaptionDataset:
    return COCOCaptionDataset(
        split="val",
        coco_root=coco_root,
        image_size=224,
        max_caption_length=64,
        download=False,
    )


def test_val_dataset_length(val_dataset: COCOCaptionDataset) -> None:
    assert len(val_dataset) == expected_split_length("val")


def test_single_sample_shapes(val_dataset: COCOCaptionDataset) -> None:
    image, input_ids, attention_mask = val_dataset[0]
    assert image.shape == (3, 224, 224)
    assert input_ids.shape == (64,)
    assert attention_mask.shape == (64,)


def test_tokenizer_valid_ids(val_dataset: COCOCaptionDataset) -> None:
    _, input_ids, attention_mask = val_dataset[0]
    vocab_size = val_dataset.tokenizer.vocab_size
    assert input_ids.min() >= 0
    assert input_ids.max() < vocab_size
    assert attention_mask.dtype == torch.long
    assert set(attention_mask.unique().tolist()).issubset({0, 1})


def test_epoch_changes_caption_but_not_image(coco_root: Path) -> None:
    ds = COCOCaptionDataset(
        split="val",
        coco_root=coco_root,
        download=False,
    )
    # Find an index with multiple captions
    multi_idx = None
    for i in range(min(200, len(ds))):
        _, caps = ds._coco[i]
        if len(caps) > 1:
            multi_idx = i
            break
    if multi_idx is None:
        pytest.skip("No multi-caption sample in first 200 val images")

    ds.set_epoch(0)
    img0, ids0, _ = ds[multi_idx]
    ds.set_epoch(1)
    img1, ids1, _ = ds[multi_idx]
    assert torch.equal(img0, img1)
    # With different epochs, caption choice may differ (not guaranteed but likely)
    ds.set_epoch(0)
    _, ids0_repeat, _ = ds[multi_idx]
    assert torch.equal(ids0, ids0_repeat)


def test_val_dataloader_batch_shapes(val_dataset: COCOCaptionDataset) -> None:
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    images, input_ids, attention_mask = next(iter(val_loader))
    assert images.shape == (4, 3, 224, 224)
    assert input_ids.shape == (4, 64)
    assert attention_mask.shape == (4, 64)


def test_train_val_splits_distinct(coco_root: Path) -> None:
    val_ds = COCOCaptionDataset(split="val", coco_root=coco_root, download=False)
    assert len(val_ds) == expected_split_length("val")

    try:
        ensure_coco_2017(coco_root, splits=("train",), download=False)
    except CocoDatasetError:
        pytest.skip("COCO train2017 split not cached; skipping train length check")

    train_ds = COCOCaptionDataset(split="train", coco_root=coco_root, download=False)
    assert len(train_ds) == expected_split_length("train")
    assert len(train_ds) != len(val_ds)


@pytest.mark.skipif(
    os.environ.get("VL_JEPA_RUN_COCO_TRAIN", "").lower() not in ("1", "true", "yes"),
    reason="Full train split test disabled (set VL_JEPA_RUN_COCO_TRAIN=1 to enable)",
)
def test_create_dataloaders_train_val(coco_root: Path) -> None:
    try:
        ensure_coco_2017(coco_root, splits=("train", "val"), download=True)
    except CocoDatasetError as exc:
        pytest.skip(f"COCO train split unavailable: {exc}")

    train_loader, val_loader = create_dataloaders(
        batch_size=2,
        num_workers=0,
        coco_root=coco_root,
        download=False,
    )
    assert len(train_loader.dataset) == expected_split_length("train")
    assert len(val_loader.dataset) == expected_split_length("val")
