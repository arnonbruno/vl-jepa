"""COCO 2017 caption dataset integration for VL-JEPA."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CocoCaptions
from torchvision.datasets.utils import download_and_extract_archive
from transformers import DistilBertTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# COCO 2017 official archives (torchvision.datasets.utils)
_COCO_ARCHIVES: Dict[str, Tuple[str, str]] = {
    "train2017": (
        "http://images.cocodataset.org/zips/train2017.zip",
        "train2017",
    ),
    "val2017": (
        "http://images.cocodataset.org/zips/val2017.zip",
        "val2017",
    ),
    "annotations": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations",
    ),
}

# Known image counts (COCO 2017 captions)
_COCO_SPLIT_LENGTHS: Dict[str, int] = {
    "train": 118_287,
    "val": 5_000,
}


class CocoDatasetError(RuntimeError):
    """Raised when COCO 2017 data cannot be loaded or downloaded."""


def _expand_path(path: Union[str, Path]) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def _default_coco_root() -> Path:
    return _expand_path("~/.cache/torch/hub/checkpoints")


def _caption_ann_path(coco_root: Path, split: str) -> Path:
    year = "train2017" if split == "train" else "val2017"
    return coco_root / "annotations" / f"captions_{year}.json"


def _image_root(coco_root: Path, split: str) -> Path:
    folder = "train2017" if split == "train" else "val2017"
    return coco_root / folder


def _split_ready(coco_root: Path, split: str) -> bool:
    ann = _caption_ann_path(coco_root, split)
    images = _image_root(coco_root, split)
    return ann.is_file() and images.is_dir() and any(images.iterdir())


def ensure_coco_2017(
    coco_root: Optional[Union[str, Path]] = None,
    *,
    splits: Tuple[str, ...] = ("train", "val"),
    download: bool = True,
) -> Path:
    """Ensure COCO 2017 caption assets exist under ``coco_root``.

    Downloads missing archives via ``torchvision.datasets.utils`` when
    ``download=True``. Raises :class:`CocoDatasetError` on failure.
    """
    root = _expand_path(coco_root) if coco_root is not None else _default_coco_root()
    root.mkdir(parents=True, exist_ok=True)

    need_annotations = any(
        not _caption_ann_path(root, s).is_file() for s in splits
    )
    archives: list[str] = []
    if need_annotations:
        archives.append("annotations")
    for split in splits:
        if not _split_ready(root, split):
            archives.append("val2017" if split == "val" else "train2017")

    if not archives:
        return root

    if not download:
        missing = ", ".join(archives)
        raise CocoDatasetError(
            f"COCO 2017 data incomplete under {root}. Missing: {missing}. "
            "Set download=True or place train2017/, val2017/, and "
            "annotations/ manually."
        )

    for key in archives:
        url, folder = _COCO_ARCHIVES[key]
        try:
            download_and_extract_archive(url, str(root), remove_finished=True)
        except Exception as exc:
            raise CocoDatasetError(
                f"Failed to download/extract COCO archive '{key}' into {root}. "
                f"Original error: {exc}"
            ) from exc
        extracted = root / folder
        if key == "annotations" and not extracted.is_dir():
            raise CocoDatasetError(
                f"Expected annotations/ under {root} after extract, not found."
            )

    for split in splits:
        if not _split_ready(root, split):
            raise CocoDatasetError(
                f"COCO 2017 {split} split still unavailable under {root} "
                f"(images + captions_{split}2017.json)."
            )

    return root


def _build_image_transform(image_size: int, split: str) -> transforms.Compose:
    if split == "train":
        image_ops = [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.50, 1.00),
                ratio=(0.75, 1.3333),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        image_ops = [
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
        ]

    return transforms.Compose(
        [
            *image_ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def _load_coco_captions(
    coco_root: Path,
    split: str,
    transform: transforms.Compose,
    *,
    download: bool,
) -> CocoCaptions:
    ensure_coco_2017(coco_root, splits=(split,), download=download)
    ann_file = str(_caption_ann_path(coco_root, split))
    image_root = str(_image_root(coco_root, split))
    try:
        return CocoCaptions(root=image_root, annFile=ann_file, transform=transform)
    except Exception as exc:
        raise CocoDatasetError(
            f"Failed to open CocoCaptions for split={split!r} under {coco_root}. "
            "Ensure pycocotools is installed (`pip install pycocotools`). "
            f"Original error: {exc}"
        ) from exc


class COCOCaptionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """COCO 2017 image–caption pairs with one random caption per epoch."""

    def __init__(
        self,
        split: str = "train",
        coco_root: Optional[Union[str, Path]] = None,
        image_size: int = 224,
        max_caption_length: int = 64,
        download: bool = True,
        tokenizer: Optional[DistilBertTokenizer] = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.split = split
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self._epoch = 0
        self.coco_root = (
            _expand_path(coco_root) if coco_root is not None else _default_coco_root()
        )

        self.tokenizer = tokenizer or DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        transform = _build_image_transform(image_size, split)
        self._coco = _load_coco_captions(
            self.coco_root, split, transform, download=download
        )

    def set_epoch(self, epoch: int) -> None:
        """Fix per-index caption RNG for this epoch (one caption per image)."""
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._coco)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, captions = self._coco[index]
        if not captions:
            caption = ""
        else:
            rng = random.Random(self._epoch * 1_000_003 + index)
            caption = rng.choice(captions)

        encoded = self.tokenizer(
            caption,
            max_length=self.max_caption_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0).to(dtype=torch.long)
        attention_mask = encoded["attention_mask"].squeeze(0).to(dtype=torch.long)
        return image, input_ids, attention_mask


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    coco_root: Optional[Union[str, Path]] = None,
    image_size: int = 224,
    max_caption_length: int = 64,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for COCO 2017 captions."""
    root = _expand_path(coco_root) if coco_root is not None else _default_coco_root()

    train_ds = COCOCaptionDataset(
        split="train",
        coco_root=root,
        image_size=image_size,
        max_caption_length=max_caption_length,
        download=download,
    )
    val_ds = COCOCaptionDataset(
        split="val",
        coco_root=root,
        image_size=image_size,
        max_caption_length=max_caption_length,
        download=download,
        tokenizer=train_ds.tokenizer,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def expected_split_length(split: str) -> int:
    """Return the canonical COCO 2017 captions image count for a split."""
    if split not in _COCO_SPLIT_LENGTHS:
        raise ValueError(f"Unknown split {split!r}")
    return _COCO_SPLIT_LENGTHS[split]
