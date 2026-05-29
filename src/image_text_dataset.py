"""Generic web-scale image-text pair dataset (CC3M / CC12M style).

COCO 2017 has only 118K images seen ~50x per long run — there is little left to
learn from it, which is why both prior runs plateaued at ~25% R@1. Pretraining
on a much larger corpus (CC3M ≈ 3.3M pairs, CC12M ≈ 12M pairs) before
fine-tuning on COCO is the highest long-term lever.

This module is the *infrastructure* for that path. It reads a lightweight
manifest of ``<image_path>\\t<caption>`` lines, so it works with any corpus that
has been downloaded to disk (e.g. via ``experiments/download_cc3m.py`` +
``img2dataset``). It shares the caption tokenizer and image transforms with the
COCO pipeline, so the same model/trainer code trains on either source.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    CaptionTokenizer,
    _build_image_transform,
    _is_openclip_text,
    build_dataloader_kwargs,
    resolve_num_workers,
)


class ImageTextDatasetError(RuntimeError):
    """Raised when a manifest is missing/empty or an image cannot be read."""


def _expand(path: Union[str, Path]) -> Path:
    return Path(os.path.expanduser(str(path)))


def read_manifest(
    manifest_path: Union[str, Path],
    *,
    delimiter: str = "\t",
) -> List[Tuple[str, str]]:
    """Parse a ``<image_path><delim><caption>`` manifest into (path, caption) pairs.

    Blank lines and lines without the delimiter are skipped. Captions may
    themselves contain the delimiter (only the first split is treated as the
    image path).
    """
    path = _expand(manifest_path)
    if not path.is_file():
        raise ImageTextDatasetError(f"Manifest not found: {path}")

    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if delimiter not in line:
                continue
            image_path, caption = line.split(delimiter, 1)
            image_path = image_path.strip()
            caption = caption.strip()
            if image_path:
                pairs.append((image_path, caption))
    if not pairs:
        raise ImageTextDatasetError(
            f"Manifest {path} contained no valid '<image>{delimiter}<caption>' lines."
        )
    return pairs


class ImageTextPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Image-caption pairs from a manifest, returning (image, input_ids, mask).

    Mirrors :class:`~src.dataset.COCOCaptionDataset`'s output contract so the
    existing trainer/dataloader code consumes it unchanged. Selects the CLIP BPE
    tokenizer + CLIP image normalization when ``text_backbone == 'openclip'``.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        images_root: Optional[Union[str, Path]] = None,
        *,
        split: str = "train",
        image_size: int = 224,
        max_caption_length: int = 64,
        text_backbone: str = "distilbert-base-uncased",
        openclip_model: str = "ViT-B-16",
        tokenizer: Optional[object] = None,
        delimiter: str = "\t",
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.split = split
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self.images_root = _expand(images_root) if images_root is not None else None
        self.pairs = read_manifest(manifest_path, delimiter=delimiter)

        self._encoder = CaptionTokenizer(
            text_backbone=text_backbone,
            openclip_model=openclip_model,
            max_caption_length=max_caption_length,
            tokenizer=tokenizer,
        )
        self.tokenizer = self._encoder.tokenizer

        mean, std = (
            (CLIP_MEAN, CLIP_STD)
            if _is_openclip_text(text_backbone)
            else (IMAGENET_MEAN, IMAGENET_STD)
        )
        self.transform = _build_image_transform(image_size, split, mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.pairs)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.is_absolute() and self.images_root is not None:
            path = self.images_root / path
        return path

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, caption = self.pairs[index]
        resolved = self._resolve_image_path(image_path)
        try:
            with Image.open(resolved) as img:
                image = self.transform(img.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise ImageTextDatasetError(
                f"Failed to read image {resolved} for manifest row {index}: {exc}"
            ) from exc

        input_ids, attention_mask = self._encoder.encode([caption])
        return image, input_ids[0], attention_mask[0]


def create_image_text_dataloader(
    manifest_path: Union[str, Path],
    images_root: Optional[Union[str, Path]] = None,
    *,
    batch_size: int = 128,
    num_workers: Optional[int] = None,
    image_size: int = 224,
    max_caption_length: int = 64,
    text_backbone: str = "distilbert-base-uncased",
    openclip_model: str = "ViT-B-16",
    shuffle: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader:
    """Build a single DataLoader over an image-text manifest (e.g. CC3M)."""
    dataset = ImageTextPairDataset(
        manifest_path,
        images_root,
        image_size=image_size,
        max_caption_length=max_caption_length,
        text_backbone=text_backbone,
        openclip_model=openclip_model,
    )
    workers = resolve_num_workers(num_workers)
    loader_kwargs = build_dataloader_kwargs(
        workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        **loader_kwargs,
    )
