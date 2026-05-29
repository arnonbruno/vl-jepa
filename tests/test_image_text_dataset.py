"""Tests for the CLIP caption tokenizer and the generic image-text dataset.

These stay network-free: the CLIP BPE tokenizer ships its vocab with open_clip,
and images are tiny PNGs written to a temp dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import CaptionTokenizer

try:
    import open_clip  # noqa: F401
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from src.image_text_dataset import (
    ImageTextDatasetError,
    ImageTextPairDataset,
    read_manifest,
)


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_clip_tokenizer_shapes_and_mask():
    """CLIP tokenizer yields (N, L) ids in CLIP's vocab with a 0-pad mask."""
    enc = CaptionTokenizer(text_backbone="openclip", max_caption_length=32)
    assert enc.kind == "clip"
    ids, mask = enc.encode(["a photo of a dog", ""])
    assert ids.shape == (2, 32)
    assert mask.shape == (2, 32)
    assert ids.dtype == torch.long and mask.dtype == torch.long
    # CLIP SOT/EOT markers and a strictly-positive vocab beyond DistilBERT's 30522.
    assert ids.max().item() >= 49406
    assert set(mask.unique().tolist()).issubset({0, 1})
    # Mask must exactly mark the non-pad (non-zero) ids.
    assert torch.equal(mask, (ids != 0).long())


def test_distilbert_tokenizer_is_default():
    """Default backbone keeps the DistilBERT WordPiece tokenizer (vocab 30522)."""
    enc = CaptionTokenizer(max_caption_length=16)
    assert enc.kind == "hf"
    ids, mask = enc.encode(["a cat on a mat"])
    assert ids.shape == (1, 16)
    assert ids.max().item() < 30522


def test_read_manifest_parses_and_skips(tmp_path):
    manifest = tmp_path / "m.tsv"
    manifest.write_text(
        "img0.jpg\ta caption\n"
        "\n"  # blank line skipped
        "no_delimiter_line\n"  # skipped
        "img1.jpg\tcaption\twith\ttabs\n",  # only first split is the path
        encoding="utf-8",
    )
    pairs = read_manifest(manifest)
    assert pairs == [
        ("img0.jpg", "a caption"),
        ("img1.jpg", "caption\twith\ttabs"),
    ]


def test_read_manifest_empty_raises(tmp_path):
    manifest = tmp_path / "empty.tsv"
    manifest.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ImageTextDatasetError):
        read_manifest(manifest)


@pytest.mark.skipif(not (_HAS_OPEN_CLIP and _HAS_PIL), reason="needs open_clip + PIL")
def test_image_text_pair_dataset_returns_model_contract(tmp_path):
    """Dataset yields (image[3,H,W], ids[L], mask[L]) matching the COCO contract."""
    images_root = tmp_path / "images"
    images_root.mkdir()
    captions = ["a red square", "a blue circle", "a green triangle"]
    lines = []
    for i, cap in enumerate(captions):
        name = f"img{i}.png"
        Image.new("RGB", (40, 30), color=(i * 40, 100, 200 - i * 30)).save(
            images_root / name
        )
        lines.append(f"{name}\t{cap}")
    manifest = tmp_path / "train.tsv"
    manifest.write_text("\n".join(lines), encoding="utf-8")

    ds = ImageTextPairDataset(
        manifest,
        images_root,
        image_size=64,
        max_caption_length=20,
        text_backbone="openclip",
    )
    assert len(ds) == 3
    image, input_ids, attention_mask = ds[0]
    assert image.shape == (3, 64, 64)
    assert input_ids.shape == (20,)
    assert attention_mask.shape == (20,)

    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    images, ids, masks = next(iter(loader))
    assert images.shape == (2, 3, 64, 64)
    assert ids.shape == (2, 20)
    assert masks.shape == (2, 20)


@pytest.mark.skipif(not _HAS_PIL, reason="PIL not installed")
def test_image_text_pair_dataset_missing_image_raises(tmp_path):
    manifest = tmp_path / "train.tsv"
    manifest.write_text("does_not_exist.png\ta caption\n", encoding="utf-8")
    ds = ImageTextPairDataset(manifest, tmp_path, image_size=64, max_caption_length=20)
    with pytest.raises(ImageTextDatasetError):
        _ = ds[0]
