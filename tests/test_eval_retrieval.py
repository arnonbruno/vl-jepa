"""Tests for the standard multi-caption retrieval metrics."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_retrieval import (
    build_image_to_texts,
    compute_retrieval_metrics,
    format_metrics,
)


def test_build_image_to_texts_inverts_map():
    text_to_image = torch.tensor([0, 0, 1, 2, 2, 2])
    buckets = build_image_to_texts(text_to_image, num_images=3)
    assert torch.equal(buckets[0], torch.tensor([0, 1]))
    assert torch.equal(buckets[1], torch.tensor([2]))
    assert torch.equal(buckets[2], torch.tensor([3, 4, 5]))


def test_build_image_to_texts_rejects_empty_image():
    # image index 1 has no captions
    with pytest.raises(ValueError):
        build_image_to_texts(torch.tensor([0, 0, 2]), num_images=3)


def test_perfect_retrieval_is_full_recall():
    """When each caption equals its image embedding, every recall is 100%."""
    num_images, dim, caps = 20, 16, 5
    images = torch.nn.functional.normalize(torch.randn(num_images, dim), dim=-1)
    text_to_image = torch.arange(num_images).repeat_interleave(caps)
    texts = images[text_to_image].clone()  # identical to source image

    m = compute_retrieval_metrics(images, texts, text_to_image)
    for k in (1, 5, 10):
        assert m[f"i2t_r{k}"] == pytest.approx(100.0)
        assert m[f"t2i_r{k}"] == pytest.approx(100.0)
    assert m["i2t_medr"] == pytest.approx(1.0)
    assert m["t2i_medr"] == pytest.approx(1.0)
    assert m["rsum"] == pytest.approx(600.0)


def test_recall_monotonic_and_bounded():
    """R@1 <= R@5 <= R@10 and all recalls live in [0, 100]."""
    torch.manual_seed(0)
    num_images, dim, caps = 50, 32, 5
    images = torch.randn(num_images, dim)
    text_to_image = torch.arange(num_images).repeat_interleave(caps)
    texts = torch.randn(num_images * caps, dim)

    m = compute_retrieval_metrics(images, texts, text_to_image)
    for d in ("i2t", "t2i"):
        assert 0.0 <= m[f"{d}_r1"] <= m[f"{d}_r5"] <= m[f"{d}_r10"] <= 100.0
        assert m[f"{d}_medr"] >= 1.0
        assert m[f"{d}_meanr"] >= 1.0


def test_signal_beats_random():
    """Adding the true-image signal into captions lifts recall above chance."""
    torch.manual_seed(1)
    num_images, dim, caps = 100, 64, 5
    images = torch.nn.functional.normalize(torch.randn(num_images, dim), dim=-1)
    text_to_image = torch.arange(num_images).repeat_interleave(caps)

    noise = torch.randn(num_images * caps, dim)
    signal = images[text_to_image]
    texts = torch.nn.functional.normalize(0.7 * signal + 0.3 * noise, dim=-1)

    m = compute_retrieval_metrics(images, texts, text_to_image)
    # Chance R@1 for 100 images is ~1%; signal must be far above that.
    assert m["t2i_r1"] > 20.0
    assert m["i2t_r1"] > 20.0
    assert m["t2i_r10"] > 60.0


def test_chunking_matches_full():
    """Chunked rank computation equals the single-chunk result."""
    torch.manual_seed(2)
    num_images, dim, caps = 40, 16, 5
    images = torch.randn(num_images, dim)
    text_to_image = torch.arange(num_images).repeat_interleave(caps)
    texts = torch.randn(num_images * caps, dim)

    full = compute_retrieval_metrics(images, texts, text_to_image, chunk=10_000)
    chunked = compute_retrieval_metrics(images, texts, text_to_image, chunk=7)
    for key in full:
        assert full[key] == pytest.approx(chunked[key])


def test_validation_errors():
    images = torch.randn(4, 8)
    texts = torch.randn(20, 8)
    bad_map = torch.arange(19)  # wrong length
    with pytest.raises(ValueError):
        compute_retrieval_metrics(images, texts, bad_map)
    with pytest.raises(ValueError):
        compute_retrieval_metrics(images, torch.randn(20, 7), torch.zeros(20, dtype=torch.long))


def test_format_metrics_runs():
    torch.manual_seed(3)
    images = torch.randn(10, 8)
    text_to_image = torch.arange(10).repeat_interleave(5)
    texts = torch.randn(50, 8)
    out = format_metrics(compute_retrieval_metrics(images, texts, text_to_image))
    assert "I2T" in out and "T2I" in out and "rsum" in out


def test_ambiguity_diagnostic_is_opt_in_and_conservative():
    """Duplicate GT caption strings are extra t2i positives, not a SOTA metric.

    Standard Recall@K is unchanged. The diagnostic reports per-caption
    positive multiplicity from exact caption-string provenance so a caption
    that is a valid description of two annotated images is not scored as a
    false negative under t2i. It is not a new retrieval leaderboard number.
    """
    from src.eval_retrieval import caption_positive_multiplicity

    images = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ]), dim=-1)
    # Two captions per image; "shared dog" is annotated on both images.
    captions = ["shared dog", "only image0", "shared dog", "only image1"]
    text_to_image = torch.tensor([0, 0, 1, 1])
    texts = torch.nn.functional.normalize(torch.tensor([
        [0.2, 0.8],   # closer to image 1 than to its source image 0
        [1.0, 0.0],
        [0.8, 0.2],   # closer to image 0 than to its source image 1
        [0.0, 1.0],
    ]), dim=-1)

    standard = compute_retrieval_metrics(images, texts, text_to_image, normalize=False)
    assert "diag_t2i_r1_any_gt" not in standard
    assert "diag_caption_multiplicity_mean" not in standard

    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    for key in ("i2t_r1", "t2i_r1", "rsum"):
        assert tagged[key] == pytest.approx(standard[key])

    mult = caption_positive_multiplicity(captions, text_to_image, num_images=2)
    assert torch.equal(mult, torch.tensor([2, 1, 2, 1]))
    assert tagged["diag_caption_multiplicity_mean"] == pytest.approx(1.5)
    assert tagged["diag_caption_multiplicity_max"] == pytest.approx(2.0)
    assert tagged["diag_frac_ambiguous_captions"] == pytest.approx(0.5)
    # Standard t2i R@1 misses the two swapped "shared dog" captions (50%).
    # Ambiguity-aware t2i treats the other annotated image as a valid positive.
    assert tagged["t2i_r1"] == pytest.approx(50.0)
    assert tagged["diag_t2i_r1_any_gt"] == pytest.approx(100.0)
    assert tagged["diag_note"].startswith("evaluation diagnostic")
