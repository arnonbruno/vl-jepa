"""Tests for the standard multi-caption retrieval metrics."""

import json
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_retrieval import (
    average_standard_metrics,
    build_image_to_texts,
    caption_positive_images,
    caption_positive_multiplicity,
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
    with pytest.raises(ValueError, match="empty"):
        compute_retrieval_metrics(
            torch.zeros(0, 8), torch.zeros(0, 8), torch.zeros(0, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="1-D"):
        compute_retrieval_metrics(
            torch.randn(2, 8), torch.randn(4, 8), torch.zeros(1, 4, dtype=torch.long),
        )


def test_format_metrics_runs():
    torch.manual_seed(3)
    images = torch.randn(10, 8)
    text_to_image = torch.arange(10).repeat_interleave(5)
    texts = torch.randn(50, 8)
    out = format_metrics(compute_retrieval_metrics(images, texts, text_to_image))
    assert "I2T" in out and "T2I" in out and "rsum" in out


def _t1_shared_caption_fixture():
    """Identical caption strings share one embedding (deterministic encoder)."""
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    captions = ["shared dog", "only image0", "shared dog", "only image1"]
    text_to_image = torch.tensor([0, 0, 1, 1])
    texts = torch.tensor([
        [0.6, 0.8],
        [1.0, 0.0],
        [0.6, 0.8],
        [0.0, 1.0],
    ])
    return images, texts, text_to_image, captions


def _assert_json_native_diagnostics(diag):
    assert isinstance(diag, dict)
    for key, value in diag.items():
        assert isinstance(key, str)
        assert type(value) in (int, float, str), (key, type(value), value)
        assert not isinstance(value, bool)


def test_ambiguity_diagnostic_is_opt_in_and_conservative():
    """Duplicate GT caption strings are extra t2i positives, not a SOTA metric.

    Standard Recall@K is unchanged. The diagnostic reports per-caption
    positive multiplicity from exact caption-string provenance so a caption
    that is a valid description of two annotated images is not scored as a
    false negative under t2i. It is not a new retrieval leaderboard number.
    """
    images, texts, text_to_image, captions = _t1_shared_caption_fixture()

    standard = compute_retrieval_metrics(images, texts, text_to_image, normalize=False)
    assert "diagnostics" not in standard
    assert all(not k.startswith("diag_") for k in standard)

    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    standard_from_tagged = {k: v for k, v in tagged.items() if k != "diagnostics"}
    assert standard_from_tagged == standard
    assert tagged["rsum"] == standard["rsum"]
    assert tagged["rsum"] == (
        tagged["i2t_r1"] + tagged["i2t_r5"] + tagged["i2t_r10"]
        + tagged["t2i_r1"] + tagged["t2i_r5"] + tagged["t2i_r10"]
    )

    diag = tagged["diagnostics"]
    _assert_json_native_diagnostics(diag)
    assert "i2t_r1_any_gt" not in diag
    assert all("i2t" not in k or "any_gt" not in k for k in diag)

    mult = caption_positive_multiplicity(captions, text_to_image, num_images=2)
    assert mult.dtype == torch.int64
    assert torch.equal(mult, torch.tensor([2, 1, 2, 1], dtype=torch.int64))
    sets = caption_positive_images(captions, text_to_image, num_images=2)
    assert [s.tolist() for s in sets] == [[0, 1], [0], [0, 1], [1]]
    assert torch.equal(torch.tensor([len(s) for s in sets], dtype=torch.int64), mult)

    assert diag["caption_multiplicity_mean"] == 1.5
    assert diag["caption_multiplicity_max"] == 2
    assert type(diag["caption_multiplicity_max"]) is int
    assert diag["frac_ambiguous_captions"] == 0.5
    assert diag["n_duplicate_text_groups"] == 1
    assert diag["n_cross_image_duplicate_groups"] == 1
    assert diag["captions_per_image_min"] == 2
    assert diag["captions_per_image_max"] == 2
    assert diag["captions_per_image_mean"] == 2.0
    # c0 pos-sim 0.6 vs distractor 0.8 → rank 1 miss; c1/c2/c3 rank 0.
    assert tagged["t2i_r1"] == 75.0
    assert tagged["t2i_r5"] == 100.0
    assert tagged["t2i_r10"] == 100.0
    assert tagged["i2t_r1"] == 100.0
    assert diag["t2i_r1_any_gt"] == 100.0
    assert "evaluation diagnostic" in diag["note"].lower()
    assert "not a retrieval result" in diag["note"].lower()
    assert "exact caption-string" in diag["positive_set_source"]
    assert diag["caption_normalizer"] == "str.strip"


def test_within_image_duplicate_captions_are_one_positive_image():
    images = torch.eye(2)
    captions = ["dup", "dup", "x"]
    text_to_image = torch.tensor([0, 0, 1])
    texts = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    mult = caption_positive_multiplicity(captions, text_to_image, num_images=2)
    assert torch.equal(mult, torch.tensor([1, 1, 1], dtype=torch.int64))
    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    diag = tagged["diagnostics"]
    assert diag["frac_ambiguous_captions"] == 0.0
    assert diag["n_duplicate_text_groups"] == 1
    assert diag["n_cross_image_duplicate_groups"] == 0
    for k in (1, 5, 10):
        assert diag[f"t2i_r{k}_any_gt"] == tagged[f"t2i_r{k}"]


def test_diagnostic_does_not_saturate_when_distractor_outranks():
    images = torch.eye(3)
    captions = ["shared", "shared", "x"]
    text_to_image = torch.tensor([0, 1, 2])
    texts = torch.tensor([
        [0.36, 0.48, 0.80],
        [0.36, 0.48, 0.80],
        [0.00, 0.00, 1.00],
    ])
    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False, ks=(1, 2),
        captions=captions, diagnostics=True,
    )
    assert tagged["t2i_r1"] == pytest.approx((1.0 / 3.0) * 100.0)
    assert tagged["diagnostics"]["t2i_r1_any_gt"] == tagged["t2i_r1"]
    assert tagged["t2i_r2"] == pytest.approx((2.0 / 3.0) * 100.0)
    assert tagged["diagnostics"]["t2i_r2_any_gt"] == 100.0
    rendered = format_metrics(tagged, ks=(1, 5, 10))
    assert "R@1=" in rendered and "R@2=" in rendered
    assert "R@5=" not in rendered
    assert "diagnostics" not in rendered
    assert "any_gt" not in rendered


def test_no_duplicate_captions_makes_diagnostic_equal_standard():
    torch.manual_seed(0)
    num_images, dim, caps = 50, 32, 5
    images = torch.randn(num_images, dim)
    text_to_image = torch.arange(num_images).repeat_interleave(caps)
    texts = torch.randn(num_images * caps, dim)
    captions = [f"c{i}" for i in range(num_images * caps)]
    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, captions=captions, diagnostics=True,
    )
    for k in (1, 5, 10):
        assert tagged["diagnostics"][f"t2i_r{k}_any_gt"] == tagged[f"t2i_r{k}"]
    assert tagged["diagnostics"]["t2i_medr_any_gt"] == tagged["t2i_medr"]
    assert tagged["diagnostics"]["t2i_meanr_any_gt"] == tagged["t2i_meanr"]
    assert torch.equal(
        caption_positive_multiplicity(captions, text_to_image, num_images),
        torch.ones(num_images * caps, dtype=torch.int64),
    )
    assert tagged["diagnostics"]["frac_ambiguous_captions"] == 0.0
    assert tagged["diagnostics"]["n_cross_image_duplicate_groups"] == 0
    assert tagged["diagnostics"]["n_duplicate_text_groups"] == 0


def test_diagnostics_off_with_captions_leaves_standard_dict_unchanged():
    images, texts, text_to_image, captions = _t1_shared_caption_fixture()
    bare = compute_retrieval_metrics(images, texts, text_to_image, normalize=False)
    ignored = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=False,
    )
    assert ignored == bare
    assert "diagnostics" not in ignored


def test_diagnostic_chunk_invariance_on_exact_fixture():
    images, texts, text_to_image, captions = _t1_shared_caption_fixture()
    a = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False, chunk=1,
        captions=captions, diagnostics=True,
    )
    b = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False, chunk=10_000,
        captions=captions, diagnostics=True,
    )
    assert a["diagnostics"] == b["diagnostics"]
    assert {k: v for k, v in a.items() if k != "diagnostics"} == (
        {k: v for k, v in b.items() if k != "diagnostics"}
    )


def test_diagnostic_json_round_trip_is_native():
    images, texts, text_to_image, captions = _t1_shared_caption_fixture()
    tagged = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    _assert_json_native_diagnostics(tagged["diagnostics"])
    roundtrip = json.loads(json.dumps(tagged))
    assert roundtrip == tagged
    assert type(roundtrip["diagnostics"]["caption_multiplicity_max"]) is int
    assert type(roundtrip["diagnostics"]["n_duplicate_text_groups"]) is int
    assert type(roundtrip["diagnostics"]["n_cross_image_duplicate_groups"]) is int
    assert type(roundtrip["diagnostics"]["t2i_r1_any_gt"]) is float


def test_diagnostic_input_validation():
    images, texts, text_to_image, captions = _t1_shared_caption_fixture()
    with pytest.raises(ValueError, match="captions"):
        compute_retrieval_metrics(
            images, texts, text_to_image, normalize=False, diagnostics=True,
        )
    with pytest.raises(ValueError, match="captions"):
        compute_retrieval_metrics(
            images, texts, text_to_image, normalize=False,
            captions=captions[:-1], diagnostics=True,
        )
    with pytest.raises(ValueError, match="string"):
        compute_retrieval_metrics(
            images, texts, text_to_image, normalize=False,
            captions=["shared dog", 3, "shared dog", "only image1"],
            diagnostics=True,
        )
    with pytest.raises(ValueError, match="string"):
        caption_positive_multiplicity(
            ["a", None, "c", "d"], text_to_image, num_images=2,
        )


def test_caption_strip_normalization_does_not_lowercase():
    captions = ["A dog. ", "A dog.", "a dog."]
    text_to_image = torch.tensor([0, 1, 2])
    mult = caption_positive_multiplicity(captions, text_to_image, num_images=3)
    assert torch.equal(mult, torch.tensor([2, 2, 1], dtype=torch.int64))


def test_zero_norm_rejected_when_normalize_true():
    images = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_to_image = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="zero-norm"):
        compute_retrieval_metrics(images, texts, text_to_image, normalize=True)


def test_zero_norm_tie_is_rank_zero_when_normalize_false():
    images = torch.eye(2)
    texts = torch.zeros(2, 2)
    text_to_image = torch.tensor([0, 1])
    captions = ["a", "b"]
    m = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    # All similarities are 0, so strict-'>' counts no strictly better image.
    assert m["t2i_r1"] == 100.0
    assert m["diagnostics"]["n_zero_norm_text_embs"] == 2
    assert m["diagnostics"]["n_zero_norm_image_embs"] == 0


def test_medr_uses_torch_lower_middle_even_n():
    """Published MedR path: torch.median lower-middle of 0-index ranks, then +1.

    ranks [0,1,2,3] → torch.median=1 → MedR=2.0, not the conventional 2.5.
    """
    images = torch.eye(4)
    texts = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.4, 0.5, 0.9, 0.0],
        [0.3, 0.5, 0.4, 0.9],
        [0.2, 0.3, 0.4, 0.1],
    ])
    text_to_image = torch.tensor([0, 1, 2, 3])
    m = compute_retrieval_metrics(images, texts, text_to_image, normalize=False)
    assert m["t2i_medr"] == 2.0


def test_average_standard_metrics_skips_diagnostics_and_strings():
    folds = [
        {
            "i2t_r1": 10.0, "t2i_r1": 20.0, "rsum": 30.0,
            "diagnostics": {"note": "x", "t2i_r1_any_gt": 99.0},
            "diag_note": "evaluation diagnostic",
        },
        {
            "i2t_r1": 30.0, "t2i_r1": 40.0, "rsum": 70.0,
            "diagnostics": {"note": "y", "t2i_r1_any_gt": 100.0},
            "diag_note": "evaluation diagnostic",
        },
    ]
    avg = average_standard_metrics(folds)
    assert avg == {"i2t_r1": 20.0, "t2i_r1": 30.0, "rsum": 50.0}
    assert "diagnostics" not in avg
    assert "diag_note" not in avg


def test_eval_1k_does_not_forward_diagnostics_or_use_caps_per_image():
    import inspect

    from experiments.evaluate_retrieval import _eval_1k

    sig = inspect.signature(_eval_1k)
    assert "caps_per_image" not in sig.parameters
    assert sig.parameters["fold_size"].default == 1000
    torch.manual_seed(4)
    n_images = 2000
    images = torch.randn(n_images, 4)
    texts = torch.randn(n_images, 4)
    text_to_image = torch.arange(n_images)
    avg = _eval_1k(images, texts, text_to_image)
    assert "diagnostics" not in avg
    assert "i2t_r1" in avg and "t2i_r1" in avg
    assert avg["rsum"] == pytest.approx(
        avg["i2t_r1"] + avg["i2t_r5"] + avg["i2t_r10"]
        + avg["t2i_r1"] + avg["t2i_r5"] + avg["t2i_r10"]
    )


def _five_caption_fold_fixture(n_images: int = 8, caps: int = 5, dim: int = 4):
    torch.manual_seed(7)
    images = torch.randn(n_images, dim)
    text_to_image = torch.arange(n_images).repeat_interleave(caps)
    texts = torch.randn(n_images * caps, dim)
    return images, texts, text_to_image


def _assert_five_caption_fold_calls(calls, fold_size: int, caps: int = 5) -> None:
    assert len(calls) == 2
    for i, call in enumerate(calls):
        n_img, n_txt, t2i, kwargs = call
        assert n_img == fold_size
        assert n_txt == fold_size * caps
        assert int(t2i.min()) >= 0
        assert int(t2i.max()) < fold_size
        assert torch.equal(t2i, torch.arange(fold_size).repeat_interleave(caps))
        assert kwargs.get("diagnostics") is not True


def test_eval_1k_five_captions_per_image_remaps_fold_indices(monkeypatch):
    from experiments.evaluate_retrieval import _eval_1k

    calls = []
    real = compute_retrieval_metrics

    def spy(image_embs, text_embs, text_to_image, **kwargs):
        calls.append((
            int(image_embs.size(0)),
            int(text_embs.size(0)),
            text_to_image.detach().cpu().clone(),
            dict(kwargs),
        ))
        assert kwargs.get("diagnostics") is not True
        return real(image_embs, text_embs, text_to_image, **kwargs)

    monkeypatch.setattr(
        "experiments.evaluate_retrieval.compute_retrieval_metrics", spy,
    )
    images, texts, text_to_image = _five_caption_fold_fixture()
    avg = _eval_1k(images, texts, text_to_image, fold_size=4)
    _assert_five_caption_fold_calls(calls, fold_size=4)
    assert "diagnostics" not in avg
    assert "t2i_r1_any_gt" not in avg


def test_coco_1k_fold_helper_uses_complete_folds_only():
    from src.eval_all_checkpoints import _coco_1k_5fold

    torch.manual_seed(5)
    images = torch.randn(2500, 4)
    texts = torch.randn(2500, 4)
    text_to_image = torch.arange(2500)
    avg = _coco_1k_5fold(images, texts, text_to_image, n_images=2500)
    assert "diagnostics" not in avg
    assert avg["t2i_r1"] >= 0.0

    small_img = images[:500]
    small_txt = texts[:500]
    small_map = text_to_image[:500]
    small = _coco_1k_5fold(small_img, small_txt, small_map, n_images=500)
    assert small["t2i_r1"] >= 0.0
    with pytest.raises(ValueError, match="n_images"):
        _coco_1k_5fold(small_img, small_txt, small_map, n_images=5000)
    with pytest.raises(ValueError):
        _coco_1k_5fold(small_img, small_txt, torch.arange(400), n_images=500)


def test_coco_1k_preserves_five_folds_on_5000_images(monkeypatch):
    from src.eval_all_checkpoints import _coco_1k_5fold

    sizes = []
    real = compute_retrieval_metrics

    def spy(image_embs, text_embs, text_to_image, **kwargs):
        sizes.append(int(image_embs.size(0)))
        assert kwargs.get("diagnostics") is not True
        return real(image_embs, text_embs, text_to_image, **kwargs)

    monkeypatch.setattr("src.eval_all_checkpoints.compute_retrieval_metrics", spy)
    torch.manual_seed(6)
    n = 5000
    images = torch.randn(n, 2)
    texts = torch.randn(n, 2)
    text_to_image = torch.arange(n)
    _coco_1k_5fold(images, texts, text_to_image, n_images=n)
    assert sizes == [1000, 1000, 1000, 1000, 1000]


def test_coco_1k_five_captions_per_image_remaps_fold_indices(monkeypatch):
    from src.eval_all_checkpoints import _coco_1k_5fold

    calls = []
    real = compute_retrieval_metrics

    def spy(image_embs, text_embs, text_to_image, **kwargs):
        calls.append((
            int(image_embs.size(0)),
            int(text_embs.size(0)),
            text_to_image.detach().cpu().clone(),
            dict(kwargs),
        ))
        assert kwargs.get("diagnostics") is not True
        return real(image_embs, text_embs, text_to_image, **kwargs)

    monkeypatch.setattr("src.eval_all_checkpoints.compute_retrieval_metrics", spy)
    images, texts, text_to_image = _five_caption_fold_fixture()
    avg = _coco_1k_5fold(
        images, texts, text_to_image, n_images=8, fold_size=4,
    )
    _assert_five_caption_fold_calls(calls, fold_size=4)
    assert "diagnostics" not in avg
    assert "t2i_r1_any_gt" not in avg


def test_diagnostics_reject_empty_after_strip_captions():
    images = torch.eye(2)
    texts = torch.eye(2)
    text_to_image = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="empty"):
        compute_retrieval_metrics(
            images, texts, text_to_image, normalize=False,
            captions=["ok", "   "], diagnostics=True,
        )
    with pytest.raises(ValueError, match="empty"):
        compute_retrieval_metrics(
            images, texts, text_to_image, normalize=False,
            captions=["ok", ""], diagnostics=True,
        )
    with pytest.raises(ValueError, match="empty"):
        caption_positive_images(
            ["dog", ""], text_to_image, num_images=2,
        )
    ignored = compute_retrieval_metrics(
        images, texts, text_to_image, normalize=False,
        captions=["", "   "], diagnostics=False,
    )
    assert "diagnostics" not in ignored
