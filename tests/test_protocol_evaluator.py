"""Reference retrieval stays strict-greater-than. Ties are diagnostics."""

from __future__ import annotations

import torch

from src.eval_retrieval import compute_retrieval_metrics
from src.protocol.evaluator import evaluate_retrieval, is_perfect_retrieval


def _five(n_images: int = 3) -> torch.Tensor:
    return torch.arange(n_images).repeat_interleave(5)


def test_constant_embeddings_are_not_selected_as_perfect() -> None:
    images = torch.ones(4, 8)
    texts = torch.ones(20, 8)
    result = evaluate_retrieval(images, texts, _five(4))
    assert result["reference"]["i2t_r1"] == 100.0
    assert result["reference"]["t2i_r1"] == 100.0
    assert result["selection_accepted"] is False
    assert result["selection_status"] == "rejected_fully_tied_embeddings"
    assert result["selection"] is None
    assert is_perfect_retrieval(result) is False
    assert result["protocol"] == "five_caption_reference_strict_gt"


def test_zero_norm_rows_are_rejected() -> None:
    images = torch.zeros(2, 4)
    texts = torch.ones(10, 4)
    try:
        evaluate_retrieval(images, texts, _five(2))
    except ValueError as exc:
        assert "zero-norm" in str(exc)
    else:
        raise AssertionError("zero-norm embeddings must be refused when normalize=True")


def test_exact_tie_pessimistic_bound_and_first_caption_disagreement() -> None:
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Image 0's first caption is the other image. Later captions match.
    # Index 5 is a non-positive copy of image 0, so the optimistic rank is 0
    # and the pessimistic rank is 1.
    text = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    text_to_image = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    result = evaluate_retrieval(image, text, text_to_image, near_tie_atol=0.0)
    assert result["selection_accepted"] is True
    assert result["reference"]["i2t_r1"] == 100.0
    assert result["tie_diagnostics"]["optimistic"]["i2t_r1"] == result["reference"]["i2t_r1"]
    assert result["tie_diagnostics"]["pessimistic"]["i2t_r1"] == 0.0
    assert result["tie_diagnostics"]["i2t_exact_tie_query_fraction"] == 1.0

    first = torch.tensor([0, 5])
    first_text = text[first]
    square = first_text @ image.T
    positive = torch.tensor([0, 1])
    hits = 0
    for row, col in enumerate(positive.tolist()):
        better = int((square[row] > square[row, col]).sum().item())
        hits += int(better < 1)
    assert hits == 0


def test_near_tie_does_not_change_the_reference() -> None:
    image = torch.tensor([[1.0, 0.0], [0.995, 0.099874921]])
    text = image.repeat_interleave(5, dim=0)
    mapping = _five(2)
    tight = evaluate_retrieval(image, text, mapping, near_tie_atol=0.0)
    loose = evaluate_retrieval(image, text, mapping, near_tie_atol=0.02)
    assert tight["reference"] == loose["reference"]
    assert tight["tie_diagnostics"]["t2i_exact_tie_query_fraction"] == 0.0
    assert loose["tie_diagnostics"]["t2i_near_tie_query_fraction"] > tight[
        "tie_diagnostics"
    ]["t2i_near_tie_query_fraction"]


def test_multiple_positives_and_repeated_caption_strings() -> None:
    image = torch.eye(2, 4)
    text = torch.zeros(10, 4)
    text[0] = image[1]
    text[1:5] = image[0]
    text[5] = image[0]
    text[6:] = image[1]
    mapping = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    captions = [
        "shared",
        "only-zero",
        "only-zero-b",
        "only-zero-c",
        "only-zero-d",
        "shared",
        "only-one",
        "only-one-b",
        "only-one-c",
        "only-one-d",
    ]
    result = evaluate_retrieval(
        image, text, mapping, captions=captions, diagnostics=True
    )
    direct = compute_retrieval_metrics(
        image, text, mapping, normalize=True, captions=captions, diagnostics=True
    )
    for key, value in result["reference"].items():
        assert value == direct[key]
    assert result["reference_diagnostics"]["n_cross_image_duplicate_groups"] == 1
    assert result["reference"]["i2t_r1"] == 100.0


def test_chunked_scores_match_dense_scores() -> None:
    generator = torch.Generator().manual_seed(4)
    image = torch.randn(6, 5, generator=generator)
    text = torch.randn(30, 5, generator=generator)
    mapping = _five(6)
    dense = evaluate_retrieval(image, text, mapping, chunk=10_000)
    chunked = evaluate_retrieval(image, text, mapping, chunk=1)
    assert dense["reference"] == chunked["reference"]
    assert dense["tie_diagnostics"]["pessimistic"] == chunked["tie_diagnostics"]["pessimistic"]


def test_remapping_texts_preserves_scores() -> None:
    generator = torch.Generator().manual_seed(5)
    image = torch.randn(3, 6, generator=generator)
    text = torch.randn(15, 6, generator=generator)
    mapping = _five(3)
    order = torch.randperm(15, generator=generator)
    base = evaluate_retrieval(image, text, mapping)
    moved = evaluate_retrieval(image, text[order], mapping[order])
    assert base["reference"] == moved["reference"]
