"""Hand-checked multi-positive losses and QA positive rules."""

from __future__ import annotations

import inspect
import math

import torch

from src.conditional.losses import (
    accumulated_anchor_infonce,
    answer_vocabulary_report,
    assert_single_task,
    cosine_positive_loss,
    mean_per_positive_logsoftmax,
    microbatch_rebuilt_infonce,
    multi_positive_infonce,
    qa_positive_mask,
)


def _toy():
    predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    return predictions, candidates, mask


def _expected_toy() -> float:
    loss0 = -math.log((math.exp(1.0) + math.exp(0.6)) / (math.exp(1.0) + math.exp(0.0) + math.exp(0.6)))
    loss1 = -math.log(math.exp(1.0) / (math.exp(0.0) + math.exp(1.0) + math.exp(0.8)))
    return (loss0 + loss1) / 2.0


def test_infonce_matches_the_hand_checked_sum_form() -> None:
    predictions, candidates, mask = _toy()
    got = multi_positive_infonce(predictions, candidates, mask, temperature=1.0)
    assert torch.allclose(got, torch.tensor(_expected_toy()), atol=1e-6)
    other = mean_per_positive_logsoftmax(predictions, candidates, mask, temperature=1.0)
    assert not torch.allclose(got, other, atol=1e-4)


def test_cosine_matches_half_squared_distance_on_unit_vectors() -> None:
    predictions, candidates, mask = _toy()
    loss = cosine_positive_loss(predictions, candidates, mask)
    dots = []
    pairs = [(0, 0), (0, 2), (1, 1)]
    for row, col in pairs:
        dots.append(float(torch.dot(predictions[row], candidates[col])))
    expected = sum(1.0 - value for value in dots) / len(dots)
    assert abs(float(loss) - expected) < 1e-6
    half_sq = []
    for row, col in pairs:
        half_sq.append(0.5 * float(torch.sum((predictions[row] - candidates[col]) ** 2)))
    assert abs(expected - (sum(half_sq) / len(half_sq))) < 1e-6


def test_full_pool_matches_split_anchors_and_not_rebuilt_microbatches() -> None:
    generator = torch.Generator().manual_seed(2)
    predictions = torch.randn(4, 3, generator=generator)
    candidates = torch.randn(4, 3, generator=generator)
    mask = torch.eye(4, dtype=torch.bool)
    full = multi_positive_infonce(predictions, candidates, mask)
    split = accumulated_anchor_infonce(predictions, candidates, mask, [3, 1])
    rebuilt = microbatch_rebuilt_infonce(predictions, candidates, mask, [3, 1])
    assert torch.allclose(full, split, atol=1e-6)
    assert not torch.allclose(full, rebuilt, atol=1e-4)

    left = predictions.detach().clone().requires_grad_(True)
    multi_positive_infonce(left, candidates, mask).backward()
    right = predictions.detach().clone().requires_grad_(True)
    accumulated_anchor_infonce(right, candidates, mask, [1, 3]).backward()
    assert torch.allclose(left.grad, right.grad, atol=1e-5)


def test_repeated_answers_share_a_column_and_are_not_mutual_negatives() -> None:
    assert "image" not in inspect.signature(qa_positive_mask).parameters
    shared = qa_positive_mask(["dog", "dog"], ["dog", "man"])
    assert torch.equal(shared, torch.tensor([[True, False], [True, False]]))
    predictions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    shared_loss = multi_positive_infonce(predictions, prototypes, shared)
    expected = -math.log(math.e / (math.e + 1.0))
    assert abs(float(shared_loss) - expected) < 1e-6
    mutual = torch.eye(2, dtype=torch.bool)
    mutual_loss = multi_positive_infonce(predictions, predictions, mutual)
    assert abs(float(mutual_loss) - math.log(2.0)) < 1e-6
    assert abs(float(shared_loss) - float(mutual_loss)) > 1e-3


def test_incompatible_questions_are_not_interchangeable_positives() -> None:
    mask = qa_positive_mask(["dog", "man"], ["dog", "man"])
    assert torch.equal(mask, torch.eye(2, dtype=torch.bool))
    try:
        qa_positive_mask(["dog", "dog"], ["dog", "dog"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate prototypes must be refused")
    try:
        qa_positive_mask(["bird"], ["dog"])
    except KeyError:
        pass
    else:
        raise AssertionError("out-of-vocabulary answers must raise in the mask")
    report = answer_vocabulary_report(["dog", "bird"], ["dog", "man"])
    assert report["n_oov"] == 1
    assert report["oov_ids"] == ["bird"]
    assert report["coverage"] == 0.5


def test_mixed_caption_and_qa_batches_are_refused() -> None:
    try:
        assert_single_task(["caption", "qa"])
    except ValueError as exc:
        assert "mixed" in str(exc)
    else:
        raise AssertionError("mixed tasks must be refused")
    assert assert_single_task(["qa", "qa"]) == "qa"
