"""Directional multi-positive losses.

InfoNCE uses the sum of positive scores in the numerator::

    L_q = -log( sum_{c in P_q} exp(s_qc / tau) / sum_c exp(s_qc / tau) )

That is not the mean of the per-positive log-softmaxes. Cosine prediction is
the mean of ``1 - dot`` over declared positives, which equals half the squared
Euclidean distance when both vectors are unit length.

The denominator is the full candidate bank. Averaging a loss that rebuilds
the bank inside each microbatch is a different objective. Splitting anchors
while keeping the full bank is the same objective.

QA positives are unique answer prototypes. The mask builder does not take an
image id. Repeated answers share a column, so they are not mutual negatives.
Caption rows and QA rows are not placed in one unlabeled negative matrix.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def _scores(predictions: torch.Tensor, candidates: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if predictions.dim() != 2 or candidates.dim() != 2:
        raise ValueError("predictions and candidates must be (N, D)")
    if predictions.size(1) != candidates.size(1):
        raise ValueError("prediction and candidate widths differ")
    return _unit(predictions) @ _unit(candidates).T / float(temperature)


def _check_mask(mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("positive_mask must be bool")
    if mask.shape != scores.shape:
        raise ValueError(f"positive_mask shape {tuple(mask.shape)} != scores {tuple(scores.shape)}")
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("every query needs at least one positive candidate")
    return mask


def multi_positive_infonce(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    scores = _scores(predictions, candidates, temperature)
    mask = _check_mask(positive_mask, scores)
    all_lse = torch.logsumexp(scores, dim=1)
    pos_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    pos_lse = torch.logsumexp(pos_scores, dim=1)
    return (all_lse - pos_lse).mean()


def mean_per_positive_logsoftmax(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """A different multi-positive reduction, kept so tests can show it disagrees."""
    scores = _scores(predictions, candidates, temperature)
    mask = _check_mask(positive_mask, scores)
    log_prob = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    gathered = log_prob.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).float()
    return (-gathered).mean()


def cosine_positive_loss(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    scores = _scores(predictions, candidates, temperature=1.0)
    mask = _check_mask(positive_mask, scores)
    return (1.0 - scores)[mask].mean()


def qa_positive_mask(
    query_answer_ids: Sequence[str],
    candidate_answer_ids: Sequence[str],
) -> torch.Tensor:
    """Boolean mask of shape (queries, unique answer prototypes).

    ``candidate_answer_ids`` must already be unique. An answer that is not in
    that vocabulary raises. Coverage without a raise is
    :func:`answer_vocabulary_report`.
    """
    if len(candidate_answer_ids) != len(set(candidate_answer_ids)):
        raise ValueError("candidate answer ids must be unique prototypes")
    index = {answer: column for column, answer in enumerate(candidate_answer_ids)}
    mask = torch.zeros(len(query_answer_ids), len(candidate_answer_ids), dtype=torch.bool)
    for row, answer in enumerate(query_answer_ids):
        column = index.get(answer)
        if column is None:
            raise KeyError(f"out-of-vocabulary answer {answer!r}")
        mask[row, column] = True
    return mask


def answer_vocabulary_report(
    query_answer_ids: Sequence[str],
    vocabulary: Sequence[str],
) -> dict:
    vocab = set(vocabulary)
    oov = [answer for answer in query_answer_ids if answer not in vocab]
    n = len(query_answer_ids)
    return {
        "n": n,
        "n_oov": len(oov),
        "coverage": 1.0 - (len(oov) / n if n else 0.0),
        "oov_ids": list(oov),
    }


def assert_single_task(task_ids: Sequence[str]) -> str:
    kinds = set(task_ids)
    if len(kinds) != 1:
        raise ValueError(
            "refusing mixed caption and QA rows in one candidate matrix: "
            + ", ".join(sorted(kinds))
        )
    if not kinds:
        raise ValueError("task_ids is empty")
    return next(iter(kinds))


def accumulated_anchor_infonce(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    splits: Sequence[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Mean InfoNCE over anchor splits that all see ``candidates``."""
    if sum(splits) != predictions.size(0):
        raise ValueError("splits must cover every anchor")
    total = predictions.new_zeros(())
    start = 0
    n = predictions.size(0)
    for size in splits:
        if size <= 0:
            raise ValueError("split sizes must be positive")
        end = start + size
        piece = multi_positive_infonce(
            predictions[start:end],
            candidates,
            positive_mask[start:end],
            temperature,
        )
        total = total + piece * size
        start = end
    return total / n


def microbatch_rebuilt_infonce(
    predictions: torch.Tensor,
    candidates: torch.Tensor,
    positive_mask: torch.Tensor,
    splits: Sequence[int],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Each microbatch uses only its own rows as the candidate bank.

    This is not the full-pool loss. It exists so tests can show the inequality.
    """
    if sum(splits) != predictions.size(0):
        raise ValueError("splits must cover every anchor")
    if candidates.size(0) != predictions.size(0):
        raise ValueError("this contrast expects one candidate row per anchor")
    total = predictions.new_zeros(())
    start = 0
    n = predictions.size(0)
    for size in splits:
        end = start + size
        piece = multi_positive_infonce(
            predictions[start:end],
            candidates[start:end],
            positive_mask[start:end, start:end],
            temperature,
        )
        total = total + piece * size
        start = end
    return total / n
