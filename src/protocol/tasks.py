"""Evaluators other than the five-caption retrieval reference.

Retrieval selection stays in :mod:`src.protocol.evaluator`. These helpers score
paired composition, answer ranking, and fixed-prompt classification. They use
FP32 cosine similarity after normalization.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


def answer_ranking(
    prediction: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_ids: Sequence[str],
) -> list[str]:
    if prediction.dim() != 1:
        raise ValueError("answer_ranking scores one prediction vector")
    if len(candidate_ids) != candidate_embeddings.size(0):
        raise ValueError("one id per candidate embedding")
    scores = _unit(prediction) @ _unit(candidate_embeddings).T
    order = torch.argsort(scores, descending=True)
    return [candidate_ids[int(index)] for index in order]


def classification_top1(
    prediction: torch.Tensor,
    class_embeddings: torch.Tensor,
    class_names: Sequence[str],
) -> str:
    return answer_ranking(prediction, class_embeddings, class_names)[0]


def paired_composition_prefers_matched(
    prediction: torch.Tensor,
    matched_target: torch.Tensor,
    foil_target: torch.Tensor,
) -> bool:
    """True when the prediction is closer in cosine to the matched target."""
    unit = _unit(prediction)
    matched = float(unit @ _unit(matched_target))
    foil = float(unit @ _unit(foil_target))
    return matched > foil
