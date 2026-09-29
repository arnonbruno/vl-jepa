"""Query and image controls for one trained or untrained predictor.

Scores are mean cosine similarity to the paired positive target. They describe
whether the prediction moves when the visual tokens or the query change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def prediction_controls(
    model: torch.nn.Module,
    visual_tokens: torch.Tensor,
    query_embeddings: torch.Tensor,
    positive_targets: torch.Tensor,
) -> dict[str, float | bool]:
    if visual_tokens.size(0) < 2:
        raise ValueError("controls need at least two examples so a shuffle can move a row")
    matched = model.predict_from_tokens(visual_tokens, query_embeddings)
    query_only = model.predict_from_tokens(torch.zeros_like(visual_tokens), query_embeddings)
    shuffled_visual = torch.roll(visual_tokens, shifts=1, dims=0)
    image_shuffled = model.predict_from_tokens(shuffled_visual, query_embeddings)
    incompatible_query = model.predict_from_tokens(visual_tokens, torch.roll(query_embeddings, shifts=1, dims=0))
    return {
        "matched": _mean_cosine(matched, positive_targets),
        "query_only": _mean_cosine(query_only, positive_targets),
        "image_shuffled": _mean_cosine(image_shuffled, positive_targets),
        "incompatible_query": _mean_cosine(incompatible_query, positive_targets),
        "visual_changes_prediction": not torch.allclose(matched, query_only),
        "query_changes_prediction": not torch.allclose(matched, incompatible_query),
        "shuffle_changes_visual": not torch.equal(visual_tokens, shuffled_visual),
    }


def _mean_cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    left = F.normalize(prediction.float(), dim=-1, eps=1e-6)
    right = F.normalize(target.float(), dim=-1, eps=1e-6)
    return float((left * right).sum(dim=-1).mean().item())
