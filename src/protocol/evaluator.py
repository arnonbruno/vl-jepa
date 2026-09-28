"""Retrieval scoring for new runs.

The reference score is ``src.eval_retrieval.compute_retrieval_metrics``:
five captions per image, rank = count of candidates strictly more similar
than the positive. Ties are awarded to the positive. That is an optimistic
bound, and it is the historical metric. This module does not replace it.

A fully tied candidate set makes that optimistic score perfect. Selection
refuses that case instead of treating it as a retrieval result. Pessimistic
ranks and near-tie counts are diagnostics. They are not the reference score.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from src.eval_retrieval import compute_retrieval_metrics

REFERENCE_NAME = "src.eval_retrieval.compute_retrieval_metrics"
TIE_POLICY_REFERENCE = "strict_greater_than_optimistic"
SELECTION_POLICY = "reference_unless_fully_tied"


def _as_float(embs: torch.Tensor) -> torch.Tensor:
    if embs.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        raise TypeError(f"embeddings must be floating point, got {embs.dtype}")
    return embs.float()


def _fully_tied_queries(sims: torch.Tensor, positive_sim: torch.Tensor) -> torch.Tensor:
    """True when every candidate similarity equals the positive similarity."""
    return (sims == positive_sim).all(dim=1)


def pessimistic_t2i_ranks(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    chunk: int = 1024,
) -> torch.Tensor:
    """0-indexed rank. Tied non-positives count as ranked ahead of the positive."""
    text_embs = _as_float(text_embs)
    image_embs = _as_float(image_embs)
    text_to_image = text_to_image.long()
    ranks = torch.empty(text_embs.size(0), dtype=torch.long)
    for start in range(0, text_embs.size(0), chunk):
        end = min(start + chunk, text_embs.size(0))
        sims = text_embs[start:end] @ image_embs.T
        pos = text_to_image[start:end].view(-1, 1)
        pos_sim = sims.gather(1, pos)
        better = (sims > pos_sim).sum(dim=1)
        tied_including_positive = (sims == pos_sim).sum(dim=1)
        ranks[start:end] = better + tied_including_positive - 1
    return ranks


def pessimistic_i2t_ranks(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    image_to_texts: Sequence[torch.Tensor],
    chunk: int = 256,
) -> torch.Tensor:
    """0-indexed rank of the best positive. Tied non-positives count as ahead."""
    image_embs = _as_float(image_embs)
    text_embs = _as_float(text_embs)
    ranks = torch.empty(image_embs.size(0), dtype=torch.long)
    for start in range(0, image_embs.size(0), chunk):
        end = min(start + chunk, image_embs.size(0))
        sims = image_embs[start:end] @ text_embs.T
        for row in range(end - start):
            cols = image_to_texts[start + row].long()
            best = sims[row, cols].max()
            better = int((sims[row] > best).sum().item())
            n_pos_at_best = int((sims[row, cols] == best).sum().item())
            n_tied = int((sims[row] == best).sum().item())
            ranks[start + row] = better + (n_tied - n_pos_at_best)
    return ranks


def _recall_from_ranks(ranks: torch.Tensor, ks: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        out[f"r{k}"] = float((ranks < k).float().mean().item()) * 100.0
    return out


def _exact_tie_fraction_t2i(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    text_to_image: torch.Tensor,
) -> float:
    flags = []
    text_to_image = text_to_image.long()
    for start in range(0, text_embs.size(0), 1024):
        end = min(start + 1024, text_embs.size(0))
        sims = text_embs[start:end] @ image_embs.T
        pos = text_to_image[start:end].view(-1, 1)
        pos_sim = sims.gather(1, pos)
        tied_others = (sims == pos_sim).sum(dim=1) - 1
        flags.append(tied_others > 0)
    return float(torch.cat(flags).float().mean().item()) if flags else 0.0


def _exact_tie_fraction_i2t(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    image_to_texts: Sequence[torch.Tensor],
) -> float:
    hits = 0
    for index in range(image_embs.size(0)):
        sims = image_embs[index] @ text_embs.T
        cols = image_to_texts[index].long()
        best = sims[cols].max()
        positive = torch.zeros(sims.size(0), dtype=torch.bool)
        positive[cols] = True
        nonpositive_ties = int(((sims == best) & ~positive).sum().item())
        hits += int(nonpositive_ties > 0)
    return hits / max(1, image_embs.size(0))


def _near_tie_fraction_t2i(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    atol: float,
) -> float:
    flags = []
    text_to_image = text_to_image.long()
    for start in range(0, text_embs.size(0), 1024):
        end = min(start + 1024, text_embs.size(0))
        sims = text_embs[start:end] @ image_embs.T
        pos = text_to_image[start:end].view(-1, 1)
        pos_sim = sims.gather(1, pos)
        eye = torch.zeros_like(sims, dtype=torch.bool)
        eye.scatter_(1, pos, True)
        near = (sims - pos_sim).abs() <= atol
        flags.append((near & ~eye).any(dim=1))
    return float(torch.cat(flags).float().mean().item()) if flags else 0.0


def _queries_fully_tied(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    image_to_texts: Sequence[torch.Tensor],
) -> bool:
    if image_embs.size(0) < 2 or text_embs.size(0) < 2:
        return False
    text_to_image = text_to_image.long()
    t2i_tied = []
    for start in range(0, text_embs.size(0), 1024):
        end = min(start + 1024, text_embs.size(0))
        sims = text_embs[start:end] @ image_embs.T
        pos_sim = sims.gather(1, text_to_image[start:end].view(-1, 1))
        t2i_tied.append(_fully_tied_queries(sims, pos_sim))
    if not bool(torch.cat(t2i_tied).all().item()):
        return False
    for index in range(image_embs.size(0)):
        sims = image_embs[index] @ text_embs.T
        best = sims[image_to_texts[index].long()].max()
        if not bool((sims == best).all().item()):
            return False
    return True


def _prepare(image_embs: torch.Tensor, text_embs: torch.Tensor, normalize: bool):
    image_embs = _as_float(image_embs)
    text_embs = _as_float(text_embs)
    if not torch.isfinite(image_embs).all() or not torch.isfinite(text_embs).all():
        raise ValueError("embeddings contain non-finite values")
    if normalize:
        if (image_embs.norm(dim=-1) == 0).any() or (text_embs.norm(dim=-1) == 0).any():
            raise ValueError(
                "normalize=True refuses zero-norm embeddings; "
                "they collapse every similarity to a tie"
            )
        image_embs = F.normalize(image_embs, dim=-1, eps=1e-6)
        text_embs = F.normalize(text_embs, dim=-1, eps=1e-6)
    return image_embs, text_embs


def evaluate_retrieval(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
    normalize: bool = True,
    near_tie_atol: float = 0.0,
    captions: Optional[Sequence[str]] = None,
    diagnostics: bool = False,
    chunk: int = 1024,
) -> Dict[str, Any]:
    """Score a five-caption (or generally multi-positive) retrieval pool.

    ``reference`` is exactly ``compute_retrieval_metrics``. ``selection`` equals
    that reference unless every query is fully tied, in which case selection is
    refused. Optimistic diagnostics match the reference. Pessimistic diagnostics
    count tied non-positives as ranked ahead.
    """
    from src.eval_retrieval import build_image_to_texts

    image_raw, text_raw = _prepare(image_embs, text_embs, normalize)
    reference = compute_retrieval_metrics(
        image_raw,
        text_raw,
        text_to_image,
        ks=ks,
        chunk=chunk,
        normalize=False,
        captions=captions,
        diagnostics=diagnostics,
    )
    image_to_texts = build_image_to_texts(text_to_image, image_raw.size(0))
    captions_per_image = [int(cols.numel()) for cols in image_to_texts]
    five_caption = all(count == 5 for count in captions_per_image)
    pessimistic_i2t = _recall_from_ranks(
        pessimistic_i2t_ranks(image_raw, text_raw, image_to_texts, chunk=chunk), ks
    )
    pessimistic_t2i = _recall_from_ranks(
        pessimistic_t2i_ranks(text_raw, image_raw, text_to_image, chunk=chunk), ks
    )
    pessimistic: Dict[str, float] = {}
    optimistic: Dict[str, float] = {}
    for k in ks:
        pessimistic[f"i2t_r{k}"] = pessimistic_i2t[f"r{k}"]
        pessimistic[f"t2i_r{k}"] = pessimistic_t2i[f"r{k}"]
        optimistic[f"i2t_r{k}"] = reference[f"i2t_r{k}"]
        optimistic[f"t2i_r{k}"] = reference[f"t2i_r{k}"]
    pessimistic["rsum"] = sum(pessimistic.values())
    optimistic["rsum"] = sum(
        reference[f"i2t_r{k}"] + reference[f"t2i_r{k}"] for k in ks
    )

    fully_tied = _queries_fully_tied(
        image_raw, text_raw, text_to_image, image_to_texts
    )
    tie_diag = {
        "tie_policy_reference": TIE_POLICY_REFERENCE,
        "reference_implementation": REFERENCE_NAME,
        "constant_or_fully_tied": fully_tied,
        "i2t_exact_tie_query_fraction": _exact_tie_fraction_i2t(
            image_raw, text_raw, image_to_texts
        ),
        "t2i_exact_tie_query_fraction": _exact_tie_fraction_t2i(
            text_raw, image_raw, text_to_image
        ),
        "i2t_near_tie_query_fraction": _near_tie_fraction_i2t_public(
            image_raw, text_raw, image_to_texts, near_tie_atol
        ),
        "t2i_near_tie_query_fraction": _near_tie_fraction_t2i(
            text_raw, image_raw, text_to_image, near_tie_atol
        ),
        "near_tie_atol": float(near_tie_atol),
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "note": (
            "Optimistic ranks count only strictly greater similarities, so an "
            "exact tie is awarded to the positive. Pessimistic ranks count "
            "tied non-positives as ahead. Neither diagnostic replaces "
            "reference when the similarities are not fully tied."
        ),
    }
    if fully_tied:
        selection_status = "rejected_fully_tied_embeddings"
        selection = None
        accepted = False
    else:
        selection_status = "reference_strict_gt"
        selection = {
            key: reference[key]
            for key in reference
            if key != "diagnostics"
        }
        accepted = True
    return {
        "protocol": (
            "five_caption_reference_strict_gt"
            if five_caption
            else "multi_caption_reference_strict_gt"
        ),
        "captions_per_image": captions_per_image,
        "selection_policy": SELECTION_POLICY,
        "selection_status": selection_status,
        "selection_accepted": accepted,
        "selection": selection,
        "reference": {key: value for key, value in reference.items() if key != "diagnostics"},
        "reference_diagnostics": reference.get("diagnostics"),
        "tie_diagnostics": tie_diag,
    }


def _near_tie_fraction_i2t_public(image_embs, text_embs, image_to_texts, atol: float) -> float:
    hits = 0
    for index in range(image_embs.size(0)):
        sims = image_embs[index] @ text_embs.T
        cols = image_to_texts[index].long()
        best = sims[cols].max()
        positive = torch.zeros(sims.size(0), dtype=torch.bool)
        positive[cols] = True
        near = (sims - best).abs() <= atol
        hits += int((near & ~positive).any().item())
    return hits / max(1, image_embs.size(0))


def is_perfect_retrieval(result: Dict[str, Any]) -> bool:
    """True only when a selection was accepted and both R@1 scores are 100."""
    if not result.get("selection_accepted"):
        return False
    selection = result.get("selection") or {}
    return selection.get("i2t_r1") == 100.0 and selection.get("t2i_r1") == 100.0


def mean_r1(result: Dict[str, Any]) -> float | None:
    """Checkpoint-selection scalar. None when selection was refused."""
    if not result.get("selection_accepted"):
        return None
    selection = result["selection"]
    return float(selection["i2t_r1"] + selection["t2i_r1"]) / 2.0
