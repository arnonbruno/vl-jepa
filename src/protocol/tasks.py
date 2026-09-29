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


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(_unit(left) @ _unit(right))


def sugarcrepe_pp_example(
    description: torch.Tensor,
    positive_a: torch.Tensor,
    positive_b: torch.Tensor,
    negative: torch.Tensor,
) -> dict[str, bool]:
    """SugarCrepe++ image-to-text and text-only hits. Both must pass.

    Image-to-text requires the worse positive to beat the negative.
    Text-only requires the two positives to beat both positive-negative pairs.
    Averaging two separate positive-versus-negative checks is a different score.
    """
    image_positive_a = _cosine(description, positive_a)
    image_positive_b = _cosine(description, positive_b)
    image_negative = _cosine(description, negative)
    text_pair = _cosine(positive_a, positive_b)
    text_negative_a = _cosine(positive_a, negative)
    text_negative_b = _cosine(positive_b, negative)
    itt_hit = min(image_positive_a, image_positive_b) > image_negative
    tot_hit = text_pair > max(text_negative_a, text_negative_b)
    return {"itt_hit": itt_hit, "tot_hit": tot_hit, "hit": itt_hit and tot_hit}


def sugarcrepe_pp_accuracy(examples: Sequence[dict[str, bool]]) -> dict[str, float | int]:
    if not examples:
        raise ValueError("SugarCrepe++ accuracy needs at least one example")
    n = len(examples)
    both = sum(bool(example["hit"]) for example in examples)
    independent = []
    for example in examples:
        independent.append(0.5 * (float(example["itt_hit"]) + float(example["tot_hit"])))
    return {
        "n": n,
        "hits": both,
        "accuracy": both / n,
        "independent_average": float(sum(independent) / n),
    }


def winoground_example(
    image_0: torch.Tensor,
    image_1: torch.Tensor,
    text_0: torch.Tensor,
    text_1: torch.Tensor,
) -> dict[str, bool]:
    """Strict Winoground text, image, and group scores. Ties are misses."""
    score_00 = _cosine(image_0, text_0)
    score_01 = _cosine(image_0, text_1)
    score_10 = _cosine(image_1, text_0)
    score_11 = _cosine(image_1, text_1)
    text_hit = score_00 > score_01 and score_11 > score_10
    image_hit = score_00 > score_10 and score_11 > score_01
    return {"text": text_hit, "image": image_hit, "group": text_hit and image_hit}


def winoground_scores(examples: Sequence[dict[str, bool]]) -> dict[str, float | int | bool]:
    if not examples:
        raise ValueError("Winoground scores need at least one example")
    n = len(examples)
    text_hits = sum(bool(example["text"]) for example in examples)
    image_hits = sum(bool(example["image"]) for example in examples)
    group_hits = sum(bool(example["group"]) for example in examples)
    return {
        "n": n,
        "text_score": text_hits / n,
        "image_score": image_hits / n,
        "group_score": group_hits / n,
        "ties_count_as_hits": False,
        "significance_claim": False,
    }


def gqa_fixed_candidate_diagnostic(
    predictions: torch.Tensor,
    gold_ids: Sequence[str],
    candidate_ids: Sequence[str],
    candidate_embeddings: torch.Tensor,
) -> dict[str, float | int | str | None]:
    """Fixed-candidate GQA diagnostic. This is not unrestricted GQA scoring."""
    if predictions.size(0) != len(gold_ids):
        raise ValueError("one prediction per gold answer")
    frozen = list(candidate_ids)
    hits = 0
    covered = 0
    for index, gold in enumerate(gold_ids):
        in_vocabulary = gold in frozen
        if in_vocabulary:
            covered += 1
            ranking = answer_ranking(predictions[index], candidate_embeddings, frozen)
            if ranking[0] == gold:
                hits += 1
    total = len(gold_ids)
    return {
        "name": "gqa_fixed_candidate_diagnostic",
        "n": total,
        "n_covered": covered,
        "n_hits": hits,
        "overall_accuracy": hits / total,
        "coverage": covered / total,
        "accuracy_given_coverage": None if covered == 0 else hits / covered,
        "overall_denominator": total,
        "coverage_denominator": total,
        "conditional_denominator": covered,
    }
