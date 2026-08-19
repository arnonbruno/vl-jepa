"""Standard cross-modal retrieval metrics (COCO/Flickr Karpathy protocol).

The training loop's :func:`src.trainer.retrieval_recall` uses a simplified
"one caption per image" square-matrix protocol. That is fine as a cheap
training signal but it is **not** the protocol every retrieval paper reports,
so its numbers are not comparable to published baselines (CLIP, BLIP, SigLIP…).
Training recall is also a fraction in ``[0, 1]``; this module reports
percentages in ``[0, 100]``. The two must not be mixed.

The community-standard COCO protocol evaluates 5 captions per image:

  * **Image->Text (i2t):** for each image, rank *all* captions; the image is
    retrieved-correctly @K if *any* of its 5 ground-truth captions appears in
    the top-K. The reported rank is that of the best-ranked correct caption.
  * **Text->Image (t2i):** for each caption, rank *all* images; correct @K if
    the caption's own image is in the top-K.

Metrics: Recall@1/5/10, median rank (Med r), mean rank (Mean r) for both
directions, plus ``rsum`` (sum of the six recalls), the standard single-number
summary used to compare retrieval systems.

Median rank uses ``torch.median`` on 0-indexed ranks (the lower of the two
central values when N is even) then adds 1 to report a 1-indexed MedR. That
is this repository's published convention; it is not the arithmetic mean of
the two central ranks.

Optional ambiguity diagnostics (``diagnostics=True``) report exact
caption-string collisions *within the current evaluation set* and the t2i
Recall@K that would result if those collisions were credited as extra image
positives. They are an evaluation diagnostic, not a retrieval result, not
comparable to published Recall@K, and not evidence of model quality. Matching
uses ``str.strip()`` only (same as the COCO loader). There is no i2t
ambiguity diagnostic: i2t already maxes over an image's captions, and with
this module's strict-``>`` tie policy an exact duplicate string has an
identical embedding and cannot strictly outrank the existing best positive.

This module is pure tensor math (no data/IO) so it can be unit-tested with
synthetic embeddings and reused by both the experiment script and notebooks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

_DIAG_NOTE = (
    "Evaluation diagnostic. Reports positive multiplicity arising from exact "
    "caption-string collisions in this evaluation set, and the t2i Recall@K "
    "that would result if those collisions were credited. Not a retrieval "
    "result. Not comparable to published Recall@K. Standard metrics are "
    "reported unchanged alongside."
)
_POSITIVE_SET_SOURCE = (
    "exact caption-string match within this evaluation set"
)
_CAPTION_NORMALIZER = "str.strip"


def build_image_to_texts(
    text_to_image: torch.Tensor,
    num_images: int,
) -> List[torch.Tensor]:
    """Invert a (Nt,) text->image map into a per-image list of caption indices."""
    text_to_image = text_to_image.long().view(-1)
    buckets: List[List[int]] = [[] for _ in range(num_images)]
    for text_idx, image_idx in enumerate(text_to_image.tolist()):
        if not 0 <= image_idx < num_images:
            raise ValueError(
                f"text_to_image[{text_idx}]={image_idx} out of range [0, {num_images})"
            )
        buckets[image_idx].append(text_idx)
    out: List[torch.Tensor] = []
    for image_idx, cols in enumerate(buckets):
        if not cols:
            raise ValueError(f"image {image_idx} has no captions in text_to_image")
        out.append(torch.tensor(cols, dtype=torch.long))
    return out


@torch.no_grad()
def _t2i_ranks(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """0-indexed rank of each caption's ground-truth image (lower = better)."""
    num_texts = text_embs.size(0)
    ranks = torch.empty(num_texts, dtype=torch.long)
    text_to_image = text_to_image.to(text_embs.device).long()
    for start in range(0, num_texts, chunk):
        end = min(start + chunk, num_texts)
        sims = text_embs[start:end] @ image_embs.t()           # (c, Ni)
        pos = text_to_image[start:end].view(-1, 1)             # (c, 1)
        pos_sim = sims.gather(1, pos)                           # (c, 1)
        # Rank = number of images strictly more similar than the true image.
        ranks[start:end] = (sims > pos_sim).sum(dim=1).cpu()
    return ranks


@torch.no_grad()
def _i2t_ranks(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    image_to_texts: Sequence[torch.Tensor],
    chunk: int,
) -> torch.Tensor:
    """0-indexed rank of each image's best ground-truth caption (lower = better)."""
    num_images = image_embs.size(0)
    ranks = torch.empty(num_images, dtype=torch.long)
    for start in range(0, num_images, chunk):
        end = min(start + chunk, num_images)
        sims = image_embs[start:end] @ text_embs.t()           # (c, Nt)
        for j in range(end - start):
            cols = image_to_texts[start + j].to(sims.device)
            best_pos_sim = sims[j, cols].max()
            ranks[start + j] = int((sims[j] > best_pos_sim).sum().item())
    return ranks


def _recall_summary(ranks: torch.Tensor, ks: Sequence[int]) -> Dict[str, float]:
    # MedR: torch.median is the lower-middle value for even N; +1 makes it 1-indexed.
    ranks_f = ranks.float()
    out: Dict[str, float] = {}
    for k in ks:
        out[f"r{k}"] = float((ranks < k).float().mean().item()) * 100.0
    out["medr"] = float(ranks_f.median().item()) + 1.0
    out["meanr"] = float(ranks_f.mean().item()) + 1.0
    return out


def _require_str_captions(captions: Sequence[str]) -> None:
    for i, cap in enumerate(captions):
        if not isinstance(cap, str):
            raise ValueError(
                f"captions[{i}] must be a string; got {type(cap).__name__}"
            )
        if not cap.strip():
            raise ValueError(
                f"captions[{i}] is empty after str.strip(); "
                "refusing to form an ambiguity group from a malformed caption"
            )


def _caption_image_groups(
    captions: Sequence[str],
    text_to_image: torch.Tensor,
    num_images: int,
) -> tuple[List[str], Dict[str, set[int]]]:
    """Map each caption to distinct source images after ``str.strip()``."""
    if text_to_image.dim() != 1:
        raise ValueError("text_to_image must be a 1-D tensor")
    if len(captions) != text_to_image.size(0):
        raise ValueError(
            f"captions has {len(captions)} entries but text_to_image has "
            f"{text_to_image.size(0)}"
        )
    _require_str_captions(captions)
    groups: Dict[str, set[int]] = defaultdict(set)
    normalized: List[str] = []
    for cap, image_idx in zip(captions, text_to_image.long().tolist()):
        if not 0 <= image_idx < num_images:
            raise ValueError(
                f"text_to_image image index {image_idx} out of range [0, {num_images})"
            )
        key = cap.strip()
        normalized.append(key)
        groups[key].add(int(image_idx))
    return normalized, groups


def caption_positive_images(
    captions: Sequence[str],
    text_to_image: torch.Tensor,
    num_images: int,
) -> List[torch.Tensor]:
    """Sorted, deduplicated image-index sets that share each caption string.

    Normalization is ``str.strip()`` only, matching
    ``experiments.evaluate_retrieval._gather_captions``. Case, punctuation, and
    internal whitespace are significant. Repeated identical captions on the
    *same* image count as one image because the t2i candidate space is images.
    """
    normalized, groups = _caption_image_groups(captions, text_to_image, num_images)
    return [
        torch.tensor(sorted(groups[key]), dtype=torch.int64) for key in normalized
    ]


def caption_positive_multiplicity(
    captions: Sequence[str],
    text_to_image: torch.Tensor,
    num_images: int,
) -> torch.Tensor:
    """Per-caption count of distinct annotated images sharing that caption.

    This is an **evaluation diagnostic**, not a training objective or a SOTA
    claim. Standard COCO t2i Recall@K treats only the source image as positive.
    When the same caption string is a ground-truth annotation on more than one
    image, ranking one of those other annotated images is a provenance-valid
    hit that the standard protocol still scores as a false negative.

    Matching is exact ``str.strip()`` equality on the provided captions. No
    learned similarity, paraphrases, or extra data are used. There is no i2t
    counterpart: i2t already maxes over an image's captions, and exact
    duplicate strings are a no-op under this module's strict-``>`` tie policy.
    """
    positives = caption_positive_images(captions, text_to_image, num_images)
    return torch.tensor([len(s) for s in positives], dtype=torch.int64)


@torch.no_grad()
def _t2i_ranks_any_positive(
    text_embs: torch.Tensor,
    image_embs: torch.Tensor,
    positives: Sequence[torch.Tensor],
    chunk: int,
) -> torch.Tensor:
    """0-indexed rank of each caption's best provenance-valid image."""
    num_texts = text_embs.size(0)
    ranks = torch.empty(num_texts, dtype=torch.long)
    for start in range(0, num_texts, chunk):
        end = min(start + chunk, num_texts)
        sims = text_embs[start:end] @ image_embs.t()
        for j in range(end - start):
            cols = positives[start + j].to(sims.device)
            best_pos_sim = sims[j, cols].max()
            ranks[start + j] = int((sims[j] > best_pos_sim).sum().item())
    return ranks


def _count_zero_norm(embs: torch.Tensor) -> int:
    return int((embs.norm(dim=-1) == 0).sum().item())


def _captions_per_image_stats(
    text_to_image: torch.Tensor, num_images: int,
) -> tuple[int, int, float]:
    counts = torch.bincount(text_to_image.long(), minlength=num_images)
    return (
        int(counts.min().item()),
        int(counts.max().item()),
        float(counts.float().mean().item()),
    )


def average_standard_metrics(
    fold_metrics: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Mean of numeric standard metric keys; skip nested diagnostics and strings."""
    if not fold_metrics:
        raise ValueError("fold_metrics must be non-empty")
    keys = [
        k for k, v in fold_metrics[0].items()
        if k != "diagnostics"
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    ]
    out: Dict[str, float] = {}
    n = len(fold_metrics)
    for key in keys:
        total = 0.0
        for metrics in fold_metrics:
            value = metrics[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"cannot average non-numeric fold metric {key!r}: {type(value).__name__}"
                )
            total += float(value)
        out[key] = total / n
    return out


@torch.no_grad()
def compute_retrieval_metrics(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    text_to_image: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
    chunk: int = 1024,
    normalize: bool = True,
    captions: Optional[Sequence[str]] = None,
    diagnostics: bool = False,
) -> Dict[str, Any]:
    """Compute the standard COCO/Flickr multi-caption retrieval metrics.

    Args:
        image_embs: ``(Ni, D)`` image embeddings.
        text_embs: ``(Nt, D)`` caption embeddings.
        text_to_image: ``(Nt,)`` index of the source image for each caption.
        ks: recall cutoffs to report.
        chunk: query-dimension chunk size to bound peak memory.
        normalize: L2-normalize embeddings before the dot product (cosine sim).
            Zero-norm rows are rejected when ``normalize=True`` because
            ``F.normalize`` would map them to zeros, every similarity would
            tie at 0, and the strict-``>`` rank policy would score those
            queries as rank 0. When ``normalize=False`` that tie behavior is
            left intact and, if diagnostics are on, counted in
            ``n_zero_norm_*``.
        captions: optional caption strings; required when ``diagnostics=True``.
            Ignored when ``diagnostics=False``.
        diagnostics: if True, add one nested ``diagnostics`` dict with
            caption-string multiplicity stats and an ambiguity-aware t2i
            Recall@K. That dict is an evaluation diagnostic, not a
            replacement for standard Recall@K and not a quality/SOTA claim.
            It must not be folded into ``rsum`` or printed as a result row.

    Returns:
        Dict with ``i2t_r{k}``, ``t2i_r{k}``, ``i2t_medr``, ``t2i_medr``,
        ``i2t_meanr``, ``t2i_meanr`` and ``rsum`` (sum of all recalls). When
        ``diagnostics=True``, also ``diagnostics`` (a nested dict). Standard
        keys are unchanged when diagnostics are off.
    """
    if image_embs.dim() != 2 or text_embs.dim() != 2:
        raise ValueError("image_embs and text_embs must be 2-D (N, D)")
    if image_embs.size(0) == 0 or text_embs.size(0) == 0:
        raise ValueError("image_embs and text_embs must be non-empty")
    if image_embs.size(1) != text_embs.size(1):
        raise ValueError(
            f"embedding dims differ: {image_embs.size(1)} vs {text_embs.size(1)}"
        )
    if text_to_image.dim() != 1:
        raise ValueError("text_to_image must be a 1-D tensor of shape (num_texts,)")
    if text_to_image.size(0) != text_embs.size(0):
        raise ValueError(
            f"text_to_image has {text_to_image.size(0)} entries but there are "
            f"{text_embs.size(0)} captions"
        )

    if diagnostics:
        if captions is None:
            raise ValueError("diagnostics=True requires captions")
        if len(captions) != text_embs.size(0):
            raise ValueError(
                f"captions has {len(captions)} entries but there are "
                f"{text_embs.size(0)} text embeddings"
            )
        _require_str_captions(captions)

    image_embs = image_embs.float()
    text_embs = text_embs.float()
    n_zero_img = _count_zero_norm(image_embs)
    n_zero_txt = _count_zero_norm(text_embs)
    if normalize:
        if n_zero_img or n_zero_txt:
            raise ValueError(
                f"normalize=True refuses zero-norm embeddings "
                f"(n_zero_image={n_zero_img}, n_zero_text={n_zero_txt}): "
                "F.normalize maps them to zeros, similarities all tie at 0, "
                "and the strict-'>' rank policy would score those queries as rank 0"
            )
        image_embs = F.normalize(image_embs, dim=-1, eps=1e-6)
        text_embs = F.normalize(text_embs, dim=-1, eps=1e-6)

    image_to_texts = build_image_to_texts(text_to_image, image_embs.size(0))

    t2i = _recall_summary(
        _t2i_ranks(text_embs, image_embs, text_to_image, chunk), ks
    )
    i2t = _recall_summary(
        _i2t_ranks(image_embs, text_embs, image_to_texts, chunk), ks
    )

    out: Dict[str, Any] = {}
    for k in ks:
        out[f"i2t_r{k}"] = i2t[f"r{k}"]
        out[f"t2i_r{k}"] = t2i[f"r{k}"]
    out["i2t_medr"] = i2t["medr"]
    out["t2i_medr"] = t2i["medr"]
    out["i2t_meanr"] = i2t["meanr"]
    out["t2i_meanr"] = t2i["meanr"]
    out["rsum"] = sum(out[f"i2t_r{k}"] + out[f"t2i_r{k}"] for k in ks)

    if diagnostics:
        assert captions is not None
        positives = caption_positive_images(
            captions, text_to_image, image_embs.size(0),
        )
        multiplicity = torch.tensor([len(s) for s in positives], dtype=torch.int64)
        normalized = [cap.strip() for cap in captions]
        occ = Counter(normalized)
        unique_pos = {}
        for key, pos in zip(normalized, positives):
            unique_pos.setdefault(key, pos)
        cap_min, cap_max, cap_mean = _captions_per_image_stats(
            text_to_image, image_embs.size(0),
        )
        diag_t2i = _recall_summary(
            _t2i_ranks_any_positive(text_embs, image_embs, positives, chunk), ks
        )
        diag: Dict[str, Any] = {
            "note": _DIAG_NOTE,
            "positive_set_source": _POSITIVE_SET_SOURCE,
            "caption_normalizer": _CAPTION_NORMALIZER,
        }
        for k in ks:
            diag[f"t2i_r{k}_any_gt"] = float(diag_t2i[f"r{k}"])
        diag["t2i_medr_any_gt"] = float(diag_t2i["medr"])
        diag["t2i_meanr_any_gt"] = float(diag_t2i["meanr"])
        diag["caption_multiplicity_mean"] = float(multiplicity.float().mean().item())
        diag["caption_multiplicity_max"] = int(multiplicity.max().item())
        diag["frac_ambiguous_captions"] = float((multiplicity > 1).float().mean().item())
        diag["n_duplicate_text_groups"] = int(sum(1 for n in occ.values() if n > 1))
        diag["n_cross_image_duplicate_groups"] = int(
            sum(1 for pos in unique_pos.values() if len(pos) > 1)
        )
        diag["captions_per_image_min"] = cap_min
        diag["captions_per_image_max"] = cap_max
        diag["captions_per_image_mean"] = cap_mean
        diag["n_zero_norm_image_embs"] = int(n_zero_img)
        diag["n_zero_norm_text_embs"] = int(n_zero_txt)
        out["diagnostics"] = diag
    return out


def format_metrics(metrics: Dict[str, Any], ks: Sequence[int] = (1, 5, 10)) -> str:
    """Render standard metric keys as a compact two-line report.

    Nested ``diagnostics`` and any non-standard keys are ignored. Requested
    ``ks`` that are missing from ``metrics`` are skipped so a non-default
    recall cutoff cannot raise after a long encode.
    """
    usable = [k for k in ks if f"i2t_r{k}" in metrics and f"t2i_r{k}" in metrics]
    if not usable or len(usable) != len(tuple(ks)):
        discovered = sorted(
            int(key[5:])
            for key in metrics
            if key.startswith("i2t_r") and key[5:].isdigit()
        )
        if discovered:
            usable = discovered

    def row(direction: str) -> str:
        recalls = " ".join(
            f"R@{k}={metrics[f'{direction}_r{k}']:5.2f}" for k in usable
        )
        return (
            f"  {direction.upper():4s} {recalls} "
            f"MedR={metrics[f'{direction}_medr']:.1f} "
            f"MeanR={metrics[f'{direction}_meanr']:.1f}"
        )

    return (
        f"{row('i2t')}\n{row('t2i')}\n  rsum={metrics['rsum']:.2f}"
    )
