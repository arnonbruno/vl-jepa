"""Standard cross-modal retrieval metrics (COCO/Flickr Karpathy protocol).

The training loop's :func:`src.trainer.retrieval_recall` uses a simplified
"one caption per image" square-matrix protocol. That is fine as a cheap
training signal but it is **not** the protocol every retrieval paper reports,
so its numbers are not comparable to published baselines (CLIP, BLIP, SigLIP…).

The community-standard COCO protocol evaluates 5 captions per image:

  * **Image->Text (i2t):** for each image, rank *all* captions; the image is
    retrieved-correctly @K if *any* of its 5 ground-truth captions appears in
    the top-K. The reported rank is that of the best-ranked correct caption.
  * **Text->Image (t2i):** for each caption, rank *all* images; correct @K if
    the caption's own image is in the top-K.

Metrics: Recall@1/5/10, median rank (Med r), mean rank (Mean r) for both
directions, plus ``rsum`` (sum of the six recalls), the standard single-number
summary used to compare retrieval systems.

This module is pure tensor math (no data/IO) so it can be unit-tested with
synthetic embeddings and reused by both the experiment script and notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


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
    ranks_f = ranks.float()
    out: Dict[str, float] = {}
    for k in ks:
        out[f"r{k}"] = float((ranks < k).float().mean().item()) * 100.0
    out["medr"] = float(ranks_f.median().item()) + 1.0
    out["meanr"] = float(ranks_f.mean().item()) + 1.0
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
) -> Dict[str, float]:
    """Compute the standard COCO/Flickr multi-caption retrieval metrics.

    Args:
        image_embs: ``(Ni, D)`` image embeddings.
        text_embs: ``(Nt, D)`` caption embeddings.
        text_to_image: ``(Nt,)`` index of the source image for each caption.
        ks: recall cutoffs to report.
        chunk: query-dimension chunk size to bound peak memory.
        normalize: L2-normalize embeddings before the dot product (cosine sim).

    Returns:
        Flat dict with ``i2t_r{k}``, ``t2i_r{k}``, ``i2t_medr``, ``t2i_medr``,
        ``i2t_meanr``, ``t2i_meanr`` and ``rsum`` (sum of all recalls).
    """
    if image_embs.dim() != 2 or text_embs.dim() != 2:
        raise ValueError("image_embs and text_embs must be 2-D (N, D)")
    if image_embs.size(1) != text_embs.size(1):
        raise ValueError(
            f"embedding dims differ: {image_embs.size(1)} vs {text_embs.size(1)}"
        )
    if text_to_image.numel() != text_embs.size(0):
        raise ValueError(
            f"text_to_image has {text_to_image.numel()} entries but there are "
            f"{text_embs.size(0)} captions"
        )

    image_embs = image_embs.float()
    text_embs = text_embs.float()
    if normalize:
        image_embs = F.normalize(image_embs, dim=-1, eps=1e-6)
        text_embs = F.normalize(text_embs, dim=-1, eps=1e-6)

    image_to_texts = build_image_to_texts(text_to_image, image_embs.size(0))

    t2i = _recall_summary(
        _t2i_ranks(text_embs, image_embs, text_to_image, chunk), ks
    )
    i2t = _recall_summary(
        _i2t_ranks(image_embs, text_embs, image_to_texts, chunk), ks
    )

    out: Dict[str, float] = {}
    for k in ks:
        out[f"i2t_r{k}"] = i2t[f"r{k}"]
        out[f"t2i_r{k}"] = t2i[f"r{k}"]
    out["i2t_medr"] = i2t["medr"]
    out["t2i_medr"] = t2i["medr"]
    out["i2t_meanr"] = i2t["meanr"]
    out["t2i_meanr"] = t2i["meanr"]
    out["rsum"] = sum(out[f"i2t_r{k}"] + out[f"t2i_r{k}"] for k in ks)
    return out


def format_metrics(metrics: Dict[str, float], ks: Sequence[int] = (1, 5, 10)) -> str:
    """Render a metrics dict as a compact two-line report."""
    def row(direction: str) -> str:
        recalls = " ".join(
            f"R@{k}={metrics[f'{direction}_r{k}']:5.2f}" for k in ks
        )
        return (
            f"  {direction.upper():4s} {recalls} "
            f"MedR={metrics[f'{direction}_medr']:.1f} "
            f"MeanR={metrics[f'{direction}_meanr']:.1f}"
        )

    return (
        f"{row('i2t')}\n{row('t2i')}\n  rsum={metrics['rsum']:.2f}"
    )
