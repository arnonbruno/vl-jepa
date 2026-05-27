"""Synthetic alignment overfit sanity test."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ProjectionHead


def _nce_loss_and_acc(
    vision_proj: torch.Tensor,
    text_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = vision_proj @ text_proj.T * 20.0
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2
    acc = (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.argmax(dim=0) == labels).float().mean()
    ) / 2
    return loss, acc


def test_alignment_overfits_tiny_synthetic_pairs() -> None:
    """Projection heads should memorize 128 fixed image/text pairs."""
    torch.manual_seed(123)
    num_pairs = 128
    hidden_dim = 64
    projection_dim = 64

    latent = F.normalize(torch.randn(num_pairs, hidden_dim), dim=-1)
    image_features = latent + 0.05 * torch.randn_like(latent)
    text_features = latent + 0.05 * torch.randn_like(latent)

    vision_head = ProjectionHead(hidden_dim, projection_dim)
    text_head = ProjectionHead(hidden_dim, projection_dim)
    optimizer = torch.optim.AdamW(
        list(vision_head.parameters()) + list(text_head.parameters()),
        lr=3e-3,
        weight_decay=0.0,
    )

    final_acc = torch.tensor(0.0)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        vision_proj = vision_head(image_features)
        text_proj = text_head(text_features)
        loss, final_acc = _nce_loss_and_acc(vision_proj, text_proj)
        loss.backward()
        optimizer.step()

    assert final_acc.item() > 0.50
