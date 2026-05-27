"""Synthetic alignment overfit sanity test."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import COCOCaptionDataset, CocoDatasetError
from src.model import ProjectionHead, VL_JEPA, sigmoid_contrastive_loss
from src.trainer import VL_JEPA_Trainer


def _siglip_loss_and_acc(
    vision_proj: torch.Tensor,
    text_proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss, acc = sigmoid_contrastive_loss(
        vision_proj,
        text_proj,
        logit_scale=torch.tensor(2.659, device=vision_proj.device),
        logit_bias=torch.tensor(-10.0, device=vision_proj.device),
    )
    return loss, acc


def _tiny_coco_loader(
    *,
    image_size: int,
    max_caption_length: int,
    num_samples: int = 32,
    batch_size: int = 8,
) -> DataLoader:
    coco_root = os.environ.get("COCO_ROOT")
    try:
        dataset = COCOCaptionDataset(
            split="train",
            coco_root=coco_root,
            image_size=image_size,
            max_caption_length=max_caption_length,
            download=False,
        )
    except (CocoDatasetError, RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"COCO dataset unavailable locally: {exc}")
    except Exception as exc:  # pragma: no cover - optional dependency edge case
        pytest.skip(f"COCO dataset could not be loaded: {exc}")

    if len(dataset) < num_samples:
        pytest.skip(f"Need at least {num_samples} COCO samples, found {len(dataset)}")
    subset = Subset(dataset, list(range(num_samples)))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def test_alignment_overfits_tiny_real_coco_pairs() -> None:
    """Frozen encoders + trainable heads should overfit tiny real image-caption pairs."""
    torch.manual_seed(123)
    image_size = 64
    loader = _tiny_coco_loader(image_size=image_size, max_caption_length=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VL_JEPA(
        hidden_dim=192,
        patch_size=16,
        image_size=image_size,
        predictor_layers=1,
        mask_ratio=0.6,
        freeze_encoders=True,
        projection_dim=96,
        contrastive_loss='siglip',
    )
    trainer = VL_JEPA_Trainer(
        model,
        device,
        learning_rate=1e-3,
        warmup_steps=0,
        max_steps=512,
        alpha=0.0,
        beta=1.0,
        gamma=0.01,
        memory_bank_size=0,
    )

    losses = []
    accs = []
    for _ in range(10):
        for images, input_ids, attention_mask in loader:
            metrics = trainer.train_step(images, input_ids, attention_mask)
            if metrics.get('skipped'):
                continue
            losses.append(metrics['total_loss'])
            accs.append(metrics['nce_acc'])

    assert len(losses) >= 10
    first_window = sum(losses[:10]) / 10
    last_window = sum(losses[-10:]) / 10
    final_acc = sum(accs[-10:]) / 10

    assert last_window < first_window
    assert final_acc > 0.05

    model.eval()
    image_embeds = []
    text_embeds = []
    eval_loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images, input_ids, attention_mask in eval_loader:
            v, t = model.get_joint_embedding(
                images.to(device),
                input_ids.to(device),
                attention_mask.to(device),
            )
            image_embeds.append(v.cpu())
            text_embeds.append(t.cpu())

    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    logits = image_embeds @ text_embeds.T
    target = torch.arange(logits.size(0))
    retrieval_acc = (
        (logits.argmax(dim=1) == target).float().mean()
        + (logits.argmax(dim=0) == target).float().mean()
    ) / 2
    assert retrieval_acc.item() >= (1.0 / logits.size(0))


def test_siglip_loss_converges_on_easy_pairs() -> None:
    """Directly optimize embeddings and verify SigLIP convergence."""
    torch.manual_seed(321)
    num_pairs = 64
    dim = 32

    anchor = F.normalize(torch.randn(num_pairs, dim), dim=-1)
    vision = torch.nn.Parameter(anchor + 0.25 * torch.randn_like(anchor))
    text = torch.nn.Parameter(anchor + 0.25 * torch.randn_like(anchor))
    optimizer = torch.optim.Adam([vision, text], lr=5e-2)

    start_loss, start_acc = _siglip_loss_and_acc(F.normalize(vision, dim=-1), F.normalize(text, dim=-1))
    final_loss = start_loss
    final_acc = start_acc
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        v = F.normalize(vision, dim=-1)
        t = F.normalize(text, dim=-1)
        final_loss, final_acc = _siglip_loss_and_acc(v, t)
        final_loss.backward()
        optimizer.step()

    assert final_loss.item() < start_loss.item()
    assert final_acc.item() > 0.75
