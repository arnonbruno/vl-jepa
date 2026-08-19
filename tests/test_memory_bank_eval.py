"""Val/eval contrastive loss must not read the training memory bank."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model import VL_JEPA, MemoryBank, compute_jepa_loss
from src.trainer import VL_JEPA_Trainer


def test_eval_step_nce_independent_of_filled_queue() -> None:
    device = torch.device("cpu")
    model = VL_JEPA(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
        contrastive_loss="infonce",
    )
    trainer = VL_JEPA_Trainer(
        model, device, warmup_steps=0, max_steps=50,
        alpha=0.0, beta=1.0, gamma=0.0, memory_bank_size=64,
    )
    assert trainer.memory_bank is not None

    images = torch.randn(4, 3, 64, 64)
    input_ids = torch.randint(0, 100, (4, 8))

    empty = trainer.eval_step(images, input_ids, mask_seed=0)
    junk = F.normalize(torch.randn(32, 96), dim=-1)
    trainer.memory_bank.enqueue(junk)
    assert trainer.memory_bank.num_filled > 0
    filled = trainer.eval_step(images, input_ids, mask_seed=0)

    assert empty["skipped"] is False
    assert filled["skipped"] is False
    assert empty["nce_loss"] == filled["nce_loss"]


def test_siglip_loss_independent_of_memory_bank() -> None:
    model = VL_JEPA(
        hidden_dim=96,
        patch_size=16,
        image_size=64,
        predictor_layers=1,
        freeze_encoders=False,
        projection_dim=96,
        contrastive_loss="siglip",
    )
    images = torch.randn(4, 3, 64, 64)
    input_ids = torch.randint(0, 100, (4, 8))
    outputs = model(images, input_ids, compute_jepa=False)

    bank = MemoryBank(64, 96, torch.device("cpu"))
    bank.enqueue(F.normalize(torch.randn(32, 96), dim=-1))
    assert bank.num_filled > 0

    empty = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0, memory_bank=None)
    filled = compute_jepa_loss(outputs, alpha=0.0, beta=1.0, gamma=0.0, memory_bank=bank)
    assert torch.equal(empty["nce_loss"], filled["nce_loss"])
    assert torch.equal(empty["total_loss"], filled["total_loss"])
