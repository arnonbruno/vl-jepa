"""Trainer for the conditional predictor.

The constructor does not seed. Call :func:`seed_everything` before building
the model, or a resumed process will not match a fresh one.

One AdamW group uses the configured learning rate. There is no 20x predictor
group, no EMA, and no WiSE-FT. Targets are detached, so the target encoder
cannot receive gradient even if the caller marked those tensors as trainable.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.conditional.config import PredictorConfig
from src.conditional.losses import (
    assert_single_task,
    cosine_positive_loss,
    multi_positive_infonce,
)
from src.conditional.model import ConditionalLatentPredictor


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExposureLedger:
    def __init__(self) -> None:
        self.images: Counter[str] = Counter()
        self.captions: Counter[str] = Counter()

    def record(self, image_ids: Sequence[str], caption_ids: Sequence[str]) -> None:
        self.images.update(str(item) for item in image_ids)
        self.captions.update(str(item) for item in caption_ids)

    def state_dict(self) -> dict[str, dict[str, int]]:
        return {"images": dict(self.images), "captions": dict(self.captions)}

    def load_state_dict(self, state: Mapping[str, Mapping[str, int]]) -> None:
        self.images = Counter(state["images"])
        self.captions = Counter(state["captions"])


class WarmupCosine:
    """Step scheduler. Step 0 uses the first warmup factor. No implicit extra step."""

    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._apply(0)

    def factor(self, step: int) -> float:
        if self.warmup_steps <= 0:
            progress = min(1.0, step / float(self.total_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        span = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / float(span)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self, step: int) -> None:
        factor = self.factor(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def step(self) -> None:
        self.last_step += 1
        self._apply(self.last_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": self.last_step,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.last_step = int(state["last_step"])
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self._apply(self.last_step)


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


class ConditionalTrainer:
    def __init__(self, model: ConditionalLatentPredictor, config: PredictorConfig, total_steps: int) -> None:
        self.model = model
        self.config = config
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("conditional predictor has no trainable parameters")
        if model.vision_encoder is not None:
            if any(parameter.requires_grad for parameter in model.vision_encoder.parameters()):
                raise RuntimeError("vision encoder must stay frozen")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        if len(self.optimizer.param_groups) != 1:
            raise RuntimeError("expected a single AdamW parameter group")
        self.total_steps = int(total_steps)
        warmup = int(round(config.warmup_fraction * self.total_steps))
        self.scheduler = WarmupCosine(self.optimizer, self.total_steps, warmup)
        self.step = 0
        self.ledger = ExposureLedger()
        self.scaler_state = None
        self.sampler_state = None
        self.grad_clip = float(config.grad_clip)
        self.last_preclip_grad_norm: float | None = None
        self.last_postclip_grad_norm: float | None = None

    def train_step(self, batch: Mapping[str, Any]) -> float:
        self.model.train()
        assert_single_task(list(batch["task_ids"]))
        visual = batch["visual_tokens"].detach()
        query = batch["query_embeddings"].detach()
        candidates = batch["candidate_targets"].detach()
        prediction = self.model.predict_from_tokens(visual, query)
        if self.config.objective == "infonce":
            loss = multi_positive_infonce(
                prediction,
                candidates,
                batch["positive_mask"],
                self.config.temperature,
            )
        elif self.config.objective == "cosine":
            loss = cosine_positive_loss(prediction, candidates, batch["positive_mask"])
        else:
            raise ValueError(self.config.objective)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.last_preclip_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip).item()
        )
        self.last_postclip_grad_norm = _grad_norm(parameters)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self.ledger.record(batch["image_ids"], batch.get("caption_ids", []))
        return float(loss.detach().item())

    def student_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def evaluated_state(self) -> dict[str, torch.Tensor]:
        """Raw student weights. This family has no EMA and no WiSE-FT mixture."""
        return self.student_state()

    def checkpoint_extra(self) -> dict[str, Any]:
        return {
            "ledger": self.ledger.state_dict(),
            "sampler_state": self.sampler_state,
        }

    def load_training_state(
        self,
        *,
        student_state: Mapping[str, torch.Tensor],
        optimizer_state: Mapping[str, Any],
        scheduler_state: Mapping[str, Any] | None,
        rng_state: Mapping[str, Any] | None,
        step: int,
        extra: Mapping[str, Any] | None,
    ) -> None:
        self.model.load_state_dict(student_state)
        self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        if rng_state is not None:
            from src.protocol.artifacts import restore_rng_state

            restore_rng_state(rng_state)
        self.step = int(step)
        if extra:
            self.ledger.load_state_dict(extra["ledger"])
            self.sampler_state = extra.get("sampler_state")
