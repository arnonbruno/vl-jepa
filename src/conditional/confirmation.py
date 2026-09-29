"""Confirmation rule and a nine-run synthetic rehearsal.

The rehearsal trains three configs on three seeds. It does not increment
``confirmation_runs_executed`` and it does not unblind the preregistered endpoints.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.conditional.config import config_from_mapping
from src.conditional.model import build_predictor
from src.conditional.trainer import ConditionalTrainer, seed_everything
from src.protocol.preregistration import PREREGISTRATION

SYNTHETIC_CONFIRMATION_SCOPE = "synthetic_confirmation_not_unblinded"


def decide_confirmation(
    *,
    gain_points: float,
    paired_ci_low: float,
    paired_ci_high: float,
    r1_degradation: Mapping[str, float],
) -> str:
    """Return ``proceed`` or ``stop`` from the preregistered thresholds.

    ``r1_degradation`` is baseline minus candidate, in percentage points.
    A positive value means the candidate is worse.
    """
    if paired_ci_high < paired_ci_low:
        raise ValueError("paired interval is reversed")
    missing = [name for name in PREREGISTRATION["protected_r1_endpoints"] if name not in r1_degradation]
    if missing:
        raise KeyError(f"missing protected R@1 endpoints: {missing}")
    extra = [name for name in r1_degradation if name not in PREREGISTRATION["protected_r1_endpoints"]]
    if extra:
        raise KeyError(f"unexpected R@1 endpoints: {extra}")
    limit = float(PREREGISTRATION["max_retrieval_r1_degradation_points"])
    gain_ok = float(gain_points) >= float(PREREGISTRATION["primary_compositional_min_gain_points"])
    if PREREGISTRATION["paired_ci_must_exclude_zero"]:
        ci_ok = paired_ci_low > 0.0
    else:
        ci_ok = True
    retrieval_ok = all(float(value) <= limit for value in r1_degradation.values())
    if gain_ok and ci_ok and retrieval_ok:
        return "proceed"
    return "stop"


def run_synthetic_confirmation(*, steps: int = 2) -> list[dict]:
    """Three configs times seeds 0, 1, 2. Labeled so it cannot be unblinded."""
    configs = ("candidate", "matched_baseline", "explanatory_ablation")
    rows = []
    for name in configs:
        for seed in PREREGISTRATION["seeds"]:
            seed_everything(seed)
            config = config_from_mapping(
                {
                    "vision_dim": 8,
                    "query_dim": 8,
                    "target_dim": 8,
                    "predictor_layers": 1,
                    "predictor_width": 32,
                    "predictor_heads": 4,
                    "objective": "cosine",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "seed": seed,
                }
            )
            model = build_predictor(config)
            trainer = ConditionalTrainer(model, config, total_steps=steps)
            batch = _batch(config, seed)
            losses = [trainer.train_step(batch) for _ in range(steps)]
            rows.append(
                {
                    "scope": SYNTHETIC_CONFIRMATION_SCOPE,
                    "config": name,
                    "seed": seed,
                    "steps": steps,
                    "losses": losses,
                    "confirmation_runs_executed": PREREGISTRATION["confirmation_runs_executed"],
                    "status": PREREGISTRATION["status"],
                }
            )
    return rows


def _batch(config, seed: int) -> dict:
    generator = torch.Generator().manual_seed(1000 + seed)
    batch = 4
    return {
        "visual_tokens": torch.randn(batch, 2, config.vision_dim, generator=generator),
        "query_embeddings": torch.randn(batch, config.query_dim, generator=generator),
        "candidate_targets": torch.randn(batch, config.target_dim, generator=generator),
        "positive_mask": torch.eye(batch, dtype=torch.bool),
        "task_ids": ["caption"] * batch,
        "image_ids": [f"synthetic:{index}" for index in range(batch)],
        "caption_ids": [f"synthetic-cap:{index}" for index in range(batch)],
    }


def confirmation_configs(rows: Sequence[Mapping]) -> set[str]:
    return {str(row["config"]) for row in rows}


def paired_bootstrap_ci(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile interval for the paired per-image mean difference.

    ``candidate`` and ``baseline`` are aligned on the same images or scenes.
    The interval is an input to :func:`decide_confirmation`, not a benchmark row.
    """
    if candidate.shape != baseline.shape or candidate.dim() != 1 or candidate.numel() < 2:
        raise ValueError("paired bootstrap needs two aligned 1-D samples of length >= 2")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    difference = candidate.float() - baseline.float()
    generator = torch.Generator().manual_seed(seed)
    size = difference.numel()
    draws = []
    for _ in range(n_resamples):
        index = torch.randint(0, size, (size,), generator=generator)
        draws.append(difference[index].mean())
    samples = torch.stack(draws)
    return {
        "mean_difference": float(difference.mean().item()),
        "ci_low": float(torch.quantile(samples, alpha / 2.0).item()),
        "ci_high": float(torch.quantile(samples, 1.0 - alpha / 2.0).item()),
    }
