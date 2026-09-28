"""Target-screen gate and a tiny synthetic harness.

``run_target_screen`` raises unless every guard is set. Passing the gate does
not train the six arms. Plumbing rows use scope ``synthetic_harness_not_a_result``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.conditional.config import PredictorConfig, config_from_mapping
from src.conditional.model import build_predictor
from src.conditional.targets import FIXED_QUERY_ENCODER_ID, TARGET_SPECS
from src.conditional.trainer import seed_everything
from src.protocol.hashing import sha256_bytes


class ScreenRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenArm:
    target_id: str
    objective: str
    query_encoder_id: str
    seed: int
    max_steps: int | None
    status: str


def planned_screen_arms() -> list[ScreenArm]:
    """Template only. The step budget is not frozen, and the status is not launched."""
    arms = []
    for target_id in TARGET_SPECS:
        for objective in ("infonce", "cosine"):
            arms.append(
                ScreenArm(
                    target_id=target_id,
                    objective=objective,
                    query_encoder_id=FIXED_QUERY_ENCODER_ID,
                    seed=0,
                    max_steps=None,
                    status="not_launched",
                )
            )
    return arms


def assert_shared_screen(arms: list[ScreenArm]) -> None:
    if len(arms) != 6:
        raise ValueError(f"the target screen has 6 arms, got {len(arms)}")
    query = {arm.query_encoder_id for arm in arms}
    seeds = {arm.seed for arm in arms}
    steps = {arm.max_steps for arm in arms}
    if len(query) != 1 or len(seeds) != 1 or len(steps) != 1:
        raise ValueError("screen arms must share query encoder, seed, and step budget")
    if any(arm.max_steps is None for arm in arms):
        raise ValueError("a frozen screen requires an integer step budget on every arm")


def run_target_screen(
    *,
    allow_execution: bool = False,
    frozen_shared_budget: bool = False,
    manifests_clean: bool = False,
    visual_cache_sentinel_present: bool = False,
) -> dict:
    if not (
        allow_execution
        and frozen_shared_budget
        and manifests_clean
        and visual_cache_sentinel_present
    ):
        raise ScreenRefused(
            "target screen refused: it needs allow_execution, a frozen shared "
            "budget, clean manifests, and a visual-cache sentinel. This tree "
            "does not create that sentinel and does not launch the six runs."
        )
    return {
        "status": "gate_open_execution_unwired",
        "runs_executed": 0,
        "note": "Passing the gate does not train the scientific arms.",
    }


def _state_sha256(module: torch.nn.Module) -> str:
    blob = bytearray()
    for key, value in module.state_dict().items():
        blob.extend(key.encode("utf-8"))
        blob.extend(b"\0")
        blob.extend(value.detach().cpu().contiguous().numpy().tobytes())
    return sha256_bytes(bytes(blob))


def run_synthetic_plumbing(*, steps: int = 2, seed: int = 0) -> list[dict]:
    """Two steps on six tiny arms. Same initialization hash. Not a result."""
    rows = []
    init_hashes = []
    objectives = ("infonce", "cosine")
    for target_id in TARGET_SPECS:
        for objective in objectives:
            seed_everything(seed)
            config = config_from_mapping(
                {
                    "vision_dim": 8,
                    "query_dim": 8,
                    "target_dim": 8,
                    "predictor_layers": 1,
                    "predictor_width": 32,
                    "predictor_heads": 4,
                    "objective": objective,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "microbatch": 4,
                    "seed": seed,
                }
            )
            model = build_predictor(config)
            init_hashes.append(_state_sha256(model))
            from src.conditional.trainer import ConditionalTrainer

            trainer = ConditionalTrainer(model, config, total_steps=steps)
            losses = []
            batch = _plumbing_batch(config)
            for _ in range(steps):
                losses.append(trainer.train_step(batch))
            rows.append(
                {
                    "scope": "synthetic_harness_not_a_result",
                    "target_id": target_id,
                    "objective": objective,
                    "seed": seed,
                    "steps": steps,
                    "losses": losses,
                    "init_sha256": init_hashes[-1],
                }
            )
    if len(set(init_hashes)) != 1:
        raise RuntimeError("synthetic arms did not share an initialization hash")
    return rows


def _plumbing_batch(config: PredictorConfig) -> dict:
    generator = torch.Generator()
    generator.manual_seed(123)
    batch = 4
    visual = torch.randn(batch, 2, config.vision_dim, generator=generator)
    query = torch.randn(batch, config.query_dim, generator=generator)
    targets = torch.randn(batch, config.target_dim, generator=generator)
    mask = torch.eye(batch, dtype=torch.bool)
    return {
        "visual_tokens": visual,
        "query_embeddings": query,
        "candidate_targets": targets,
        "positive_mask": mask,
        "task_ids": ["caption"] * batch,
        "image_ids": [f"synthetic:{i}" for i in range(batch)],
        "caption_ids": [f"synthetic-cap:{i}" for i in range(batch)],
    }


def benchmark_rows(rows: list[dict]) -> list[dict]:
    """Drop plumbing rows. They must not appear as benchmark measurements."""
    return [row for row in rows if row.get("scope") != "synthetic_harness_not_a_result"]
