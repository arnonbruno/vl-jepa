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


NON_BENCHMARK_SCOPES = frozenset(
    {
        "synthetic_harness_not_a_result",
        "synthetic_screen_not_a_result",
        "synthetic_mechanism_not_a_result",
        "synthetic_confirmation_not_unblinded",
        "throughput_probe_not_a_result",
    }
)


def benchmark_rows(rows: list[dict]) -> list[dict]:
    """Drop rehearsal rows. They are not benchmark measurements."""
    return [row for row in rows if row.get("scope") not in NON_BENCHMARK_SCOPES]


@dataclass(frozen=True)
class SharedSchedule:
    """Caption-only steps, then QA steps. Both phases are required and shared."""

    caption_steps: int
    qa_steps: int

    def __post_init__(self) -> None:
        if self.caption_steps < 1 or self.qa_steps < 1:
            raise ValueError("the shared schedule needs a caption phase and a QA phase")

    @property
    def total_steps(self) -> int:
        return self.caption_steps + self.qa_steps


def frozen_screen_arms(max_steps: int, seed: int = 0) -> list[ScreenArm]:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    arms = [
        ScreenArm(
            target_id=target_id,
            objective=objective,
            query_encoder_id=FIXED_QUERY_ENCODER_ID,
            seed=seed,
            max_steps=max_steps,
            status="ready",
        )
        for target_id in TARGET_SPECS
        for objective in ("infonce", "cosine")
    ]
    assert_shared_screen(arms)
    return arms


def interpret_screen(
    *,
    text_margin: float,
    visual_probe: float,
    prediction: float,
    text_cut: float,
    probe_cut: float,
    prediction_cut: float,
) -> str:
    """Map a screen's three measurements to the next action.

    Cuts are supplied by the caller. This function does not invent them.
    """
    text_ok = text_margin >= text_cut
    probe_ok = visual_probe >= probe_cut
    pred_ok = prediction >= prediction_cut
    if probe_ok and not text_ok:
        return "change_target"
    if text_ok and probe_ok and not pred_ok:
        return "change_supervision"
    if not probe_ok:
        return "change_visual"
    return "stop"


def execute_shared_screen(
    arms: list[ScreenArm],
    batches_by_target: dict[str, dict],
    *,
    scope: str,
    architecture: dict,
) -> list[dict]:
    """Train every arm for the shared step budget on the shared visual cache.

    ``scope`` must name a non-benchmark rehearsal. The COCO screen remains
    ``run_target_screen``, which does not call this function.
    """
    if scope not in NON_BENCHMARK_SCOPES:
        raise ScreenRefused(f"refusing to record scope {scope!r} as a screen result")
    assert_shared_screen(arms)
    _assert_shared_cache(batches_by_target)
    rows = []
    init_hashes = []
    for arm in arms:
        if arm.target_id not in batches_by_target:
            raise KeyError(f"no batch for target {arm.target_id}")
        seed_everything(arm.seed)
        batch = batches_by_target[arm.target_id]
        config = _screen_config(arm, batch, architecture)
        model = build_predictor(config)
        init_hashes.append(_state_sha256(model))
        from src.conditional.trainer import ConditionalTrainer

        trainer = ConditionalTrainer(model, config, total_steps=int(arm.max_steps))
        losses = [trainer.train_step(batch) for _ in range(int(arm.max_steps))]
        rows.append(
            {
                "scope": scope,
                "target_id": arm.target_id,
                "objective": arm.objective,
                "seed": arm.seed,
                "steps": arm.max_steps,
                "losses": losses,
                "init_sha256": init_hashes[-1],
                "image_counts": dict(trainer.ledger.images),
                "query_encoder_id": arm.query_encoder_id,
            }
        )
    if len(set(init_hashes)) != 1:
        raise RuntimeError("screen arms did not share an initialization")
    counts = [row["image_counts"] for row in rows]
    if any(count != counts[0] for count in counts[1:]):
        raise RuntimeError("screen arms did not share image exposure")
    return rows


def _screen_config(arm: ScreenArm, batch: dict, architecture: dict) -> PredictorConfig:
    required = {"predictor_layers", "predictor_width", "predictor_heads", "lr"}
    missing = required - set(architecture)
    if missing:
        raise KeyError(f"architecture missing {sorted(missing)}")
    return config_from_mapping(
        {
            "vision_dim": int(batch["visual_tokens"].shape[-1]),
            "query_dim": int(batch["query_embeddings"].shape[-1]),
            "target_dim": int(batch["candidate_targets"].shape[-1]),
            "predictor_layers": architecture["predictor_layers"],
            "predictor_width": architecture["predictor_width"],
            "predictor_heads": architecture["predictor_heads"],
            "objective": arm.objective,
            "lr": architecture["lr"],
            "weight_decay": float(architecture.get("weight_decay", 0.0)),
            "seed": arm.seed,
        }
    )


def execute_scheduled_screen(
    arms: list[ScreenArm],
    caption_batches: dict[str, dict],
    qa_batches: dict[str, dict],
    *,
    scope: str,
    architecture: dict,
    schedule: SharedSchedule,
) -> list[dict]:
    """Train caption steps, then QA steps, on one shared visual cache.

    The two phases never share a candidate matrix. Every arm uses ``schedule``.
    """
    if scope not in NON_BENCHMARK_SCOPES:
        raise ScreenRefused(f"refusing to record scope {scope!r} as a screen result")
    assert_shared_screen(arms)
    if any(int(arm.max_steps) != schedule.total_steps for arm in arms):
        raise ValueError("every arm must use the shared schedule length")
    _assert_shared_cache(caption_batches)
    _assert_shared_cache(qa_batches)
    _assert_phase_pair(caption_batches, qa_batches)
    rows = []
    init_hashes = []
    for arm in arms:
        seed_everything(arm.seed)
        caption = caption_batches[arm.target_id]
        qa = qa_batches[arm.target_id]
        config = _screen_config(arm, caption, architecture)
        if int(qa["candidate_targets"].shape[-1]) != config.target_dim:
            raise ValueError(f"{arm.target_id} QA target width differs from the caption target")
        model = build_predictor(config)
        init_hashes.append(_state_sha256(model))
        from src.conditional.trainer import ConditionalTrainer

        trainer = ConditionalTrainer(model, config, total_steps=schedule.total_steps)
        caption_losses = [trainer.train_step(caption) for _ in range(schedule.caption_steps)]
        qa_losses = [trainer.train_step(qa) for _ in range(schedule.qa_steps)]
        rows.append(
            {
                "scope": scope,
                "target_id": arm.target_id,
                "objective": arm.objective,
                "seed": arm.seed,
                "steps": schedule.total_steps,
                "schedule": {
                    "caption_steps": schedule.caption_steps,
                    "qa_steps": schedule.qa_steps,
                },
                "caption_losses": caption_losses,
                "qa_losses": qa_losses,
                "grad_norms": list(trainer.grad_norm_trace),
                "init_sha256": init_hashes[-1],
                "image_counts": dict(trainer.ledger.images),
                "query_encoder_id": arm.query_encoder_id,
            }
        )
    if len(set(init_hashes)) != 1:
        raise RuntimeError("screen arms did not share an initialization")
    counts = [row["image_counts"] for row in rows]
    if any(count != counts[0] for count in counts[1:]):
        raise RuntimeError("screen arms did not share image exposure")
    schedules = [tuple(row["schedule"].values()) for row in rows]
    if len(set(schedules)) != 1:
        raise RuntimeError("screen arms did not share the caption/QA schedule")
    return rows


def _assert_phase_pair(caption_batches: dict[str, dict], qa_batches: dict[str, dict]) -> None:
    for target_id, caption in caption_batches.items():
        qa = qa_batches[target_id]
        if list(caption["task_ids"]) != ["caption"] * len(caption["task_ids"]):
            raise ValueError(f"{target_id} caption phase contains a non-caption row")
        if list(qa["task_ids"]) != ["qa"] * len(qa["task_ids"]):
            raise ValueError(f"{target_id} QA phase contains a non-QA row")
        if not torch.equal(caption["visual_tokens"], qa["visual_tokens"]):
            raise ValueError(f"{target_id} caption and QA phases use different visual tokens")
        if list(caption["image_ids"]) != list(qa["image_ids"]):
            raise ValueError(f"{target_id} caption and QA phases use different image ids")


def _assert_shared_cache(batches_by_target: dict[str, dict]) -> None:
    if set(batches_by_target) != set(TARGET_SPECS):
        raise ValueError("screen batches must cover every target id exactly once")
    reference_id = next(iter(TARGET_SPECS))
    reference = batches_by_target[reference_id]
    for target_id, batch in batches_by_target.items():
        if not torch.equal(reference["visual_tokens"], batch["visual_tokens"]):
            raise ValueError(f"{target_id} visual tokens differ from the shared cache")
        if not torch.equal(reference["query_embeddings"], batch["query_embeddings"]):
            raise ValueError(f"{target_id} query embeddings differ from the shared encoder")
        if list(reference["image_ids"]) != list(batch["image_ids"]):
            raise ValueError(f"{target_id} image ids differ from the shared cache")
