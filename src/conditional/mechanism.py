"""Mechanism-screen definitions. Nothing here launches training.

M0 through M5 share one exposure-ledger key so a later run cannot change the
image count while changing the supervision. The residual adapter is identity
at initialization because its last linear layer is zero. Freeze it before any
predictor comparison. These objects are the plan, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MechanismCondition:
    condition_id: str
    target: str
    supervision: str
    purpose: str
    exposure_ledger_key: str
    status: str = "not_trained"


SHARED_EXPOSURE_LEDGER = "mechanism_screen_shared_exposure"

MECHANISM_CONDITIONS: tuple[MechanismCondition, ...] = (
    MechanismCondition(
        "M0",
        "original_fixed",
        "matched_non_role_qa",
        "Conditional baseline",
        SHARED_EXPOSURE_LEDGER,
    ),
    MechanismCondition(
        "M1",
        "calibrated_fixed",
        "matched_non_role_qa",
        "Target intervention",
        SHARED_EXPOSURE_LEDGER,
    ),
    MechanismCondition(
        "M2",
        "original_fixed",
        "grounded_role_qa",
        "Grounding intervention",
        SHARED_EXPOSURE_LEDGER,
    ),
    MechanismCondition(
        "M3",
        "calibrated_fixed",
        "grounded_role_qa",
        "Interaction of calibration and grounded role questions",
        SHARED_EXPOSURE_LEDGER,
    ),
    MechanismCondition(
        "M4",
        "original_fixed",
        "matched_compositional_declarative",
        "Check whether ordinary semantic supervision explains a gain",
        SHARED_EXPOSURE_LEDGER,
    ),
    MechanismCondition(
        "M5",
        "original_fixed",
        "role_qa_answers_permuted_within_groups",
        "Diagnostic for reliance on valid grounding labels; not a competitive baseline",
        SHARED_EXPOSURE_LEDGER,
    ),
)


class ResidualTargetAdapter(nn.Module):
    """512 -> 128 -> 512 residual MLP. Output equals the input at initialization."""

    def __init__(self, dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.fc2(F.gelu(self.fc1(inputs)))

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()


def mechanism_claims(scores: Mapping[str, float], *, match_atol: float = 0.0) -> dict:
    """Say which claims the six scores support.

    M5 is recorded and is not a competitive baseline. M3 beating M0 is not
    enough for a synergy claim. A match between M3 and M4 blocks a claim that
    query conditioning is essential.
    """
    if match_atol < 0:
        raise ValueError("match_atol must be non-negative")
    missing = [item.condition_id for item in MECHANISM_CONDITIONS if item.condition_id not in scores]
    if missing:
        raise KeyError(f"mechanism scores missing {missing}")
    values = {item.condition_id: float(scores[item.condition_id]) for item in MECHANISM_CONDITIONS}
    matched = {name: values[name] for name in ("M0", "M1", "M2", "M4")}
    strongest = max(matched, key=matched.get)
    m0, m1, m2, m3, m4 = (values[name] for name in ("M0", "M1", "M2", "M3", "M4"))
    return {
        "m3_beats_m0": m3 > m0,
        "strongest_matched_baseline": strongest,
        "m3_survives_strongest_matched_baseline": m3 > matched[strongest],
        "claim_role_supervision_synergy": m3 > m0 and m3 > m1 and m3 > m2 and m3 > m4,
        "claim_query_conditioning_essential": (m3 - m4) > match_atol,
        "claim_target_calibration_alone": m1 > m0 and m1 >= m2 and m1 >= m3 and m1 >= m4,
        "m5_excluded_from_baselines": True,
    }


def answer_prior(answers: Sequence[str]) -> dict[str, float]:
    if not answers:
        raise ValueError("answer prior needs at least one answer")
    counts: dict[str, int] = {}
    for answer in answers:
        counts[str(answer)] = counts.get(str(answer), 0) + 1
    total = float(len(answers))
    return {answer: count / total for answer, count in sorted(counts.items())}


def answer_prior_l1(left: Sequence[str], right: Sequence[str]) -> float:
    prior_left = answer_prior(left)
    prior_right = answer_prior(right)
    keys = set(prior_left) | set(prior_right)
    return float(sum(abs(prior_left.get(key, 0.0) - prior_right.get(key, 0.0)) for key in keys))


def fit_residual_adapter(
    adapter: ResidualTargetAdapter,
    original: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    steps: int,
    lr: float = 1e-2,
    margin: float = 1.5,
) -> list[float]:
    """Fit the adapter, then freeze it. Returns the per-step loss."""
    if steps < 1:
        raise ValueError("adapter steps must be positive")
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = adapter_margin_loss(
            adapter(original),
            original,
            positive,
            negative,
            margin=margin,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    adapter.freeze()
    return losses


def run_synthetic_mechanism(*, steps: int = 2, adapter_steps: int = 20, seed: int = 0) -> list[dict]:
    """Train M0–M5 on one shared image list. Not a COCO result."""
    from src.conditional.config import config_from_mapping
    from src.conditional.model import build_predictor
    from src.conditional.trainer import ConditionalTrainer, seed_everything

    bundle = _mechanism_bundle(seed)
    adapter = ResidualTargetAdapter(dim=bundle["dim"], hidden=8)
    before = adapter(bundle["anchor"]).detach()
    fit_residual_adapter(
        adapter,
        bundle["anchor"],
        bundle["paraphrase"],
        bundle["negative"],
        steps=adapter_steps,
    )
    after = adapter(bundle["anchor"]).detach()
    if torch.equal(before, after):
        raise RuntimeError("adapter did not move")
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("adapter must be frozen before the predictor comparison")

    rows = []
    init_hashes = []
    for condition in MECHANISM_CONDITIONS:
        seed_everything(seed)
        config = config_from_mapping(
            {
                "vision_dim": bundle["dim"],
                "query_dim": bundle["dim"],
                "target_dim": bundle["dim"],
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
        init_hashes.append(_module_sha256(model))
        trainer = ConditionalTrainer(model, config, total_steps=steps)
        batch = _condition_batch(condition.condition_id, bundle, adapter)
        losses = [trainer.train_step(batch) for _ in range(steps)]
        rows.append(
            {
                "scope": "synthetic_mechanism_not_a_result",
                "condition_id": condition.condition_id,
                "target": condition.target,
                "supervision": condition.supervision,
                "exposure_ledger_key": condition.exposure_ledger_key,
                "seed": seed,
                "steps": steps,
                "losses": losses,
                "init_sha256": init_hashes[-1],
                "image_counts": dict(trainer.ledger.images),
                "answer_ids": list(batch["answer_ids"]),
                "uses_adapter": condition.target == "calibrated_fixed",
            }
        )
    if len(set(init_hashes)) != 1:
        raise RuntimeError("mechanism conditions did not share an initialization")
    counts = [row["image_counts"] for row in rows]
    if any(count != counts[0] for count in counts[1:]):
        raise RuntimeError("mechanism conditions did not share image exposure")
    return rows


def _mechanism_bundle(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    dim = 16
    n = 4
    visual = torch.randn(n, 2, dim, generator=generator)
    anchor = torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1)
    paraphrase = torch.nn.functional.normalize(anchor + 0.05 * torch.randn(n, dim, generator=generator), dim=-1)
    negative = torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1)
    order = [1, 0, 3, 2]
    return {
        "dim": dim,
        "visual": visual,
        "image_ids": [f"img-{index}" for index in range(n)],
        "anchor": anchor,
        "paraphrase": paraphrase,
        "negative": negative,
        "query_nonrole": torch.randn(n, dim, generator=generator),
        "query_role": torch.randn(n, dim, generator=generator),
        "query_declarative": torch.randn(n, dim, generator=generator),
        "target_nonrole": torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1),
        "target_role": torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1),
        "target_declarative": torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=-1),
        "answers": [f"ans-{index}" for index in range(n)],
        "perm": order,
    }


def _condition_batch(condition_id: str, bundle: dict, adapter: ResidualTargetAdapter) -> dict:
    n = len(bundle["image_ids"])
    if condition_id in {"M0", "M1"}:
        query = bundle["query_nonrole"]
        target = bundle["target_nonrole"]
        answers = bundle["answers"]
    elif condition_id in {"M2", "M3"}:
        query = bundle["query_role"]
        target = bundle["target_role"]
        answers = bundle["answers"]
    elif condition_id == "M4":
        query = bundle["query_declarative"]
        target = bundle["target_declarative"]
        answers = bundle["answers"]
    elif condition_id == "M5":
        query = bundle["query_role"]
        target = bundle["target_role"][bundle["perm"]]
        answers = [bundle["answers"][index] for index in bundle["perm"]]
    else:
        raise KeyError(condition_id)
    if condition_id in {"M1", "M3"}:
        with torch.no_grad():
            target = adapter(target)
    return {
        "visual_tokens": bundle["visual"],
        "query_embeddings": query,
        "candidate_targets": target.detach(),
        "positive_mask": torch.eye(n, dtype=torch.bool),
        "task_ids": ["qa"] * n,
        "image_ids": list(bundle["image_ids"]),
        "caption_ids": [f"{condition_id}:{answer}" for answer in answers],
        "answer_ids": answers,
    }


def _module_sha256(module: torch.nn.Module) -> str:
    from src.protocol.hashing import sha256_bytes

    blob = bytearray()
    for key, value in module.state_dict().items():
        blob.extend(key.encode("utf-8"))
        blob.extend(b"\0")
        blob.extend(value.detach().cpu().contiguous().numpy().tobytes())
    return sha256_bytes(bytes(blob))


def adapter_margin_loss(
    adapted: torch.Tensor,
    original: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float = 0.2,
    anchor_weight: float = 1.0,
) -> torch.Tensor:
    """Paraphrase-versus-negative hinge plus a penalty toward the original target."""
    adapted_u = F.normalize(adapted.float(), dim=-1, eps=1e-6)
    original_u = F.normalize(original.float(), dim=-1, eps=1e-6)
    positive_u = F.normalize(positive.float(), dim=-1, eps=1e-6)
    negative_u = F.normalize(negative.float(), dim=-1, eps=1e-6)
    gap = (adapted_u * positive_u).sum(dim=-1) - (adapted_u * negative_u).sum(dim=-1)
    hinge = torch.relu(margin - gap).mean()
    anchor = (1.0 - (adapted_u * original_u).sum(dim=-1)).mean()
    return hinge + anchor_weight * anchor
