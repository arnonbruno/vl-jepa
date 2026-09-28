"""Mechanism-screen definitions. Nothing here launches training.

M0 through M5 share one exposure-ledger key so a later run cannot change the
image count while changing the supervision. The residual adapter is identity
at initialization because its last linear layer is zero. Freeze it before any
predictor comparison. These objects are the plan, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

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
