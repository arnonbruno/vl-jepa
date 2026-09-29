"""Descriptors for a target space. They are not training objectives.

Effective rank and isotropy describe a sample of embeddings. A linear probe
that fails does not by itself show that the attribute is absent from the
visual tokens; a small nonlinear probe is the check that comes next.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_rank(embeddings: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum."""
    if embeddings.dim() != 2 or embeddings.size(0) < 2:
        raise ValueError("effective rank needs a (N, D) sample with N >= 2")
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if float(total) == 0.0:
        return 0.0
    weights = energy / total
    return float((weights.sum().square() / weights.square().sum()).item())


def isotropy_mean_norm(embeddings: torch.Tensor) -> float:
    """Norm of the mean of unit vectors. Near 0 is more directionally spread."""
    unit = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    return float(unit.mean(dim=0).norm().item())


def paraphrase_margin(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> float:
    anchor = F.normalize(anchor.float(), dim=-1, eps=1e-6)
    positive = F.normalize(positive.float(), dim=-1, eps=1e-6)
    negative = F.normalize(negative.float(), dim=-1, eps=1e-6)
    return float((anchor * positive - anchor * negative).sum(dim=-1).mean().item())


def neighborhood_mean_cosine(embeddings: torch.Tensor, k: int) -> float:
    """Mean cosine from each vector to its ``k`` nearest other vectors."""
    if embeddings.dim() != 2 or embeddings.size(0) < 2:
        raise ValueError("neighborhood statistics need a (N, D) sample with N >= 2")
    if k < 1 or k >= embeddings.size(0):
        raise ValueError("k must be at least 1 and smaller than the sample")
    unit = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
    similarity = unit @ unit.T
    similarity.fill_diagonal_(float("-inf"))
    return float(similarity.topk(k, dim=1).values.mean().item())


def exact_collision_pairs(embeddings: torch.Tensor, *, atol: float = 0.0) -> int:
    """Count unordered pairs whose vectors are equal within ``atol``."""
    if embeddings.dim() != 2 or embeddings.size(0) < 2:
        raise ValueError("collision check needs a (N, D) sample with N >= 2")
    distance = torch.cdist(embeddings.float(), embeddings.float())
    upper = torch.triu(torch.ones_like(distance, dtype=torch.bool), diagonal=1)
    return int((distance <= atol)[upper].sum().item())


def collision_action(embeddings: torch.Tensor) -> str:
    """Exact copies cannot be split by a function of the pooled vector."""
    if exact_collision_pairs(embeddings, atol=0.0) > 0:
        return "switch_encoder_or_prepooling_tokens"
    return "adapter_may_reshape_near_collisions"


def separation_sufficient(prediction: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> bool:
    """Sufficient condition from the execution plan, not a necessary one.

    For unit targets, nearest-target ranking prefers ``positive`` whenever
    ``||z - t+|| < ||t+ - t-|| / 2``. Exact collisions have distance 0 and
    cannot be resolved by predicting into that space.
    """
    distance = torch.norm(positive.float() - negative.float())
    error = torch.norm(prediction.float() - positive.float())
    return bool(error.item() < (distance.item() / 2.0))


def linear_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> float:
    """Least-squares one-vs-rest on centered features. Returns test accuracy."""
    x_train = features.float()[train_idx]
    y_train = labels.long()[train_idx]
    x_test = features.float()[test_idx]
    y_test = labels.long()[test_idx]
    classes = int(labels.max().item()) + 1
    mean = x_train.mean(dim=0, keepdim=True)
    x_train = x_train - mean
    x_test = x_test - mean
    eye = torch.eye(classes, device=features.device, dtype=x_train.dtype)
    targets = eye[y_train]
    xtx = x_train.T @ x_train
    xtx = xtx + 1e-3 * torch.eye(xtx.size(0), dtype=xtx.dtype)
    weight = torch.linalg.solve(xtx, x_train.T @ targets)
    pred = (x_test @ weight).argmax(dim=-1)
    return float((pred == y_test).float().mean().item())


def small_nonlinear_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    steps: int = 80,
    hidden: int = 32,
    seed: int = 0,
) -> float:
    """One hidden layer. Used only as a diagnostic, not as the predictor."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x_train = features.float()[train_idx]
    y_train = labels.long()[train_idx]
    x_test = features.float()[test_idx]
    y_test = labels.long()[test_idx]
    classes = int(labels.max().item()) + 1
    layer1 = torch.nn.Linear(features.size(1), hidden)
    layer2 = torch.nn.Linear(hidden, classes)
    with torch.no_grad():
        layer1.weight.copy_(torch.randn(layer1.weight.shape, generator=generator) * 0.02)
        layer2.weight.copy_(torch.randn(layer2.weight.shape, generator=generator) * 0.02)
        layer1.bias.zero_()
        layer2.bias.zero_()
    optimizer = torch.optim.Adam(list(layer1.parameters()) + list(layer2.parameters()), lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = layer2(torch.relu(layer1(x_train)))
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = layer2(torch.relu(layer1(x_test))).argmax(dim=-1)
    return float((pred == y_test).float().mean().item())
