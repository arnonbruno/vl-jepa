"""Pooling used by the target encoders.

Mean pooling includes the prompt tokens and ignores padding. Last-token pooling
takes the last non-pad position, which is the right-hand token under left padding.
CLIP EOT pooling is not implemented here.
"""

from __future__ import annotations

import torch


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.dim() != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("mean_pool expects hidden (B, S, D) and a matching mask")
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("mean_pool refuses an empty sequence")
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.dim() != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("last_token_pool expects hidden (B, S, D) and a matching mask")
    mask = attention_mask.long()
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("last_token_pool refuses an empty sequence")
    offset = mask.flip(dims=(1,)).argmax(dim=1)
    index = mask.size(1) - 1 - offset
    gathered = hidden.gather(1, index.view(-1, 1, 1).expand(-1, 1, hidden.size(-1)))
    return gathered.squeeze(1)


def pool_hidden(pooling: str, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if pooling == "mean":
        return mean_pool(hidden, attention_mask)
    if pooling == "last_token":
        return last_token_pool(hidden, attention_mask)
    if pooling == "eot":
        raise ValueError("CLIP EOT pooling uses the EOT position, not this mask pooler")
    raise ValueError(f"unknown pooling {pooling!r}")
