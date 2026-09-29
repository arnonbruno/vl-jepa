"""Frozen target encoders. Missing weights block execution.

EmbeddingGemma mean-pools, applies the released 768 -> 3072 -> 768
projections, then truncates to 512 and L2-normalizes. Qwen pools the last
non-pad token. Neither path may fall back to random weights, and a missing
target is not dropped from the screen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from src.conditional.pooling import last_token_pool, mean_pool
from src.conditional.targets import TARGET_SPECS, mrl_truncate

GEMMA_INFERENCE_DTYPES = (torch.float32, torch.bfloat16)
PLACEHOLDER_PINS = frozenset(
    {"", "unresolved", "placeholder", "dummy", "todo", "none", "clip_vitb16_text"}
)


class WeightsUnavailable(RuntimeError):
    """A required checkpoint is absent. Execution must stop."""


class GemmaPostPoolProjection(nn.Module):
    """Two linear maps applied after mean pooling. Not a random target."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(768, 3072)
        self.down = nn.Linear(3072, 768)
        self.loaded_from_checkpoint = False

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.dtype not in GEMMA_INFERENCE_DTYPES:
            raise ValueError(
                "EmbeddingGemma inference allows float32 or bfloat16 activations, "
                f"got {pooled.dtype}"
            )
        return self.down(self.up(pooled))


def load_gemma_projection(state_dict: Mapping[str, torch.Tensor] | None) -> GemmaPostPoolProjection:
    if not state_dict:
        raise WeightsUnavailable("EmbeddingGemma projection weights are not loaded")
    module = GemmaPostPoolProjection()
    module.load_state_dict(dict(state_dict))
    module.loaded_from_checkpoint = True
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def project_then_truncate(pooled: torch.Tensor, projection: GemmaPostPoolProjection, *, dim: int = 512) -> torch.Tensor:
    """Truncate the projected native vector, not the pre-projection hidden state."""
    native = projection(pooled)
    if native.size(-1) != 768:
        raise ValueError(f"EmbeddingGemma native width must be 768, got {native.size(-1)}")
    return mrl_truncate(native, dim, 768)


def encode_gemma_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    projection: GemmaPostPoolProjection,
    *,
    dim: int = 512,
) -> torch.Tensor:
    if not getattr(projection, "loaded_from_checkpoint", False):
        raise WeightsUnavailable("EmbeddingGemma projection weights are not loaded")
    if hidden.dtype not in GEMMA_INFERENCE_DTYPES:
        raise ValueError(
            "EmbeddingGemma inference allows float32 or bfloat16 activations, "
            f"got {hidden.dtype}"
        )
    return project_then_truncate(mean_pool(hidden, attention_mask), projection, dim=dim)


def encode_qwen_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    native_dim: int = 1024,
    dim: int = 512,
) -> torch.Tensor:
    """Last non-pad token, then leading-dim truncation and L2 normalization."""
    if hidden.size(-1) != native_dim:
        raise ValueError(f"Qwen hidden width {hidden.size(-1)} != native {native_dim}")
    return mrl_truncate(last_token_pool(hidden, attention_mask), dim, native_dim)


def clip_eot_indices(input_ids: torch.Tensor) -> torch.Tensor:
    """Official CLIP pool index: the EOT id is the maximum id in the row."""
    if input_ids.dim() != 2:
        raise ValueError("CLIP token ids must be (B, context)")
    return input_ids.argmax(dim=-1)


def encode_clip_text_hidden(hidden: torch.Tensor, input_ids: torch.Tensor, text_projection: torch.Tensor) -> torch.Tensor:
    """EOT hidden state times ``text_projection``. Context length stays 77."""
    if input_ids.size(1) != 77:
        raise ValueError(f"CLIP text uses context length 77, got {input_ids.size(1)}")
    if hidden.shape[:2] != input_ids.shape:
        raise ValueError("CLIP hidden states and token ids must share a batch and context")
    index = clip_eot_indices(input_ids)
    gathered = hidden.gather(1, index.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))).squeeze(1)
    projected = gathered.float() @ text_projection.float()
    return torch.nn.functional.normalize(projected, dim=-1, eps=1e-6)


def assert_query_pin(checkpoint_sha256: str, preprocessing: str) -> None:
    """The query encoder is a checkpoint and a preprocessing recipe, not a name."""
    digest = (checkpoint_sha256 or "").strip().lower()
    recipe = (preprocessing or "").strip().lower()
    if digest in PLACEHOLDER_PINS or len(digest) < 64:
        raise ValueError("query encoder must be pinned by a checkpoint sha256, not by its name")
    if recipe in PLACEHOLDER_PINS:
        raise ValueError("query encoder preprocessing must be declared")


def embeddinggemma_weights_present() -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    return any("embeddinggemma-300m" in path.name for path in hub.iterdir())


def require_screen_weights(available: Mapping[str, bool]) -> None:
    """Every screen target must be loadable. None are dropped or randomly initialized."""
    missing = [target_id for target_id in TARGET_SPECS if not available.get(target_id, False)]
    if missing:
        raise WeightsUnavailable(
            "target-screen execution blocked; weights unavailable for " + ", ".join(missing)
        )
