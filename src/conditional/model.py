"""Compact predictor from frozen visual tokens and a frozen query.

``prediction = normalize(predictor(visual_tokens, query_tokens))``.
The target encoder is not a submodule. Candidate captions and answers are
encoded outside this module and never become predictor inputs.

The CLIP CLS token is kept. ``keep_cls`` documents that token 0 is part of
the visual sequence the predictor attends over. The output is a mean over
query tokens, then a linear map to ``target_dim``, then L2 normalization.
There is no mask argument on the vision path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.conditional.config import PredictorConfig


class _CrossBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.cross_attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, query: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        self_in = self.norm_self(query)
        self_out, _ = self.self_attn(self_in, self_in, self_in, need_weights=False)
        query = query + self_out
        cross_out, _ = self.cross_attn(
            self.norm_q(query),
            self.norm_kv(visual),
            self.norm_kv(visual),
            need_weights=False,
        )
        query = query + cross_out
        query = query + self.ff(self.norm_ff(query))
        return query


class ConditionalLatentPredictor(nn.Module):
    def __init__(self, config: PredictorConfig, vision_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.vision_encoder = vision_encoder
        if vision_encoder is not None:
            for parameter in vision_encoder.parameters():
                parameter.requires_grad = False
        self.visual_proj = nn.Linear(config.vision_dim, config.predictor_width)
        self.query_proj = nn.Linear(config.query_dim, config.predictor_width)
        self.blocks = nn.ModuleList(
            _CrossBlock(config.predictor_width, config.predictor_heads)
            for _ in range(config.predictor_layers)
        )
        self.out_norm = nn.LayerNorm(config.predictor_width)
        self.out_proj = nn.Linear(config.predictor_width, config.target_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        return self

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen visual tokens, including CLS when the encoder emits it.

        No mask is forwarded. Masking cached tokens later is feature dropout,
        not a claim that the encoder skipped those patches.
        """
        if self.vision_encoder is None:
            raise RuntimeError("encode_image requires a vision encoder")
        was_training = self.vision_encoder.training
        self.vision_encoder.eval()
        with torch.no_grad():
            tokens = self.vision_encoder(images)
        if was_training:
            self.vision_encoder.train()
        if tokens.dim() != 3:
            raise ValueError(f"vision encoder must return (B, T, D), got {tuple(tokens.shape)}")
        return tokens

    def predict(self, images: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.predict_from_tokens(self.encode_image(images), query_embeddings)

    def predict_from_tokens(
        self,
        visual_tokens: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if visual_tokens.dim() != 3:
            raise ValueError(f"visual_tokens must be (B, T, D), got {tuple(visual_tokens.shape)}")
        if visual_tokens.size(-1) != self.config.vision_dim:
            raise ValueError(
                f"visual width {visual_tokens.size(-1)} != config.vision_dim {self.config.vision_dim}"
            )
        query = self._query_tokens(query_embeddings)
        if query.size(0) != visual_tokens.size(0):
            raise ValueError("visual batch and query batch differ")
        visual = self.visual_proj(visual_tokens)
        hidden = self.query_proj(query)
        for block in self.blocks:
            hidden = block(hidden, visual)
        pooled = self.out_norm(hidden).mean(dim=1)
        return F.normalize(self.out_proj(pooled), dim=-1, eps=1e-6)

    def _query_tokens(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if query_embeddings.dim() == 2:
            if query_embeddings.size(-1) != self.config.query_dim:
                raise ValueError(
                    f"query width {query_embeddings.size(-1)} != config.query_dim {self.config.query_dim}"
                )
            return query_embeddings.unsqueeze(1)
        if query_embeddings.dim() == 3:
            if query_embeddings.size(-1) != self.config.query_dim:
                raise ValueError(
                    f"query width {query_embeddings.size(-1)} != config.query_dim {self.config.query_dim}"
                )
            return query_embeddings
        raise ValueError("query_embeddings must be (B, D) or (B, Q, D)")

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_predictor(
    config: PredictorConfig,
    vision_encoder: nn.Module | None = None,
) -> ConditionalLatentPredictor:
    """Construct the predictor. Seeding, if any, is the caller's job."""
    return ConditionalLatentPredictor(config, vision_encoder=vision_encoder)
