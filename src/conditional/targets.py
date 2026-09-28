"""Target-encoder contracts. Specs do not download weights.

Pooling and prompts follow the model cards named here. A shared output
dimension is not permission to reuse another model's tokenizer or pooler.
The query encoder stays CLIP text across every target arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    hub_id: str | None
    native_dim: int
    output_dim: int
    pooling: str
    tokenizer: str
    context_length: int | None
    mrl: bool
    role_prompts: dict[str, str]
    documents_receive_instruction: bool
    notes: str


CLIP_VITB16_TEXT = TargetSpec(
    target_id="clip_vitb16_text",
    hub_id=None,
    native_dim=512,
    output_dim=512,
    pooling="eot",
    tokenizer="clip_bpe",
    context_length=77,
    mrl=False,
    role_prompts={},
    documents_receive_instruction=False,
    notes=(
        "OpenAI CLIP ViT-B/16 text tower. Pool the EOT token and apply "
        "text_projection. encode_text uses the full 77 positional embeddings; "
        "do not slice the sequence to the last non-pad token before the transformer."
    ),
)

EMBEDDING_GEMMA_300M = TargetSpec(
    target_id="embeddinggemma_300m_mrl512",
    hub_id="google/embeddinggemma-300m",
    native_dim=768,
    output_dim=512,
    pooling="mean",
    tokenizer="gemma",
    context_length=None,
    mrl=True,
    role_prompts={
        "query": "task: search result | query: ",
        "document": "title: none | text: ",
        "qa": "task: question answering | query: ",
    },
    documents_receive_instruction=True,
    notes=(
        "Mean-pool the prompt together with the text. Matryoshka truncation "
        "keeps the leading dimensions and renormalizes. Captions used as "
        "documents take the document prompt. The role has to be explicit."
    ),
)

QWEN3_EMBEDDING_0_6B = TargetSpec(
    target_id="qwen3_embedding_0.6b_mrl512",
    hub_id="Qwen/Qwen3-Embedding-0.6B",
    native_dim=1024,
    output_dim=512,
    pooling="last_token",
    tokenizer="qwen2",
    context_length=None,
    mrl=True,
    role_prompts={
        "query": "Instruct: {task}\nQuery: {text}",
    },
    documents_receive_instruction=False,
    notes=(
        "Last-token pooling, left-padding aware. Queries receive "
        "Instruct: {task}\\nQuery: {text}. Documents do not. Do not mean-pool Qwen."
    ),
)

TARGET_SPECS = {
    spec.target_id: spec
    for spec in (CLIP_VITB16_TEXT, EMBEDDING_GEMMA_300M, QWEN3_EMBEDDING_0_6B)
}

FIXED_QUERY_ENCODER_ID = "clip_vitb16_text"


def mrl_truncate(embedding: torch.Tensor, dim: int, native_dim: int) -> torch.Tensor:
    if dim > native_dim:
        raise ValueError(f"cannot truncate native dim {native_dim} to {dim}")
    if embedding.size(-1) < dim:
        raise ValueError(f"embedding width {embedding.size(-1)} < requested {dim}")
    if embedding.size(-1) != native_dim and embedding.size(-1) != dim:
        raise ValueError(
            f"expected native width {native_dim} or already-truncated {dim}, "
            f"got {embedding.size(-1)}"
        )
    return F.normalize(embedding[..., :dim].float(), dim=-1, eps=1e-6)


def format_target_text(spec: TargetSpec, text: str, role: str, task: str = "Retrieve relevant captions") -> str:
    if spec.target_id == "clip_vitb16_text":
        return text
    if spec.target_id == "embeddinggemma_300m_mrl512":
        if role not in spec.role_prompts:
            raise KeyError(f"EmbeddingGemma role must be one of {sorted(spec.role_prompts)}")
        return spec.role_prompts[role] + text
    if spec.target_id == "qwen3_embedding_0.6b_mrl512":
        if role == "document":
            return text
        if role != "query":
            raise KeyError("Qwen role must be 'query' or 'document'")
        return spec.role_prompts["query"].format(task=task, text=text)
    raise KeyError(spec.target_id)
