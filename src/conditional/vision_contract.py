"""Frozen CLIP ViT-B/16 spatial tokens used by the conditional predictor.

The cached representation is the full token sequence after ``visual.ln_post``,
including the CLS token, and before ``visual.proj``. Patch order is row-major.
Learned positional embeddings are added before ``ln_pre``. Preprocessing is a
224 center crop with the OpenAI CLIP mean and standard deviation. No mask is
applied, so this cache is not evidence about unseen patches.
"""

from __future__ import annotations

from typing import Any

import torch

CLIP_VITB16_SPATIAL: dict[str, Any] = {
    "model_name": "ViT-B-16",
    "pretrained_tag": "openai",
    "extraction_layer": "visual.ln_post",
    "normalization_point": "ln_post_before_visual_proj",
    "keep_cls": True,
    "token_order": "cls_then_row_major_patches",
    "positional_treatment": "add_learned_positional_embedding_before_ln_pre_native_grid",
    "preprocessing": "resize_224_center_crop_clip_mean_std",
    "image_size": 224,
    "patch_size": 16,
    "grid": 14,
    "n_tokens": 197,
    "dim": 768,
    "storage_precision": "float16",
    "mask_before_encoder": False,
}


def extract_spatial_tokens(encoder: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return (B, 197, 768) tokens from a frozen ``OpenCLIPVisionEncoder``."""
    if images.dim() != 4 or tuple(images.shape[-2:]) != (224, 224):
        raise ValueError(f"CLIP ViT-B/16 expects NCHW images at 224, got {tuple(images.shape)}")
    if any(parameter.requires_grad for parameter in encoder.visual.parameters()):
        raise RuntimeError("visual extractor must stay frozen")
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        tokens = encoder(images, mask=None)
    if was_training:
        encoder.train()
    expected = (images.size(0), CLIP_VITB16_SPATIAL["n_tokens"], CLIP_VITB16_SPATIAL["dim"])
    if tuple(tokens.shape) != expected:
        raise ValueError(f"spatial tokens must be {expected}, got {tuple(tokens.shape)}")
    if tokens.size(-1) == 512:
        raise ValueError("cached visual tokens are pre-projection 768-d states, not CLIP image embeddings")
    return tokens
