"""Compare the existing OpenCLIP wrapper with ``encode_image`` / ``encode_text``.

Vision uses the CLS token after ``ln_post``, multiplied by ``visual.proj``.
Text uses the EOT token after ``ln_final``, multiplied by ``text_projection``.
The text tower is run at context length 77, which is the length whose
positional table ``encode_text`` applies. This module does not slice that table.
"""

from __future__ import annotations

import torch

from src.model import (
    LinearProjection,
    OpenCLIPLanguageEncoder,
    OpenCLIPVisionEncoder,
    create_openclip_model_and_transforms,
)


def clip_pooled_from_wrapper(
    shared_model,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    *,
    model_name: str = "ViT-B-16",
) -> tuple[torch.Tensor, torch.Tensor]:
    vision = OpenCLIPVisionEncoder(
        model_name,
        pretrained=None,
        freeze=True,
        shared_model=shared_model,
    )
    vision.eval()
    text = OpenCLIPLanguageEncoder(
        model_name,
        pretrained=None,
        freeze=True,
        shared_model=shared_model,
    )
    text.eval()
    image_proj = LinearProjection(
        shared_model.visual.proj.shape[0],
        shared_model.visual.proj.shape[1],
        init_weight=shared_model.visual.proj,
    )
    text_proj = LinearProjection(
        shared_model.text_projection.shape[0],
        shared_model.text_projection.shape[1],
        init_weight=shared_model.text_projection,
    )
    with torch.no_grad():
        cls = vision(images)[:, 0, :]
        image_out = image_proj.raw(cls)
        if text_ids.size(1) != shared_model.positional_embedding.shape[0]:
            raise ValueError(
                "CLIP text parity requires the full positional context "
                f"({shared_model.positional_embedding.shape[0]}), got {text_ids.size(1)}"
            )
        attention = (text_ids != 0).long()
        hidden = text(text_ids, attention)
        eot = attention.sum(dim=1) - 1
        gathered = hidden.gather(1, eot.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))).squeeze(1)
        text_out = text_proj.raw(gathered)
    return image_out, text_out


def load_openai_vitb16():
    model, _, preprocess = create_openclip_model_and_transforms("ViT-B-16", "openai")
    model.eval()
    return model, preprocess
