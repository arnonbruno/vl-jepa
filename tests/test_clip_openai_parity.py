"""OpenAI CLIP weights: wrapper CLS/EOT matches encode_image and encode_text."""

from __future__ import annotations

import torch

from src.conditional.clip_parity import clip_pooled_from_wrapper, load_openai_vitb16

open_clip = None
try:
    import open_clip as _open_clip

    open_clip = _open_clip
except ImportError:
    pass


def test_openai_weights_match_encode_image_and_encode_text() -> None:
    if open_clip is None:
        import pytest

        pytest.skip("open_clip_torch is not installed")
    model, _preprocess = load_openai_vitb16()
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    captions = ["a photo of a cat on a mat", "dog"]
    text_ids = tokenizer(captions, context_length=77)
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(2, 3, 224, 224, generator=generator)
    with torch.no_grad():
        reference_image = model.encode_image(images)
        reference_text = model.encode_text(text_ids)
        ours_image, ours_text = clip_pooled_from_wrapper(model, images, text_ids)
    assert torch.allclose(ours_image, reference_image, atol=1e-4)
    assert torch.allclose(ours_text, reference_text, atol=1e-4)
    assert text_ids.shape[1] == 77
