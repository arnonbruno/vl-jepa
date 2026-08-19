"""OpenAI OpenCLIP construction must request QuickGELU; other pretrained paths must not."""

from __future__ import annotations

import types

from src.model import _create_openclip_model


def test_openai_pretrained_requests_force_quick_gelu(monkeypatch) -> None:
    captured = {}

    def fake_create(model_name, pretrained=None, **kwargs):
        captured["name"] = model_name
        captured["pretrained"] = pretrained
        captured["kwargs"] = dict(kwargs)
        return object(), None, None

    monkeypatch.setattr(
        "src.model.open_clip",
        types.SimpleNamespace(create_model_and_transforms=fake_create),
    )
    _create_openclip_model("ViT-B-16", "openai")
    assert captured["pretrained"] == "openai"
    assert captured["kwargs"].get("force_quick_gelu") is True


def test_non_openai_pretrained_does_not_request_force_quick_gelu(monkeypatch) -> None:
    captured = {}

    def fake_create(model_name, pretrained=None, **kwargs):
        captured["pretrained"] = pretrained
        captured["kwargs"] = dict(kwargs)
        return object(), None, None

    monkeypatch.setattr(
        "src.model.open_clip",
        types.SimpleNamespace(create_model_and_transforms=fake_create),
    )
    _create_openclip_model("ViT-B-16", "laion2b_s34b_b88k")
    assert captured["pretrained"] == "laion2b_s34b_b88k"
    assert captured["kwargs"].get("force_quick_gelu") is not True

    captured.clear()
    _create_openclip_model("ViT-B-16", None)
    assert captured["pretrained"] is None
    assert captured["kwargs"].get("force_quick_gelu") is not True
