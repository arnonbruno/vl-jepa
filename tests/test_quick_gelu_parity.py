"""OpenAI OpenCLIP construction must request QuickGELU; other pretrained paths must not."""

from __future__ import annotations

import inspect
import types

import pytest

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


def test_force_quick_gelu_typeerror_fallback(monkeypatch) -> None:
    calls = []

    def rejecting(model_name, pretrained=None, **kwargs):
        calls.append(dict(kwargs))
        if "force_quick_gelu" in kwargs:
            raise TypeError(
                "create_model_and_transforms() got an unexpected keyword "
                "argument 'force_quick_gelu'"
            )
        return object(), None, None

    monkeypatch.setattr(
        "src.model.open_clip",
        types.SimpleNamespace(create_model_and_transforms=rejecting),
    )
    model = _create_openclip_model("ViT-B-16", "openai")
    assert model is not None
    assert len(calls) == 2
    assert calls[0].get("force_quick_gelu") is True
    assert "force_quick_gelu" not in calls[1]


def test_installed_openclip_force_quick_gelu_signature() -> None:
    open_clip = pytest.importorskip("open_clip")
    sig = inspect.signature(open_clip.create_model_and_transforms)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepts = "force_quick_gelu" in sig.parameters or accepts_var_kw
    if accepts:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=None, force_quick_gelu=True,
        )
        assert model is not None
    else:
        with pytest.raises(TypeError, match="force_quick_gelu"):
            open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained=None, force_quick_gelu=True,
            )
