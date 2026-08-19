"""Eval backend loads the current trainer checkpoint schema and config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.model import VL_JEPA

try:
    import open_clip  # noqa: F401
    _HAS_OPEN_CLIP = True
except ImportError:
    _HAS_OPEN_CLIP = False


def _tiny_config(**model_over):
    mcfg = {
        "hidden_dim": 96,
        "patch_size": 16,
        "image_size": 64,
        "mask_ratio": 0.75,
        "predictor_layers": 1,
        "vision_backbone": "custom",
        "text_backbone": "custom",
        "freeze_encoders": False,
        "projection_dim": 96,
        "projection_type": "mlp",
        "text_pool": "mean",
        "contrastive_loss": "infonce",
    }
    mcfg.update(model_over)
    return {"model": mcfg, "data": {"max_caption_length": 16}}


def _save_ckpt(tmp_path: Path, model: VL_JEPA, cfg: dict, **extra) -> Path:
    path = tmp_path / "checkpoint.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }
    payload.update(extra)
    torch.save(payload, path)
    return path


def test_backend_loads_model_state_dict_and_config(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend

    cfg = _tiny_config(projection_type="mlp", text_pool="mean")
    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
        projection_type="mlp", text_pool="mean",
    )
    path = _save_ckpt(tmp_path, model, cfg)
    device = torch.device("cpu")
    encode_image, encode_text, tokenizer, image_size, max_len = _build_vljepa_backend(
        path, device,
    )
    assert image_size == 64
    assert max_len == 16
    assert tokenizer.kind == "hf"

    model.eval()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        vis_e, txt_e = encode_image(images), encode_text(input_ids, attn)
        vis_j, txt_j = model.get_joint_embedding(images, input_ids, attn)
    assert torch.allclose(vis_e, vis_j, atol=1e-5)
    assert torch.allclose(txt_e, txt_j, atol=1e-5)


def test_backend_refuses_checkpoint_without_config(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend

    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
    )
    path = tmp_path / "no_config.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    with pytest.raises(KeyError, match="config"):
        _build_vljepa_backend(path, torch.device("cpu"))


def test_backend_prefers_model_eval_state(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _vljepa_weights_from_ckpt

    live = {"a": torch.tensor([1.0])}
    eval_w = {"a": torch.tensor([2.0])}
    w = _vljepa_weights_from_ckpt(
        {"model_state_dict": live, "model_eval_state": eval_w},
    )
    assert torch.equal(w["a"], eval_w["a"])
    w2 = _vljepa_weights_from_ckpt({"model_state_dict": live})
    assert torch.equal(w2["a"], live["a"])


def test_unified_eval_backend_accepts_model_state_dict(tmp_path: Path) -> None:
    from src.eval_all_checkpoints import _build_vljepa_backend

    cfg = _tiny_config()
    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
        projection_type="mlp", text_pool="mean",
    )
    path = _save_ckpt(tmp_path, model, cfg)
    encode_images, encode_texts, image_size, _preprocess = _build_vljepa_backend(
        str(path), "cpu",
    )
    assert image_size == 64
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 100, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        vis = encode_images(images)
        vis_j, txt_j = model.get_joint_embedding(images, input_ids, attn)
        # Unified encode_texts tokenizes strings; compare against the same
        # VL-JEPA text tower using the backend tokenizer ids.
        from experiments.evaluate_retrieval import _build_vljepa_backend as _er
        _enc_i, encode_text, tokenizer, _isz, _ml = _er(path, torch.device("cpu"))
        ids, mask = tokenizer.encode(["a small red bird", "a dog on a couch"])
        txt = encode_texts(["a small red bird", "a dog on a couch"])
        txt_direct = encode_text(ids, mask)
    assert vis.shape == vis_j.shape
    assert torch.allclose(vis.cpu(), vis_j.cpu(), atol=1e-5)
    assert txt.shape == txt_direct.shape
    assert torch.allclose(txt.cpu(), txt_direct.cpu(), atol=1e-5)
    assert txt.shape[-1] == vis.shape[-1]


def test_backend_refuses_incomplete_model_config(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend

    cfg = _tiny_config()
    del cfg["model"]["projection_type"]
    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
        projection_type="mlp", text_pool="mean",
    )
    path = _save_ckpt(tmp_path, model, cfg)
    with pytest.raises(KeyError, match="projection_type"):
        _build_vljepa_backend(path, torch.device("cpu"))


def test_backend_accepts_complete_legacy_flat_config(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend

    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
        projection_type="mlp", text_pool="mean",
    )
    flat = {
        "hidden_dim": 96,
        "patch_size": 16,
        "image_size": 64,
        "predictor_layers": 1,
        "vision_backbone": "custom",
        "text_backbone": "custom",
        "projection_dim": 96,
        "projection_type": "mlp",
        "text_pool": "mean",
    }
    path = tmp_path / "flat.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": flat}, path)
    encode_image, encode_text, _tok, image_size, _max_len = _build_vljepa_backend(
        path, torch.device("cpu"),
    )
    assert image_size == 64
    images = torch.randn(1, 3, 64, 64)
    ids = torch.randint(0, 100, (1, 8))
    attn = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        vis = encode_image(images)
        txt = encode_text(ids, attn)
    assert vis.shape[0] == 1 and txt.shape[0] == 1


def test_backend_mismatched_state_dict_fails_strict(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend

    cfg = _tiny_config()
    model = VL_JEPA(
        hidden_dim=96, patch_size=16, image_size=64, predictor_layers=1,
        freeze_encoders=False, projection_dim=96,
        projection_type="mlp", text_pool="mean",
    )
    state = model.state_dict()
    dropped = next(k for k in state if state[k].ndim > 0)
    bad = {k: v for k, v in state.items() if k != dropped}
    path = tmp_path / "mismatch.pt"
    torch.save({"model_state_dict": bad, "config": cfg}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        _build_vljepa_backend(path, torch.device("cpu"))


@pytest.mark.skipif(not _HAS_OPEN_CLIP, reason="open_clip_torch not installed")
def test_backend_uses_clip_tokenizer_from_config(tmp_path: Path) -> None:
    from experiments.evaluate_retrieval import _build_vljepa_backend
    from src.dataset import CaptionTokenizer

    cfg = _tiny_config()
    cfg["model"].update(
        {
            "vision_backbone": "openclip",
            "text_backbone": "openclip",
            "openclip_model": "ViT-B-16",
            "openclip_pretrained": None,
            "hidden_dim": 768,
            "image_size": 224,
            "patch_size": 16,
            "projection_dim": 512,
            "projection_type": "clip",
            "text_pool": "eot",
        }
    )
    model = VL_JEPA(
        hidden_dim=768, patch_size=16, image_size=224, predictor_layers=1,
        vision_backbone="openclip", text_backbone="openclip",
        openclip_model="ViT-B-16", openclip_pretrained=None,
        freeze_encoders=True, projection_dim=512,
        projection_type="clip", text_pool="eot",
    )
    path = _save_ckpt(tmp_path, model, cfg)
    _enc_i, _enc_t, tokenizer, image_size, _max_len = _build_vljepa_backend(
        path, torch.device("cpu"),
    )
    assert isinstance(tokenizer, CaptionTokenizer)
    assert tokenizer.kind == "clip"
    assert image_size == 224
    assert cfg["model"]["projection_type"] == "clip"
    assert cfg["model"]["text_pool"] == "eot"
