"""Diagnostic: compare raw CLIP zero-shot vs our VL-JEPA wiring vs trained ckpt
on the exact COCO val retrieval protocol used in training.

This isolates whether the 43% ceiling is a *wiring* loss (our reimplementation
fails to reproduce CLIP zero-shot) or a *training* loss (overfitting drifts away
from a good start).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import create_dataloaders
from src.model import VL_JEPA
from src.trainer import retrieval_recall

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_ROOT = Path("~/.cache/torch/hub/checkpoints").expanduser()
MAX_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


@torch.no_grad()
def raw_clip_recall(val_loader):
    """True CLIP zero-shot using open_clip's own encode_image/encode_text."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.to(DEVICE).eval()

    img_feats, txt_feats = [], []
    seen = 0
    for images, input_ids, _ in val_loader:
        images = images.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        if input_ids.size(1) < 77:
            pad = torch.zeros(input_ids.size(0), 77 - input_ids.size(1),
                              dtype=input_ids.dtype, device=DEVICE)
            input_ids = torch.cat([input_ids, pad], dim=1)
        with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
            i = model.encode_image(images)
            t = model.encode_text(input_ids)
        img_feats.append(F.normalize(i.float(), dim=-1).cpu())
        txt_feats.append(F.normalize(t.float(), dim=-1).cpu())
        seen += images.size(0)
        if seen >= MAX_IMAGES:
            break

    img = F.normalize(torch.cat(img_feats), dim=-1)
    txt = F.normalize(torch.cat(txt_feats), dim=-1)
    sim = img @ txt.T
    tgt = torch.arange(sim.size(0))
    out = {}
    for k in (1, 5, 10):
        i2t = (sim.argsort(1, descending=True)[:, :k] == tgt[:, None]).any(1).float().mean().item()
        t2i = (sim.T.argsort(1, descending=True)[:, :k] == tgt[:, None]).any(1).float().mean().item()
        out[f"i2t_r{k}"] = i2t
        out[f"t2i_r{k}"] = t2i
    return out


def build_model(projection_type="clip"):
    return VL_JEPA(
        vision_backbone="openclip", text_backbone="openclip",
        openclip_model="ViT-B-16", openclip_pretrained="openai",
        freeze_encoders=True, projection_dim=512,
        projection_type=projection_type, text_pool="eot",
        contrastive_loss="siglip",
    ).to(DEVICE).eval()


@torch.no_grad()
def vljepa_recall(model, val_loader):
    img_feats, txt_feats = [], []
    seen = 0
    for images, input_ids, attn in val_loader:
        images = images.to(DEVICE); input_ids = input_ids.to(DEVICE)
        attn = attn.to(DEVICE) if attn is not None else None
        v, l = model.get_joint_embedding(images, input_ids, attn)
        img_feats.append(v.float().cpu()); txt_feats.append(l.float().cpu())
        seen += images.size(0)
        if seen >= MAX_IMAGES:
            break
    img = F.normalize(torch.cat(img_feats), dim=-1)
    txt = F.normalize(torch.cat(txt_feats), dim=-1)
    sim = img @ txt.T
    tgt = torch.arange(sim.size(0))
    out = {}
    for k in (1, 5, 10):
        i2t = (sim.argsort(1, descending=True)[:, :k] == tgt[:, None]).any(1).float().mean().item()
        t2i = (sim.T.argsort(1, descending=True)[:, :k] == tgt[:, None]).any(1).float().mean().item()
        out[f"i2t_r{k}"] = i2t; out[f"t2i_r{k}"] = t2i
    return out


def fmt(d):
    return " ".join(f"{k}={v:.3f}" for k, v in d.items())


def main():
    print(f"Device={DEVICE} max_images={MAX_IMAGES}")
    _, val_loader = create_dataloaders(
        batch_size=128, num_workers=8, coco_root=COCO_ROOT, image_size=224,
        max_caption_length=64, download=False, text_backbone="openclip",
        openclip_model="ViT-B-16",
    )

    print("\n[1] RAW open_clip encode_image/encode_text (true zero-shot reference)")
    print("   ", fmt(raw_clip_recall(val_loader)))

    model = build_model()
    print("\n[2] FRESH VL_JEPA wiring (frozen, clip proj, eot pool) — should match [1]")
    print("   ", fmt(vljepa_recall(model, val_loader)))

    res_model = build_model("clip_residual")
    print("\n[2b] FRESH VL_JEPA clip_residual (zero-init residual) — must also match [1]")
    print("   ", fmt(vljepa_recall(res_model, val_loader)))

    ckpt_path = Path("experiments/exp_jepa_768d_50ep/checkpoint_epoch5.pt")
    if ckpt_path.is_file():
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd["model_state_dict"], strict=False)
        model.eval()
        print(f"\n[3] Trained checkpoint_epoch5 (reported best ~43%)")
        print("   ", fmt(vljepa_recall(model, val_loader)))


if __name__ == "__main__":
    main()
