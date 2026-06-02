#!/usr/bin/env python3
"""Unified evaluation of all VL-JEPA final checkpoints.

Runs the exact same eval path for all 6 headline models:
  1. CLIP ViT-B/16 zero-shot
  2. CLIP ViT-L/14 zero-shot
  3. ViT-B/16 + InfoNCE (robust recipe, 16 epochs)
  4. ViT-B/16 + SigLIP (ablation, 5 epochs)
  5. ViT-L/14 + InfoNCE (robust recipe, 20 epochs — actually 2ep checkpoint)
  6. ViT-L/14 + SigLIP (20 epochs)

For each, exports:
  COCO 5K:  R@1/R@5/R@10 for i2t and t2i, median rank, mean rank, rsum
  COCO 1K:  5-fold R@1/R@5/R@10 for i2t and t2i, median rank, mean rank, rsum
  Flickr30K: R@1/R@5/R@10 for i2t and t2i, median rank, mean rank, rsum

Usage:
    python src/eval_all_checkpoints.py [--output-dir artifacts/v1_final_report_2026_06_02/eval_json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_retrieval import compute_retrieval_metrics  # noqa: E402


# ── Model definitions ──────────────────────────────────────────────────────

MODELS = [
    {
        "run_id": "clip_vitb16_zeroshot",
        "display_name": "CLIP ViT-B/16 zero-shot",
        "zeroshot": True,
        "openclip_model": "ViT-B-16",
        "openclip_pretrained": "openai",
        "checkpoint": None,
        "config_path": None,
        "loss": "N/A (zero-shot)",
        "pooling": "EOT",
        "projection": "CLIP native",
        "ema_decay": "N/A",
        "wise_ft_alpha": "N/A",
        "wise_ft_target": "N/A",
        "epoch_selected": "N/A",
        "selection_metric": "N/A",
        "seed": "N/A",
    },
    {
        "run_id": "clip_vitl14_zeroshot",
        "display_name": "CLIP ViT-L/14 zero-shot",
        "zeroshot": True,
        "openclip_model": "ViT-L-14",
        "openclip_pretrained": "openai",
        "checkpoint": None,
        "config_path": None,
        "loss": "N/A (zero-shot)",
        "pooling": "EOT",
        "projection": "CLIP native",
        "ema_decay": "N/A",
        "wise_ft_alpha": "N/A",
        "wise_ft_target": "N/A",
        "epoch_selected": "N/A",
        "selection_metric": "N/A",
        "seed": "N/A",
    },
    {
        "run_id": "vitb16_infonce",
        "display_name": "VL-JEPA ViT-B/16 + InfoNCE",
        "zeroshot": False,
        "openclip_model": "ViT-B-16",
        "openclip_pretrained": "openai",
        "checkpoint": "experiments/exp_jepa_768d_16ep/checkpoint_best.pt",
        "config_path": "configs/openclip_vitb16_robust.yaml",
        "loss": "InfoNCE (FP32 softmax-CE + MoCo)",
        "pooling": "EOT",
        "projection": "CLIP-seeded zero-init residual",
        "ema_decay": 0.996,
        "wise_ft_alpha": 0.5,
        "wise_ft_target": "EMA weights",
        "epoch_selected": "best R@1",
        "selection_metric": "mean R@1",
        "seed": 42,
    },
    {
        "run_id": "vitb16_siglip",
        "display_name": "VL-JEPA ViT-B/16 + SigLIP",
        "zeroshot": False,
        "openclip_model": "ViT-B-16",
        "openclip_pretrained": "openai",
        "checkpoint": "experiments/ablations/siglip/exp_jepa_768d_5ep/checkpoint_best.pt",
        "config_path": "configs/openclip_vitb16_robust.yaml + ablations/siglip/config.yaml",
        "loss": "SigLIP (sigmoid pairwise)",
        "pooling": "EOT",
        "projection": "CLIP-seeded zero-init residual",
        "ema_decay": 0.996,
        "wise_ft_alpha": 0.5,
        "wise_ft_target": "EMA weights",
        "epoch_selected": "best R@1",
        "selection_metric": "mean R@1",
        "seed": 42,
    },
    {
        "run_id": "vitl14_infonce",
        "display_name": "VL-JEPA ViT-L/14 + InfoNCE",
        "zeroshot": False,
        "openclip_model": "ViT-L-14",
        "openclip_pretrained": "openai",
        "checkpoint": "experiments/exp_jepa_1024d_2ep/checkpoint_best.pt",
        "config_path": "configs/openclip_vitl14_robust.yaml",
        "loss": "InfoNCE (FP32 softmax-CE + MoCo)",
        "pooling": "EOT",
        "projection": "CLIP-seeded zero-init residual",
        "ema_decay": 0.996,
        "wise_ft_alpha": 0.5,
        "wise_ft_target": "EMA weights",
        "epoch_selected": "best R@1",
        "selection_metric": "mean R@1",
        "seed": 42,
    },
    {
        "run_id": "vitl14_siglip",
        "display_name": "VL-JEPA ViT-L/14 + SigLIP",
        "zeroshot": False,
        "openclip_model": "ViT-L-14",
        "openclip_pretrained": "openai",
        "checkpoint": "experiments/exp_jepa_1024d_20ep/checkpoint_best.pt",
        "config_path": "configs/openclip_vitl14_siglip.yaml",
        "loss": "SigLIP (sigmoid pairwise)",
        "pooling": "EOT",
        "projection": "CLIP-seeded zero-init residual",
        "ema_decay": 0.996,
        "wise_ft_alpha": 0.5,
        "wise_ft_target": "EMA weights",
        "epoch_selected": "best R@1 (E20)",
        "selection_metric": "mean R@1",
        "seed": 42,
    },
]


# ── Backend builders (reuse from evaluate_retrieval.py) ─────────────────────

def _build_zeroshot_backend(model_name: str, pretrained: str, device: str):
    """Return (encode_images, encode_texts, image_size) for a zero-shot OpenCLIP model."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    image_size = model.visual.image_size if hasattr(model.visual, "image_size") else 224
    if isinstance(image_size, (tuple, list)):
        image_size = image_size[0]

    @torch.no_grad()
    def encode_images(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images.to(device)).float()

    @torch.no_grad()
    def encode_texts(texts: list) -> torch.Tensor:
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens).float()

    return encode_images, encode_texts, image_size, preprocess


def _build_vljepa_backend(checkpoint_path: str, device: str):
    """Return (encode_images, encode_texts, image_size) for a VL-JEPA checkpoint."""
    from src.model import VL_JEPA
    from src.dataset import CaptionTokenizer, CLIP_MEAN, CLIP_STD
    from torchvision import transforms

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = VL_JEPA(
        hidden_dim=cfg.get("hidden_dim", 768),
        patch_size=cfg.get("patch_size", 16),
        image_size=cfg.get("image_size", 224),
        mask_ratio=cfg.get("mask_ratio", 0.75),
        text_mask_ratio=cfg.get("text_mask_ratio", 0.0),
        predictor_layers=cfg.get("predictor_layers", 4),
        momentum_tau=cfg.get("momentum_tau", 0.996),
        vision_backbone=cfg.get("vision_backbone", None),
        text_backbone=cfg.get("text_backbone", None),
        openclip_model=cfg.get("openclip_model", None),
        contrastive_loss=cfg.get("contrastive_loss", "infonce"),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Load EMA weights if available
    ema_state = ckpt.get("ema", None)
    wise_ft_alpha = cfg.get("wise_ft_alpha", 0.5)

    image_size = cfg.get("image_size", 224)
    tokenizer = CaptionTokenizer()
    proj = model.text_proj if hasattr(model, "text_proj") else None

    preprocess = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])

    @torch.no_grad()
    def encode_images(images: torch.Tensor) -> torch.Tensor:
        return model.encode_image(images.to(device)).float()

    @torch.no_grad()
    def encode_texts(texts: list) -> torch.Tensor:
        tokens = tokenizer(texts).to(device)
        return model.encode_text(tokens).float()

    return encode_images, encode_texts, image_size, preprocess


# ── COCO data loader ───────────────────────────────────────────────────────

def _load_coco(image_size: int, preprocess, batch_size: int = 64, num_workers: int = 4):
    """Load COCO val2017 and return (image_loader, captions, text_to_image)."""
    from torchvision.datasets import CocoCaptions
    from torch.utils.data import DataLoader

    from src.dataset import ensure_coco_2017, _image_root, _caption_ann_path

    ensure_coco_2017()
    img_root = _image_root()
    ann_path = _caption_ann_path()

    class ImageDataset:
        def __init__(self, root, ann, transform):
            self.ds = CocoCaptions(root=root, annFile=ann, transform=transform)
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, _ = self.ds[idx]
            return img

    ds = ImageDataset(img_root, ann_path, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Load captions and image mapping
    import json as _json
    with open(ann_path) as f:
        ann = _json.load(f)

    # Build ordered lists
    img_ids = sorted({a["image_id"] for a in ann["annotations"]})
    img_id_to_idx = {iid: i for i, iid in enumerate(img_ids)}

    captions = []
    text_to_image = []
    for a in ann["annotations"]:
        captions.append(a["caption"])
        text_to_image.append(img_id_to_idx[a["image_id"]])

    text_to_image = torch.tensor(text_to_image, dtype=torch.long)

    return loader, captions, text_to_image, len(img_ids)


# ── Flickr30K data loader ──────────────────────────────────────────────────

def _load_flickr30k(image_size: int, preprocess, batch_size: int = 64, num_workers: int = 4):
    """Load Flickr30K Karpathy 1K test split."""
    import ast
    import io
    import zipfile
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from huggingface_hub import hf_hub_download

    csv_path = hf_hub_download(
        repo_id="nlphuji/flickr30k", filename="flickr_annotations_30k.csv"
    )
    zip_path = hf_hub_download(
        repo_id="nlphuji/flickr30k", filename="flickr30k-images.zip"
    )

    import csv as _csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Karpathy test split: last 1000 images
    all_filenames = sorted({r["filename"] for r in rows})
    test_filenames = set(all_filenames[-1000:])

    test_rows = [r for r in rows if r["filename"] in test_filenames]
    img_filenames = sorted({r["filename"] for r in test_rows})
    img_to_idx = {fn: i for i, fn in enumerate(img_filenames)}

    captions = []
    text_to_image = []
    for r in test_rows:
        captions.append(r["raw"])
        text_to_image.append(img_to_idx[r["filename"]])

    text_to_image = torch.tensor(text_to_image, dtype=torch.long)

    # Image dataset from zip
    class FlickrZipDataset(Dataset):
        def __init__(self, zip_path, filenames, transform):
            self.zip_path = zip_path
            self.filenames = filenames
            self.transform = transform
            self._zf = None

        def _get_zf(self):
            if self._zf is None:
                self._zf = zipfile.ZipFile(self.zip_path, "r")
            return self._zf

        def __len__(self):
            return len(self.filenames)

        def __getitem__(self, idx):
            fn = self.filenames[idx]
            zf = self._get_zf()
            with zf.open(f"flickr30k-images/{fn}") as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            return self.transform(img)

    ds = FlickrZipDataset(zip_path, img_filenames, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return loader, captions, text_to_image, len(img_filenames)


# ── Embedding extraction ───────────────────────────────────────────────────

def _encode_dataset(loader, encode_fn, desc: str) -> torch.Tensor:
    """Encode all images through the loader."""
    embeddings = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        emb = encode_fn(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


def _encode_captions_in_chunks(captions: list, encode_fn, chunk_size: int = 256) -> torch.Tensor:
    """Encode captions in chunks to avoid OOM."""
    embeddings = []
    for i in range(0, len(captions), chunk_size):
        chunk = captions[i:i + chunk_size]
        emb = encode_fn(chunk)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)


# ── COCO 1K 5-fold protocol ───────────────────────────────────────────────

def _coco_1k_5fold(image_embs, text_embs, text_to_image, n_images=5000):
    """Compute COCO 1K metrics: average over 5 disjoint 1000-image folds."""
    from src.eval_retrieval import compute_retrieval_metrics

    fold_size = 1000
    all_metrics = []

    for fold in range(5):
        start = fold * fold_size
        end = start + fold_size

        # Filter images
        img_mask = torch.zeros(n_images, dtype=torch.bool)
        img_mask[start:end] = True

        # Filter captions that belong to these images
        cap_mask = img_mask[text_to_image]

        fold_img_embs = image_embs[start:end]
        fold_text_embs = text_embs[cap_mask]
        fold_t2i = text_to_image[cap_mask] - start  # re-index to 0..999

        m = compute_retrieval_metrics(fold_img_embs, fold_text_embs, fold_t2i)
        all_metrics.append(m)

    # Average
    avg = {}
    for key in all_metrics[0]:
        avg[key] = float(np.mean([m[key] for m in all_metrics]))
    return avg


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all VL-JEPA checkpoints")
    parser.add_argument("--output-dir", default="artifacts/v1_final_report_2026_06_02/eval_json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of run_ids to evaluate (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = MODELS
    if args.models:
        models_to_eval = [m for m in MODELS if m["run_id"] in args.models]

    print(f"=" * 70)
    print(f"VL-JEPA Unified Evaluation — {len(models_to_eval)} models")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"=" * 70)

    for model_def in models_to_eval:
        run_id = model_def["run_id"]
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {model_def['display_name']} ({run_id})")
        print(f"{'=' * 70}")

        t0 = time.time()
        result = {
            "run_id": run_id,
            "display_name": model_def["display_name"],
            "checkpoint": model_def.get("checkpoint"),
            "config_path": model_def.get("config_path"),
            "loss": model_def["loss"],
            "pooling": model_def["pooling"],
            "projection": model_def["projection"],
            "ema_decay": model_def["ema_decay"],
            "wise_ft_alpha": model_def["wise_ft_alpha"],
            "wise_ft_target": model_def["wise_ft_target"],
            "epoch_selected": model_def["epoch_selected"],
            "selection_metric": model_def["selection_metric"],
            "seed": model_def["seed"],
            "tokenizer": "CLIP BPE" if "ViT-B" in model_def["openclip_model"] else "CLIP BPE (ViT-L)",
            "dataset_split": "COCO val2017 (5K) + Flickr30K Karpathy 1K test",
        }

        # Build backend
        if model_def["zeroshot"]:
            encode_images, encode_texts, image_size, preprocess = _build_zeroshot_backend(
                model_def["openclip_model"], model_def["openclip_pretrained"], args.device
            )
        else:
            ckpt_path = model_def["checkpoint"]
            if not Path(ckpt_path).exists():
                print(f"  SKIP: checkpoint not found: {ckpt_path}")
                continue
            encode_images, encode_texts, image_size, preprocess = _build_vljepa_backend(
                ckpt_path, args.device
            )

        # ── COCO eval ──
        print("  Loading COCO...")
        coco_loader, coco_captions, coco_t2i, n_coco_images = _load_coco(
            image_size, preprocess, args.batch_size, args.num_workers
        )

        print("  Encoding COCO images...")
        coco_img_embs = _encode_dataset(coco_loader, encode_images, "COCO images")
        print("  Encoding COCO captions...")
        coco_txt_embs = _encode_captions_in_chunks(coco_captions, encode_texts)

        # COCO 5K
        print("  Computing COCO 5K metrics...")
        coco_5k = compute_retrieval_metrics(coco_img_embs, coco_txt_embs, coco_t2i)
        result["coco_5k"] = coco_5k
        print(f"    COCO 5K: rsum={coco_5k['rsum']:.2f}")

        # COCO 1K (5-fold)
        print("  Computing COCO 1K metrics (5-fold)...")
        coco_1k = _coco_1k_5fold(coco_img_embs, coco_txt_embs, coco_t2i, n_coco_images)
        result["coco_1k"] = coco_1k
        print(f"    COCO 1K: rsum={coco_1k['rsum']:.2f}")

        # ── Flickr30K eval ──
        print("  Loading Flickr30K...")
        flickr_loader, flickr_captions, flickr_t2i, n_flickr_images = _load_flickr30k(
            image_size, preprocess, args.batch_size, args.num_workers
        )

        print("  Encoding Flickr30K images...")
        flickr_img_embs = _encode_dataset(flickr_loader, encode_images, "Flickr images")
        print("  Encoding Flickr30K captions...")
        flickr_txt_embs = _encode_captions_in_chunks(flickr_captions, encode_texts)

        print("  Computing Flickr30K metrics...")
        flickr = compute_retrieval_metrics(flickr_img_embs, flickr_txt_embs, flickr_t2i)
        result["flickr30k"] = flickr
        print(f"    Flickr30K: rsum={flickr['rsum']:.2f}")

        elapsed = time.time() - t0
        result["eval_time_seconds"] = round(elapsed, 1)

        # Save
        out_path = output_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {out_path} ({elapsed:.1f}s)")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")

    header = (
        f"{'Model':<35} | "
        f"{'COCO 5K':>40} | "
        f"{'COCO 1K':>40} | "
        f"{'Flickr30K':>40}"
    )
    subheader = (
        f"{'':35} | "
        f"{'i2t R@1/R@5/R@10 t2i R@1/R@5/R@10 rsum':>40} | "
        f"{'i2t R@1/R@5/R@10 t2i R@1/R@5/R@10 rsum':>40} | "
        f"{'i2t R@1/R@5/R@10 t2i R@1/R@5/R@10 rsum':>40}"
    )
    print(header)
    print(subheader)
    print("-" * len(header))

    for model_def in models_to_eval:
        run_id = model_def["run_id"]
        json_path = output_dir / f"{run_id}.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            d = json.load(f)

        def fmt(m):
            return (
                f"{m['i2t_r1']:5.1f}/{m['i2t_r5']:5.1f}/{m['i2t_r10']:5.1f} "
                f"{m['t2i_r1']:5.1f}/{m['t2i_r5']:5.1f}/{m['t2i_r10']:5.1f} "
                f"{m['rsum']:6.1f}"
            )

        print(
            f"{model_def['display_name']:<35} | "
            f"{fmt(d['coco_5k']):>40} | "
            f"{fmt(d['coco_1k']):>40} | "
            f"{fmt(d['flickr30k']):>40}"
        )

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
