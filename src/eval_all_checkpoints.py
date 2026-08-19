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

    extra = {"force_quick_gelu": True} if pretrained == "openai" else {}
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, **extra,
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
    """Return (encode_images, encode_texts, image_size, preprocess) for a VL-JEPA ckpt.

    Reuses the loader from ``experiments.evaluate_retrieval``: reads
    ``model_eval_state`` (or ``model_state_dict``) and architecture fields
    from the nested checkpoint config.
    """
    from experiments.evaluate_retrieval import (
        _build_vljepa_backend as _er_backend,
        _clip_eval_transform,
    )

    encode_image, encode_text, tokenizer, image_size, _max_len = _er_backend(
        Path(checkpoint_path), torch.device(device),
    )
    preprocess = _clip_eval_transform(image_size)

    @torch.no_grad()
    def encode_images(images: torch.Tensor) -> torch.Tensor:
        return encode_image(images.to(device)).float()

    @torch.no_grad()
    def encode_texts(texts: list) -> torch.Tensor:
        ids, attn = tokenizer.encode(list(texts))
        return encode_text(ids, attn).float()

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
    """Load Flickr30K Karpathy 1K test split (``split=='test'``)."""
    from torch.utils.data import DataLoader

    from experiments.evaluate_flickr30k import (
        _ZipImageDataset,
        _ensure_flickr_assets,
        _load_test_split,
    )

    csv_path, zip_path = _ensure_flickr_assets()
    filenames, captions, text_to_image = _load_test_split(csv_path)
    ds = _ZipImageDataset(zip_path, filenames, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader, captions, text_to_image, len(filenames)


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
