#!/usr/bin/env python3
"""Epoch sweep: evaluate all saved ViT-L/14 SigLIP checkpoints on COCO 5K + Flickr30K.

Memory-safe: loads one checkpoint at a time, explicit cleanup between epochs.
Each checkpoint is ~10GB (model + EMA), fits in 32GB RAM one at a time.

Outputs per-epoch JSON with full R@1/R@5/R@10 + median/mean rank + rsum.
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.evaluate_retrieval import (
    _build_vljepa_backend,
    _clip_eval_transform,
    _encode_images_loader,
    _encode_texts,
    _gather_captions,
    _load_coco,
    _ImageDataset,
)
from experiments.evaluate_flickr30k import (
    _ensure_flickr_assets,
    _load_test_split,
    _ZipImageDataset,
    _encode_images as _encode_flickr_images,
)
from src.eval_retrieval import compute_retrieval_metrics
from src.dataset import CaptionTokenizer

OUTPUT_DIR = Path("/var/mnt/DATA/OpenClaw/workspace/vl-jepa-aaai/artifacts/v1_final_report_2026_06_02/epoch_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CKPT = Path("experiments/exp_jepa_1024d_20ep/checkpoint_best.pt")
EPOCHS = [4, 8, 12, 16, 20]
DEVICE = torch.device("cuda")
BATCH_SIZE = 64
NUM_WORKERS = 4
COCO_ROOT = Path.home() / ".cache/torch/hub/checkpoints"


def _extract_config(ckpt_path: Path) -> dict:
    """Extract config from best checkpoint, return it without keeping the checkpoint in memory."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    del ckpt
    gc.collect()
    return config


def _patch_and_load(ckpt_path: Path, config: dict, tmp_path: Path):
    """Load epoch checkpoint, patch with config, save temp, free original."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt["config"] = config
    torch.save(ckpt, tmp_path)
    del ckpt
    gc.collect()


def main():
    # Extract config once from best checkpoint
    print("Extracting config from best checkpoint...")
    config = _extract_config(BEST_CKPT)

    # Pre-load COCO (shared across epochs — just metadata, not images)
    print("Loading COCO val2017 metadata...")
    coco = _load_coco(COCO_ROOT)
    texts_5k, t2i_5k = _gather_captions(coco, captions_per_image=5)

    mcfg = config.get("model", {})
    max_len = config.get("data", {}).get("max_caption_length", 64)
    tokenizer = CaptionTokenizer(
        text_backbone="openclip",
        openclip_model=mcfg.get("openclip_model", "ViT-L-14"),
        max_caption_length=max_len,
    )
    input_ids_5k, attn_mask_5k = tokenizer.encode(texts_5k)

    # Pre-load Flickr30K metadata
    print("Loading Flickr30K metadata...")
    csv_path, zip_path = _ensure_flickr_assets()
    flickr_filenames, flickr_texts, flickr_t2i = _load_test_split(csv_path)
    flickr_input_ids, flickr_attn_mask = tokenizer.encode(flickr_texts)

    all_results = {}

    for ep in EPOCHS:
        ckpt_path = Path(f"experiments/exp_jepa_1024d_20ep/checkpoint_epoch{ep}.pt")
        if not ckpt_path.exists():
            print(f"\nSKIP epoch {ep}: {ckpt_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Epoch {ep}: {ckpt_path}")
        print(f"{'='*60}")

        t0 = time.time()

        # Patch checkpoint with config (saves temp, frees original)
        tmp_path = Path(f"/tmp/vitl14_siglip_ep{ep}.pt")
        print("  Patching checkpoint with config...")
        _patch_and_load(ckpt_path, config, tmp_path)

        # Build backend (loads model to GPU)
        print("  Building backend...")
        encode_image, encode_text, tok, image_size, max_len_val = _build_vljepa_backend(
            tmp_path, DEVICE
        )

        # ── COCO 5K ──
        print("  COCO 5K: encoding images...")
        img_ds = _ImageDataset(coco, _clip_eval_transform(image_size))
        img_embs = _encode_images_loader(encode_image, img_ds, DEVICE, BATCH_SIZE, NUM_WORKERS)

        print("  COCO 5K: encoding texts...")
        txt_embs = _encode_texts(encode_text, input_ids_5k, attn_mask_5k, DEVICE, BATCH_SIZE)

        print("  COCO 5K: computing metrics...")
        coco_5k = compute_retrieval_metrics(img_embs, txt_embs, t2i_5k)
        print(f"    i2t: R@1={coco_5k['i2t_r1']:.2f} R@5={coco_5k['i2t_r5']:.2f} R@10={coco_5k['i2t_r10']:.2f}")
        print(f"    t2i: R@1={coco_5k['t2i_r1']:.2f} R@5={coco_5k['t2i_r5']:.2f} R@10={coco_5k['t2i_r10']:.2f}")
        print(f"    rsum={coco_5k['rsum']:.2f}")

        # Free COCO embeddings before Flickr
        del img_embs, txt_embs
        torch.cuda.empty_cache()

        # ── Flickr30K ──
        print("  Flickr30K: encoding images...")
        flickr_ds = _ZipImageDataset(zip_path, flickr_filenames, _clip_eval_transform(image_size))
        flickr_img_embs = _encode_flickr_images(encode_image, flickr_ds, DEVICE, BATCH_SIZE, NUM_WORKERS)

        print("  Flickr30K: encoding texts...")
        flickr_txt_embs = _encode_texts(encode_text, flickr_input_ids, flickr_attn_mask, DEVICE, BATCH_SIZE)

        print("  Flickr30K: computing metrics...")
        flickr_metrics = compute_retrieval_metrics(flickr_img_embs, flickr_txt_embs, flickr_t2i)
        print(f"    i2t: R@1={flickr_metrics['i2t_r1']:.2f} R@5={flickr_metrics['i2t_r5']:.2f} R@10={flickr_metrics['i2t_r10']:.2f}")
        print(f"    t2i: R@1={flickr_metrics['t2i_r1']:.2f} R@5={flickr_metrics['t2i_r5']:.2f} R@10={flickr_metrics['t2i_r10']:.2f}")
        print(f"    rsum={flickr_metrics['rsum']:.2f}")

        elapsed = time.time() - t0

        result = {
            "run_id": f"vitl14_siglip_epoch{ep}",
            "epoch": ep,
            "coco_5k": coco_5k,
            "flickr30k": flickr_metrics,
            "eval_time_seconds": round(elapsed, 1),
        }
        all_results[ep] = result

        # Save individual
        out_path = OUTPUT_DIR / f"vitl14_siglip_epoch{ep}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {out_path} ({elapsed:.1f}s)")

        # Full cleanup
        del encode_image, encode_text, flickr_img_embs, flickr_txt_embs
        del img_ds, flickr_ds
        tmp_path.unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined
    combined_path = OUTPUT_DIR / "vitl14_siglip_epoch_sweep.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    # Summary table
    print(f"\n{'='*90}")
    print("EPOCH SWEEP SUMMARY — ViT-L/14 + SigLIP")
    print(f"{'='*90}")
    print(f"{'Ep':>3} | {'COCO 5K i2t R@1':>14} {'t2i R@1':>8} {'rsum':>8} | {'Flickr i2t R@1':>14} {'t2i R@1':>8} {'rsum':>8}")
    print("-" * 90)
    for ep in sorted(all_results.keys()):
        r = all_results[ep]
        c, f = r["coco_5k"], r["flickr30k"]
        print(f"{ep:>3} | {c['i2t_r1']:>14.2f} {c['t2i_r1']:>8.2f} {c['rsum']:>8.2f} | {f['i2t_r1']:>14.2f} {f['t2i_r1']:>8.2f} {f['rsum']:>8.2f}")


if __name__ == "__main__":
    main()
