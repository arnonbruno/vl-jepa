"""
Precompute frozen vision/language encoder outputs for COCO 2017.

Saves ``<output_dir>/train.pt`` and ``val.pt`` with:
  - local_patch_emb, global_patch_emb: unmasked patch+CLS embeddings (fp16)
  - language_emb: full token sequence embeddings (fp16)
  - attention_mask: int64 token mask

Use with ``experiments/exp_jepa_training.py --cached-data <output_dir>``.
Recommended: ``--no-multi-crop`` during precompute and training for deterministic crops.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, print_config
from src.dataset import COCOCaptionDataset, build_dataloader_kwargs, resolve_num_workers
from src.model import VL_JEPA, make_multicrop_views


def _expand(path: str) -> str:
    return os.path.expanduser(path)


@torch.no_grad()
def _encode_batch(
    model: VL_JEPA,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    *,
    use_multi_crop: bool,
    global_crop_size: int,
    local_crop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = images.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    if use_multi_crop:
        views = make_multicrop_views(
            images,
            global_size=global_crop_size,
            local_size=local_crop_size,
            training=False,
        )
        local_images = views["local"]
        global_images = views["global"]
    else:
        local_images = global_images = images

    model.eval()
    local_emb = model.context_encoder(local_images, mask=None)
    global_emb = model.context_encoder(global_images, mask=None)
    language_emb = model.language_encoder(input_ids, attention_mask)

    return (
        local_emb.cpu().half(),
        global_emb.cpu().half(),
        language_emb.cpu().half(),
        attention_mask.cpu().long(),
    )


def _precompute_split(
    model: VL_JEPA,
    dataset: COCOCaptionDataset,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    use_multi_crop: bool,
    global_crop_size: int,
    local_crop_size: int,
    max_batches: int | None,
) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **build_dataloader_kwargs(num_workers),
    )

    local_chunks: list[torch.Tensor] = []
    global_chunks: list[torch.Tensor] = []
    language_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, input_ids, attention_mask = batch
        local_emb, global_emb, language_emb, attn = _encode_batch(
            model,
            images,
            input_ids,
            attention_mask,
            device,
            use_multi_crop=use_multi_crop,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
        )
        local_chunks.append(local_emb)
        global_chunks.append(global_emb)
        language_chunks.append(language_emb)
        mask_chunks.append(attn)

    return {
        "local_patch_emb": torch.cat(local_chunks, dim=0),
        "global_patch_emb": torch.cat(global_chunks, dim=0),
        "language_emb": torch.cat(language_chunks, dim=0),
        "attention_mask": torch.cat(mask_chunks, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute COCO encoder embeddings")
    parser.add_argument("--config", type=str, default="configs/mvp_pretrained_siglip.yaml")
    parser.add_argument("--output-dir", type=str, default="data/embedding_cache")
    parser.add_argument("--coco-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per split (debug)")
    parser.add_argument(
        "--multi-crop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply multi-crop before encoding (default: from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    coco_root = _expand(args.coco_root or data_cfg.get("coco_root", "~/.cache/torch/hub/checkpoints"))
    use_multi_crop = (
        train_cfg.get("use_multi_crop", False)
        if args.multi_crop is None
        else args.multi_crop
    )
    global_crop_size = train_cfg.get("global_crop_size", model_cfg["image_size"])
    local_crop_size = train_cfg.get("local_crop_size", 96)
    image_size = model_cfg["image_size"]
    max_caption_length = data_cfg.get("max_caption_length", 64)
    workers = resolve_num_workers(args.num_workers or data_cfg.get("num_workers"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print_config(cfg)
    print(f"  COCO root: {coco_root}")
    print(f"  Multi-crop: {use_multi_crop}")
    print(f"  Output: {args.output_dir}")

    model = VL_JEPA(
        hidden_dim=model_cfg["hidden_dim"],
        patch_size=model_cfg["patch_size"],
        image_size=model_cfg["image_size"],
        mask_ratio=model_cfg["mask_ratio"],
        predictor_layers=model_cfg["predictor_layers"],
        momentum_tau=model_cfg["momentum_tau"],
        text_mask_ratio=model_cfg.get("text_mask_ratio", 0.0),
        vision_backbone=model_cfg.get("vision_backbone", "custom"),
        text_backbone=model_cfg.get("text_backbone", "custom"),
        freeze_encoders=model_cfg.get("freeze_encoders", True),
        projection_dim=model_cfg.get("projection_dim", 256),
        contrastive_loss=model_cfg.get("contrastive_loss", "infonce"),
    ).to(device)
    model.eval()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "config": args.config,
        "use_multi_crop": use_multi_crop,
        "global_crop_size": global_crop_size,
        "local_crop_size": local_crop_size,
        "image_size": image_size,
        "vision_backbone": model_cfg.get("vision_backbone"),
        "text_backbone": model_cfg.get("text_backbone"),
    }

    for split in ("train", "val"):
        print(f"\nPrecomputing {split} split...")
        dataset = COCOCaptionDataset(
            split=split,
            coco_root=coco_root,
            image_size=image_size,
            max_caption_length=max_caption_length,
            download=True,
        )
        payload = _precompute_split(
            model,
            dataset,
            device,
            batch_size=args.batch_size,
            num_workers=workers,
            use_multi_crop=use_multi_crop,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            max_batches=args.max_batches,
        )
        payload["metadata"] = {**metadata, "split": split, "num_samples": payload["local_patch_emb"].size(0)}
        out_path = out_dir / f"{split}.pt"
        torch.save(payload, out_path)
        size_gb = out_path.stat().st_size / 1e9
        print(f"  Saved {out_path} ({payload['local_patch_emb'].size(0)} samples, {size_gb:.2f} GB)")

    print(f"\nDone. Train with: --cached-data {out_dir}")


if __name__ == "__main__":
    main()
