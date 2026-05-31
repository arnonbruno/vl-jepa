"""AAAI-grade COCO image-text retrieval evaluation.

Reports the *standard* multi-caption protocol (5 captions/image) so the numbers
are directly comparable to published baselines, unlike the training loop's
simplified one-caption square-matrix recall.

Backends (pick one):
  * ``--checkpoint PATH``      evaluate a trained VL-JEPA checkpoint.
  * ``--zeroshot``             evaluate an open_clip model zero-shot (baseline).

Protocols:
  * ``5k``  full 5000-image COCO val2017 test (the standard "COCO 5K" number).
  * ``1k``  average over 5 disjoint 1000-image folds ("COCO 1K", the older
            protocol many papers also report).
  * ``both`` (default) reports both.

Examples
--------
Zero-shot CLIP ViT-B/16 baseline (no training needed)::

    python experiments/evaluate_retrieval.py --zeroshot \
        --openclip-model ViT-B-16 --openclip-pretrained openai

Zero-shot CLIP ViT-L/14 baseline (stronger ceiling)::

    python experiments/evaluate_retrieval.py --zeroshot \
        --openclip-model ViT-L-14 --openclip-pretrained openai

Trained VL-JEPA checkpoint::

    python experiments/evaluate_retrieval.py \
        --checkpoint experiments/exp_jepa_768d_16ep/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    CaptionTokenizer,
    _caption_ann_path,
    _image_root,
    ensure_coco_2017,
)
from src.eval_retrieval import compute_retrieval_metrics, format_metrics  # noqa: E402
from src.model import VL_JEPA  # noqa: E402

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None

from torchvision import transforms
from torchvision.datasets import CocoCaptions


def _clip_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])


class _ImageDataset(Dataset):
    """Returns one transformed image per COCO image index."""

    def __init__(self, coco: CocoCaptions, transform):
        self.coco = coco
        self.transform = transform

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, idx: int):
        img = self.coco._load_image(self.coco.ids[idx])
        return self.transform(img)


def _load_coco(coco_root: Path) -> CocoCaptions:
    ensure_coco_2017(coco_root, splits=("val",), download=False)
    return CocoCaptions(
        root=str(_image_root(coco_root, "val")),
        annFile=str(_caption_ann_path(coco_root, "val")),
    )


def _gather_captions(
    coco: CocoCaptions,
    captions_per_image: int,
) -> Tuple[List[str], torch.Tensor]:
    """Flatten captions to a list and the (Nt,) text->image index map."""
    texts: List[str] = []
    text_to_image: List[int] = []
    for image_idx in range(len(coco)):
        anns = coco.coco.imgToAnns[coco.ids[image_idx]]
        caps = [a["caption"].strip() for a in anns][:captions_per_image]
        for cap in caps:
            texts.append(cap)
            text_to_image.append(image_idx)
    return texts, torch.tensor(text_to_image, dtype=torch.long)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_images_loader(
    encode_fn,
    image_ds: _ImageDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        image_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    feats = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feats.append(F.normalize(encode_fn(images).float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def _encode_texts(
    encode_fn,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    for start in range(0, input_ids.size(0), batch_size):
        end = start + batch_size
        ids = input_ids[start:end].to(device)
        attn = attention_mask[start:end].to(device) if attention_mask is not None else None
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feats.append(F.normalize(encode_fn(ids, attn).float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


def _build_vljepa_backend(checkpoint: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})
    model = VL_JEPA(
        hidden_dim=mcfg.get("hidden_dim", 768),
        patch_size=mcfg.get("patch_size", 16),
        image_size=mcfg.get("image_size", 224),
        mask_ratio=mcfg.get("mask_ratio", 0.75),
        predictor_layers=mcfg.get("predictor_layers", 4),
        vision_backbone=mcfg.get("vision_backbone", "openclip"),
        text_backbone=mcfg.get("text_backbone", "openclip"),
        openclip_model=mcfg.get("openclip_model", "ViT-B-16"),
        openclip_pretrained=mcfg.get("openclip_pretrained", "openai"),
        freeze_encoders=mcfg.get("freeze_encoders", True),
        projection_dim=mcfg.get("projection_dim", 512),
        projection_type=mcfg.get("projection_type", "clip_residual"),
        text_pool=mcfg.get("text_pool", "eot"),
        contrastive_loss=mcfg.get("contrastive_loss", "infonce"),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    image_size = mcfg.get("image_size", 224)
    max_len = cfg.get("data", {}).get("max_caption_length", 64)
    tokenizer = CaptionTokenizer(
        text_backbone="openclip",
        openclip_model=mcfg.get("openclip_model", "ViT-B-16"),
        max_caption_length=max_len,
    )

    def encode_image(images):
        return model.vision_proj.raw(model.context_encoder(images)[:, 0, :].float())

    def encode_text(ids, attn):
        lang = model.language_encoder(ids, attn).float()
        return model.language_proj.raw(model._pool_language(lang, attn))

    return encode_image, encode_text, tokenizer, image_size, max_len


def _build_zeroshot_backend(model_name: str, pretrained: str, device: torch.device):
    if open_clip is None:
        raise ImportError("open_clip_torch is required for --zeroshot")
    # OpenAI CLIP weights were trained with QuickGELU; open_clip's default
    # configs use plain GELU, which silently degrades the model (a noticeable
    # retrieval drop). Force QuickGELU so the baseline is the real CLIP.
    extra = {"force_quick_gelu": True} if pretrained == "openai" else {}
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, **extra,
    )
    model = model.to(device).eval()
    image_size = model.visual.image_size
    image_size = image_size[0] if isinstance(image_size, (tuple, list)) else int(image_size)
    max_len = model.context_length
    tokenizer = CaptionTokenizer(
        text_backbone="openclip",
        openclip_model=model_name,
        max_caption_length=max_len,
    )

    def encode_image(images):
        return model.encode_image(images)

    def encode_text(ids, attn):
        return model.encode_text(ids)

    return encode_image, encode_text, tokenizer, image_size, max_len


# ---------------------------------------------------------------------------
# Protocol drivers
# ---------------------------------------------------------------------------

def _eval_5k(image_embs, text_embs, text_to_image) -> Dict[str, float]:
    return compute_retrieval_metrics(image_embs, text_embs, text_to_image, normalize=False)


def _eval_1k(image_embs, text_embs, text_to_image, caps_per_image: int) -> Dict[str, float]:
    """Average metrics over 5 disjoint 1000-image folds (classic COCO 1K)."""
    num_images = image_embs.size(0)
    fold_size = 1000
    n_folds = num_images // fold_size
    if n_folds == 0:
        return _eval_5k(image_embs, text_embs, text_to_image)
    keys = None
    acc: Dict[str, float] = {}
    for fold in range(n_folds):
        lo, hi = fold * fold_size, (fold + 1) * fold_size
        img_fold = image_embs[lo:hi]
        text_mask = (text_to_image >= lo) & (text_to_image < hi)
        txt_fold = text_embs[text_mask]
        t2i_fold = text_to_image[text_mask] - lo
        m = compute_retrieval_metrics(img_fold, txt_fold, t2i_fold, normalize=False)
        if keys is None:
            keys = list(m.keys())
            acc = {k: 0.0 for k in keys}
        for k in keys:
            acc[k] += m[k]
    return {k: v / n_folds for k, v in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO retrieval evaluation")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="VL-JEPA checkpoint path")
    src.add_argument("--zeroshot", action="store_true", help="open_clip zero-shot baseline")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--coco-root", type=str, default="~/.cache/torch/hub/checkpoints")
    parser.add_argument("--protocol", choices=("5k", "1k", "both"), default="both")
    parser.add_argument("--captions-per-image", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap images (quick smoke); default: full val set")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path to write the metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco_root = Path(args.coco_root).expanduser().resolve()

    if args.checkpoint:
        encode_image, encode_text, tokenizer, image_size, max_len = _build_vljepa_backend(
            Path(args.checkpoint).expanduser().resolve(), device,
        )
        backend = f"VL-JEPA checkpoint: {args.checkpoint}"
    else:
        encode_image, encode_text, tokenizer, image_size, max_len = _build_zeroshot_backend(
            args.openclip_model, args.openclip_pretrained, device,
        )
        backend = f"open_clip zero-shot: {args.openclip_model} ({args.openclip_pretrained})"

    print("=" * 70)
    print("COCO image-text retrieval evaluation")
    print("=" * 70)
    print(f"Backend : {backend}")
    print(f"Device  : {device} | image_size={image_size} | ctx_len={max_len}")

    coco = _load_coco(coco_root)
    transform = _clip_eval_transform(image_size)
    image_ds = _ImageDataset(coco, transform)

    texts, text_to_image = _gather_captions(coco, args.captions_per_image)

    if args.max_images is not None:
        keep = args.max_images
        image_ds.coco = coco  # keep reference; subset via mask below
        mask = text_to_image < keep
        text_to_image = text_to_image[mask]
        texts = [t for t, m in zip(texts, mask.tolist()) if m]

        # Subset images by limiting the dataset length through a view.
        class _Sub(_ImageDataset):
            def __len__(self_inner):
                return keep
        image_ds = _Sub(coco, transform)
        num_images = keep
    else:
        num_images = len(coco)

    print(f"Images  : {num_images} | Captions: {len(texts)} "
          f"({args.captions_per_image}/image)")

    t0 = time.time()
    print("Encoding images...")
    image_embs = _encode_images_loader(
        encode_image, image_ds, device, args.batch_size, args.num_workers,
    )
    print(f"  {image_embs.shape} in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Encoding captions...")
    input_ids, attention_mask = tokenizer.encode(texts)
    text_embs = _encode_texts(
        encode_text, input_ids, attention_mask, device, args.batch_size,
    )
    print(f"  {text_embs.shape} in {time.time() - t1:.1f}s")

    results: Dict[str, Dict[str, float]] = {}
    if args.protocol in ("5k", "both"):
        m5k = _eval_5k(image_embs, text_embs, text_to_image)
        results["5k"] = m5k
        print(f"\n--- COCO {num_images}-image (5K protocol) ---")
        print(format_metrics(m5k))
    if args.protocol in ("1k", "both"):
        m1k = _eval_1k(image_embs, text_embs, text_to_image, args.captions_per_image)
        results["1k"] = m1k
        print(f"\n--- COCO 1K protocol (5-fold avg) ---")
        print(format_metrics(m1k))

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"backend": backend, "num_images": num_images, "results": results}, f, indent=2)
        print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
