"""Cross-dataset Flickr30K image-text retrieval evaluation (Karpathy 1K test).

Measures **zero-shot transfer** of a VL-JEPA checkpoint that was fine-tuned on
COCO to a *different* retrieval benchmark, Flickr30K. This is the standard
cross-dataset generalization probe reported by CLIP/BLIP/SigLIP: a COCO-tuned
model is evaluated on the 1000-image Flickr30K Karpathy test split (5 captions /
image) with no further training.

The same multi-caption protocol and metrics as
:mod:`experiments.evaluate_retrieval` are reused (Recall@1/5/10, MedR, MeanR,
rsum), so the numbers are directly comparable to published Flickr30K results.

Data source: the ``nlphuji/flickr30k`` HuggingFace dataset repo, which ships the
images (``flickr30k-images.zip``) and Karpathy-split annotations
(``flickr_annotations_30k.csv``). Both are downloaded + cached on first run via
``huggingface_hub`` (no ``datasets`` loader script needed).

Examples
--------
Zero-shot CLIP baselines (cross-dataset reference)::

    python experiments/evaluate_flickr30k.py --zeroshot \
        --openclip-model ViT-B-16 --openclip-pretrained openai
    python experiments/evaluate_flickr30k.py --zeroshot \
        --openclip-model ViT-L-14 --openclip-pretrained openai

COCO-trained VL-JEPA checkpoints (transfer)::

    python experiments/evaluate_flickr30k.py \
        --checkpoint experiments/exp_jepa_768d_16ep/checkpoint_best.pt
    python experiments/evaluate_flickr30k.py \
        --checkpoint experiments/exp_jepa_1024d_20ep/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_retrieval import compute_retrieval_metrics, format_metrics  # noqa: E402
from experiments.evaluate_retrieval import (  # noqa: E402
    _build_vljepa_backend,
    _build_zeroshot_backend,
    _clip_eval_transform,
    _encode_texts,
)

FLICKR_REPO = "nlphuji/flickr30k"
_ANNOTATIONS_CSV = "flickr_annotations_30k.csv"
_IMAGES_ZIP = "flickr30k-images.zip"


def _ensure_flickr_assets() -> Tuple[Path, Path]:
    """Download + cache the Flickr30K annotations CSV and image zip."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for Flickr30K evaluation "
            "(`pip install huggingface_hub`)."
        ) from exc
    csv_path = Path(hf_hub_download(FLICKR_REPO, _ANNOTATIONS_CSV, repo_type="dataset"))
    zip_path = Path(hf_hub_download(FLICKR_REPO, _IMAGES_ZIP, repo_type="dataset"))
    return csv_path, zip_path


def _load_test_split(csv_path: Path) -> Tuple[List[str], List[str], torch.Tensor]:
    """Parse the Karpathy ``test`` split: filenames, flat captions, t2i map."""
    import csv

    filenames: List[str] = []
    texts: List[str] = []
    text_to_image: List[int] = []
    image_idx = 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            caps = ast.literal_eval(row["raw"])
            filenames.append(str(row["filename"]))
            for cap in caps:
                texts.append(str(cap).strip())
                text_to_image.append(image_idx)
            image_idx += 1
    return filenames, texts, torch.tensor(text_to_image, dtype=torch.long)


class _ZipImageDataset(Dataset):
    """Reads Flickr30K images by filename out of the cached image zip."""

    def __init__(self, zip_path: Path, filenames: List[str], transform):
        self.zip_path = str(zip_path)
        self.filenames = filenames
        self.transform = transform
        self._zip: zipfile.ZipFile | None = None
        # Flickr30K zip may nest images under a top-level folder; resolve once.
        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()
        base = ""
        for n in names:
            if n.endswith(filenames[0]):
                base = n[: -len(filenames[0])]
                break
        self._base = base

    def _zf(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        with self._zf().open(self._base + self.filenames[idx]) as fh:
            img = Image.open(io.BytesIO(fh.read())).convert("RGB")
        return self.transform(img)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        return state


@torch.no_grad()
def _encode_images(encode_fn, ds, device, batch_size, num_workers) -> torch.Tensor:
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    feats = []
    for images in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feats.append(F.normalize(encode_fn(images).float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flickr30K retrieval evaluation")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="VL-JEPA checkpoint path")
    src.add_argument("--zeroshot", action="store_true", help="open_clip zero-shot baseline")
    parser.add_argument("--openclip-model", type=str, default="ViT-B-16")
    parser.add_argument("--openclip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        encode_image, encode_text, tokenizer, image_size, _ = _build_vljepa_backend(
            Path(args.checkpoint).expanduser().resolve(), device,
        )
        backend = f"VL-JEPA checkpoint: {args.checkpoint}"
    else:
        encode_image, encode_text, tokenizer, image_size, _ = _build_zeroshot_backend(
            args.openclip_model, args.openclip_pretrained, device,
        )
        backend = f"open_clip zero-shot: {args.openclip_model} ({args.openclip_pretrained})"

    print("=" * 70)
    print("Flickr30K image-text retrieval (Karpathy 1K test, cross-dataset)")
    print("=" * 70)
    print(f"Backend : {backend}")
    print(f"Device  : {device} | image_size={image_size}")

    csv_path, zip_path = _ensure_flickr_assets()
    filenames, texts, text_to_image = _load_test_split(csv_path)
    print(f"Images  : {len(filenames)} | Captions: {len(texts)} "
          f"({len(texts) // max(1, len(filenames))}/image)")

    transform = _clip_eval_transform(image_size)
    image_ds = _ZipImageDataset(zip_path, filenames, transform)

    t0 = time.time()
    print("Encoding images...")
    image_embs = _encode_images(encode_image, image_ds, device, args.batch_size, args.num_workers)
    print(f"  {tuple(image_embs.shape)} in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Encoding captions...")
    input_ids, attention_mask = tokenizer.encode(texts)
    text_embs = _encode_texts(encode_text, input_ids, attention_mask, device, args.batch_size)
    print(f"  {tuple(text_embs.shape)} in {time.time() - t1:.1f}s")

    metrics = compute_retrieval_metrics(image_embs, text_embs, text_to_image, normalize=False)
    print("\n--- Flickr30K 1K test ---")
    print(format_metrics(metrics))

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {"backend": backend, "num_images": len(filenames),
                 "results": {"flickr1k": metrics}}, f, indent=2,
            )
        print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
