"""Prepare Conceptual Captions (CC3M / CC12M) for VL-JEPA pretraining.

COCO is too small to push past the ~25% R@1 ceiling (every image is seen ~50x).
Pretraining on Conceptual Captions before fine-tuning on COCO is the highest
long-term lever. CC3M ships as a TSV of ``<caption>\\t<image_url>`` rows; the
images themselves (~500GB at full resolution) must be fetched separately. This
script handles the lightweight, network-light parts of that pipeline:

  1. ``download-tsv``  — fetch the official CC3M/CC12M caption+URL TSV.
  2. ``make-img2dataset`` — print the recommended ``img2dataset`` command that
     downloads + resizes the images into webdataset/parquet shards.
  3. ``build-manifest`` — turn a directory of downloaded images plus the URL TSV
     into the ``<image_path>\\t<caption>`` manifest that
     :class:`src.image_text_dataset.ImageTextPairDataset` consumes.

The actual multi-hundred-GB image download is intentionally left to
``img2dataset`` (parallel, resumable, dedupe-aware) rather than reimplemented
here. Run, e.g.::

    python experiments/download_cc3m.py download-tsv --dataset cc3m --out data/cc3m
    python experiments/download_cc3m.py make-img2dataset --tsv data/cc3m/cc3m.tsv \\
        --out data/cc3m/images
    # ... run the printed img2dataset command ...
    python experiments/download_cc3m.py build-manifest \\
        --images data/cc3m/images --out data/cc3m/train_manifest.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Official Conceptual Captions caption+URL TSVs (Google AI).
CC_TSV_URLS = {
    "cc3m": "https://storage.googleapis.com/conceptual-captions-v1-1-labels/Train_GCC-training.tsv",
    "cc12m": "https://storage.googleapis.com/conceptual_12m/cc12m.tsv",
}

# Image extensions img2dataset typically writes when --output_format=files.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _download_tsv(args: argparse.Namespace) -> int:
    from torchvision.datasets.utils import download_url

    url = CC_TSV_URLS.get(args.dataset)
    if url is None:
        print(f"Unknown dataset {args.dataset!r}; choose from {list(CC_TSV_URLS)}")
        return 1
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.dataset}.tsv"
    print(f"Downloading {args.dataset} caption/URL TSV from {url}")
    download_url(url, str(out_dir), filename=filename)
    print(f"Saved to {out_dir / filename}")
    return 0


def _make_img2dataset(args: argparse.Namespace) -> int:
    tsv = Path(args.tsv).expanduser()
    out = Path(args.out).expanduser()
    if not tsv.is_file():
        print(f"TSV not found: {tsv} (run `download-tsv` first)")
        return 1
    print("Install once:  pip install img2dataset")
    print("Then run (downloads + resizes images into shards):\n")
    print(
        f"img2dataset --url_list {tsv} --input_format tsv \\\n"
        "    --url_col url --caption_col caption \\\n"
        f"    --output_folder {out} --output_format files \\\n"
        f"    --image_size {args.image_size} --resize_mode keep_ratio \\\n"
        "    --processes_count 16 --thread_count 64 --enable_wandb False"
    )
    print(
        "\nNote: CC3M TSVs are 'caption<TAB>url' with no header; pass "
        "--input_format tsv.gz / adjust --url_col,--caption_col if needed."
    )
    return 0


def _build_manifest(args: argparse.Namespace) -> int:
    """Build a <image_path>\\t<caption> manifest from downloaded image files.

    img2dataset with ``--output_format files`` writes ``NNN.jpg`` alongside
    ``NNN.txt`` (the caption) inside shard subfolders. We pair them up.
    """
    images_root = Path(args.images).expanduser()
    if not images_root.is_dir():
        print(f"Images directory not found: {images_root}")
        return 1
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as manifest:
        for image_file in sorted(images_root.rglob("*")):
            if image_file.suffix.lower() not in _IMAGE_EXTS:
                continue
            caption_file = image_file.with_suffix(".txt")
            if not caption_file.is_file():
                skipped += 1
                continue
            caption = caption_file.read_text(encoding="utf-8").strip().replace("\t", " ")
            caption = caption.replace("\n", " ").strip()
            if not caption:
                skipped += 1
                continue
            rel = image_file.relative_to(images_root)
            manifest.write(f"{rel}\t{caption}\n")
            written += 1

    print(f"Wrote {written} pairs to {out_path} (skipped {skipped} without captions)")
    print(
        "Train with ImageTextPairDataset(images_root=" f"{images_root}, "
        f"manifest_path={out_path})."
    )
    return 0 if written else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CC3M/CC12M for VL-JEPA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_tsv = sub.add_parser("download-tsv", help="Fetch the caption/URL TSV")
    p_tsv.add_argument("--dataset", choices=list(CC_TSV_URLS), default="cc3m")
    p_tsv.add_argument("--out", default="data/cc3m", help="Output directory")
    p_tsv.set_defaults(func=_download_tsv)

    p_img = sub.add_parser(
        "make-img2dataset", help="Print the img2dataset command to fetch images"
    )
    p_img.add_argument("--tsv", required=True, help="Path to the caption/URL TSV")
    p_img.add_argument("--out", default="data/cc3m/images", help="Image output dir")
    p_img.add_argument("--image-size", type=int, default=256)
    p_img.set_defaults(func=_make_img2dataset)

    p_man = sub.add_parser(
        "build-manifest", help="Build <image>\\t<caption> manifest from images"
    )
    p_man.add_argument("--images", required=True, help="Downloaded images directory")
    p_man.add_argument("--out", required=True, help="Output manifest path")
    p_man.set_defaults(func=_build_manifest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
