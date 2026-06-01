"""Qualitative COCO retrieval analysis: where VL-JEPA helps vs where it fails.

Runs the *same* COCO val2017 multi-caption protocol used for the quantitative
tables, but instead of aggregate recalls it dumps concrete examples comparing a
trained VL-JEPA checkpoint against its CLIP zero-shot init:

  * **Text->Image:** for a sample of captions, the rank the query's
    ground-truth image gets under each model, and the top-K retrieved image ids.
    "Wins" (VL-JEPA ranks the GT image much higher than zero-shot) illustrate
    what the COCO fine-tune buys; "losses" are reported too for honesty.
  * **Image->Text:** for a sample of images, the top-K retrieved captions and
    whether any ground-truth caption is among them.

Outputs a Markdown report (``experiments/qualitative_examples.md``) and a JSON
dump. Optionally renders a montage PNG of a few text->image examples
(``--figure``) for the paper.

Example::

    python experiments/qualitative_retrieval.py \
        --checkpoint experiments/exp_jepa_1024d_20ep/checkpoint_best.pt \
        --baseline-model ViT-L-14 --num-examples 12 --figure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.evaluate_retrieval import (  # noqa: E402
    _ImageDataset,
    _build_vljepa_backend,
    _build_zeroshot_backend,
    _clip_eval_transform,
    _encode_images_loader,
    _encode_texts,
    _gather_captions,
    _load_coco,
)


def _embed(backend, coco, device, batch_size, num_workers):
    encode_image, encode_text, tokenizer, image_size, _ = backend
    transform = _clip_eval_transform(image_size)
    image_ds = _ImageDataset(coco, transform)
    image_embs = _encode_images_loader(encode_image, image_ds, device, batch_size, num_workers)
    texts, text_to_image = _gather_captions(coco, 5)
    input_ids, attention_mask = tokenizer.encode(texts)
    text_embs = _encode_texts(encode_text, input_ids, attention_mask, device, batch_size)
    return F.normalize(image_embs, dim=-1), F.normalize(text_embs, dim=-1), texts, text_to_image


def _t2i_rank(text_emb, image_embs, gt_image) -> int:
    sims = text_emb @ image_embs.t()
    return int((sims > sims[gt_image]).sum().item())


def _t2i_topk(text_emb, image_embs, k) -> List[int]:
    sims = text_emb @ image_embs.t()
    return sims.topk(k).indices.tolist()


def _i2t_topk(image_emb, text_embs, k) -> List[int]:
    sims = image_emb @ text_embs.t()
    return sims.topk(k).indices.tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="Qualitative retrieval analysis")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline-model", default="ViT-L-14")
    p.add_argument("--baseline-pretrained", default="openai")
    p.add_argument("--coco-root", default="~/.cache/torch/hub/checkpoints")
    p.add_argument("--num-examples", type=int, default=12)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--figure", action="store_true", help="render a montage PNG")
    p.add_argument("--out-md", default="experiments/qualitative_examples.md")
    p.add_argument("--out-json", default="experiments/qualitative_examples.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coco = _load_coco(Path(args.coco_root).expanduser().resolve())

    print("Embedding with VL-JEPA checkpoint ...")
    jepa = _build_vljepa_backend(Path(args.checkpoint).expanduser().resolve(), device)
    j_img, j_txt, texts, t2i = _embed(jepa, coco, device, args.batch_size, args.num_workers)

    print("Embedding with CLIP zero-shot baseline ...")
    base = _build_zeroshot_backend(args.baseline_model, args.baseline_pretrained, device)
    b_img, b_txt, _, _ = _embed(base, coco, device, args.batch_size, args.num_workers)

    g = torch.Generator().manual_seed(args.seed)
    sample = torch.randperm(len(texts), generator=g)[: args.num_examples].tolist()

    rows: List[Dict] = []
    for ti in sample:
        gt = int(t2i[ti].item())
        jr = _t2i_rank(j_txt[ti], j_img, gt)
        br = _t2i_rank(b_txt[ti], b_img, gt)
        rows.append({
            "caption": texts[ti],
            "gt_image_idx": gt,
            "gt_image_id": int(coco.ids[gt]),
            "vljepa_rank": jr + 1,
            "zeroshot_rank": br + 1,
            "rank_delta": (br - jr),
            "vljepa_top5_image_ids": [int(coco.ids[i]) for i in _t2i_topk(j_txt[ti], j_img, args.topk)],
        })

    rows.sort(key=lambda r: r["rank_delta"], reverse=True)

    # Image->text examples (first few sampled images by their first caption).
    i2t_rows: List[Dict] = []
    img_sample = sorted({int(t2i[ti].item()) for ti in sample})[: min(6, args.num_examples)]
    for img_idx in img_sample:
        gt_caps = {i for i, m in enumerate(t2i.tolist()) if m == img_idx}
        j_top = _i2t_topk(j_img[img_idx], j_txt, args.topk)
        i2t_rows.append({
            "image_id": int(coco.ids[img_idx]),
            "vljepa_top_captions": [texts[i] for i in j_top],
            "vljepa_hit@k": any(i in gt_caps for i in j_top),
        })

    # ---- write markdown ----
    lines = ["# Qualitative COCO retrieval: VL-JEPA vs CLIP zero-shot\n"]
    lines.append(f"Checkpoint: `{args.checkpoint}` vs CLIP {args.baseline_model} zero-shot.\n")
    lines.append("## Text -> Image (caption query, rank of the correct image; lower = better)\n")
    lines.append("| caption | zero-shot rank | VL-JEPA rank | Δ |")
    lines.append("|---|---|---|---|")
    for r in rows:
        cap = r["caption"].replace("|", "/")[:80]
        lines.append(f"| {cap} | {r['zeroshot_rank']} | {r['vljepa_rank']} | {r['rank_delta']:+d} |")
    wins = sum(1 for r in rows if r["rank_delta"] > 0)
    losses = sum(1 for r in rows if r["rank_delta"] < 0)
    lines.append(f"\n{wins}/{len(rows)} sampled captions improve under VL-JEPA, "
                 f"{losses} regress (Δ = zero-shot_rank - vljepa_rank).\n")
    lines.append("## Image -> Text (top retrieved captions)\n")
    for r in i2t_rows:
        lines.append(f"**image {r['image_id']}** (hit@{args.topk}: {r['vljepa_hit@k']})")
        for c in r["vljepa_top_captions"]:
            lines.append(f"  - {c}")
        lines.append("")
    Path(args.out_md).write_text("\n".join(lines))
    Path(args.out_json).write_text(json.dumps({"t2i": rows, "i2t": i2t_rows}, indent=2))
    print(f"Wrote {args.out_md} and {args.out_json}")
    print(f"t2i wins: {wins}/{len(rows)}, regressions: {losses}")

    if args.figure:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping figure")
            return
        n = min(3, len(rows))
        fig, axes = plt.subplots(n, args.topk, figsize=(2.2 * args.topk, 2.5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        for ri in range(n):
            r = rows[ri]
            for ci, img_id in enumerate(r["vljepa_top5_image_ids"]):
                ax = axes[ri, ci]
                img = coco._load_image(img_id)
                ax.imshow(img)
                ax.axis("off")
                correct = img_id == r["gt_image_id"]
                ax.set_title("GT" if correct else "", color="green", fontsize=10)
            axes[ri, 0].set_ylabel(r["caption"][:40], fontsize=7)
        fig.suptitle("VL-JEPA text->image top-5 (green=correct)", fontsize=11)
        fig.tight_layout()
        out_png = Path(args.out_md).with_suffix(".png")
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
