"""Component ablation harness for VL-JEPA (AAAI-grade study).

Each ablation toggles a single component relative to the robust recipe
(``configs/openclip_vitb16_robust.yaml``), trains it, and evaluates the best
checkpoint with the *standard* COCO retrieval protocol
(:mod:`experiments.evaluate_retrieval`). Results are tabulated as Markdown +
JSON so the contribution of every component is measurable.

The harness writes an isolated config + output dir per variant (so runs never
clobber each other) and shells out to the training and evaluation scripts.

Quick smoke (1 epoch, few batches — verifies plumbing, not science)::

    python experiments/run_ablations.py --smoke --only full

Full study (each variant a short fine-tune; tune --epochs to taste)::

    python experiments/run_ablations.py --epochs 8

Ablation grid (each row = robust recipe with ONE change):
  full          the complete robust recipe (reference)
  no_ema        disable model-weight EMA
  no_wise_ft    disable WiSE-FT (evaluate raw fine-tuned weights)
  no_robust     disable both EMA and WiSE-FT
  mean_pool     mean text pooling instead of CLIP EOT pooling
  random_proj   random MLP projection instead of CLIP-seeded linear+residual
  siglip        SigLIP sigmoid loss instead of InfoNCE
  with_jepa     re-enable the JEPA MSE objective (alpha=0.1)
  frozen        keep encoders frozen the whole run (no unfreeze)
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "openclip_vitb16_robust.yaml"

# Each variant is a nested override applied on top of the base robust config.
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_ema": {"training": {"use_model_ema": False}},
    "no_wise_ft": {"training": {"wise_ft_alpha": 1.0}},
    "no_robust": {"training": {"use_model_ema": False, "wise_ft_alpha": 1.0}},
    "mean_pool": {"model": {"text_pool": "mean"}},
    "random_proj": {"model": {"projection_type": "mlp"}},
    "siglip": {"model": {"contrastive_loss": "siglip"}},
    "with_jepa": {"loss": {"alpha": 0.1, "beta": 0.9}},
    "frozen": {"training": {"unfreeze_after_epoch": None}},
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _run(cmd: List[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _train_one(name: str, overrides: Dict[str, Any], args) -> Path:
    out_dir = ROOT / "experiments" / "ablations" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(BASE_CONFIG.read_text())
    cfg = _deep_update(base, overrides)
    cfg.setdefault("output", {})["output_dir"] = str(out_dir)
    cfg["training"]["epochs"] = args.epochs

    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cmd = [
        sys.executable, str(ROOT / "experiments" / "exp_jepa_training.py"),
        "--config", str(cfg_path), "--fresh", "--no-tensorboard",
        "--num-workers", str(args.num_workers),
    ]
    if args.max_train_batches is not None:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    _run(cmd)

    hidden = cfg["model"]["hidden_dim"]
    ckpt = out_dir / f"exp_jepa_{hidden}d_{args.epochs}ep" / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint produced for ablation {name!r}: {ckpt}")
    return ckpt


def _eval_one(name: str, ckpt: Path, args) -> Dict[str, float]:
    out_json = ckpt.parent / "retrieval_5k.json"
    cmd = [
        sys.executable, str(ROOT / "experiments" / "evaluate_retrieval.py"),
        "--checkpoint", str(ckpt), "--protocol", "5k",
        "--num-workers", str(args.num_workers), "--batch-size", "128",
        "--output", str(out_json),
    ]
    if args.max_images is not None:
        cmd += ["--max-images", str(args.max_images)]
    _run(cmd)
    data = json.loads(out_json.read_text())
    return data["results"]["5k"]


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Variant | i2t R@1 | i2t R@5 | t2i R@1 | t2i R@5 | rsum |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        m = r["metrics"]
        body += (
            f"| {r['name']} | {m['i2t_r1']:.2f} | {m['i2t_r5']:.2f} | "
            f"{m['t2i_r1']:.2f} | {m['t2i_r5']:.2f} | {m['rsum']:.2f} |\n"
        )
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser(description="VL-JEPA ablation study")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Cap train batches/epoch (smoke testing)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Cap eval images (smoke testing)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset of ablation names to run (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, 20 train batches, 500 eval images")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        if args.max_train_batches is None:
            args.max_train_batches = 20
        if args.max_images is None:
            args.max_images = 500

    names = args.only or list(ABLATIONS.keys())
    rows: List[Dict[str, Any]] = []
    for name in names:
        if name not in ABLATIONS:
            raise SystemExit(f"Unknown ablation {name!r}. Choices: {list(ABLATIONS)}")
        print(f"\n{'=' * 70}\nABLATION: {name}\n{'=' * 70}")
        ckpt = _train_one(name, ABLATIONS[name], args)
        metrics = _eval_one(name, ckpt, args)
        rows.append({"name": name, "metrics": metrics})

    table = _markdown_table(rows)
    out_dir = ROOT / "experiments" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_results.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "ablation_results.md").write_text(table)
    print(f"\n{'=' * 70}\nAblation results\n{'=' * 70}\n{table}")
    print(f"Saved to {out_dir / 'ablation_results.md'}")


if __name__ == "__main__":
    main()
