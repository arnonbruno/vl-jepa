#!/usr/bin/env python3
"""Paper-ready component-ablation figure for VL-JEPA.

Reads ``experiments/ablations/ablation_results.json`` (the list of
``{"name", "metrics"}`` records written by ``run_ablations.py``) and renders a
single bar chart: rsum per ablation variant, sorted weakest→strongest, with the
``full`` recipe drawn as a reference line and each bar annotated with its rsum
delta vs ``full``. Bars that beat ``full`` are green, bars that hurt are red.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "experiments" / "ablations" / "ablation_results.json"

# Human-readable label for each variant in the ablation grid.
PRETTY: Dict[str, str] = {
    "full": "full recipe",
    "no_ema": "− model EMA",
    "no_wise_ft": "− WiSE-FT",
    "no_robust": "− EMA & WiSE-FT",
    "mean_pool": "mean pool (vs EOT)",
    "random_proj": "random MLP proj",
    "siglip": "SigLIP loss (vs InfoNCE)",
    "with_jepa": "+ JEPA MSE loss",
    "frozen": "frozen encoders",
}


def load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} is not a non-empty list of ablation records")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-path", default=None,
                        help="PNG path (default: <results dir>/ablation_figure.png)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(Path(args.results_json))
    by_name = {r["name"]: r["metrics"]["rsum"] for r in rows}
    if "full" not in by_name:
        raise SystemExit("'full' variant missing — cannot compute deltas")
    full_rsum = by_name["full"]

    # Sort weakest → strongest so the degradation ladder reads left→right.
    ordered = sorted(by_name.items(), key=lambda kv: kv[1])
    names = [n for n, _ in ordered]
    values = [v for _, v in ordered]
    labels = [PRETTY.get(n, n) for n in names]
    deltas = [v - full_rsum for v in values]

    def bar_color(name: str, delta: float) -> str:
        if name == "full":
            return "#1f77b4"        # reference: blue
        return "#2ca02c" if delta > 0 else "#d62728"  # green up, red down

    colors = [bar_color(n, d) for n, d in zip(names, deltas)]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.6)

    ax.axhline(full_rsum, linestyle="--", linewidth=1.3, color="#1f77b4", alpha=0.8,
               label=f"full recipe (rsum {full_rsum:.1f})")

    # Headroom so the delta annotations above each bar are not clipped.
    ymin = min(values) - 25
    ymax = max(values) + 25
    ax.set_ylim(ymin, ymax)

    for rect, name, delta in zip(bars, names, deltas):
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        if name != "full":
            sign = "+" if delta > 0 else "−"
            ax.annotate(f"{sign}{abs(delta):.1f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 18), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8,
                        color="#2ca02c" if delta > 0 else "#d62728")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("COCO 5K rsum (Σ i2t/t2i R@1+R@5+R@10)")
    ax.set_title("VL-JEPA component ablation — each bar toggles ONE component of the robust recipe")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    output_path = args.output_path
    if output_path is None:
        output_path = str(Path(args.results_json).parent / "ablation_figure.png")
    fig.savefig(output_path, dpi=300)
    print(f"Saved ablation figure to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
