#!/usr/bin/env python3
"""Standalone training metrics visualization.

Reads a ``metrics.json`` produced by the JEPA training loop (a list of
per-epoch dicts) and renders a 2x2 summary figure covering loss curves,
NCE accuracy, and retrieval recall. Optionally overlays a second run for
comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt


def load_metrics(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} does not contain a non-empty list of epoch records")
    return data


def series(metrics: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Return (epochs, values) for records that contain ``key``."""
    xs: list[float] = []
    ys: list[float] = []
    for i, record in enumerate(metrics):
        if key in record and record[key] is not None:
            xs.append(record.get("epoch", i + 1))
            ys.append(record[key])
    return xs, ys


def best_epoch(metrics: list[dict[str, Any]]) -> tuple[Optional[int], Optional[float]]:
    """Best epoch = lowest val_loss (fallback: highest val_i2t_r1)."""
    candidates = [(m["val_loss"], m.get("epoch")) for m in metrics if m.get("val_loss") is not None]
    if candidates:
        best = min(candidates, key=lambda t: t[0])
        return best[1], best[0]
    candidates = [(m["val_i2t_r1"], m.get("epoch")) for m in metrics if m.get("val_i2t_r1") is not None]
    if candidates:
        best = max(candidates, key=lambda t: t[0])
        return best[1], best[0]
    return None, None


def mark_best(ax, epoch: Optional[int], label: str, color: str = "black") -> None:
    if epoch is None:
        return
    ax.axvline(epoch, linestyle="--", linewidth=1.2, color=color, alpha=0.7)
    ax.annotate(
        label,
        xy=(epoch, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        fontsize=8,
        color=color,
        ha="left",
        va="top",
        rotation=90,
    )


# Distinct color families for run A vs run B.
COLORS_A = {"a": "#1f77b4", "b": "#d62728", "c": "#2ca02c", "d": "#ff7f0e"}
COLORS_B = {"a": "#17becf", "b": "#e377c2", "c": "#8c564b", "d": "#bcbd22"}


def plot_run(
    axes,
    metrics: list[dict[str, Any]],
    colors: dict[str, str],
    run_label: str,
    mark_color: str,
) -> None:
    (ax_loss, ax_acc), (ax_r1, ax_all) = axes
    suffix = f" ({run_label})" if run_label else ""

    # Top-left: train vs val loss.
    x, y = series(metrics, "train_loss")
    if x:
        ax_loss.plot(x, y, color=colors["a"], label=f"train_loss{suffix}")
    x, y = series(metrics, "val_loss")
    if x:
        ax_loss.plot(x, y, color=colors["b"], label=f"val_loss{suffix}")

    # Top-right: NCE accuracy.
    x, y = series(metrics, "train_nce_acc")
    if x:
        ax_acc.plot(x, y, color=colors["a"], label=f"train_nce_acc{suffix}")
    x, y = series(metrics, "val_nce_acc")
    if x:
        ax_acc.plot(x, y, color=colors["b"], label=f"val_nce_acc{suffix}")

    # Bottom-left: retrieval R@1.
    x, y = series(metrics, "val_i2t_r1")
    if x:
        ax_r1.plot(x, y, color=colors["a"], label=f"i2t R@1{suffix}")
    x, y = series(metrics, "val_t2i_r1")
    if x:
        ax_r1.plot(x, y, color=colors["b"], label=f"t2i R@1{suffix}")

    # Bottom-right: all retrieval metrics.
    retrieval = [
        ("val_i2t_r1", "i2t R@1", colors["a"], "-"),
        ("val_i2t_r5", "i2t R@5", colors["a"], "--"),
        ("val_i2t_r10", "i2t R@10", colors["a"], ":"),
        ("val_t2i_r1", "t2i R@1", colors["b"], "-"),
        ("val_t2i_r5", "t2i R@5", colors["b"], "--"),
        ("val_t2i_r10", "t2i R@10", colors["b"], ":"),
    ]
    for key, label, color, style in retrieval:
        x, y = series(metrics, key)
        if x:
            ax_all.plot(x, y, color=color, linestyle=style, label=f"{label}{suffix}")

    epoch, _ = best_epoch(metrics)
    best_label = f"best{suffix} (ep {epoch})" if epoch is not None else ""
    for ax in (ax_loss, ax_acc, ax_r1, ax_all):
        mark_best(ax, epoch, best_label, color=mark_color)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-path", required=True, help="Path to metrics.json")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to save the PNG (default: <metrics dir>/training_plot.png)",
    )
    parser.add_argument("--show", action="store_true", help="Display the interactive plot")
    parser.add_argument(
        "--compare",
        default=None,
        help="Path to a second metrics.json to overlay for comparison",
    )
    parser.add_argument("--title", default=None, help="Figure title")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    metrics = load_metrics(args.metrics_path)
    compare_metrics = load_metrics(args.compare) if args.compare else None

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.metrics_path)), "training_plot.png")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_loss, ax_acc), (ax_r1, ax_all) = axes

    label_a = "run A" if compare_metrics else ""
    plot_run(axes, metrics, COLORS_A, label_a, mark_color="#1f77b4")
    if compare_metrics:
        plot_run(axes, compare_metrics, COLORS_B, "run B", mark_color="#17becf")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")

    ax_acc.set_title("NCE accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")

    ax_r1.set_title("Retrieval R@1")
    ax_r1.set_xlabel("epoch")
    ax_r1.set_ylabel("recall@1")

    ax_all.set_title("Retrieval (R@1 / R@5 / R@10)")
    ax_all.set_xlabel("epoch")
    ax_all.set_ylabel("recall")

    for ax in (ax_loss, ax_acc, ax_r1, ax_all):
        ax.legend(fontsize=8, loc="best")

    title = args.title or "Training metrics"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
