"""Flickr evaluation must use split=='test', not sorted filename tails."""

from __future__ import annotations

import ast
import csv
import inspect
from pathlib import Path

import torch

from experiments.evaluate_flickr30k import _load_test_split


def _write_ann_csv(path: Path) -> None:
    # Filename order would pick zzz.jpg as the tail; the Karpathy test row is aaa.jpg.
    rows = [
        {
            "filename": "zzz.jpg",
            "split": "train",
            "raw": "['train caption a', 'train caption b']",
        },
        {
            "filename": "mmm.jpg",
            "split": "val",
            "raw": "['val caption a', 'val caption b']",
        },
        {
            "filename": "aaa.jpg",
            "split": "test",
            "raw": "['test caption one', 'test caption two', 'c3', 'c4', 'c5']",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "raw"])
        writer.writeheader()
        writer.writerows(rows)


def test_flickr_loader_uses_split_test_not_filename_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "flickr_annotations_30k.csv"
    _write_ann_csv(csv_path)
    filenames, texts, text_to_image = _load_test_split(csv_path)

    assert filenames == ["aaa.jpg"]
    assert "zzz.jpg" not in filenames
    assert "mmm.jpg" not in filenames
    assert texts == ["test caption one", "test caption two", "c3", "c4", "c5"]
    assert torch.equal(text_to_image, torch.zeros(5, dtype=torch.long))
    caps = ast.literal_eval(
        next(r for r in csv.DictReader(csv_path.open()) if r["split"] == "test")["raw"]
    )
    assert len(texts) == len(caps)


def test_eval_all_checkpoints_flickr_delegates_to_split_test() -> None:
    from src import eval_all_checkpoints as m

    src = inspect.getsource(m._load_flickr30k)
    assert "[-1000:]" not in src
    assert "_load_test_split" in src
    assert "split" in inspect.getsource(m)
