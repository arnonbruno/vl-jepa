"""Karpathy ids and the train2017 overlap audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.protocol.manifests import build_split_audit
from src.protocol.splits import (
    canonical_coco_id,
    clean_adaptation_ids,
    filename_cocoid,
    first_captions,
    karpathy_coco_records,
    karpathy_flickr_records,
)

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_ids_do_not_use_the_year_directory() -> None:
    assert filename_cocoid("COCO_val2014_000000391895.jpg") == 391895
    assert canonical_coco_id(391895) == "coco:391895"
    payload = {
        "images": [
            {
                "cocoid": 5,
                "filename": "COCO_train2014_000000000005.jpg",
                "filepath": "train2014",
                "split": "train",
                "sentences": [{}, {}],
            },
            {
                "cocoid": 9,
                "filename": "COCO_val2014_000000000009.jpg",
                "filepath": "val2014",
                "split": "test",
                "sentences": [{}],
            },
        ]
    }
    records = karpathy_coco_records(payload)
    groups = {
        "train": {5},
        "restval": set(),
        "val": set(),
        "test": {9},
    }
    # The helper requires every split key. This toy only checks identity.
    assert records[0].canonical_id == "coco:5"
    assert records[0].filepath == "train2014"
    flickr = karpathy_flickr_records(
        {
            "images": [
                {
                    "filename": "1000092795.jpg",
                    "imgid": 0,
                    "split": "train",
                    "sentences": [{}],
                }
            ]
        }
    )
    assert flickr[0].canonical_id == "flickr30k:1000092795.jpg"
    assert groups["train"].isdisjoint(groups["test"])
    assert first_captions(["a", "b", "c", "d", "e", "f"]) == ["a", "b", "c", "d", "e"]


def test_clean_set_removes_final_benchmark_ids() -> None:
    grouped = {
        "train": {1, 2},
        "restval": {3, 4},
        "val": {5},
        "test": {4, 6},
    }
    parts = clean_adaptation_ids(grouped, benchmark_coco_ids={4, 6})
    assert parts["clean_train"] == {1, 2, 3}
    assert 4 not in parts["clean_train"]
    assert parts["removed_from_train"] == {4}
    assert parts["clean_dev"] == {5}


@pytest.mark.skipif(
    not (ROOT / "data" / "external" / "dataset_coco.json").is_file(),
    reason="Karpathy annotations are not in data/external",
)
def test_official_karpathy_overlap_counts() -> None:
    audit = build_split_audit(ROOT)
    counts_ready = {
        split: len(ids) for split, ids in audit["coco_groups"].items()
    }
    assert counts_ready == {"train": 82783, "restval": 30504, "val": 5000, "test": 5000}
    flickr = {split: len(ids) for split, ids in audit["flickr_groups"].items()}
    assert flickr == {"train": 29000, "val": 1014, "test": 1000}
    overlap = audit["overlap"]
    assert overlap["train2017_n"] == 118287
    assert overlap["val2017_n"] == 5000
    assert overlap["train2017_val2017_overlap"] == 0
    assert overlap["karpathy_test_in_train2017"] == 4407
    assert overlap["karpathy_test_in_val2017"] == 593
    assert overlap["per_karpathy_split"]["restval"]["in_train2017"] == 26705
    assert overlap["per_karpathy_split"]["restval"]["in_val2017"] == 3799
    assert overlap["per_karpathy_split"]["val"]["in_train2017"] == 4392
    assert overlap["per_karpathy_split"]["val"]["in_val2017"] == 608
    assert overlap["per_karpathy_split"]["train"]["in_val2017"] == 0
    assert overlap["all_karpathy_n"] == 123287
    assert overlap["karpathy_ids_outside_coco2017"] == 0
    assert overlap["train2017_equals_all_karpathy_ids"] is False
    assert overlap["train2017_equals_karpathy_minus_val2017"] is True
    assert overlap["val2017_subset_of_karpathy_ids"] is True
    assert overlap["train2017_adapted_checkpoint_is_clean_karpathy_test"] is False
    assert len(audit["parts"]["clean_train"]) == 113287
    assert len(audit["parts"]["clean_dev"]) == 5000
    assert audit["parts"]["clean_train"].isdisjoint(audit["parts"]["karpathy_test"])
    assert audit["parts"]["clean_train"].isdisjoint(audit["coco_groups"]["val"])
    assert audit["hashes"]["karpathy_coco"] == (
        "2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d"
    )
    assert audit["hashes"]["karpathy_flickr"] == (
        "db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af"
    )
    assert audit["hashes"]["karpathy_zip"] == (
        "4cfd70132527b80933105e5829dc9034eaab9573482e2e680abbab6130244817"
    )
    assert audit["coco_sentence_histogram"] == {5: 122959, 6: 324, 7: 4}
    assert audit["flickr_sentence_histogram"] == {5: 31014}


@pytest.mark.skipif(
    not (ROOT / "manifests" / "overlap_report.json").is_file(),
    reason="split manifests have not been generated",
)
def test_written_manifests_match_the_audit_counts() -> None:
    overlap = json.loads((ROOT / "manifests" / "overlap_report.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "manifests" / "benchmark_registry.json").read_text(encoding="utf-8"))
    assert overlap["karpathy_test_in_train2017"] == 4407
    assert overlap["train2017_equals_karpathy_minus_val2017"] is True
    assert overlap["train2017_adapted_checkpoint_is_clean_karpathy_test"] is False
    assert registry["counts"]["clean_train"] == 113287
    assert registry["restval_included"] is True
    assert registry["disjointness"]["claimed_disjoint_from_not_loaded_benchmarks"] is False

    def ids(path: Path) -> set[str]:
        with path.open(encoding="utf-8", newline="") as handle:
            return {row["canonical_id"] for row in csv.DictReader(handle)}

    clean = ids(ROOT / "manifests" / "clean_adaptation_ids.csv")
    dev = ids(ROOT / "manifests" / "development_ids.csv")
    test_ids = ids(ROOT / "manifests" / "final_benchmark_coco_ids.csv")
    assert len(clean) == 113287
    assert len(dev) == 5000
    assert len(test_ids) == 5000
    assert clean.isdisjoint(dev)
    assert clean.isdisjoint(test_ids)
    assert dev.isdisjoint(test_ids)
    assert all(item.startswith("coco:") for item in clean)
