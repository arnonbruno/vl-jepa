"""Build split manifests from the local Karpathy and COCO annotation files.

The clean training set is Karpathy ``train`` plus ``restval``, minus Karpathy
test image ids. ``restval`` is included on purpose. The set is a
Karpathy-derived, benchmark-disjoint training set. It is not COCO train2017.

GQA, Visual Genome, SugarCrepe++, Winoground, and Oxford-IIIT Pets ids are
not in this tree. This module does not claim the clean set is disjoint from
them. Base-model pretraining exposure is not something these files can rule
out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.protocol.hashing import sha256_file
from src.protocol.splits import (
    assert_disjoint_splits,
    clean_adaptation_ids,
    coco2017_overlap_report,
    coco_caption_image_ids,
    filename_cocoid_mismatches,
    ids_by_split,
    karpathy_coco_records,
    karpathy_flickr_records,
    load_json,
    write_id_csv,
    write_id_set_csv,
    canonical_coco_id,
    canonical_flickr_id,
    sentence_histogram,
)

KARPATHY_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

NOT_LOADED_BENCHMARKS = (
    "sugarcrepe++",
    "winoground",
    "gqa",
    "visual_genome",
    "oxford_iiit_pets",
)


def default_paths(root: Path) -> dict[str, Path]:
    external = root / "data" / "external"
    return {
        "karpathy_zip": external / "caption_datasets.zip",
        "karpathy_coco": external / "dataset_coco.json",
        "karpathy_flickr": external / "dataset_flickr30k.json",
        "coco_train2017": external / "coco_ann" / "annotations" / "captions_train2017.json",
        "coco_val2017": external / "coco_ann" / "annotations" / "captions_val2017.json",
        "coco_ann_zip": external / "coco_ann" / "annotations_trainval2017.zip",
    }


def data_available(root: Path) -> bool:
    paths = default_paths(root)
    return paths["karpathy_coco"].is_file() and paths["coco_train2017"].is_file()


def build_split_audit(root: Path) -> dict[str, Any]:
    paths = default_paths(root)
    for key in ("karpathy_coco", "karpathy_flickr", "coco_train2017", "coco_val2017"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    coco_records = karpathy_coco_records(load_json(paths["karpathy_coco"]))
    flickr_records = karpathy_flickr_records(load_json(paths["karpathy_flickr"]))
    mismatches = filename_cocoid_mismatches(coco_records)
    if mismatches:
        raise ValueError(f"{len(mismatches)} Karpathy filenames disagree with cocoid")
    coco_groups = ids_by_split(coco_records)
    flickr_groups = ids_by_split(flickr_records)
    assert_disjoint_splits(coco_groups)
    assert_disjoint_splits(flickr_groups)
    train2017 = coco_caption_image_ids(paths["coco_train2017"])
    val2017 = coco_caption_image_ids(paths["coco_val2017"])
    overlap = coco2017_overlap_report(coco_groups, train2017, val2017)
    benchmark = set(coco_groups["test"])
    parts = clean_adaptation_ids(coco_groups, benchmark)
    if parts["clean_train"] & benchmark:
        raise RuntimeError("clean train still contains Karpathy test ids")
    if parts["clean_dev"] & benchmark:
        raise RuntimeError("development still contains Karpathy test ids")
    if parts["clean_train"] & parts["clean_dev"]:
        raise RuntimeError("clean train and development overlap")
    coco_sentences = sentence_histogram(coco_records)
    flickr_sentences = sentence_histogram(flickr_records)
    hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }
    return {
        "coco_records": coco_records,
        "flickr_records": flickr_records,
        "coco_groups": coco_groups,
        "flickr_groups": flickr_groups,
        "overlap": overlap,
        "parts": parts,
        "hashes": hashes,
        "paths": {name: str(path.relative_to(root)) for name, path in paths.items()},
        "coco_sentence_histogram": coco_sentences,
        "flickr_sentence_histogram": flickr_sentences,
    }


def write_manifests(root: Path, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    audit = build_split_audit(root) if audit is None else audit
    out = root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    write_id_csv(out / "karpathy_coco_ids.csv", audit["coco_records"])
    write_id_csv(out / "karpathy_flickr30k_ids.csv", audit["flickr_records"])
    write_id_set_csv(
        out / "clean_adaptation_ids.csv",
        [canonical_coco_id(image_id) for image_id in audit["parts"]["clean_train"]],
    )
    write_id_set_csv(
        out / "development_ids.csv",
        [canonical_coco_id(image_id) for image_id in audit["parts"]["clean_dev"]],
    )
    write_id_set_csv(
        out / "final_benchmark_coco_ids.csv",
        [canonical_coco_id(image_id) for image_id in audit["parts"]["karpathy_test"]],
    )
    write_id_set_csv(
        out / "final_benchmark_flickr30k_ids.csv",
        [canonical_flickr_id(filename) for filename in audit["flickr_groups"]["test"]],
    )
    overlap_path = out / "overlap_report.json"
    overlap_path.write_text(json.dumps(audit["overlap"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = manifest_summary(audit)
    (out / "benchmark_registry.json").write_text(
        json.dumps(summary["benchmark_registry"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "provenance.json").write_text(
        json.dumps(summary["provenance"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "licenses.md").write_text(summary["licenses_md"], encoding="utf-8")
    return summary


def manifest_summary(audit: dict[str, Any]) -> dict[str, Any]:
    parts = audit["parts"]
    flickr = audit["flickr_groups"]
    overlap = audit["overlap"]
    counts = {
        "karpathy_coco_train": len(audit["coco_groups"]["train"]),
        "karpathy_coco_restval": len(audit["coco_groups"]["restval"]),
        "karpathy_coco_val": len(audit["coco_groups"]["val"]),
        "karpathy_coco_test": len(audit["coco_groups"]["test"]),
        "clean_train": len(parts["clean_train"]),
        "clean_dev": len(parts["clean_dev"]),
        "removed_from_train": len(parts["removed_from_train"]),
        "restval_included_in_clean_train": True,
        "flickr30k_train": len(flickr["train"]),
        "flickr30k_val": len(flickr["val"]),
        "flickr30k_test": len(flickr["test"]),
    }
    registry = {
        "label": "Karpathy-derived, benchmark-disjoint training set",
        "not_labeled_as": "unmodified COCO train2017",
        "restval_included": True,
        "counts": counts,
        "loaded_final_benchmarks": {
            "coco_karpathy_test": {
                "id_space": "coco",
                "n": counts["karpathy_coco_test"],
                "canonical_form": "coco:{cocoid}",
            },
            "flickr30k_karpathy_test": {
                "id_space": "flickr30k",
                "n": counts["flickr30k_test"],
                "canonical_form": "flickr30k:{filename}",
                "note": (
                    "Flickr ids are a different space from COCO. Removing them "
                    "does not remove COCO images, and a COCO-trained checkpoint "
                    "is not thereby a clean Flickr model in the pretraining sense."
                ),
            },
        },
        "not_loaded": list(NOT_LOADED_BENCHMARKS),
        "disjointness": {
            "clean_train_disjoint_from_karpathy_coco_test": True,
            "clean_train_disjoint_from_karpathy_coco_val": True,
            "claimed_disjoint_from_not_loaded_benchmarks": False,
            "pretraining_contamination": "unknown_not_ruled_out",
            "train2017_adapted_checkpoint_is_clean_karpathy_test": overlap[
                "train2017_adapted_checkpoint_is_clean_karpathy_test"
            ],
        },
        "development_split": "karpathy_coco_val_minus_loaded_final_benchmark_ids",
        "images_on_disk": False,
        "coco_sentence_histogram": {str(k): v for k, v in audit["coco_sentence_histogram"].items()},
        "flickr_sentence_histogram": {str(k): v for k, v in audit["flickr_sentence_histogram"].items()},
        "five_caption_rule": "first_five_sentences_in_file_order",
    }
    provenance = {
        "karpathy_url": KARPATHY_URL,
        "coco_annotations_url": COCO_ANN_URL,
        "paths": audit["paths"],
        "sha256": audit["hashes"],
        "canonical_coco_id": "coco:{cocoid}",
        "canonical_flickr_id": "flickr30k:{filename}",
        "filename_cocoid_checked": True,
        "images_downloaded": False,
    }
    licenses = "\n".join(
        [
            "# Annotation sources",
            "",
            "Manifests are id lists. Image files were not downloaded.",
            "",
            f"- Karpathy split file: {KARPATHY_URL}",
            "  Local zip, `dataset_coco.json`, and `dataset_flickr30k.json` are hashed in `provenance.json`.",
            f"- COCO 2017 captions: {COCO_ANN_URL}",
            "  Used to compare `cocoid` with train2017 and val2017.",
            "  Terms: https://cocodataset.org/#termsofuse",
            "- Flickr30k captions are the Karpathy split in that zip. This directory does not relicense them.",
            "- SugarCrepe++, Winoground, GQA, Visual Genome, and Oxford-IIIT Pets were not loaded.",
            "",
        ]
    )
    return {
        "counts": counts,
        "overlap": overlap,
        "benchmark_registry": registry,
        "provenance": provenance,
        "licenses_md": licenses,
        "coco_sentence_histogram": audit["coco_sentence_histogram"],
        "flickr_sentence_histogram": audit["flickr_sentence_histogram"],
    }
