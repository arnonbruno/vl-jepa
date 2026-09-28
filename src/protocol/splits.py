"""Karpathy split contracts keyed by canonical image ids.

Canonical COCO identity is the integer ``cocoid``, written ``coco:{cocoid}``.
It is not the 2014 or 2017 filename and not the directory the file was found
in. Flickr30k identity in the Karpathy JSON is the image filename, written
``flickr30k:{filename}``, because that file has no COCO id.

``restval`` is a real Karpathy split: the val2014 images that the original
protocol adds to training. This module's clean adaptation set is Karpathy
``train`` plus ``restval``, minus any final-benchmark ids in the COCO id
space. That is a Karpathy-derived, benchmark-disjoint training set. It is not
unmodified COCO train2017. COCO train2017 contains thousands of Karpathy test
images; a checkpoint adapted on train2017 is not a clean Karpathy-test model.
Removing those ids after the checkpoint has seen them does not undo that.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_COCO_FILENAME = re.compile(r"^COCO_(?:train|val)2014_(\d+)\.jpg$")


@dataclass(frozen=True)
class SplitRecord:
    canonical_id: str
    dataset: str
    source_id: str
    split: str
    filename: str
    filepath: str | None
    n_sentences: int
    cocoid: int | None


def filename_cocoid(filename: str) -> int | None:
    match = _COCO_FILENAME.match(filename)
    if match is None:
        return None
    return int(match.group(1))


def canonical_coco_id(cocoid: int) -> str:
    return f"coco:{int(cocoid)}"


def canonical_flickr_id(filename: str) -> str:
    return f"flickr30k:{filename}"


def karpathy_coco_records(payload: Mapping) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for image in payload["images"]:
        cocoid = int(image["cocoid"])
        records.append(
            SplitRecord(
                canonical_id=canonical_coco_id(cocoid),
                dataset="coco",
                source_id=str(cocoid),
                split=str(image["split"]),
                filename=str(image["filename"]),
                filepath=image.get("filepath"),
                n_sentences=len(image["sentences"]),
                cocoid=cocoid,
            )
        )
    return records


def karpathy_flickr_records(payload: Mapping) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for image in payload["images"]:
        filename = str(image["filename"])
        records.append(
            SplitRecord(
                canonical_id=canonical_flickr_id(filename),
                dataset="flickr30k",
                source_id=filename,
                split=str(image["split"]),
                filename=filename,
                filepath=image.get("filepath"),
                n_sentences=len(image["sentences"]),
                cocoid=None,
            )
        )
    return records


def load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def coco_caption_image_ids(path: Path) -> set[int]:
    payload = load_json(path)
    return {int(image["id"]) for image in payload["images"]}


def ids_by_split(records: Sequence[SplitRecord]) -> dict[str, set[int] | set[str]]:
    grouped: dict[str, set] = {}
    for record in records:
        key: int | str
        if record.dataset == "coco":
            if record.cocoid is None:
                raise ValueError(f"COCO record {record.canonical_id} has no cocoid")
            key = record.cocoid
        else:
            key = record.source_id
        grouped.setdefault(record.split, set()).add(key)
    return grouped


def assert_disjoint_splits(grouped: Mapping[str, set]) -> None:
    names = list(grouped)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = grouped[left] & grouped[right]
            if overlap:
                raise ValueError(
                    f"splits {left!r} and {right!r} share {len(overlap)} ids"
                )


def clean_adaptation_ids(
    coco_by_split: Mapping[str, set[int]],
    benchmark_coco_ids: set[int],
) -> dict[str, set[int]]:
    """Karpathy train+restval and Karpathy val, minus final-benchmark COCO ids.

    ``restval`` is included in training on purpose and is documented as such.
    Karpathy ``val`` is development, not training. Karpathy ``test`` is a final
    benchmark and is not a member of either set once it is passed in
    ``benchmark_coco_ids``.
    """
    for name in ("train", "restval", "val", "test"):
        if name not in coco_by_split:
            raise KeyError(f"Karpathy COCO split {name!r} is missing")
    classic_train = set(coco_by_split["train"]) | set(coco_by_split["restval"])
    development = set(coco_by_split["val"])
    return {
        "clean_train": classic_train - benchmark_coco_ids,
        "clean_dev": development - benchmark_coco_ids,
        "removed_from_train": classic_train & benchmark_coco_ids,
        "removed_from_dev": development & benchmark_coco_ids,
        "classic_train": classic_train,
        "karpathy_test": set(coco_by_split["test"]),
    }


def overlap_count(left: set, right: set) -> int:
    return len(left & right)


def coco2017_overlap_report(
    coco_by_split: Mapping[str, set[int]],
    train2017: set[int],
    val2017: set[int],
) -> dict:
    per_split = {}
    for split, ids in sorted(coco_by_split.items()):
        per_split[split] = {
            "n": len(ids),
            "in_train2017": overlap_count(ids, train2017),
            "in_val2017": overlap_count(ids, val2017),
            "in_neither": len(ids - train2017 - val2017),
        }
    all_ids: set[int] = set()
    for ids in coco_by_split.values():
        all_ids |= set(ids)
    test_ids = set(coco_by_split["test"])
    return {
        "per_karpathy_split": per_split,
        "train2017_n": len(train2017),
        "val2017_n": len(val2017),
        "all_karpathy_n": len(all_ids),
        "train2017_val2017_overlap": overlap_count(train2017, val2017),
        "train2017_equals_all_karpathy_ids": train2017 == all_ids,
        "train2017_equals_karpathy_minus_val2017": train2017 == (all_ids - val2017),
        "val2017_subset_of_karpathy_ids": val2017 <= all_ids,
        "karpathy_ids_outside_coco2017": len(all_ids - train2017 - val2017),
        "karpathy_test_in_train2017": overlap_count(test_ids, train2017),
        "karpathy_test_in_val2017": overlap_count(test_ids, val2017),
        "train2017_adapted_checkpoint_is_clean_karpathy_test": (
            overlap_count(test_ids, train2017) == 0
        ),
    }


def sentence_histogram(records: Iterable[SplitRecord]) -> dict[int, int]:
    return dict(sorted(Counter(record.n_sentences for record in records).items()))


def first_captions(sentences: Sequence[str], n: int = 5) -> list[str]:
    """Five-caption evaluation uses file order, not a new sample of sentences.

    Some Karpathy COCO images have six or seven sentences. Those extras stay
    in the id manifest. A five-caption score uses the first five and must say so.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if len(sentences) < n:
        raise ValueError(f"need at least {n} sentences, got {len(sentences)}")
    return list(sentences[:n])


def filename_cocoid_mismatches(records: Iterable[SplitRecord]) -> list[SplitRecord]:
    bad = []
    for record in records:
        if record.dataset != "coco":
            continue
        parsed = filename_cocoid(record.filename)
        if parsed != record.cocoid:
            bad.append(record)
    return bad


def write_id_csv(path: Path, records: Sequence[SplitRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "canonical_id",
        "dataset",
        "source_id",
        "split",
        "filename",
        "filepath",
        "n_sentences",
        "cocoid",
    ]
    ordered = sorted(records, key=lambda record: (record.dataset, record.split, record.canonical_id))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            row = asdict(record)
            if row["filepath"] is None:
                row["filepath"] = ""
            if row["cocoid"] is None:
                row["cocoid"] = ""
            writer.writerow(row)


def write_id_set_csv(path: Path, canonical_ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["canonical_id"])
        for canonical_id in sorted(canonical_ids):
            writer.writerow([canonical_id])
