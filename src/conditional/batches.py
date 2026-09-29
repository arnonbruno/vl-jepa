"""Screen batches. Caption and QA stay in separate matrices and interleave.

A missing exclusion manifest blocks real execution. It is not read as an empty
exclusion set. Answer candidates are frozen from training answers before any
evaluation example is seen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from src.conditional.losses import qa_positive_mask
from src.conditional.ledger import assert_gqa_screening_cap

GQA_SCREEN_IMAGE_CAP_APPLIES_TO_EVERY_ARM = True


class MissingExclusionManifest(FileNotFoundError):
    pass


def qualify_image_id(source: str, raw: str) -> str:
    if source not in {"coco", "flickr30k", "gqa"}:
        raise ValueError(f"unknown image source {source!r}")
    text = str(raw)
    if ":" in text or text == "":
        raise ValueError("raw image ids are unqualified and non-empty")
    return f"{source}:{text}"


def assert_source_qualified(image_id: str) -> None:
    source, separator, raw = image_id.partition(":")
    if separator != ":" or source not in {"coco", "flickr30k", "gqa"} or raw == "" or ":" in raw:
        raise ValueError(f"image id {image_id!r} is not source-qualified")


def ids_refer_to_same_image(left: str, right: str) -> bool:
    """Bare numbers from different datasets are different images."""
    assert_source_qualified(left)
    assert_source_qualified(right)
    return left == right


def load_exclusion_manifest(path: Path | None) -> set[str]:
    if path is None or not Path(path).is_file():
        raise MissingExclusionManifest(
            "a missing final-benchmark exclusion manifest blocks execution; "
            "it is not an empty exclusion set"
        )
    excluded = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        assert_source_qualified(text)
        excluded.add(text)
    return excluded


def reject_excluded(image_ids: Sequence[str], excluded: set[str]) -> None:
    overlap = sorted(set(image_ids) & excluded)
    if overlap:
        raise ValueError("screen examples overlap the final benchmark: " + ", ".join(overlap))


def freeze_answer_vocabulary(training_answers: Sequence[str]) -> tuple[str, ...]:
    if not training_answers:
        raise ValueError("the answer vocabulary is frozen from training answers")
    frozen: list[str] = []
    for answer in training_answers:
        if answer not in frozen:
            frozen.append(str(answer))
    return tuple(frozen)


def evaluation_candidates(frozen: Sequence[str], gold_answers: Sequence[str]) -> tuple[str, ...]:
    """Return the frozen list. Gold evaluation answers are not appended."""
    del gold_answers
    return tuple(frozen)


def interleaved_task_schedule(caption_steps: int, qa_steps: int) -> list[str]:
    """Share one caption/QA interleave. This is not caption-then-QA."""
    if caption_steps < 1 or qa_steps < 1:
        raise ValueError("interleaving needs both a caption phase and a QA phase")
    sequence: list[str] = []
    caption_done = 0
    qa_done = 0
    while caption_done < caption_steps or qa_done < qa_steps:
        caption_fraction = caption_done / caption_steps
        qa_fraction = qa_done / qa_steps
        if caption_done < caption_steps and caption_fraction <= qa_fraction:
            sequence.append("caption")
            caption_done += 1
        else:
            sequence.append("qa")
            qa_done += 1
    return sequence


def caption_positive_mask(image_ids: Sequence[str]) -> torch.Tensor:
    for image_id in image_ids:
        assert_source_qualified(image_id)
    ids = list(image_ids)
    mask = torch.zeros(len(ids), len(ids), dtype=torch.bool)
    for row, left in enumerate(ids):
        for column, right in enumerate(ids):
            mask[row, column] = left == right
    return mask


def build_qa_positive_mask(query_answer_ids: Sequence[str], frozen_vocabulary: Sequence[str]) -> torch.Tensor:
    return qa_positive_mask(query_answer_ids, frozen_vocabulary)


def assert_shared_arm_contract(arm_rows: Sequence[dict]) -> None:
    if len(arm_rows) != 6:
        raise ValueError("the target screen has 6 arms")
    schedules = {tuple(row["schedule"]) for row in arm_rows}
    seeds = {row["seed"] for row in arm_rows}
    weights = {row["caption_qa_weight"] for row in arm_rows}
    examples = {tuple(row["example_ids"]) for row in arm_rows}
    if len(schedules) != 1 or len(seeds) != 1 or len(weights) != 1 or len(examples) != 1:
        raise ValueError("screen arms must share examples, schedule, seed, and caption/QA weight")
    if not GQA_SCREEN_IMAGE_CAP_APPLIES_TO_EVERY_ARM:
        raise RuntimeError("the GQA screening cap is shared by every arm")


def assert_screen_gqa_cap(*, images: int, pairs: int) -> None:
    assert_gqa_screening_cap(images=images, pairs=pairs)
