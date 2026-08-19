"""Unified COCO loader must use the evaluate_retrieval 5-caption protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from experiments.evaluate_retrieval import _gather_captions, _load_coco as er_load_coco


def _write_fake_coco(root: Path) -> None:
    """Tiny val2017 tree: unsorted image ids, extra captions on one image."""
    img_dir = root / "val2017"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    # Unsorted ids so loader order must follow CocoCaptions (sorted ids), not JSON order.
    image_ids = (20, 3)
    images = []
    annotations = []
    ann_id = 1
    for i, image_id in enumerate(image_ids):
        file_name = f"{image_id:012d}.jpg"
        Image.new("RGB", (16, 16), color=(i * 80 + 40, 60, 90)).save(img_dir / file_name)
        images.append(
            {"id": image_id, "file_name": file_name, "height": 16, "width": 16}
        )
        n_caps = 7 if image_id == 20 else 5
        for cap_i in range(n_caps):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "caption": f"caption-{image_id}-{cap_i}",
                }
            )
            ann_id += 1

    payload = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
    }
    (ann_dir / "captions_val2017.json").write_text(json.dumps(payload))


def test_unified_load_coco_five_captions_sorted_ids_and_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling src.eval_all_checkpoints._load_coco must not TypeError and must match protocol.

    The old loader called ``_image_root()`` / ``_caption_ann_path()`` with no
    arguments (missing coco_root, split) and flattened every annotation, so an
    image with 7 captions would contribute 7 texts instead of 5.
    """
    from src.eval_all_checkpoints import _load_coco

    _write_fake_coco(tmp_path)
    monkeypatch.setattr("src.dataset._default_coco_root", lambda: tmp_path)
    preprocess = transforms.Compose([transforms.ToTensor()])
    loader, captions, text_to_image, n_images = _load_coco(
        image_size=16,
        preprocess=preprocess,
        batch_size=2,
        num_workers=0,
        coco_root=tmp_path,
    )

    coco = er_load_coco(tmp_path)
    expected_caps, expected_map = _gather_captions(coco, captions_per_image=5)

    assert n_images == 2
    assert len(coco.ids) == 2
    assert list(coco.ids) == sorted(coco.ids)
    assert list(coco.ids) == [3, 20]
    assert len(captions) == 10
    assert captions == expected_caps
    assert torch.equal(text_to_image, expected_map)
    assert torch.equal(text_to_image, torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    # Extra captions on image 20 must be dropped (not 5+7=12).
    assert all(not c.endswith("-5") and not c.endswith("-6") for c in captions)
    assert len(loader.dataset) == n_images
    batch = next(iter(loader))
    assert batch.shape[0] == 2
    assert batch.shape[1] == 3

    raw = json.loads((tmp_path / "annotations" / "captions_val2017.json").read_text())
    from collections import Counter
    per_image = Counter(a["image_id"] for a in raw["annotations"])
    assert per_image[20] == 7
    assert per_image[3] == 5

    from src.eval_retrieval import compute_retrieval_metrics

    dummy_images = torch.eye(n_images)
    dummy_texts = torch.nn.functional.one_hot(
        text_to_image, num_classes=n_images,
    ).float()
    tagged = compute_retrieval_metrics(
        dummy_images, dummy_texts, text_to_image, normalize=False,
        captions=captions, diagnostics=True,
    )
    diag = tagged["diagnostics"]
    assert diag["captions_per_image_min"] == 5
    assert diag["captions_per_image_max"] == 5
    assert diag["captions_per_image_mean"] == 5.0
