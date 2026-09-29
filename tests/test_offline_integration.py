"""Offline integration. Missing weights and a dry run do not launch research runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.conditional.batches import (
    MissingExclusionManifest,
    assert_shared_arm_contract,
    caption_positive_mask,
    evaluation_candidates,
    freeze_answer_vocabulary,
    ids_refer_to_same_image,
    interleaved_task_schedule,
    load_exclusion_manifest,
    qualify_image_id,
)
from src.conditional.encoders import (
    GemmaPostPoolProjection,
    WeightsUnavailable,
    clip_eot_indices,
    embeddinggemma_weights_present,
    encode_clip_text_hidden,
    encode_gemma_hidden,
    encode_qwen_hidden,
    load_gemma_projection,
    require_screen_weights,
)
from src.conditional.execute import (
    openai_clip_weights_present,
    real_integration_status,
    run_target_screen_adapter,
)
from src.conditional.locks import (
    ConfirmationExperimentConfig,
    EvaluationCheckpointManifest,
    EvaluationProtocolRecord,
    FreezeRefused,
    rehearsal_visibility,
)
from src.conditional.pooling import last_token_pool
from src.conditional.stores import (
    authorizes_training,
    production_sentinel_path,
    text_fingerprint,
    visual_fingerprint,
    write_test_sentinel,
    write_vector_cache,
    assert_production_sentinel_absent,
)
from src.conditional.targets import TARGET_SPECS, mrl_truncate
from src.conditional.vision_contract import CLIP_VITB16_SPATIAL, extract_spatial_tokens
from src.model import OpenCLIPLanguageEncoder, OpenCLIPVisionEncoder
from src.protocol.preregistration import PREREGISTRATION
from src.protocol.tasks import (
    gqa_fixed_candidate_diagnostic,
    sugarcrepe_pp_accuracy,
    sugarcrepe_pp_example,
    winoground_example,
    winoground_scores,
)

SHA = "a" * 64
OTHER = "b" * 64


def _schedule() -> list[str]:
    return interleaved_task_schedule(2, 2)


def _arms(schedule: list[str] | None = None) -> list[dict]:
    schedule = list(schedule or _schedule())
    example_ids = ["coco:1", "gqa:7"]
    rows = []
    for target_id in TARGET_SPECS:
        for objective in ("infonce", "cosine"):
            rows.append(
                {
                    "target_id": target_id,
                    "objective": objective,
                    "schedule": schedule,
                    "seed": 0,
                    "caption_qa_weight": 1.0,
                    "example_ids": example_ids,
                }
            )
    return rows


def _assets(**overrides):
    schedule = _schedule()
    payload = {
        "scope": "synthetic_screen_not_a_result",
        "dry_run": False,
        "authorize_execution": True,
        "weights_available": {target_id: True for target_id in TARGET_SPECS},
        "query_checkpoint_sha256": SHA,
        "query_preprocessing": "resize_224_center_crop_clip_mean_std",
        "gqa_images": 2,
        "gqa_pairs": 4,
        "caption_steps": 2,
        "qa_steps": 2,
        "schedule": schedule,
        "arms": _arms(schedule),
        "caption_batch": {"task_ids": ["caption", "caption"], "image_ids": ["coco:1", "coco:1"]},
        "qa_batch": {"task_ids": ["qa", "qa"], "image_ids": ["coco:1", "gqa:7"]},
    }
    payload.update(overrides)
    return payload


def _four_caches(root: Path) -> dict:
    ids = ["coco:1", "gqa:7"]
    caches = {}
    visual = visual_fingerprint(checkpoint_sha256=SHA, manifest_sha256=OTHER)
    from src.conditional.cache import ShardedTokenCache

    visual_root = root / "visual"
    ShardedTokenCache(visual_root, visual, n_tokens=2, dim=4, shard_rows=2).write(
        ids, np.zeros((2, 2, 4), dtype=np.float32)
    )
    caches["visual"] = {"root": str(visual_root), "digest": visual.digest(), "ids": ids}
    for kind, pooling in ("captions", "mean"), ("answers", "mean"), ("queries", "eot"):
        fingerprint = text_fingerprint(
            kind=kind,
            checkpoint_sha256=SHA,
            tokenizer="declared",
            prompt=kind,
            pooling=pooling,
            projection="released" if kind != "queries" else "clip_text_projection",
            truncation="native" if kind == "queries" else "mrl512",
            normalization="l2",
            manifest_sha256=OTHER,
        )
        caches[kind] = write_vector_cache(
            root / kind,
            fingerprint,
            ids,
            np.zeros((2, 4), dtype=np.float32),
        )
    return caches


def test_qwen_padding_does_not_use_mask_sum() -> None:
    hidden = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
            [[3.0, 0.0], [0.0, 4.0], [9.0, 9.0], [8.0, 8.0]],
        ]
    )
    left_pad = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    pooled = last_token_pool(hidden, left_pad)
    left_sum_index = int(left_pad[0].sum()) - 1
    assert not torch.equal(pooled[0], hidden[0, left_sum_index])
    assert torch.equal(pooled[0], hidden[0, 3])
    right_sum_index = int(left_pad[1].sum()) - 1
    assert torch.equal(pooled[1], hidden[1, right_sum_index])
    encoded = encode_qwen_hidden(hidden, left_pad, native_dim=2, dim=2)
    assert encoded.shape == (2, 2)
    assert torch.allclose(encoded[0], torch.nn.functional.normalize(hidden[0, 3], dim=-1))


def test_gemma_truncates_after_the_released_projection() -> None:
    module = GemmaPostPoolProjection()
    with torch.no_grad():
        module.up.weight.zero_()
        module.up.bias.zero_()
        module.up.bias[0] = 1
        module.down.weight.zero_()
        module.down.bias.zero_()
        module.down.weight[0, 0] = 1
        module.down.weight[600, 0] = 4
    try:
        encode_gemma_hidden(torch.zeros(1, 2, 768), torch.ones(1, 2), module)
    except WeightsUnavailable:
        pass
    else:
        raise AssertionError("an unloaded EmbeddingGemma projection must block encoding")
    loaded = load_gemma_projection(module.state_dict())
    hidden = torch.zeros(1, 2, 768)
    mask = torch.ones(1, 2)
    encoded = encode_gemma_hidden(hidden, mask, loaded)
    native = loaded(hidden[:, 0, :])
    truncated_first = torch.nn.functional.normalize(hidden[:, 0, :512], dim=-1, eps=1e-6)
    assert encoded.shape == (1, 512)
    assert not torch.allclose(encoded, truncated_first)
    assert float(native[0, 600]) == pytest.approx(4.0)
    assert float(encoded[0, 0]) == pytest.approx(1.0)
    try:
        encode_gemma_hidden(hidden.half(), mask, loaded)
    except ValueError as exc:
        assert "float16" in str(exc)
    else:
        raise AssertionError("EmbeddingGemma inference excludes float16 activations")


def test_missing_target_weights_block_the_screen() -> None:
    available = {target_id: True for target_id in TARGET_SPECS}
    available["qwen3_embedding_0.6b_mrl512"] = False
    try:
        require_screen_weights(available)
    except WeightsUnavailable as exc:
        assert "qwen3_embedding_0.6b_mrl512" in str(exc)
    else:
        raise AssertionError("a missing target arm must block execution")


def test_clip_eot_index_is_the_maximum_id() -> None:
    ids = torch.zeros(1, 77, dtype=torch.long)
    ids[0, 3] = 49407
    ids[0, 10] = 100
    assert int(clip_eot_indices(ids)) == 3
    assert int((ids != 0).sum()) - 1 != 3
    try:
        encode_clip_text_hidden(torch.zeros(1, 76, 4), ids[:, :76], torch.eye(4))
    except ValueError as exc:
        assert "77" in str(exc)
    else:
        raise AssertionError("CLIP text keeps context length 77")


def test_spatial_contract_and_cache_roundtrip(tmp_path) -> None:
    import open_clip

    model = open_clip.create_model("ViT-B-16", pretrained=None)
    model.eval()
    encoder = OpenCLIPVisionEncoder("ViT-B-16", pretrained=None, freeze=True, shared_model=model)
    images = torch.randn(2, 3, 224, 224)
    before = [parameter.detach().clone() for parameter in encoder.visual.parameters()]
    tokens = extract_spatial_tokens(encoder, images)
    assert tuple(tokens.shape) == (2, 197, 768)
    assert CLIP_VITB16_SPATIAL["keep_cls"] is True
    assert CLIP_VITB16_SPATIAL["extraction_layer"] == "visual.ln_post"
    assert CLIP_VITB16_SPATIAL["mask_before_encoder"] is False
    encoder.eval()
    with torch.no_grad():
        direct = encoder(images, mask=None)
    assert torch.allclose(tokens, direct, atol=1e-6)
    from src.conditional.stores import build_visual_cache_from_encoder

    fingerprint = visual_fingerprint(checkpoint_sha256=SHA, manifest_sha256=OTHER)
    built = build_visual_cache_from_encoder(
        encoder,
        images,
        ["coco:1", "gqa:7"],
        tmp_path / "visual",
        fingerprint,
        sources={"coco", "gqa"},
    )
    assert built["cached_vs_live_max_abs"] <= 5e-2
    after = [parameter.detach().clone() for parameter in encoder.visual.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    try:
        build_visual_cache_from_encoder(
            encoder,
            images[:1],
            ["coco:1"],
            tmp_path / "coco-only",
            fingerprint,
            sources={"coco"},
        )
    except ValueError as exc:
        assert "GQA" in str(exc)
    else:
        raise AssertionError("a COCO-only visual cache must be refused")
    text_encoder = OpenCLIPLanguageEncoder("ViT-B-16", pretrained=None, freeze=True, shared_model=model)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    text_ids = tokenizer(["a dog", "a cat"]).long()
    hidden = text_encoder(text_ids, (text_ids != 0).long())
    pooled = encode_clip_text_hidden(hidden, text_ids, model.text_projection)
    reference = torch.nn.functional.normalize(model.encode_text(text_ids).float(), dim=-1)
    assert torch.allclose(pooled, reference, atol=1e-5)


def test_prompt_change_invalidates_a_text_fingerprint() -> None:
    left = text_fingerprint(
        kind="captions",
        checkpoint_sha256=SHA,
        tokenizer="gemma",
        prompt="title: none | text: ",
        pooling="mean",
        projection="768-3072-768",
        truncation="mrl512",
        normalization="l2",
        manifest_sha256=OTHER,
    )
    right = text_fingerprint(
        kind="captions",
        checkpoint_sha256=SHA,
        tokenizer="gemma",
        prompt="task: search result | query: ",
        pooling="mean",
        projection="768-3072-768",
        truncation="mrl512",
        normalization="l2",
        manifest_sha256=OTHER,
    )
    assert left.digest() != right.digest()


def test_sentinel_requires_contents(tmp_path) -> None:
    caches = _four_caches(tmp_path / "caches")
    sentinel = tmp_path / "sentinel.json"
    write_test_sentinel(sentinel, caches, cached_vs_live_max_abs=0.0)
    allowed, _message = authorizes_training(sentinel)
    assert allowed is True
    stale = json.loads(sentinel.read_text(encoding="utf-8"))
    stale["caches"]["visual"]["digest"] = "stale"
    sentinel.write_text(json.dumps(stale), encoding="utf-8")
    allowed, message = authorizes_training(sentinel)
    assert allowed is False
    assert "stale" in message
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    allowed, message = authorizes_training(empty)
    assert allowed is False
    assert "empty" in message
    assert_production_sentinel_absent(tmp_path)
    assert not production_sentinel_path(Path(__file__).resolve().parents[1]).exists()


def test_batch_contract_and_interleaving(tmp_path) -> None:
    schedule = _schedule()
    assert schedule == ["caption", "qa", "caption", "qa"]
    assert schedule != ["caption", "caption", "qa", "qa"]
    assert qualify_image_id("coco", "1") == "coco:1"
    assert ids_refer_to_same_image("coco:1", "gqa:1") is False
    mask = caption_positive_mask(["coco:1", "coco:1", "gqa:7"])
    assert torch.equal(mask[0], torch.tensor([True, True, False]))
    frozen = freeze_answer_vocabulary(["dog", "man", "dog"])
    assert evaluation_candidates(frozen, ["bird"]) == ("dog", "man")
    assert_shared_arm_contract(_arms())
    exclusion = tmp_path / "exclude.txt"
    exclusion.write_text("coco:9\n", encoding="utf-8")
    assert load_exclusion_manifest(exclusion) == {"coco:9"}
    try:
        load_exclusion_manifest(tmp_path / "missing.txt")
    except MissingExclusionManifest as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("a missing exclusion manifest must block execution")


def test_panel_formulas() -> None:
    matched = sugarcrepe_pp_example(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert matched == {"itt_hit": True, "tot_hit": True, "hit": True}
    partial = sugarcrepe_pp_example(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.6, 0.8]),
        torch.tensor([0.0, 1.0]),
    )
    assert partial["itt_hit"] is True
    assert partial["tot_hit"] is False
    assert partial["hit"] is False
    summary = sugarcrepe_pp_accuracy([matched, partial])
    assert summary["accuracy"] == 0.5
    assert summary["independent_average"] != summary["accuracy"]
    constant = winoground_example(
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert constant == {"text": False, "image": False, "group": False}
    scored = winoground_scores([constant])
    assert scored["text_score"] == 0.0
    assert scored["group_score"] == 0.0
    assert scored["ties_count_as_hits"] is False
    assert scored["significance_claim"] is False
    hit = winoground_example(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert hit["group"] is True
    diagnostic = gqa_fixed_candidate_diagnostic(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        ["dog", "bird"],
        ["dog", "cat"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert diagnostic["name"] == "gqa_fixed_candidate_diagnostic"
    assert diagnostic["n"] == 2
    assert diagnostic["n_covered"] == 1
    assert diagnostic["n_hits"] == 1
    assert diagnostic["overall_accuracy"] == 0.5
    assert diagnostic["coverage"] == 0.5
    assert diagnostic["accuracy_given_coverage"] == 1.0
    assert diagnostic["overall_denominator"] == 2
    assert diagnostic["conditional_denominator"] == 1


def test_confirmation_locks_reject_placeholders() -> None:
    protocol = EvaluationProtocolRecord(
        metrics=("coco_karpathy_test_i2t_r1",),
        splits=("coco_karpathy_test",),
        selection_rule="development_loss",
    ).freeze()
    assert protocol.status == "frozen"
    try:
        ConfirmationExperimentConfig(
            methods=("candidate", "matched_baseline", "explanatory_ablation"),
            target_artifact_sha256="dummy",
            adapter_artifact_sha256=SHA,
            hyperparameters_sha256=OTHER,
            seeds=(0, 1, 2),
        ).freeze()
    except FreezeRefused:
        pass
    else:
        raise AssertionError("a dummy adapter hash is not a freeze")
    frozen = ConfirmationExperimentConfig(
        methods=("candidate", "matched_baseline", "explanatory_ablation"),
        target_artifact_sha256=SHA,
        adapter_artifact_sha256=OTHER,
        hyperparameters_sha256="c" * 64,
        seeds=(0, 1, 2),
    ).freeze()
    assert frozen.status == "frozen"
    hashes = {f"run-{index}": f"{index:064x}" for index in range(9)}
    assert EvaluationCheckpointManifest(hashes).freeze().status == "frozen"
    try:
        EvaluationCheckpointManifest({f"run-{index}": SHA for index in range(8)}).freeze()
    except FreezeRefused:
        pass
    else:
        raise AssertionError("the checkpoint manifest has one hash per confirmation run")
    visibility = rehearsal_visibility()
    assert visibility["synthetic_outputs"] == "visible"
    assert visibility["real_confirmation_benchmarks"] == "unopened"
    assert visibility["real_status"] == "preregistered_not_unblinded"
    assert int(visibility["confirmation_runs_executed"]) == 0
    assert PREREGISTRATION["confirmation_runs_executed"] == 0


def test_adapter_dry_run_synthetic_and_real_paths(tmp_path) -> None:
    calls = []

    def trainer_fn(arm, schedule, caption_batch, qa_batch):
        calls.append((arm["target_id"], arm["objective"], list(schedule), caption_batch["task_ids"]))
        return {"development_score": 0.2, "checkpoint_sha256": SHA}

    dry = run_target_screen_adapter(_assets(dry_run=True, run_dir=tmp_path / "dry"), trainer_fn)
    assert dry["status"] == "dry_run"
    assert dry["runs_executed"] == 0
    assert dry["trainer_invocations"] == 0
    assert calls == []
    assert not (tmp_path / "dry" / "record.json").exists()

    blocked = run_target_screen_adapter(
        _assets(weights_available={target_id: target_id != "clip_vitb16_text" for target_id in TARGET_SPECS}),
        trainer_fn,
    )
    assert blocked["status"] == "blocked"
    assert blocked["runs_executed"] == 0
    assert "weights_unavailable" in blocked["reasons"]

    unpinned = run_target_screen_adapter(_assets(query_checkpoint_sha256="clip_vitb16_text"), trainer_fn)
    assert "query_encoder_unpinned" in unpinned["reasons"]

    rehearsal = run_target_screen_adapter(_assets(run_dir=tmp_path / "rehearsal"), trainer_fn)
    assert rehearsal["status"] == "rehearsal"
    assert rehearsal["runs_executed"] == 0
    assert rehearsal["trainer_invocations"] == 6
    assert rehearsal["schedule"] == ["caption", "qa", "caption", "qa"]
    assert rehearsal["wrote_final_results"] is False
    assert rehearsal["development_score"] == 0.2
    assert (tmp_path / "rehearsal" / "record.json").is_file()
    assert not (tmp_path / "rehearsal" / "final_results.csv").exists()

    caches = _four_caches(tmp_path / "real-caches")
    sentinel = tmp_path / "real-sentinel.json"
    write_test_sentinel(sentinel, caches, cached_vs_live_max_abs=0.0)
    exclusion = tmp_path / "exclude.txt"
    exclusion.write_text("# none\n", encoding="utf-8")
    calls.clear()
    refused = run_target_screen_adapter(
        _assets(scope="real_target_screen", exclusion_manifest=None, sentinel=sentinel),
        trainer_fn,
    )
    assert "missing_exclusion_manifest" in refused["reasons"]
    assert calls == []
    real = run_target_screen_adapter(
        _assets(
            scope="real_target_screen",
            exclusion_manifest=exclusion,
            sentinel=sentinel,
            run_dir=tmp_path / "real",
        ),
        trainer_fn,
    )
    assert real["status"] == "executed"
    assert real["runs_executed"] == 6
    assert real["trainer_invocations"] == 6
    assert real["wrote_final_results"] is False

    def fail_on_third(arm, schedule, caption_batch, qa_batch):
        del arm, schedule, caption_batch, qa_batch
        fail_on_third.n += 1
        if fail_on_third.n == 3:
            raise RuntimeError("stop")
        return {"development_score": 0.1}

    fail_on_third.n = 0
    failed = run_target_screen_adapter(
        _assets(scope="real_target_screen", exclusion_manifest=exclusion, sentinel=sentinel),
        fail_on_third,
    )
    assert failed["status"] == "failed"
    assert failed["runs_executed"] == 2
    assert failed["trainer_invocations"] == 2


def test_skipped_weight_check_is_unverified() -> None:
    assert real_integration_status(weight_test_ran=False) == "unverified"
    assert real_integration_status(weight_test_ran=True) == "checked"


@pytest.mark.skipif(
    not embeddinggemma_weights_present(),
    reason="EmbeddingGemma weights are absent; real integration unverified",
)
def test_gemma_mrl_matches_the_released_native_embedding() -> None:
    """Truncate the official 768-d embedding. Do not truncate transformer states."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("google/embeddinggemma-300m")
    native = model.encode(
        ["title: none | text: a small dog"],
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    assert tuple(native.shape) == (1, 768)
    ours = mrl_truncate(native.float(), 512, 768)
    manual = torch.nn.functional.normalize(native.float()[:, :512], dim=-1, eps=1e-6)
    assert torch.allclose(ours, manual, atol=1e-5)
    assert ours.shape[-1] == 512


@pytest.mark.skipif(
    not openai_clip_weights_present(),
    reason="OpenAI CLIP weights are absent; real integration unverified",
)
def test_openai_clip_weight_integration(tmp_path) -> None:
    import open_clip

    model = open_clip.create_model("ViT-B-16", pretrained="openai")
    model.eval()
    encoder = OpenCLIPVisionEncoder("ViT-B-16", pretrained=None, freeze=True, shared_model=model)
    images = torch.randn(2, 3, 224, 224)
    before = [parameter.detach().clone() for parameter in encoder.parameters()]
    tokens = extract_spatial_tokens(encoder, images)
    from src.conditional.config import config_from_mapping
    from src.conditional.model import build_predictor
    from src.conditional.trainer import ConditionalTrainer
    from src.protocol.artifacts import load_checkpoint, save_checkpoint, capture_rng_state

    config = config_from_mapping(
        {
            "vision_dim": 768,
            "query_dim": 8,
            "target_dim": 8,
            "predictor_layers": 1,
            "predictor_width": 32,
            "predictor_heads": 4,
            "objective": "cosine",
            "seed": 0,
        }
    )
    predictor = build_predictor(config)
    trainer = ConditionalTrainer(predictor, config, total_steps=1)
    batch = {
        "visual_tokens": tokens,
        "query_embeddings": torch.randn(2, 8),
        "candidate_targets": torch.nn.functional.normalize(torch.randn(2, 8), dim=-1),
        "positive_mask": torch.eye(2, dtype=torch.bool),
        "task_ids": ["caption", "caption"],
        "image_ids": ["coco:1", "gqa:7"],
        "caption_ids": ["c0", "c1"],
    }
    loss = trainer.train_step(batch)
    assert loss == loss
    after = [parameter.detach().clone() for parameter in encoder.parameters()]
    assert all(torch.equal(left, right) for left, right in zip(before, after))
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(
        path,
        student_state=trainer.student_state(),
        evaluated_state=trainer.evaluated_state(),
        evaluated_state_name="raw",
        optimizer_state=trainer.optimizer.state_dict(),
        scheduler_state=trainer.scheduler.state_dict(),
        scaler_state=None,
        rng_state=capture_rng_state(),
        step=trainer.step,
        config=config.to_dict(),
    )
    restored = load_checkpoint(path)
    assert restored["step"] == 1
    assert real_integration_status(weight_test_ran=True) == "checked"
    assert isinstance(digest, str) and len(digest) == 64
