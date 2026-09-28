"""Synthetic reference run for the conditional predictor.

The GPU probe times random tokens. It does not encode CLIP.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from src.conditional.config import PredictorConfig
from src.conditional.model import build_predictor
from src.conditional.trainer import ConditionalTrainer, seed_everything
from src.protocol.artifacts import (
    append_registry,
    capture_rng_state,
    config_sha256,
    create_run_dir,
    git_state,
    save_checkpoint,
)
from src.protocol.environment import distribution_versions


def synthetic_batch(config: PredictorConfig, batch: int, tokens: int, seed: int, device: torch.device) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    visual = torch.randn(batch, tokens, config.vision_dim, generator=generator, device="cpu")
    query = torch.randn(batch, config.query_dim, generator=generator, device="cpu")
    targets = torch.randn(batch, config.target_dim, generator=generator, device="cpu")
    return {
        "visual_tokens": visual.to(device),
        "query_embeddings": query.to(device),
        "candidate_targets": targets.to(device),
        "positive_mask": torch.eye(batch, dtype=torch.bool, device=device),
        "task_ids": ["caption"] * batch,
        "image_ids": [f"synthetic:{index}" for index in range(batch)],
        "caption_ids": [f"synthetic-cap:{index}" for index in range(batch)],
    }


def run_phase0_reference(
    repo: Path,
    *,
    run_id: str = "phase0-reference-seed0",
    steps: int = 5,
    seed: int = 0,
    batch: int = 4,
    tokens: int = 4,
) -> dict[str, Any]:
    repo = Path(repo)
    seed_everything(seed)
    config = PredictorConfig(seed=seed)
    model = build_predictor(config)
    trainer = ConditionalTrainer(model, config, total_steps=steps)
    example = synthetic_batch(config, batch, tokens, seed=123, device=torch.device("cpu"))
    started = time.perf_counter()
    losses = [trainer.train_step(example) for _ in range(steps)]
    elapsed = time.perf_counter() - started
    run_dir = create_run_dir(repo / "runs", run_id)
    student = trainer.student_state()
    evaluated = trainer.evaluated_state()
    checkpoint = run_dir / "checkpoint.pt"
    digest = save_checkpoint(
        checkpoint,
        student_state=student,
        evaluated_state=evaluated,
        evaluated_state_name="raw",
        optimizer_state=trainer.optimizer.state_dict(),
        scheduler_state=trainer.scheduler.state_dict(),
        scaler_state=None,
        rng_state=capture_rng_state(),
        step=trainer.step,
        config=config.to_dict(),
        extra=trainer.checkpoint_extra(),
    )
    git = git_state(repo)
    versions = distribution_versions()
    record = {
        "run_id": run_id,
        "scope": "phase0_reference_not_a_benchmark",
        "seed": seed,
        "steps": steps,
        "losses": losses,
        "seconds": elapsed,
        "trainable_params": model.trainable_parameter_count(),
        "total_params": model.total_parameter_count(),
        "checkpoint_sha256": digest,
        "evaluated_state": "raw",
        "evaluated_equals_student": True,
        "git_commit": git["commit"],
        "dirty": git["dirty"],
        "dirty_sha256": git["dirty_sha256"],
        "config_sha256": config_sha256(config.to_dict()),
        "config": config.to_dict(),
        "image_exposures": sum(trainer.ledger.images.values()),
        "caption_exposures": sum(trainer.ledger.captions.values()),
        "microbatch": batch,
        "optimizer_batch": batch,
        "candidate_pool": batch,
        "positive_rule": "identity_synthetic_captions",
        "precision": "fp32",
        "context_length": config.context_length,
        "target_dim": config.target_dim,
        "pooling": config.pooling,
        "dataset": "synthetic_reference",
        "versions": versions,
        "note": (
            "Synthetic tokens. Not a COCO, Flickr, or compositional benchmark. "
            "CLIP was not encoded in this loop."
        ),
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    append_registry(
        repo / "experiment_registry.csv",
        {
            "run_id": run_id,
            "git_commit": git["commit"],
            "dirty": git["dirty"],
            "dirty_sha256": git["dirty_sha256"],
            "seed": seed,
            "config_sha256": record["config_sha256"],
            "scope": record["scope"],
            "evaluated_state": "raw",
            "checkpoint_sha256": digest,
            "trainable_params": record["trainable_params"],
            "total_params": record["total_params"],
            "image_exposures": record["image_exposures"],
            "caption_exposures": record["caption_exposures"],
            "microbatch": batch,
            "optimizer_batch": batch,
            "candidate_pool": batch,
            "positive_rule": record["positive_rule"],
            "precision": "fp32",
            "context_length": config.context_length,
            "target_dim": config.target_dim,
            "pooling": config.pooling,
        },
    )
    return record


def gpu_throughput_probe(repo: Path, run_id: str = "phase0-reference-seed0") -> dict[str, Any]:
    """Time one optimizer step on random tokens. Does not replace the CPU checkpoint."""
    run_dir = Path(repo) / "runs" / run_id
    out_path = run_dir / "throughput.json"
    if not torch.cuda.is_available():
        payload = {"status": "cuda_unavailable", "scope": "throughput_probe_not_a_result"}
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
    config = PredictorConfig()
    model = build_predictor(config).to("cuda")
    trainer = ConditionalTrainer(model, config, total_steps=2)
    batch = synthetic_batch(config, batch=8, tokens=197, seed=7, device=torch.device("cuda"))
    trainer.train_step(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train_step(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = {
        "status": "measured",
        "scope": "throughput_probe_not_a_result",
        "device": torch.cuda.get_device_name(0),
        "batch": 8,
        "tokens": 197,
        "seconds_per_step": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "excludes": "clip_image_and_text_encoding",
        "note": "Random tokens. CLIP encoding is not included.",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
