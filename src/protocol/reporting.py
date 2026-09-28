"""Render the handoff documents from computed records.

The writers do not invent benchmark scores. Historical numbers are copied out
of the pinned JSON files and labeled with the protocol those files actually
used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.conditional.cache import (
    caption_target_bytes,
    gibibytes,
    vision_cache_bytes,
)
from src.conditional.mechanism import MECHANISM_CONDITIONS
from src.conditional.screen import planned_screen_arms
from src.protocol.historical import UNRESOLVED_VITL_FILES, unresolved_checkpoint_paths
from src.protocol.preregistration import PREREGISTRATION


FINAL_FIELDS = [
    "row_id",
    "scope",
    "protocol",
    "split",
    "seed",
    "metric",
    "value",
    "uncertainty",
    "cost",
    "status",
    "source",
]


def _historical_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    mapping = {
        "experiments/exp_baseline_vitb16_zeroshot.json": "historical_vitb16_zeroshot",
        "experiments/exp_vljepa_vitb16_robust.json": "historical_vitb16_robust",
    }
    for relative, row_id in mapping.items():
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        metrics = payload["results"]["5k"]
        rows.append(
            {
                "row_id": row_id,
                "scope": "archived_regression_only",
                "protocol": "historical_protocol_b_coco_val2017",
                "split": "coco_val2017_5k_not_karpathy_test",
                "seed": "not_recorded",
                "metric": "rsum",
                "value": repr(metrics["rsum"]),
                "uncertainty": "not_recorded",
                "cost": "not_recorded",
                "status": "archived_not_remeasured",
                "source": relative,
            }
        )
    return rows


def write_final_results(root: Path, reference: dict[str, Any] | None) -> None:
    rows = _historical_rows(root)
    if reference is not None:
        rows.append(
            {
                "row_id": reference["run_id"],
                "scope": reference["scope"],
                "protocol": "synthetic_token_reference",
                "split": "synthetic_reference",
                "seed": str(reference["seed"]),
                "metric": "final_step_loss",
                "value": repr(reference["losses"][-1]),
                "uncertainty": "single_run",
                "cost": f"{reference['seconds']:.6f}s_cpu",
                "status": "not_a_benchmark",
                "source": f"runs/{reference['run_id']}/record.json",
            }
        )
    rows.append(
        {
            "row_id": "target_screen",
            "scope": "scientific",
            "protocol": "not_run",
            "split": "not_run",
            "seed": "not_run",
            "metric": "not_applicable",
            "value": "",
            "uncertainty": "",
            "cost": "0 runs",
            "status": "not_launched",
            "source": "src/conditional/screen.py",
        }
    )
    rows.append(
        {
            "row_id": "mechanism_screen",
            "scope": "scientific",
            "protocol": "not_run",
            "split": "not_run",
            "seed": "not_run",
            "metric": "not_applicable",
            "value": "",
            "uncertainty": "",
            "cost": "0 runs",
            "status": "not_launched",
            "source": "src/conditional/mechanism.py",
        }
    )
    rows.append(
        {
            "row_id": "confirmation",
            "scope": "scientific",
            "protocol": "not_run",
            "split": "not_run",
            "seed": "0,1,2_preregistered",
            "metric": "not_applicable",
            "value": "",
            "uncertainty": "",
            "cost": "0 runs",
            "status": "preregistered_not_unblinded",
            "source": "src/protocol/preregistration.py",
        }
    )
    path = root / "final_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FINAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_integrity_report(
    *,
    summary: dict[str, Any],
    reference: dict[str, Any],
    historical_ok: bool,
    vitl_backends: dict[str, str],
) -> str:
    counts = summary["counts"]
    overlap = summary["overlap"]
    versions = reference.get("versions") or {}
    vision_gib = gibibytes(vision_cache_bytes())
    caption_gib = gibibytes(caption_target_bytes())
    same_backend = len(set(vitl_backends.values())) == 1
    lines = [
        "# Integrity report",
        "",
        "## Historical files",
        "",
        f"Pinned hashes match the files on disk: {historical_ok}.",
        "Regression pair: `exp_baseline_vitb16_zeroshot.json`, `exp_vljepa_vitb16_robust.json`.",
        "Protocol B on COCO val2017. The robust checkpoint was adapted on train2017.",
        "",
        f"Karpathy test images inside COCO train2017: {overlap['karpathy_test_in_train2017']}.",
        "A checkpoint already adapted on train2017 is not a clean Karpathy-test model.",
        "New runs start from the original pretrained backbone.",
        "",
        "ViT-L rows stay unresolved. Both adapted JSON files name the same backend path"
        if same_backend
        else "ViT-L backend paths differ and stay unresolved",
        "and store different scores.",
        "",
    ]
    for path, backend in vitl_backends.items():
        lines.append(f"- `{path}` backend `{backend}` sha256 `{UNRESOLVED_VITL_FILES[path]}`")
    lines.extend(
        [
            "",
            "## Splits",
            "",
            "Canonical COCO id is `coco:{cocoid}`. Every Karpathy filename's numeric id",
            "matched `cocoid`. Canonical Flickr id is `flickr30k:{filename}`.",
            "",
            "restval is included in the clean training set.",
            f"Clean train size: {counts['clean_train']} "
            f"(Karpathy train {counts['karpathy_coco_train']} + restval {counts['karpathy_coco_restval']}).",
            f"Development size: {counts['clean_dev']} (Karpathy val).",
            f"Karpathy test held out: {counts['karpathy_coco_test']}.",
            "Karpathy COCO sentence counts are "
            f"{summary['coco_sentence_histogram']}. Flickr30k counts are "
            f"{summary['flickr_sentence_histogram']}.",
            "Five-caption evaluation uses the first five sentences in file order.",
            "Extra sentences stay in the id manifest.",
            f"Removed from train because they were final-benchmark ids: {counts['removed_from_train']}.",
            "",
            "Label: Karpathy-derived, benchmark-disjoint training set.",
            "train2017 is the Karpathy ids that are not in val2017, and that remainder",
            "still contains Karpathy test images.",
            f"train2017 equals every Karpathy id: {overlap['train2017_equals_all_karpathy_ids']}.",
            "train2017 equals Karpathy ids minus val2017: "
            f"{overlap['train2017_equals_karpathy_minus_val2017']}.",
            f"train2017/val2017 overlap: {overlap['train2017_val2017_overlap']}.",
            f"Karpathy ids outside COCO 2017: {overlap['karpathy_ids_outside_coco2017']}.",
            "",
            f"Flickr30k Karpathy val has {counts['flickr30k_val']} images, and the test has "
            f"{counts['flickr30k_test']}. Flickr ids are not COCO ids.",
            "",
            "SugarCrepe++, Winoground, GQA, Visual Genome, and Oxford-IIIT Pets were not",
            "loaded. The clean set is not claimed to be disjoint from them.",
            "Pretraining exposure is unknown. COCO images are not on disk.",
            "",
            "## Evaluator",
            "",
            "Selection uses `src.eval_retrieval.compute_retrieval_metrics` (strict `>` ranks).",
            "Fully tied embeddings set `selection_accepted` false.",
            "Pessimistic ranks and near-tie rates are diagnostics.",
            "",
            "## Reference run",
            "",
            f"Run `{reference['run_id']}` scope `{reference['scope']}` seed {reference['seed']}.",
            f"Git `{reference['git_commit']}`, dirty={reference['dirty']} at save time.",
            f"Trainable parameters {reference['trainable_params']}.",
            f"Checkpoint sha256 `{reference['checkpoint_sha256']}`.",
            "Evaluated state `raw` equals the student. No EMA, no WiSE-FT.",
            "Tokens are synthetic. Preprocessing time and retrieval evaluation time are zero.",
        ]
    )
    probe = reference.get("throughput_probe")
    if isinstance(probe, dict) and probe.get("status") == "measured":
        lines.append(
            f"GPU probe: {probe.get('device')}, "
            f"{probe.get('seconds_per_step')} s/step, batch {probe.get('batch')}, "
            f"{probe.get('tokens')} tokens, peak {probe.get('peak_vram_bytes')} bytes. "
            "CLIP encoding excluded."
        )
    hashes = reference.get("manifest_hashes_not_used_for_training")
    if isinstance(hashes, dict):
        lines.append("Manifest hashes were recorded after the run and were not training inputs.")
    lines.extend(
        [
            "",
            "## Environment",
            "",
            "Versions imported by the reference run. `environment.lock` lists the same packages.",
        ]
    )
    for name, version in versions.items():
        lines.append(f"- {name}={version}")
    lines.extend(
        [
            "",
            "## Cache",
            "",
            f"113287 x 197 x 768 x 2 bytes = {vision_gib:.4f} GiB.",
            f"Five 512-d fp16 caption targets per image = {caption_gib:.4f} GiB per target encoder.",
            "COCO token cache was not built. `cache_manifest.json` is a synthetic parity check.",
            "",
            "## Not launched",
            "",
            "Target screen arms: "
            + ", ".join(f"{arm.target_id}/{arm.objective}" for arm in planned_screen_arms())
            + ". Status not_launched. Shared step budget not frozen.",
            "Mechanism conditions "
            + ", ".join(item.condition_id for item in MECHANISM_CONDITIONS)
            + " were not trained.",
            f"Confirmation runs executed: {PREREGISTRATION['confirmation_runs_executed']}.",
            f"Final analysis status: {PREREGISTRATION['status']}.",
            "",
        ]
    )
    return "\n".join(lines)


def render_target_diagnostics() -> str:
    vision_gib = gibibytes(vision_cache_bytes())
    caption_gib = gibibytes(caption_target_bytes())
    return "\n".join(
        [
            "# Target diagnostics",
            "",
            "No target-screen arm has been trained. Six arms: CLIP ViT-B/16 text,",
            "EmbeddingGemma-300M, and Qwen3-Embedding-0.6B, each with InfoNCE and cosine.",
            "Query encoder is CLIP ViT-B/16 text for every arm. Step budget is not frozen.",
            "`run_target_screen` requires a frozen budget, clean manifests, and a",
            "visual-cache sentinel. The sentinel was not created. EmbeddingGemma and",
            "Qwen weights were not downloaded.",
            "",
            "CLIP text: native 512, EOT, context 77.",
            "EmbeddingGemma: native 768, mean pool, MRL leading 512 then renorm.",
            "Prompts: `task: search result | query: `, `title: none | text: `,",
            "`task: question answering | query: `.",
            "Qwen3-Embedding-0.6B: native 1024, last token, MRL leading 512 then renorm.",
            "Queries use `Instruct: {task}\\nQuery: {text}`. Documents stay raw.",
            "",
            "Descriptors, not objectives: effective rank, mean unit-vector norm,",
            "paraphrase margin, linear probe, small nonlinear probe.",
            "",
            "If text distinction is poor and the visual probe is good, change the target.",
            "If both are good and conditional prediction is poor, change the supervision.",
            "If visual probes are weak, change layer, view, resolution, or backbone",
            "before adding a loss.",
            "",
            f"Clean-set visual cache estimate: {vision_gib:.4f} GiB.",
            f"Five 512-d fp16 captions per image: {caption_gib:.4f} GiB per encoder.",
            "`FEATURE_DROPOUT_MASKS_BEFORE_ENCODER` is false.",
            "Synthetic plumbing uses scope `synthetic_harness_not_a_result`.",
            "",
        ]
    )


def render_mechanism_screen() -> str:
    lines = [
        "# Mechanism screen",
        "",
        "M0–M5 are defined and were not trained. Shared exposure ledger:",
        f"`{MECHANISM_CONDITIONS[0].exposure_ledger_key}`.",
        "",
        "| ID | Target | Supervision | Purpose |",
        "|---|---|---|---|",
    ]
    for item in MECHANISM_CONDITIONS:
        lines.append(
            f"| {item.condition_id} | {item.target} | {item.supervision} | {item.purpose} |"
        )
    lines.extend(
        [
            "",
            "Residual adapter: 512 → 128 → 512, last linear layer zero, identity at init.",
            "Freeze it before comparing predictors. Loss: paraphrase-versus-negative",
            "hinge plus an anchor penalty. No adapter was fit.",
            "",
            "M3 has to beat the strongest matched baseline, not only M0.",
            "M5 is a label permutation. Status: not_trained.",
            "",
        ]
    )
    return "\n".join(lines)


def render_blinded_analysis() -> str:
    data = PREREGISTRATION
    return "\n".join(
        [
            "# Blinded final analysis",
            "",
            f"Status: `{data['status']}`.",
            f"Confirmation runs executed: {data['confirmation_runs_executed']}.",
            f"Seeds if confirmation is launched: {data['seeds']}.",
            "",
            "Thresholds are stopping rules. Confirmation scores do not exist yet.",
            "",
            f"- At least {data['primary_compositional_min_gain_points']} points on one declared compositional endpoint.",
            "- The paired confidence interval on that difference excludes zero."
            if data["paired_ci_must_exclude_zero"]
            else "- Paired interval rule not set.",
            f"- No more than {data['max_retrieval_r1_degradation_points']} point of degradation",
            "  on each protected retrieval R@1 versus the strongest matched baseline:",
        ]
        + [f"  - `{name}`" for name in data["protected_r1_endpoints"]]
        + [
            "",
            "Metrics stay separate. Winoground alone is not a significance claim.",
            "Tuning after these endpoints are opened is development evidence.",
            "",
        ]
    )


def write_cache_manifest(root: Path, parity_max_abs: float, fingerprint_digest: str) -> None:
    payload = {
        "dataset": "synthetic_reference",
        "fingerprint": fingerprint_digest,
        "parity_max_abs": parity_max_abs,
        "precision": "float16",
        "feature_dropout_masks_before_encoder": False,
        "coco_cache_built": False,
        "note": "Synthetic float16 parity. The 113287-image cache was not built.",
    }
    (root / "cache_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
