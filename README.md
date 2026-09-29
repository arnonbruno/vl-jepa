# VL-JEPA

Two systems live in this repository.

The original stack in `src/model.py` and `src/trainer.py` is a CLIP retrieval model. It fine-tunes OpenCLIP vision and text towers with SigLIP or InfoNCE, optional EMA, and WiSE-FT. Retrieval numbers for that stack use `src.eval_retrieval.compute_retrieval_metrics`: five captions per image, strict greater-than ranks. The training loop's own recall helper is a different, one-caption protocol.

`src/conditional/` is a separate family, `conditional_latent_predictor`. A frozen vision tower emits spatial tokens. A frozen CLIP ViT-B/16 text encoder emits the query. A small cross-attention predictor maps those to a normalized target. The target encoder changes across arms. The predictor does not use a masked teacher, a queue, a 20× learning-rate group, or WiSE-FT. Its temperature default is 1.

## Recorded retrieval results

These rows are copied from the historical JSON files. They are Protocol B on COCO val2017 (5,000 images), not Karpathy test. No seed was stored with them. The robust checkpoint was adapted on COCO train2017, which still contains Karpathy test images, so it is not a clean Karpathy-test model.

| Run | rsum | Source |
|---|---|---|
| CLIP ViT-B/16 zero-shot | 373.288 | `experiments/exp_baseline_vitb16_zeroshot.json` |
| VL-JEPA ViT-B/16 robust | 393.096 | `experiments/exp_vljepa_vitb16_robust.json` |

Two ViT-L JSON files name the same checkpoint path and store different scores. That conflict is unresolved. `final_results.csv` lists the same historical rows and marks the target screen, mechanism screen, and confirmation as not launched.

The conditional predictor has a synthetic reference run (`runs/phase0-reference-seed0`) and a dry-run screen path. It has no COCO-trained result.

## Conditional predictor

`prediction = normalize(predictor(frozen_visual_tokens, frozen_query))`.

The starting predictor is 4 blocks, width 384, 6 heads. Vision tokens are CLIP ViT-B/16 after `visual.ln_post`: the CLS token plus 196 patches, 768-d, before `visual.proj`. The query encoder is pinned by checkpoint hash and preprocessing, not by name alone.

Target arms, each with multi-positive InfoNCE and with cosine:

- CLIP ViT-B/16 text, native 512, EOT, context 77
- EmbeddingGemma-300M: mean pool, the released 768→3072→768 projections, then leading-512 truncation and L2 normalization. Inference is float32 or bfloat16
- Qwen3-Embedding-0.6B: last non-pad token, then leading-512 truncation and L2 normalization

Caption steps and question-answer steps use separate candidate matrices and a shared interleaved schedule. Missing target weights block execution. The screen adapter defaults to a dry run. A visual-cache sentinel authorizes training only after its fingerprint, ids, shard checksums, and a cached-versus-live check match. This tree does not contain that sentinel.

Confirmation thresholds are preregistered and the final benchmarks are unopened: gain of at least 2.0 points, a paired interval whose lower bound is above zero, and at most 1.0 point of R@1 degradation on each protected COCO and Flickr30K Karpathy-test endpoint.

## Splits

Canonical ids are `coco:{cocoid}` and `flickr30k:{filename}`. The clean training list is Karpathy train plus restval (113,287 images), disjoint from Karpathy val and test. Call it a Karpathy-derived, benchmark-disjoint training set. `manifests/` holds the id lists, the overlap report, and `licenses.md`.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Python 3.9 or newer. COCO images are not part of this repository. Tests that need the official train2017 image folder skip when it is absent. EmbeddingGemma and Qwen weight checks skip when those checkpoints are not cached; a skip means that integration was not verified.

Train the original stack with `experiments/exp_jepa_training.py` and a config under `configs/`. The conditional screen entry point is `src.conditional.execute.run_target_screen_adapter`. It does not launch the 6 / 6 / 9 / 6 research runs on import.

## Layout

```
src/model.py, src/trainer.py     original CLIP retrieval stack
src/eval_retrieval.py            Protocol B retrieval
src/conditional/                 query-conditioned predictor, caches, dry-run screen
src/protocol/                    splits, run records, preregistration
manifests/                       Karpathy-derived id lists
experiments/                     training and evaluation scripts for the original stack
configs/                         YAML for the original stack
final_results.csv                recorded rows and not-launched stages
```
