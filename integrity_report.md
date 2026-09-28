# Integrity report

## Historical files

Pinned hashes match the files on disk: True.
Regression pair: `exp_baseline_vitb16_zeroshot.json`, `exp_vljepa_vitb16_robust.json`.
Protocol B on COCO val2017. The robust checkpoint was adapted on train2017.

Karpathy test images inside COCO train2017: 4407.
A checkpoint already adapted on train2017 is not a clean Karpathy-test model.
New runs start from the original pretrained backbone.

ViT-L rows stay unresolved. Both adapted JSON files name the same backend path
and store different scores.

- `experiments/exp_vljepa_vitl14_robust.json` backend `VL-JEPA checkpoint: experiments/exp_jepa_1024d_20ep/checkpoint_best.pt` sha256 `f5d63100f86156614315d40f81d7ff4566313b50211ca6b4f125162fc8f5bdf9`
- `experiments/exp_vljepa_vitl14_siglip.json` backend `VL-JEPA checkpoint: experiments/exp_jepa_1024d_20ep/checkpoint_best.pt` sha256 `dadcdd7d54e7a426906fccb8e336d74cc752539e5c88a339c7775384d719286b`

## Splits

Canonical COCO id is `coco:{cocoid}`. Every Karpathy filename's numeric id
matched `cocoid`. Canonical Flickr id is `flickr30k:{filename}`.

restval is included in the clean training set.
Clean train size: 113287 (Karpathy train 82783 + restval 30504).
Development size: 5000 (Karpathy val).
Karpathy test held out: 5000.
Karpathy COCO sentence counts are {5: 122959, 6: 324, 7: 4}. Flickr30k counts are {5: 31014}.
Five-caption evaluation uses the first five sentences in file order.
Extra sentences stay in the id manifest.
Removed from train because they were final-benchmark ids: 0.

Label: Karpathy-derived, benchmark-disjoint training set.
train2017 is the Karpathy ids that are not in val2017, and that remainder
still contains Karpathy test images.
train2017 equals every Karpathy id: False.
train2017 equals Karpathy ids minus val2017: True.
train2017/val2017 overlap: 0.
Karpathy ids outside COCO 2017: 0.

Flickr30k Karpathy val has 1014 images, and the test has 1000. Flickr ids are not COCO ids.

SugarCrepe++, Winoground, GQA, Visual Genome, and Oxford-IIIT Pets were not
loaded. The clean set is not claimed to be disjoint from them.
Pretraining exposure is unknown. COCO images are not on disk.

## Evaluator

Selection uses `src.eval_retrieval.compute_retrieval_metrics` (strict `>` ranks).
Fully tied embeddings set `selection_accepted` false.
Pessimistic ranks and near-tie rates are diagnostics.

## Reference run

Run `phase0-reference-seed0` scope `phase0_reference_not_a_benchmark` seed 0.
Git `dfeaaf6298422d926cd67b9ea6586a9bf4d4d996`, dirty=True at save time.
Trainable parameters 10159616.
Checkpoint sha256 `2b60e3f8236fc505ece9dc167bb62e88a80f7f58e7482f299fc290170e1ed669`.
Evaluated state `raw` equals the student. No EMA, no WiSE-FT.
Tokens are synthetic. Preprocessing time and retrieval evaluation time are zero.
GPU probe: NVIDIA GeForce RTX 5080, 0.009271114016883075 s/step, batch 8, 197 tokens, peak 286137856 bytes. CLIP encoding excluded.
Manifest hashes were recorded after the run and were not training inputs.

## Environment

Versions imported by the reference run. `environment.lock` lists the same packages.
- torch=2.14.0
- torchvision=0.29.0
- numpy=2.4.4
- open_clip_torch=3.3.0
- transformers=4.57.6
- huggingface_hub=0.36.2
- timm=1.0.30
- pycocotools=2.0.11
- pillow=12.0.0
- pyyaml=6.0.3

## Cache

113287 x 197 x 768 x 2 bytes = 31.9255 GiB.
Five 512-d fp16 caption targets per image = 0.5402 GiB per target encoder.
COCO token cache was not built. `cache_manifest.json` is a synthetic parity check.

## Not launched

Target screen arms: clip_vitb16_text/infonce, clip_vitb16_text/cosine, embeddinggemma_300m_mrl512/infonce, embeddinggemma_300m_mrl512/cosine, qwen3_embedding_0.6b_mrl512/infonce, qwen3_embedding_0.6b_mrl512/cosine. Status not_launched. Shared step budget not frozen.
Mechanism conditions M0, M1, M2, M3, M4, M5 were not trained.
Confirmation runs executed: 0.
Final analysis status: preregistered_not_unblinded.
