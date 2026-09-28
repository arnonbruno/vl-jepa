# What this repository actually is

This file is a reading of the code, configs, tests, and committed result files as of `dfeaaf6` (2026-08-19). It is not a paraphrase of `README.md`. The README still describes an early training stage. The code that produced the numbers in `experiments/*.json` is a later system, and a few of those numbers disagree with each other.

Package: `vl-jepa` 0.2.0 (`pyproject.toml`). Language: Python, PyTorch. The repository calls itself an implementation of VL-JEPA (arXiv:2512.10942). What is actually implemented, and what the saved scores measure, is a CLIP retrieval fine-tune on COCO captions, with a JEPA masked-patch predictor that the best recipes turn off.

There is no `LICENSE` file. The README claims CC-BY 4.0. Weights, COCO images, and training logs are not in git.

---

## The one-sentence version

A single RTX-class machine fine-tunes an OpenAI CLIP model (ViT-B/16 or ViT-L/14, loaded through `open_clip`) on COCO 2017 captions, then measures image–text retrieval on COCO val and, separately, on Flickr30K. The fine-tune keeps CLIP’s pretrained alignment (end-of-text pooling and CLIP’s own projection matrices), adds a zero-initialized residual adapter, optionally unfreezes the last few transformer blocks of both towers, and at evaluation time averages the weights with an exponential moving average and with the original CLIP initialization (WiSE-FT). Recall is the only task.

---

## What this is

It is a research training and evaluation codebase for **cross-modal retrieval**: given an image, rank captions; given a caption, rank images. Cosine similarity in a shared vector space is the ranker. There is no caption decoder, no VQA head, no detection head, and no generative loss.

It contains three encoder generations, all still runnable:

1. **From-scratch transformers** (`VisionEncoder`, `LanguageEncoder`). Random ViT-style and BERT-style stacks. This is `configs/default.yaml`. The README’s own comparison says this regime, with InfoNCE and a 65,536-key queue, sat near chance. The code is kept because the tests and the default config still use it.
2. **timm CLIP vision + DistilBERT text** (`TimmVisionEncoder`, `HFLanguageEncoder`). `configs/mvp_pretrained_siglip.yaml`. Vision starts from `vit_base_patch16_clip_224.openai`. Text is DistilBERT with mean pooling. The two towers were never trained together, so the projection heads have to invent the alignment on COCO.
3. **One OpenCLIP model for both towers** (`OpenCLIPVisionEncoder`, `OpenCLIPLanguageEncoder`). This is every `configs/openclip_*.yaml` file and every committed score. Vision and text are slices of a single `open_clip` module so they start from the same CLIP checkpoint.

On top of whichever vision tower is chosen, the model always constructs:

- an EMA copy of the vision encoder (the JEPA teacher),
- a small transformer predictor,
- a linear `vision_pred_head` that maps predictor outputs back to the vision width,
- two projection modules into a joint space,
- a learnable `logit_scale` (initialized at `log(1/0.07)`) and `logit_bias` (initialized at `-10`, used by SigLIP).

Whether those JEPA pieces receive a gradient depends on the loss weight `alpha`. The configs that the result files point at set `alpha: 0.0`.

---

## What this is not

- **It is not, in the runs that produced the saved scores, a JEPA.** Those configs set the masked-patch MSE weight to zero. The trainer then skips the masked forward, the teacher forward, and the predictor (`compute_jepa=False` in `VL_JEPA.forward`). The predictor still exists and still sits in the optimizer, but it gets no gradient. Calling the headline system “VL-JEPA” describes the class name and the unused branch, not the objective that was optimized.
- **It is not a from-scratch vision-language model.** The numbers start from OpenAI CLIP weights (`openclip_pretrained: openai`). A fresh model with `projection_type: clip` or `clip_residual` and `text_pool: eot` is constructed so that its joint embedding matches `open_clip`’s `encode_image` / `encode_text`. That equality is what `tests/test_smoke.py::test_clip_proj_plus_eot_matches_openclip_encode` locks in.
- **It is not a paper reproduction with paper tables.** No paper table, no paper hyperparameter dump, and no comparison to the arXiv model’s reported scores is stored in the tree. The README’s “~24–25% R@1” is an older metric from an older run (see the two protocols below). It is not the COCO number in `experiments/exp_vljepa_*.json`.
- **It is not trained on CC3M or CC12M.** `src/image_text_dataset.py` and `experiments/download_cc3m.py` can build a manifest of image–caption pairs. `experiments/exp_jepa_training.py` never imports them. Every training entry point loads COCO, or a cache of COCO encoder outputs.
- **It is not a general multimodal benchmark.** The only reported task is retrieval. Qualitative examples are retrieval ranks and top captions, not generated text.
- **The saved weights are not in the repository.** `*.pt` is gitignored. The JSON files name checkpoint paths. Those files are absent here, so the scores cannot be re-checked from this checkout alone.
- **Training is not seeded.** `src/eval_all_checkpoints.py` writes `"seed": 42` into its model catalog. The training script never calls `torch.manual_seed` or `random.seed`. DataLoader shuffle is unseeded. Caption choice inside an epoch is deterministic (`random.Random(epoch * 1_000_003 + index)`), and the JEPA eval mask can be seeded, but a full run is not reproducible from a declared seed.
- **`contrastive_on_global: true` does nothing.** It appears in several YAML files and is read by no Python. Contrastive features always come from the unmasked image view. That is hard-coded in `VL_JEPA.forward`, not switched by the flag.

---

## Two different Recall numbers live in this repo

Mixing them is how the README and the JSON files appear to contradict each other.

### Protocol A — what the training loop optimizes and checkpoints on

`src/trainer.py::retrieval_recall`.

- One caption per image. The COCO dataset picks it with the epoch RNG. Validation stays at epoch 0, so the val caption is fixed per image.
- The similarity matrix is square: N images by N captions.
- A hit is “the paired row/column index is inside the top-K”. Image→text does **not** get credit for the other four COCO captions.
- The returned values are fractions in `[0, 1]`. The training log prints them with a percent format, so `0.349` is displayed as `34.90%`.

Checkpoint selection uses the mean of image→text R@1 and text→image R@1 under this protocol (`experiments/exp_jepa_training.py`). Validation loss is logged and is **not** the selection criterion. The comments in `configs/openclip_vitb16_robust.yaml` say why: on the earlier SigLIP runs, validation loss kept falling after retrieval had already peaked.

The “~25% plateau”, the “~30% CLIP zero-shot”, and the “43% ceiling” in the README and in the older config comments are Protocol A numbers. A commit message on `cfbfa91` says the same thing in the project’s own words: the apparent ceiling was an artifact of this one-caption recall.

### Protocol B — what the JSON files report

`src/eval_retrieval.py::compute_retrieval_metrics`, driven by `experiments/evaluate_retrieval.py` and `experiments/evaluate_flickr30k.py`.

- Up to 5 captions per image (`captions_per_image=5`). Extra annotations beyond five are dropped. This is tested in `tests/test_eval_coco_loader.py`.
- **Image→text:** rank all captions; the image is correct at K if any of its ground-truth captions is in the top-K. The stored rank is the rank of the best correct caption.
- **Text→image:** rank all images; the caption is correct at K if its own image is in the top-K. Another image that happens to carry the same caption string still counts as wrong. An optional diagnostic (`diagnostics=True`) reports what text→image recall would be if those exact string collisions were also counted. That diagnostic is not part of `rsum` and is not in the committed JSON files.
- Rank is the number of candidates **strictly** more similar than the positive. Ties do not count as better. Median rank uses `torch.median` (the lower of the two central values when N is even) and then adds 1, so it is 1-indexed. Mean rank is the mean of 0-indexed ranks, plus 1.
- Recalls are percentages in `[0, 100]`. `rsum` is the sum of the six recalls (R@1, R@5, R@10 in both directions). The maximum is 600.
- **COCO 5K** is all 5,000 images in COCO 2017 val, in `CocoCaptions` order (image ids sorted by the COCO API, not JSON file order).
- **COCO 1K** is the mean of the complete disjoint 1,000-image blocks of that same ordering (five blocks when N=5,000). It is not a separate Karpathy fold file shipped in the repo. A trailing partial block is dropped.
- **Flickr30K** is the Karpathy test split: rows of `flickr_annotations_30k.csv` whose `split` column equals `test`, from the Hugging Face dataset `nlphuji/flickr30k`. Filename order is not used as the split. `tests/test_flickr_split.py` guards that.

Zero-shot baselines in the JSON files are raw `open_clip` `encode_image` / `encode_text` on the same Protocol B, with QuickGELU requested for OpenAI weights (`create_openclip_model_and_transforms`). OpenAI CLIP was trained with QuickGELU; a default GELU build would not be the same model.

Protocol B numbers were written in late May and on 2026-06-02. After that, `src/eval_retrieval.py`, `experiments/evaluate_retrieval.py`, `src/model.py`, and `src/trainer.py` changed by hundreds of lines (diagnostics, checkpoint contracts, stricter config checks). The core rank rule (`similarity > positive`) dates from the original Protocol B commit and is still the rule. The JSON files were not regenerated after those later edits. Treat them as historical measurements, not as a fresh dump of today’s script.

---

## How a training step actually works

Entry point: `experiments/exp_jepa_training.py`. It loads one YAML file. It does **not** merge that file over `configs/default.yaml`. Missing keys fall back to the argparse and trainer defaults, which are not always the defaults written in `default.yaml`.

Each step:

1. The COCO training set returns `(image, input_ids, attention_mask)`. Train images are random-resized-cropped, flipped, color-jittered, sometimes grayscaled, normalized, then randomly erased. Val images are resized and center-cropped. OpenCLIP backbones use CLIP mean/std `(0.48145466, 0.4578275, 0.40821073)` / `(0.26862954, 0.26130258, 0.27577711)`. The DistilBERT path uses ImageNet mean/std. Evaluation scripts always use CLIP mean/std, which matches the OpenCLIP runs and would mismatch an old timm/DistilBERT checkpoint.
2. If `use_multi_crop` is on, the trainer makes another global crop and a local crop inside the model (`make_multicrop_views`). The robust configs leave this off, so the only crop is the dataset’s.
3. If `alpha != 0`, the context encoder sees a block-masked view (default 75% of patches, I-JEPA-style random rectangles, exact masked count). The EMA teacher encodes the same view with no mask, under `eval()` and `no_grad`. The predictor reads the masked context. MSE is averaged over masked patch positions only, per sample, skipping the CLS token. If `alpha == 0`, this whole branch is replaced with zeros so a NaN in an unused tensor cannot poison the loss (`tests/test_alpha_zero_nan.py`).
4. Contrastive features are the CLS of an **unmasked** forward (the global crop when multi-crop is on) and a pooled text vector. `text_pool: mean` averages non-padding tokens. `text_pool: eot` takes the last non-padding token. For a CLIP tokenizer that is the end-of-text token, which is the only position that has attended to the whole caption under CLIP’s causal mask. Mean pooling throws that vector away. That was the bug behind the early OpenCLIP runs that could not beat their own zero-shot baseline.
5. Both vectors go through the projection module and are L2-normalized. The loss is computed in FP32.

The loss in `compute_jepa_loss` is:

```
L = alpha * MSE + beta * contrastive + gamma * variance + delta * hard_negative
```

- **Contrastive** is either InfoNCE or SigLIP (`model.contrastive_loss`).
  - InfoNCE is symmetric cross-entropy. Image→text may also see a FIFO queue of previous text embeddings (`MemoryBank`). Text→image is in-batch only. The queue stores the current student text projection, detached, not the EMA teacher. The class docstring says “momentum keys”; the enqueue call does not use a momentum encoder. SigLIP ignores the queue even if `memory_bank_size` is set (`tests/test_memory_bank_eval.py`). The ViT-L SigLIP config still sets `memory_bank_size: 16384`. That queue is allocated and filled and does not enter the loss.
  - SigLIP is a pairwise sigmoid loss on the B×B similarity matrix, with optional label smoothing. It does not depend on a softmax over the batch, which is why the early small-batch runs switched to it. Gradient accumulation does **not** add SigLIP negatives: each micro-batch has its own B×B matrix, and the gradients are averaged. Only a larger real batch adds negatives. The comment in `configs/mvp_pretrained_siglip.yaml` says this explicitly.
- **Variance** is a VICReg-style penalty pushing the per-dimension standard deviation of the pre-normalization projections toward at least 1. The robust configs set `gamma: 0` and the term is skipped entirely.
- **Hard negative** is a VSE++ max-violation hinge: for each anchor, only the worst in-batch negative contributes, with a margin (default 0.2). `delta` is `loss.hard_negative_weight`. The robust and SigLIP headline configs set it to 0. The older `openclip_vitb16.yaml` uses 0.2. `openclip_vitb16_v2.yaml` uses 0.05.

`logit_scale` is clamped to `[log(1/100), log(100)]` after every optimizer step.

### Optimizer

AdamW, betas `(0.9, 0.95)`. Parameter groups:

| Group | Learning rate | Weight decay |
|---|---|---|
| Ordinary weights | base LR, with warmup then cosine | configured decay |
| Biases, LayerNorms, `logit_scale`, `logit_bias` | base LR | 0 |
| Predictor | 20× base | configured decay |
| Projection | 10× base if `projection_type` is `mlp`; 1× if `clip` or `clip_residual` | configured decay |
| Blocks unfrozen mid-run | `encoder_unfreeze_lr`, constant (no warmup, no cosine) | same decay as group 0 |

The 10× projection LR exists so a random MLP can move. On a CLIP-initialized matrix it would erase the pretrained alignment in the first updates, so the trainer drops it to 1× for `clip` and `clip_residual`.

Unfreezing happens once, at `unfreeze_after_epoch` (1-based epoch compared inside the trainer). `unfreeze_vision_last_blocks` turns on the last N vision blocks, the final vision norm, and the mask token. `unfreeze_text_last_blocks` does the same for the text tower when `unfreeze_text_blocks > 0`. The new tensors become a new optimizer group and the scheduler is rebuilt by replaying every step already taken. Until that epoch the towers stay frozen and only the projection, the logit parameters, and (if `alpha != 0`) the predictor learn.

AMP is on for CUDA. Non-finite inputs, outputs, loss, or gradients skip the step, reset the accumulation window, and can write a JSON diagnostic if `--nan-diagnostics-dir` is set. A non-finite weight marks the run as corrupted and the training script tells you to resume or restart. `tests/test_accum_skip.py` covers a NaN in the middle of an accumulation window.

### Two different moving averages

These are easy to confuse because both are called EMA.

| Mechanism | What moves | Default in the robust configs | Used when `alpha = 0`? |
|---|---|---|---|
| JEPA teacher `momentum_update` | vision encoder only, `tau` cosine from 0.996 toward 1.0 | still updated every optimizer step | the teacher is not read by the loss |
| `ModelEMA` | every floating tensor in the whole module, decay 0.999 | on | yes; this is what evaluation reads first |

`ModelEMA` is a Polyak average meant to avoid chasing a late-epoch overfit. It updates on optimizer steps, not on every micro-batch.

### WiSE-FT

`wise_ft_interpolate` builds `(1 - alpha) * zero_shot_init + alpha * fine_tuned`. `wise_ft_alpha: 0.5` in the robust configs means the evaluated network is halfway between the CLIP initialization (snapshotted at trainer construction, on CPU) and the current weights. The current weights are the `ModelEMA` shadow when model EMA is on, otherwise the live student.

Evaluation swaps those tensors in, scores retrieval, and swaps the live student back (`eval_weights`). The best checkpoint stores both:

- `model_state_dict`: the live student
- `model_eval_state`: the tensors that were actually scored (EMA, then a single WiSE-FT mix)

`experiments/evaluate_retrieval.py` loads `model_eval_state` when it is present, and refuses a checkpoint that has no `config`. `tests/test_best_ckpt_keys.py` checks that saving from inside `eval_weights` does not write the interpolated copy as the student, and that WiSE-FT is not applied twice.

Interpolation runs on CPU, one tensor at a time, because doing it on GPU for ViT-L/14 held three full copies in VRAM (`c481d59`).

### Phased loss

`loss_weights_for_epoch` exists only for `--phase-training`. Epochs 1–3 use `(alpha, beta, gamma) = (0, 1, 0.01)`. From epoch 4 onward they use `(0.1, 0.9, 0.01)`. This **replaces** the YAML loss weights for those epochs. Only `configs/mvp_pretrained_siglip.yaml` turns it on. The OpenCLIP configs leave it off and use the YAML weights for the whole run.

### Where runs are written

The directory name is `exp_jepa_{hidden_dim}d_{epochs}ep` under `output.output_dir`. It does not include the config name, the loss, or the backbone. `configs/openclip_vitl14_robust.yaml` and `configs/openclip_vitl14_siglip.yaml` both say hidden dim 1024 and 20 epochs, so both write `experiments/exp_jepa_1024d_20ep/`. A later run replaces `checkpoint_best.pt`. That collision is visible in the result files and is documented with the numbers below.

`checkpoint_best.pt` updates whenever Protocol A mean R@1 improves. Periodic checkpoints are `checkpoint_epoch{N}.pt` every `checkpoint_interval` epochs. `metrics.json` in that directory is the per-epoch training log. Those files are gitignored. `experiments/training_plot.png` is a plot of some such log; the log itself is not in the tree. `experiments/plot_training.py` can redraw one from a `metrics.json`, and its “best epoch” helper still prefers lowest validation loss, which is the criterion the trainer stopped using.

### Cached-encoder path

`experiments/precompute_embeddings.py` dumps frozen encoder outputs to `train.pt` / `val.pt`. `src/cached_dataset.py` loads them. Training with `--cached-data` runs `forward_from_cache`: masking is applied to stored tokens with the learnable mask token, and the encoders are not executed. Multi-crop inside the trainer is ignored on this path. The JEPA teacher update is also skipped. This path matches a frozen-encoder run. It does not match the robust recipe, which unfreezes blocks.

---

## Projection types

| `projection_type` | What is built | At step 0 |
|---|---|---|
| `mlp` (default) | LayerNorm → Linear → GELU → Linear, then L2-normalize. Final linear is Xavier with gain 0.1. | Random. CLIP’s alignment is discarded. |
| `clip` | Bias-free `nn.Linear`. Weight is the transpose of CLIP’s `visual.proj` / `text_projection`, because CLIP stores `(in, out)` and `nn.Linear` stores `(out, in)`. `projection_dim` is overwritten to CLIP’s embed dim (512 for ViT-B/16, 768 for ViT-L/14). | Matches CLIP `encode_*` together with EOT pooling. |
| `clip_residual` | The same linear, plus a LayerNorm → Linear → GELU → Linear residual on the projected vector. The residual’s last layer is zero-initialized, so `raw(x) = linear(x)` at init. | Same match to CLIP, with room to learn a nonlinear correction later. |

`text_pool: eot` is required for the match. `text_pool: mean` on a causal CLIP tower does not reproduce `encode_text`.

The OpenCLIP vision wrapper reimplements the tower (`conv1`, add CLS and interpolated positions, `ln_pre`, transformer, `ln_post`) so it can insert a mask token after the patch projection. It does not call `visual.proj` itself; the projection module does. The text wrapper returns every token, and pooling happens in `VL_JEPA._pool_language`.

`hidden_dim` in the ViT-L YAML (1024) is a comment for humans. The module overwrites `self.hidden_dim` from the loaded tower. The predictor’s head count is chosen so the width divides evenly (`_pick_num_heads`), which is why ViT-L’s 1024-d blocks work; a hard-coded 12 heads would not.

---

## Data

**COCO 2017 captions.** Official image counts expected by the code: 118,287 train, 5,000 val (`expected_split_length`). Default root is `~/.cache/torch/hub/checkpoints`, which is an unusual place for a dataset (it is torchvision’s hub cache). Layout: `train2017/`, `val2017/`, `annotations/captions_train2017.json`, `annotations/captions_val2017.json`. `ensure_coco_2017` can download the official zips. `pycocotools` is required.

Each image has about five captions. Training uses one of them per epoch. Tokens are cached on disk under `{coco_root}/.vl_jepa_token_cache/` so workers do not re-tokenize, and the cache is excluded from the dataset’s pickle state so DataLoader workers do not each deserialize a giant tensor (`__getstate__`). DistilBERT caches and CLIP caches use different filenames because the vocabularies differ (WordPiece 30,522 versus CLIP BPE 49,408 with start- and end-of-text tokens). Feeding DistilBERT ids into the CLIP text tower is a silent failure mode the tokenizer class exists to prevent.

CLIP captions are truncated to `max_caption_length` (64 in every committed config). CLIP’s native context is 77. The zero-shot evaluator uses the model’s context length (77). The checkpoint evaluator uses the length stored in the checkpoint config (64). COCO captions usually fit in 64. The two setups are not the same tokenizer call.

**Flickr30K** is evaluation only. Images come from `flickr30k-images.zip` inside the `nlphuji/flickr30k` dataset repo, via `huggingface_hub`. That package is not in `requirements.txt` or `pyproject.toml`.

**CC3M / CC12M** is a download helper plus a manifest dataset. `download_cc3m.py` fetches the caption/URL TSV and prints an `img2dataset` command. It does not download the images itself. Nothing in the training script points at the manifest. There is no committed CC3M result.

---

## Configurations, in the order the project actually moved

Each file is a full standalone config. The comments at the top of the later files are the project’s own lab notes. They refer to Protocol A unless they explicitly say “standard protocol” or `rsum`.

| File | Role |
|---|---|
| `configs/default.yaml` | Random encoders, InfoNCE, projection dim 256, multi-crop, MSE weight 0.5, variance weight 0.1, memory bank 65,536, 15 epochs, batch 32. The early regime. |
| `configs/mvp_pretrained_siglip.yaml` | timm CLIP ViT-B/16 + DistilBERT, SigLIP, phase training, batch 128 with grad accumulation 2, unfreeze last 4 vision blocks at epoch 5. This is the generation the README still presents as current. |
| `configs/openclip_vitb16.yaml` | Both towers from OpenCLIP ViT-B/16, but projection type and text pool are left at the defaults (`mlp`, `mean`). Hard-negative weight 0.2. Still relearns alignment. |
| `configs/openclip_vitb16_aligned.yaml` | `projection_type: clip`, `text_pool: eot`, base LR 5e-5, light MSE (`alpha: 0.05`), no hard negatives, unfreeze 6 vision blocks at epoch 2, no text unfreeze. The “start from real CLIP zero-shot” fix. |
| `configs/openclip_vitb16_v2.yaml` | Adds `clip_residual`, unfreezes 6 text blocks as well, 16 epochs so the cosine actually decays, hard-negative weight 0.05, still SigLIP and `alpha: 0.05`. Written to attack a Protocol A peak around 43% followed by a decline. |
| `configs/openclip_vitb16_robust.yaml` | The reference recipe for the ViT-B JSON. InfoNCE, `alpha: 0`, weight decay 0.5, batch 256, memory bank 16,384, model EMA 0.999, WiSE-FT 0.5, 16 epochs. |
| `configs/openclip_vitl14_robust.yaml` | Same idea on ViT-L/14. Batch 64 × accumulation 4 (effective batch 256). Unfreeze 4 blocks per tower. Encoder LR 8e-6. 20 epochs. |
| `configs/openclip_vitl14_siglip.yaml` | The ViT-L recipe with SigLIP instead of InfoNCE. Same 20-epoch, 1024-d output directory as the robust ViT-L config. |

The ablation YAMLs under `experiments/ablations/*/config.yaml` are snapshots of 5-epoch ViT-B runs. Their `output.output_dir` points at `/var/mnt/DATA/OpenClaw/workspace/vl-jepa/...`, a path on another machine. Re-running `experiments/run_ablations.py` would rewrite those paths. The committed table is `experiments/ablations/ablation_results.md`. `experiments/plot_ablations.py` expects `experiments/ablations/ablation_results.json`, which is not in the tree. `experiments/ablations/ablation_figure.png` is.

---

## Results that are actually in the tree

All figures below are Protocol B percentages. `rsum` is the sum of six recalls. Deltas are against the matching OpenAI CLIP zero-shot row.

### COCO val, 5,000 images, 5 captions each

| Run | File | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | rsum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP ViT-B/16 zero-shot | `experiments/exp_baseline_vitb16_zeroshot.json` | 52.52 | 77.32 | 84.98 | 32.57 | 57.44 | 68.46 | 373.29 |
| VL-JEPA ViT-B/16, InfoNCE robust, checkpoint named `exp_jepa_768d_16ep` | `experiments/exp_vljepa_vitb16_robust.json` | 52.64 | 78.08 | 86.08 | 36.55 | 64.33 | 75.42 | 393.10 |
| CLIP ViT-L/14 zero-shot | `experiments/exp_baseline_vitl14_zeroshot.json` | 56.70 | 80.26 | 86.96 | 36.13 | 60.74 | 70.98 | 391.78 |
| VL-JEPA ViT-L/14, JSON labeled robust, same path as the SigLIP file | `experiments/exp_vljepa_vitl14_robust.json` | 57.24 | 80.24 | 87.44 | 38.46 | 64.01 | 74.26 | 401.65 |
| VL-JEPA ViT-L/14, JSON labeled SigLIP, rounded, incomplete | `experiments/exp_vljepa_vitl14_siglip.json` | 64.84 | 86.86 | 92.58 | 48.91 | 75.30 | 84.09 | 452.58 |

What those rows support:

- The ViT-B InfoNCE fine-tune barely moves image→text R@1 (+0.12) and gains about 4 points of text→image R@1. `rsum` goes from 373.3 to 393.1. Most of the `rsum` gain is text→image at R@5 and R@10, where the zero-shot model was weaker. Median image→text rank stays 1.0. Median text→image rank goes from 4 to 3. Mean text→image rank goes from 24.3 to 15.8.
- That ViT-B fine-tune’s `rsum` (393.1) is about the same as **untuned** CLIP ViT-L/14 (391.8). Scaling the frozen CLIP model by one size class and fine-tuning the smaller one land in the same place on this split.
- The ViT-L “robust” JSON is a smaller gain on top of a stronger zero-shot model: image→text R@1 +0.5, text→image R@1 +2.3, `rsum` +9.9.
- The ViT-L SigLIP JSON is the only large jump: image→text R@1 56.7 → 64.8, text→image R@1 36.1 → 48.9, `rsum` 391.8 → 452.6. It is also the least trustworthy row in the table. The other JSON files are full-precision dumps with median and mean rank. This one is rounded to two decimals, has no median or mean rank, and its COCO 1K block omits R@5 and R@10. Its `backend` string names `experiments/exp_jepa_1024d_20ep/checkpoint_best.pt`, which is the same path the robust ViT-L JSON names, with different numbers. Those cannot be the same weights. The output-directory formula explains how both runs would claim that path: the second training run overwrites the file, and each JSON records whatever occupied the path when that evaluation was launched.

`src/eval_all_checkpoints.py` later catalogs six “headline” models and **disagrees** with the robust ViT-L JSON. It points ViT-L InfoNCE at `experiments/exp_jepa_1024d_2ep/checkpoint_best.pt` and comments that the 20-epoch recipe “actually” used a 2-epoch checkpoint. It points ViT-L SigLIP at the 20-epoch path. It also records `ema_decay: 0.996` for the trained models. That number is the JEPA teacher momentum (`momentum_tau`), not the model EMA decay, which the configs set to `0.999`. The catalog’s seed field is 42, which the trainer does not set. The 2-epoch checkpoint is not in git, so this catalog cannot be checked against the 401.65 JSON. Both stories are in the repo. The JSON `backend` field and the catalog do not name the same file.

### COCO 1K (mean of five 1,000-image blocks)

| Run | i2t R@1 | t2i R@1 | rsum |
|---|---:|---:|---:|
| CLIP ViT-B/16 zero-shot | 71.92 | 51.98 | 481.51 |
| ViT-B/16 robust | 73.62 | 57.36 | 500.47 |
| CLIP ViT-L/14 zero-shot | 75.44 | 54.63 | 492.52 |
| ViT-L/14 robust JSON | 76.10 | 57.80 | 501.98 |
| ViT-L/14 SigLIP JSON | 82.58 | 68.41 | 534.81 |

1K recall is higher than 5K recall because each query is ranked against 1,000 images or 5,000 captions, not 5,000 images or 25,000 captions. The SigLIP 1K `rsum` is not comparable in detail to the others: the file does not store R@5 or R@10, yet `rsum` is present, so those missing terms were included in the sum and then omitted from the JSON.

### Flickr30K Karpathy test (1,000 images, transfer from COCO, no Flickr training)

| Run | File | i2t R@1 | t2i R@1 | rsum |
|---|---|---:|---:|---:|
| CLIP ViT-B/16 zero-shot | `experiments/exp_flickr_zeroshot_vitb16.json` | 82.70 | 62.20 | 518.16 |
| ViT-B/16 robust | `experiments/exp_flickr_vljepa_vitb16.json` | 76.90 | 61.60 | 502.76 |
| CLIP ViT-L/14 zero-shot | `experiments/exp_flickr_zeroshot_vitl14.json` | 86.10 | 64.72 | 527.16 |
| ViT-L/14 robust JSON | `experiments/exp_flickr_vljepa_vitl14.json` | 85.40 | 68.26 | 533.04 |
| ViT-L/14 SigLIP JSON | `experiments/exp_flickr_vljepa_vitl14_siglip.json` | 87.60 | 75.72 | 550.88 |

The ViT-B COCO fine-tune is worse than its own CLIP initialization on Flickr (`rsum` 518.2 → 502.8), mostly image→text R@1 (82.7 → 76.9). The ViT-L robust JSON is a small net gain and still slightly worse at image→text R@1. The SigLIP JSON is a gain on both directions. The same checkpoint-path collision applies to the two ViT-L Flickr files: both name `exp_jepa_1024d_20ep/checkpoint_best.pt`.

### Ablations (ViT-B/16, 5 epochs, Protocol B COCO 5K)

Source: `experiments/ablations/ablation_results.md`. These are short runs of the robust recipe with one change each, not the 16-epoch checkpoint. The “full” row (`rsum` 390.36) is close to the 16-epoch ViT-B JSON (393.10) and is not the same run. There is one run per cell, no error bar.

| Variant | What changed relative to the robust ViT-B recipe | i2t R@1 | t2i R@1 | rsum |
|---|---|---:|---:|---:|
| siglip | InfoNCE → SigLIP | 57.92 | 39.30 | 406.19 |
| with_jepa | `alpha` 0 → 0.1, `beta` 0.9 | 53.58 | 36.56 | 392.61 |
| full | nothing | 54.22 | 36.87 | 390.36 |
| no_ema | model EMA off, WiSE-FT still 0.5 | 52.44 | 35.73 | 388.05 |
| no_wise_ft | WiSE-FT alpha 1.0 (raw fine-tune), EMA still on | 50.94 | 34.45 | 383.20 |
| no_robust | both EMA and WiSE-FT off | 47.82 | 31.41 | 368.69 |
| frozen | `unfreeze_after_epoch: null` | 50.54 | 31.33 | 362.16 |
| mean_pool | EOT pooling → mean pooling | 41.84 | 30.88 | 341.02 |
| random_proj | `clip_residual` → random MLP | 22.62 | 15.69 | 235.69 |

Reading the table as the code defines the variants:

- Replacing the CLIP-seeded projection with a random MLP collapses retrieval (`rsum` 235). That is the early-bug result, measured cleanly: without the pretrained matrices the model does not recover CLIP-level retrieval in five epochs on COCO.
- Mean pooling instead of the end-of-text token costs about 49 `rsum` points against the full recipe.
- Turning off both robustness tricks costs about 22 `rsum` points. WiSE-FT accounts for more of that than the weight EMA (383 vs 388 when each is removed alone, 369 when both are removed).
- Leaving the encoders frozen costs about 28 `rsum` points. The last blocks are doing something.
- Turning the JEPA MSE back on (`alpha: 0.1`) changes `rsum` by about +2 relative to the full recipe. On a single 5-epoch run that is not evidence that the predictive loss is the source of the retrieval number.
- Swapping InfoNCE for SigLIP is the largest positive change in the grid (+16 `rsum`, and image→text R@1 54.2 → 57.9). It is also only five epochs, so it is not the 16-epoch ViT-B number, and it is not the ViT-L SigLIP JSON.

### Qualitative retrieval

`experiments/qualitative_examples.md` and `experiments/qualitative_examples.json`, produced by `experiments/qualitative_retrieval.py`.

The report compares a checkpoint it names as `experiments/exp_jepa_1024d_20ep/checkpoint_best.pt` with CLIP ViT-L/14 zero-shot, on 16 sampled COCO captions for text→image. Six captions rank the correct image higher under the fine-tune, six rank it worse, four tie at rank 1. The largest listed gains are generic food and animal captions (a plates-of-food caption moves from rank 124 to 42; an elephants caption from 10 to 2). The largest listed regressions are a yellow sign (rank 35 → 78) and a person at a keyboard (53 → 75). Several image→text examples show the top-5 captions are paraphrases of the image even when the exact ground-truth string is missing (`hit@5: False` on a bathroom and a table-of-food).

That file was committed on 2026-05-31 (`d8c7802`), before the SigLIP result commit. At that time the 20-epoch directory was the ViT-L InfoNCE run. The path string in the markdown was later overloaded by the SigLIP run. The examples in the markdown are frozen text from the earlier evaluation; they are not a description of whatever file sits at that path now.

---

## Tests, and the contracts they enforce

`tests/` is the specification of behavior the later commits tried to stop from regressing. `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`.

| File | What it locks |
|---|---|
| `tests/test_smoke.py` | Shapes, masking, multi-crop, memory bank, SigLIP and hard-negative losses, EMA teacher update, checkpoint round-trip, OpenCLIP tower shapes, CLIP projection math, EOT pooling, residual projection equals the linear projection at init, text-tower unfreeze, model EMA, WiSE-FT endpoints, eval-weight swap and restore. Includes a GPU test that skips without CUDA. |
| `tests/test_quick_gelu_parity.py` | OpenAI pretrained tags request `force_quick_gelu`; other tags do not; a TypeError that is specifically “unexpected keyword force_quick_gelu” falls back; any other TypeError is raised. |
| `tests/test_eval_retrieval.py` | Perfect retrieval, monotonic recall, chunking, strict rank, median-rank convention, 1K fold remapping, and the ambiguity diagnostic (opt-in, not part of the standard dict, exact `str.strip` match, case preserved, duplicate captions on the same image count as one image). |
| `tests/test_eval_backend.py` | Checkpoint loading prefers `model_eval_state`, refuses a missing config, refuses an incomplete architecture config, accepts a complete legacy flat config, uses the CLIP tokenizer when the config says `openclip`, and fails `strict=True` on a mismatched state dict. |
| `tests/test_eval_coco_loader.py` | The unified loader keeps five captions per image and follows sorted COCO ids. |
| `tests/test_best_ckpt_keys.py` | Best-checkpoint payload contains config and eval state, and WiSE-FT is applied once. |
| `tests/test_alpha_zero_nan.py` | `alpha = 0` and `gamma = 0` skip NaN-prone unused tensors; the predictor does not run. |
| `tests/test_accum_skip.py` | A non-finite micro-batch resets accumulation and does not enqueue the memory bank; a short final window is rescaled so it is not under-weighted. |
| `tests/test_memory_bank_eval.py` | Eval loss does not depend on a filled queue; SigLIP does not use the queue. |
| `tests/test_alignment_overfit.py` | On a tiny set of real COCO pairs, the contrastive head can overfit, and SigLIP can drive an easy batch down. This is a capacity check, not a benchmark. |
| `tests/test_dataset.py` | COCO val length, token cache, epoch-dependent caption choice, train and val are different splits. Needs COCO on disk; it will skip or fail without it. |
| `tests/test_flickr_split.py` | Flickr loader filters `split == test`. |
| `tests/test_image_text_dataset.py` | Manifest parsing and the `(image, ids, mask)` contract for the unused CC3M path. |
| `tests/test_cached_dataset.py` | Cached tensor bundle round-trip and `forward_from_cache` shapes. |

---

## Dependencies

`requirements.txt` pins Torch, torchvision, NumPy, matplotlib, TensorBoard, PyYAML, Pillow, tqdm, scikit-learn, pytest, pycocotools, timm, and transformers. `pyproject.toml` repeats most of those and omits timm. Neither file lists `open_clip_torch`, which every headline config imports, or `huggingface_hub`, which Flickr evaluation imports. `img2dataset` is mentioned only as a command the CC3M helper prints.

Python: `pyproject.toml` says `>=3.9`. The README says 3.11+.

Claimed hardware, from config comments rather than from a log in the tree: one RTX 3090 (24 GB). The README’s opening line says “10 GB VRAM” and later says the 3090 is 24 GB. ViT-L/14 is configured with gradient checkpointing and batch 64 because the comments say a larger batch does not fit.

---

## File inventory

### Package and docs

| Path | What it is |
|---|---|
| `pyproject.toml` | Package metadata, setuptools package find for `src*`, pytest config. Version 0.2.0. |
| `requirements.txt` | Pip pins. Missing `open_clip_torch`. |
| `README.md` | Early-stage narrative: DistilBERT diagram, phased JEPA, Protocol A scores around 25% R@1. Useful as history. Stale as a description of the JSON results. |
| `.gitignore` | Ignores `data/`, `*.pt`, `logs/`, and `experiments/exp_*.json`. The result JSON files that are present were committed before that ignore rule, or were force-added; gitignore does not remove tracked files. New experiment JSON dropped into `experiments/` will not show up in `git status`. |
| `PROJECT_UNDERSTANDING.md` | This document. |

Narrative markdown that used to live at the repo root (`RESULTS.md`, `COMPLETE_REPORT.md`, `INVESTIGATION.md`, `SOTA_RESEARCH_GPT.md`, and others) was deleted in `3e42853` (2026-06-15). The README still links `INVESTIGATION.md` and `SOTA_RESEARCH_GPT.md`. Those links are dangling.

### `src/`

| Path | What it is |
|---|---|
| `src/__init__.py` | Re-exports the model pieces and `VL_JEPA_Trainer`. Does not export the OpenCLIP encoders, the projection classes, or the eval module. |
| `src/model.py` | Masking, three vision encoders, three text encoders, predictor, three projection types, `VL_JEPA`, `MemoryBank`, variance / hard-negative / SigLIP / InfoNCE losses. About 1,800 lines. This is the model. |
| `src/trainer.py` | NaN handling, checkpoint validation, Protocol A recall, `ModelEMA`, WiSE-FT, the training step, accumulation, unfreezing, eval-weight swap, save/load. About 1,500 lines. |
| `src/dataset.py` | COCO download, CLIP vs ImageNet normalization, `CaptionTokenizer`, on-disk token cache, train/val loaders. |
| `src/image_text_dataset.py` | Generic `<image path><TAB><caption>` dataset. Not used by the trainer. |
| `src/cached_dataset.py` | Loads precomputed encoder tensors. |
| `src/config.py` | YAML load, deep merge for CLI overrides, derived `max_steps` when `data.samples` is set. CLI override map does not include `wise_ft_alpha`, `use_model_ema`, or `model_ema_decay`; those are YAML-only. |
| `src/eval_retrieval.py` | Protocol B math only. No files, no models. |
| `src/eval_all_checkpoints.py` | A driver that tries to score six named models on COCO 5K, COCO 1K, and Flickr, and write a combined JSON. The model list has the catalog bugs described above. Default output directory is `artifacts/v1_final_report_2026_06_02/eval_json`, which is not in the repo. |

### `experiments/`

| Path | What it is |
|---|---|
| `experiments/exp_jepa_training.py` | The training program. COCO or cached embeddings. Writes checkpoints and `metrics.json`. |
| `experiments/evaluate_retrieval.py` | Protocol B on COCO for one checkpoint or one zero-shot OpenCLIP model. |
| `experiments/evaluate_flickr30k.py` | Protocol B on Flickr30K Karpathy test. |
| `experiments/eval_epoch_sweep.py` | Loads epoch checkpoints 4, 8, 12, 16, 20 of a ViT-L run and scores COCO and Flickr. Hard-codes an output directory under `/var/mnt/DATA/OpenClaw/...`. Not something you can run unchanged on this machine. |
| `experiments/diag_zeroshot.py` | Compares raw OpenCLIP, a fresh VL-JEPA with CLIP projection and EOT pooling, and an optional checkpoint, using **Protocol A**. This is the script the v2 config comment cites for “i2t 34.9% / t2i 30.2%”. Those percentages are not Protocol B. |
| `experiments/run_ablations.py` | Trains the nine variants and evaluates each with Protocol B 5K. |
| `experiments/plot_ablations.py` | Bar chart of `rsum` from a JSON file that is not committed. |
| `experiments/plot_training.py` | Curves from a training `metrics.json`. Best-epoch helper still uses validation loss. |
| `experiments/precompute_embeddings.py` | Dumps frozen encoder outputs for the cached path. |
| `experiments/download_cc3m.py` | TSV download and an `img2dataset` command printer. |
| `experiments/qualitative_retrieval.py` | Side-by-side ranks versus a zero-shot CLIP model. |
| `experiments/qualitative_examples.md` | The 16-caption write-up described above. |
| `experiments/qualitative_examples.json` | The machine-readable dump behind that write-up. |
| `experiments/training_plot.png` | A saved training figure. Source `metrics.json` is not in the tree. |
| `experiments/ablations/ablation_results.md` | The nine-row table. |
| `experiments/ablations/ablation_figure.png` | The bar chart. |
| `experiments/ablations/{full,no_ema,no_wise_ft,no_robust,mean_pool,random_proj,siglip,with_jepa,frozen}/config.yaml` | The exact override that defines each row. |
| `experiments/exp_baseline_vitb16_zeroshot.json` | Protocol B zero-shot ViT-B/16, COCO 5K and 1K. |
| `experiments/exp_baseline_vitl14_zeroshot.json` | Protocol B zero-shot ViT-L/14, COCO 5K and 1K. |
| `experiments/exp_vljepa_vitb16_robust.json` | Protocol B for `exp_jepa_768d_16ep/checkpoint_best.pt`. |
| `experiments/exp_vljepa_vitl14_robust.json` | Protocol B whose backend path collides with the SigLIP file. |
| `experiments/exp_vljepa_vitl14_siglip.json` | Rounded Protocol B numbers, same backend path, different scores. |
| `experiments/exp_flickr_zeroshot_vitb16.json` | Flickr zero-shot ViT-B/16. |
| `experiments/exp_flickr_zeroshot_vitl14.json` | Flickr zero-shot ViT-L/14. |
| `experiments/exp_flickr_vljepa_vitb16.json` | Flickr transfer of the ViT-B robust checkpoint. |
| `experiments/exp_flickr_vljepa_vitl14.json` | Flickr transfer, robust JSON, path collision. |
| `experiments/exp_flickr_vljepa_vitl14_siglip.json` | Flickr transfer, SigLIP JSON, same path. |

### `tests/`

Listed in the table above. `tests/__init__.py` is empty.

---

## Conditional predictor

`src/conditional/` is a second model family. It does not replace `VL_JEPA`. The predictor reads frozen visual tokens and a frozen query, and it is trained against a target encoder that is not a submodule. Default shape: 4 cross-attention blocks, width 384, 6 heads, target dim 512, temperature 1, one AdamW group. No teacher, queue, EMA, or WiSE-FT.

`src/protocol/` records Karpathy ids as `coco:{cocoid}` and Flickr ids as `flickr30k:{filename}`. The clean training list is Karpathy train plus restval (113,287), disjoint from Karpathy val and test. COCO train2017 still contains 4,407 Karpathy test images, so the historical ViT-B robust checkpoint is a train2017 adaptation. Manifests, `integrity_report.md`, and `final_results.csv` are that audit. The reference run `phase0-reference-seed0` uses synthetic tokens. The six target arms, M0–M5, and the confirmation seeds have not been trained.

---

## How the pieces connect

```
configs/*.yaml
        │
        ▼
experiments/exp_jepa_training.py
        │  builds VL_JEPA + VL_JEPA_Trainer
        │  data: src/dataset.py (COCO)  or  src/cached_dataset.py
        │  loss: src/model.py::compute_jepa_loss
        │  selection: Protocol A mean R@1 on val, under EMA/WiSE-FT weights
        ▼
experiments/exp_jepa_{dim}d_{epochs}ep/checkpoint_best.pt
        │  contains model_state_dict, model_eval_state, config
        ▼
experiments/evaluate_retrieval.py  ──►  COCO 5K / 1K JSON
experiments/evaluate_flickr30k.py  ──►  Flickr JSON
        │
        └── both call src/eval_retrieval.py (Protocol B)
            and load model_eval_state
```

Zero-shot rows skip the checkpoint and call `open_clip` directly.

A new run that should be comparable to the saved ViT-B number is:

```bash
python experiments/exp_jepa_training.py \
  --config configs/openclip_vitb16_robust.yaml \
  --fresh
```

That requires COCO at the configured root, a CUDA GPU, and `open_clip_torch`. Scoring it the way the JSON files were scored is:

```bash
python experiments/evaluate_retrieval.py \
  --checkpoint experiments/exp_jepa_768d_16ep/checkpoint_best.pt
```

The training log’s `R@1 i2t/t2i` line is Protocol A. It will not match the script above.

---

## Measured picture

The repository is a CLIP fine-tuning stack for retrieval, with a JEPA branch that the measured recipes disable. The gain over OpenAI CLIP on COCO, for the fully documented ViT-B run, is real on text→image and negligible on image→text R@1, and that same run gets worse on Flickr. A 5-epoch ablation says the CLIP projection and end-of-text pooling are load-bearing, WiSE-FT matters more than the weight EMA, and SigLIP beats InfoNCE on that short grid. The largest number in the tree (ViT-L SigLIP, COCO 5K `rsum` 452.58) sits in an incomplete JSON that shares a checkpoint path with a different result, and the in-repo model catalog does not agree with that path. The weights those files refer to are not here.
