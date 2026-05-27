# Next Session Context — VL-JEPA Development

Date: 2026-05-27

## Current State

- **Branch:** main, commit 0bb8a2f (pushed)
- **Tests:** 35/35 passing
- **GPU:** RTX 3090 (~10GB usable VRAM)
- **Last training:** killed at epoch 26, NCE@1 stuck at ~3.2% (random baseline)
- **No running processes**

## What Happened

Two independent research agents (GLM-5 and GPT-5.5-high) analyzed the codebase and SOTA literature. Both reached the same conclusions:

### Root Cause
The model has **random-initialized vision and text encoders** trying to learn vision-language alignment from scratch on COCO only (118K images, batch 32). This is structurally impossible — published VL-JEPA/V-JEPA/CLIP/SigLIP all use pretrained backbones.

### What We Tried (all failed to move NCE@1 above random)
1. Student-to-student contrastive (vs detached teacher)
2. VICReg variance regularization
3. Context encoder CLS for contrastive
4. Projection head LR 10x
5. MoCo memory bank (65K negatives)
6. Gradient clipping (max_norm=2.0)
7. Student projections in queue (vs stale teacher)

### Why Each Failed
- Random encoders have no semantic geometry for contrastive learning to exploit
- Local masked 96×96 crop used for contrastive (caption describes full 224×224 image)
- InfoNCE softmax needs huge batches (CLIP uses 32K+); we have batch 32
- Single linear projection heads too weak

## Recommended Implementation Order (P0)

### Step 1: Pretrained Encoders (frozen)
- Replace custom `VisionEncoder` with `timm` MAE ViT-B/16 (`vit_base_patch16_224.mae`)
- Replace custom `LanguageEncoder` with HuggingFace `DistilBertModel`
- Freeze both for first 5 epochs
- VRAM: lower than current when frozen; fits at batch 16-32

### Step 2: Fix Contrastive View
- Use **global unmasked student CLS** for contrastive alignment (not local masked)
- Keep local masked views for JEPA MSE only
- File: `src/model.py`, function `VL_JEPA.forward`

### Step 3: SigLIP Loss
- Replace InfoNCE with sigmoid pairwise loss (no softmax dependency on batch size)
- Learnable `logit_scale` + `logit_bias`
- Disable memory bank initially (`memory_bank_size: 0`)
- File: `src/model.py` + `src/trainer.py`

### Step 4: Projection Heads
- 2-layer MLP: `LayerNorm → Linear(768, 768) → GELU → Linear(768, 256) → L2 normalize`
- Queue dimension drops from 768→256 (saves VRAM)

### Step 5: Phased Training
- Phase A (epochs 1-5): `alpha=0, beta=1, gamma=0.01` — alignment only
- Phase B (epochs 6-20): `alpha=0.2, beta=0.8, gamma=0.01` — add JEPA gently
- Phase C (epochs 21+): `alpha=0.3, beta=0.7, gamma=0.01`

### Step 6: Proper Metrics
- Full-val Recall@K (i2t and t2i, R@1/5/10)
- Tiny overfit test (128 pairs, must reach >50% NCE@1)

## SOTA Research Reports
- `SOTA_RESEARCH.md` — GLM-5 agent findings
- `SOTA_RESEARCH_GPT.md` — GPT-5.5-high agent findings (includes exact config recipe)

## Implementation Notes
- All code changes via `cursor-agent` (spawn-cursor.sh)
- Default model: `composer-2.5` for cursor-agent
- Use `stream-parser.py` to monitor agent progress
- Use `say-to-agent.sh <session_id>` for mid-task feedback
- Test with `pytest tests/ -v` after each change
- 1-epoch verification run before committing to 50-epoch training
