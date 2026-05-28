# VL-JEPA SOTA Performance Overhaul

## Goal
Improve the VL-JEPA model to match or exceed CLIP ViT-B/16 retrieval performance on COCO 2017 (target: i2t R@1 ≥ 45%, t2i R@1 ≥ 45%). Hardware: RTX 3090 (24GB VRAM), 32GB RAM, 16 CPUs.

## Current State
Training ran 22 epochs. Best metrics:
- i2t R@1: 14.4% | R@5: 38.3% | R@10: 53.1%
- t2i R@1: 14.7% | R@5: 39.2% | R@10: 53.4%
- Plateauing since epoch 15. Loss converged.

## Root Cause Analysis (verified by multi-model research)
Seven bottlenecks, ranked by impact:

### 1. Wrong pretrained backbone (HIGHEST IMPACT)
Config says "pretrained_siglip" but actually uses `vit_base_patch16_224.mae` — an MAE-pretrained ViT. MAE learns to reconstruct masked patches, NOT to align images with text. The features are not multimodal. **Fix: switch to CLIP-pretrained ViT-B/16** via timm: `vit_base_patch16_clip_224.openai`. This model was trained with CLIP contrastive objective on 400M image-text pairs. Its features are already aligned with language. This alone should double or triple R@1.

### 2. Batch size 32 is too small for contrastive learning
Only 31 negatives per sample. SigLIP/CLIP needs 256+ for good gradients. RTX 3090 can't fit larger batches directly. **Fix: add gradient accumulation** to simulate effective batch size of 256 (8 accumulation steps × 32). Add `--gradient-accumulation-steps` CLI arg.

### 3. Projection dim 256 is too small
Compressing 768-dim features to 256 loses information. CLIP uses 512, SigLIP uses 768-1152. **Fix: increase projection_dim to 512.**

### 4. Only 2/12 vision blocks unfrozen
83% of ViT is frozen. With CLIP pretrained features, we should unfreeze more aggressively. **Fix: unfreeze last 4 blocks by default, with encoder_unfreeze_lr=5e-5** (5x current).

### 5. DistilBERT uses CLS token pooling
DistilBERT CLS token is NOT trained for sentence representation. Mean pooling over non-padded tokens is much better. **Fix: change HFLanguageEncoder.forward to use mean pooling.**

### 6. No text masking (text_mask_ratio=0.0)
JEPA-style text masking is disabled, removing the text-side prediction signal. This is fine for phase 1 (contrastive only) but should be enabled later.

### 7. Memory bank disabled
memory_bank_size=0. With gradient accumulation simulating larger batches, this is less critical. Can add later.

## Required Changes

### Step 1: Switch backbone to CLIP-pretrained ViT-B/16
In `configs/mvp_pretrained_siglip.yaml`, change:
```yaml
vision_backbone: vit_base_patch16_clip_224.openai
```

In `src/model.py`, the `TimmVisionEncoder` class already handles timm backbones. Verify it works with CLIP ViT (check `embed_dim`, `num_features`, patch embedding access). The CLIP ViT in timm uses the same architecture as MAE ViT so this should be straightforward.

**CRITICAL**: The CLIP ViT has different patch embedding weights. The `mask_token` parameter is already learnable, which is correct. But verify that `self.backbone.patch_embed` works the same way. Also check that position embedding interpolation works for the CLIP ViT (it uses the same 224/16=14 grid).

### Step 2: Add gradient accumulation
In `experiments/exp_jepa_training.py`:
- Add `--gradient-accumulation-steps` CLI arg (default: 1)
- In the training loop, accumulate gradients over N steps before `optimizer.step()`
- Scale the loss by 1/N before `.backward()`
- Only call `optimizer.step()` and `scheduler.step()` every N steps
- Add `optimizer.zero_grad()` after each step
- Print effective batch size in the header

### Step 3: Increase projection_dim to 512
In `configs/mvp_pretrained_siglip.yaml`:
```yaml
projection_dim: 512
```

### Step 4: Unfreeze 4 blocks with higher LR
In `configs/mvp_pretrained_siglip.yaml`:
```yaml
unfreeze_vision_blocks: 4
encoder_unfreeze_lr: 5.0e-5
```

### Step 5: Mean pooling for DistilBERT
In `src/model.py`, `HFLanguageEncoder.forward()`:
- Currently uses `outputs.last_hidden_state[:, 0]` (CLS token)
- Change to mean pooling: `outputs.last_hidden_state.masked_fill(~attention_mask.unsqueeze(-1).bool(), 0).sum(1) / attention_mask.sum(1, keepdim=True)`
- This gives a much better sentence representation from DistilBERT

### Step 6: Update config defaults
Update `configs/mvp_pretrained_siglip.yaml` with all changes:
```yaml
model:
  vision_backbone: vit_base_patch16_clip_224.openai
  projection_dim: 512
  freeze_encoders: true

training:
  batch_size: 32
  gradient_accumulation_steps: 8  # effective batch = 256
  unfreeze_vision_blocks: 4
  encoder_unfreeze_lr: 5.0e-5
  epochs: 50
```

## Verification

1. **Run all existing tests**: `python3.14 -m pytest tests/ -v`
2. **1-epoch smoke test**: `python3.14 experiments/exp_jepa_training.py --config configs/mvp_pretrained_siglip.yaml --epochs 1 --no-tensorboard`
   - Verify: no NaN, no crashes, loss decreasing
   - Verify: NCE@1 > 70% in first epoch (CLIP features should give immediate boost)
   - Verify: R@1 > 15% (should already beat the 22-epoch MAE model)
3. **Check VRAM usage**: should stay under 22GB with batch_size=32 + gradient_accumulation=8
4. **Commit and push** after verification passes

## Constraints
- Do NOT change the loss function (keep SigLIP sigmoid contrastive)
- Do NOT change the data pipeline (COCO 2017, same dataloader)
- Do NOT add new dependencies beyond what's already installed (timm, transformers, torch)
- Do NOT modify test files unless tests need updating for the new backbone
- Keep the `--fresh` flag behavior (overwrite existing metrics)
- The `--gradient-checkpointing` flag should still work with gradient accumulation
- Commit message: describe what changed, NOT the tool that made it

## Files to Modify
- `configs/mvp_pretrained_siglip.yaml` — backbone, projection_dim, unfreeze, gradient_accumulation_steps
- `src/model.py` — HFLanguageEncoder mean pooling, verify CLIP ViT compatibility
- `experiments/exp_jepa_training.py` — gradient accumulation in training loop, CLI arg
- `tests/` — update any tests that assert specific backbone or projection dimensions
