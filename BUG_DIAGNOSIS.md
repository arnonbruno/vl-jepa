# VL-JEPA Contrastive Learning Bug - Root Cause Analysis

## Critical Bug Found: Wrong Contrastive Targets

### Location
`src/model.py` lines 716-725 in `compute_jepa_loss()`

### Current Code (BROKEN)
```python
target_language_proj = outputs.get('target_language_proj')
target_vision_proj = outputs.get('target_vision_proj')
if target_language_proj is None or target_vision_proj is None:
    logits_i2t = vision_proj @ language_proj.T * scale
    logits_t2i = logits_i2t.T
else:
    tgt_lang = target_language_proj.detach().float()
    tgt_vis = target_vision_proj.detach().float()
    logits_i2t = vision_proj @ tgt_lang.T * scale
    logits_t2i = language_proj @ tgt_vis.T * scale
```

### The Problem

When target projections exist (EMA teachers), the code computes:
- `vision_proj @ target_language_proj.T` (student vision vs TEACHER language)
- `language_proj @ target_vision_proj.T` (student language vs TEACHER vision)

**Why this causes no learning:**
1. Teacher embeddings are `detached()` — no gradients flow back
2. With momentum tau=0.997, teacher ≈ student after first few batches
3. Matching student→teacher is trivially easy (nearly identical embeddings)
4. No gradient signal → NCE stuck at random baseline (~ln(32) ≈ 3.47)

### Evidence from Training

| Epoch | Train Loss | Val NCE | Val NCE@1 | MSE |
|-------|-----------|---------|-----------|-----|
| 1 | 3.458 | 3.457 | 3.17% | 0.00225 |
| 2 | 3.470 | 3.460 | 3.18% | 2.0e-05 |
| 10 | 3.467 | 3.457 | 3.20% | 2.8e-07 |

- MSE converged 4 orders of magnitude (reconstruction works)
- NCE flat at ~3.46 (random baseline for batch_size=32)
- NCE@1 at 3.2% = 1/32 = random chance

### The Fix

Use student-to-student contrastive learning (correct for contrastive objectives):

```python
# Always use student embeddings for contrastive (both get gradients)
logits_i2t = vision_proj @ language_proj.T * scale
logits_t2i = logits_i2t.T
```

Or optionally, use cross-modal student-teacher (CLIP-style but with EMA):
```python
# Student vision vs Student language (both receive gradients)
logits_i2t = vision_proj @ language_proj.T * scale  
logits_t2i = language_proj @ vision_proj.T * scale  # Transpose of the same matrix
```

### Why This Works

- Both `vision_proj` and `language_proj` receive gradients
- The model must learn to align vision and language representations
- No teacher "shortcut" — real contrastive learning signal

### Additional Context

- Logit scale: 2.659 (ln(1/0.07)) — reasonable, no change needed
- EMA momentum: 0.997 — fine for teacher stability, but shouldn't be used for contrastive targets
- Beta (NCE weight): 0.5 — may want to increase after fix to speed up contrastive learning

### Related Code

The teacher projections are still needed for the MSE loss (JEPA prediction):
- `target_patches` for MSE = correct use of teacher
- `target_vision_proj` / `target_language_proj` for contrastive = WRONG

### After Fix

The training should show:
- NCE loss decreasing from ~3.46 toward lower values
- NCE@1 increasing from 3% toward meaningful accuracy (>10%)
- Gradient norm should increase slightly (more signal flowing)

### Verification

Run 1-epoch training after fix and check:
1. NCE loss < 3.40 (vs 3.46 before)
2. NCE@1 > 5% (vs 3.2% before)
3. Gradients flowing to projection layers
