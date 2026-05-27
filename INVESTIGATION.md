# VL-JEPA Training Investigation

**Last updated**: May 27, 2026 (architecture as of commit `8b6298d`)

---

## Executive summary

Training on COCO 2017 was **stable** after fixing AMP/InfoNCE NaNs, but **contrastive learning did not improve** for many epochs: NCE stayed at `ln(batch_size) ≈ 3.47` and NCE@1 at ~3.12% (random for B=32). A sequence of targeted fixes (student NCE, context CLS, variance reg, projection LR, memory bank, student queue keys) addressed root causes. The current stack is documented in `README.md`; contrastive metrics remain an active tuning area.

---

## Timeline

### Phase 1 — NaN instability (resolved)

1. Training crashed around batch ~1690 with NaN (exit -15).
2. **Diagnosis**: InfoNCE computed in FP16 under AMP → softmax overflow.
3. **Fix** (`93491ad`): Run InfoNCE in FP32; cap GradScaler growth; EMA schedule fixes (`c6caa1e`).
4. **Result**: Training completes epochs without NaN; checkpoint validation added.

### Phase 2 — Contrastive not learning (in-batch only)

**Symptoms** (10 epochs, B=32, COCO):

| Epoch | Train NCE | Val NCE | Val NCE@1 | Train MSE |
|-------|-----------|---------|-----------|-----------|
| 1 | 3.457 | 3.457 | 3.17% | 0.002 |
| 10 | 3.467 | 3.457 | 3.20% | ~0 |

- MSE dropped 4+ orders of magnitude (reconstruction works).
- NCE flat at random baseline; gradient norms fell to ~0.03–0.05 by epoch 10.

**Hypotheses tested**:

| Hypothesis | Finding |
|------------|---------|
| Teacher–student too similar | Student–teacher InfoNCE had weak/no useful gradient |
| Wrong vision token for NCE | Predictor CLS is wrong; **context encoder CLS** is correct |
| Embedding collapse | **Variance regularization** (`γ=0.1`) helps without killing NCE |
| Projection heads too slow | **10× LR** on `vision_proj` / `language_proj` |
| Too few negatives (B=32) | **65K MoCo queue** needed for i2t |

### Phase 3 — Memory bank fixes

| Commit | Change | Outcome |
|--------|--------|---------|
| `5243a7d` | Student (not teacher) projections in InfoNCE | Required for gradient flow |
| `adc126b` | Context CLS for vision queries | Aligns contrastive with encoder representation |
| `319dd72` | VICReg variance loss | Anti-collapse |
| `bae0884` | Projection LR 10× | Heads adapt faster |
| `08927de` | 65,536 FIFO language key queue | Many more i2t negatives |
| `08927de` (initial) | Enqueued **teacher** `target_language_proj` | **Failed**: stale τ=0.996 teacher → nearly identical queue entries → NCE **rose** toward ~7.27 |
| **`8b6298d`** | Enqueue **student** `language_proj` | Queue diversity restored; meaningful negatives |

**Root cause of stale-teacher queue** (`8b6298d`):

```
Previous: memory_bank.enqueue(target_language_proj)  # EMA, updates slowly
Effect:   65K keys nearly identical → softmax saturated → NCE climbed (4.75 → 7.27)

Fixed:    memory_bank.enqueue(language_proj)         # student, every step
Effect:   Diverse negatives; i2t loss scale changes (not comparable to ln(32))
```

---

## Current contrastive configuration

```yaml
# configs/default.yaml (excerpt)
loss:
  alpha: 0.5
  beta: 0.5
  gamma: 0.1          # variance regularization

training:
  batch_size: 32
  max_grad_norm: 2.0  # was 1.0, then 3.0; 2.0 best for student-student NCE
  memory_bank_size: 65536
```

**InfoNCE details** (`src/model.py`):

- **i2t**: `vision_proj` (context CLS) vs `[batch language_proj ∥ queue]`
- **t2i**: `language_proj` vs in-batch `vision_proj` only
- Keys in queue: detached, L2-normalized **student** `language_proj`
- Teacher projections (`target_*`) used only for **MSE patch targets**, not NCE

---

## Metrics interpretation (post memory bank)

| Metric | Old (B=32 only) | New (B=32 + 65K queue) |
|--------|-----------------|-------------------------|
| Random NCE baseline | `ln(32) ≈ 3.47` | i2t: `ln(32 + N_fill) → ln(65568) ≈ 11.1` when full |
| NCE@1 random | 1/32 = 3.125% | Still ~3.1% early on (hard task) |
| What to watch | NCE@1, val loss | Downward val NCE trends, not absolute value vs 3.47 |

**Example** (`experiments/exp_jepa_768d_50ep`, after `8b6298d`):

- Best val loss: **1.92** (epoch 7)
- Best val NCE in that run: **3.64** (epoch 7)
- MSE → ~0 by epoch 2–3

---

## What works

- I-JEPA reconstruction (MSE on masked patches)
- Stable long training with FP32 NCE + GradScaler cap
- Checkpoint save/resume including `memory_bank_state_dict`
- COCO 2017 dataloaders with DistilBERT tokenization
- Student queue memory bank (after `8b6298d`)

## Open issues

1. **NCE@1** still near random on validation in many runs — alignment needs more epochs / hyperparameter search.
2. Some 50-epoch runs hit **non-finite batches** after epoch 5–6 (MSE spike → inf); investigate LR / MSE weight / AMP interaction.
3. Multi-crop + memory bank increases compute; tune `alpha`/`beta` if MSE dominates.

---

## Files reference

| File | Relevant logic |
|------|----------------|
| `src/model.py` | `compute_jepa_loss`, `MemoryBank`, `variance_loss` |
| `src/trainer.py` | EMA schedule, param groups (10×/20× LR), `memory_bank.enqueue` |
| `configs/default.yaml` | `memory_bank_size`, `max_grad_norm`, loss weights |
| `experiments/exp_jepa_training.py` | CLI, COCO loaders, logging |

---

## Key commits (chronological)

```
c6caa1e  Fix NaN instability (AMP scaler, EMA schedule)
93491ad  InfoNCE FP32 + NaN diagnostics
5243a7d  Student embeddings for InfoNCE
f88da89  Configurable grad clipping
a712827  max_grad_norm = 2.0
319dd72  Variance regularization
adc126b  Context encoder CLS for contrastive
bae0884  Projection head LR 10×
08927de  MoCo memory bank (65K)
8b6298d  Student projections in memory bank (fix stale teacher queue)
```
