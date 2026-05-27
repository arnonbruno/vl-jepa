# VL-JEPA Training Investigation - May 26, 2026

## Timeline

1. **Initial problem:** Training crashed at batch ~1690 with NaN (exit code -15)
2. **First fix:** Cursor-agent diagnosed InfoNCE running in FP16 under AMP → overflow. Fixed by running InfoNCE in FP32.
3. **Result:** Training stable, passed batch 1690, completed epochs 1-10 without NaN.
4. **New problem:** Model not learning contrastive objective despite stable training.

## Symptoms

### Training Behavior
- MSE converged to near-zero by epoch 2 (reconstruction working)
- NCE loss stuck at ~3.46 (near ln(32) ≈ 3.47 = random baseline)
- NCE@1 accuracy stuck at 3.18-3.20% (random = 1/32 = 3.125%)
- No improvement over 10 epochs (20% of training)

### Metrics Summary

| Epoch | Train Loss | Val Loss | Train NCE | Val NCE | Val NCE@1 | Train MSE |
|-------|-----------|----------|-----------|---------|-----------|-----------|
| 1 | 3.458 | 3.458 | 3.457 | 3.457 | 3.17% | 0.00225 |
| 2 | 3.470 | 3.460 | 3.470 | 3.460 | 3.18% | 2.0e-05 |
| 3 | 3.469 | 3.460 | 3.469 | 3.460 | 3.18% | 4.7e-06 |
| 5 | 3.467 | 3.457 | 3.467 | 3.457 | 3.19% | 3.7e-07 |
| 10 | 3.467 | 3.457 | 3.467 | 3.457 | 3.20% | 2.8e-07 |

**Key observations:**
- MSE dropped 4 orders of magnitude (learning reconstruction)
- NCE barely moved (not learning contrastive)
- Loss actually increased slightly from epoch 1 to epoch 2
- Gradient norm very low (0.03-0.05) by epoch 10

## Possible Causes

1. **Teacher-student collapse:** EMA teacher might be too similar to student, providing no learning signal
2. **Negative sampling issue:** Contrastive negatives not informative
3. **Temperature/scale problem:** Logit scale might be wrong
4. **Embedding collapse:** Vision and language embeddings might be collapsing to same values
5. **Teacher momentum too high:** τ = 0.997 might prevent teacher from diverging enough
6. **Multi-crop configuration issue:** Possible mismatch in how crops are processed

## Files to Investigate

- `src/model.py` — InfoNCE implementation, temperature, negative sampling
- `src/trainer.py` — EMA update, teacher momentum schedule
- `configs/default.yaml` — temperature, momentum, multi-crop settings
- `experiments/exp_jepa_training.py` — training loop, loss weighting

## Relevant Code Points

### From previous cursor-agent session (commit 93491ad):
- InfoNCE now runs in FP32 (was FP16 causing NaN)
- NaN diagnostics added with `--nan-diagnostics-dir`
- Pre/post forward checks for weight corruption

### Current config (from logs):
- τ (EMA momentum): 0.997
- Learning rate: 1e-4 (max)
- Batch size: 32
- Multi-crop: enabled (global + local crops)

## What We Know Works

- Reconstruction (MSE) converges
- Training is stable (no NaN after fix)
- Checkpoints save correctly
- GPU memory usage consistent (~6.8GB)

## What We Need to Fix

The contrastive learning component is not receiving gradient signal or the objective itself is misconfigured. Need to investigate:
1. Are teacher embeddings actually different from student?
2. Are negatives being sampled correctly?
3. Is temperature/scaling correct?
4. Is the contrastive loss actually being backpropagated?

## Next Steps

Spawn cursor-agent with this context to investigate the contrastive learning pipeline and propose fixes.
