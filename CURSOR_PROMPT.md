## VL-JEPA Commodity Hardware R&D — Critical Fixes + Architecture Adaptation

You are implementing R&D changes to make VL-JEPA work on a single RTX 3090 (10GB VRAM). This is not a toy exercise — we need a working JEPA+contrastive pipeline that can actually learn meaningful representations.

Read all source files first: src/model.py, src/trainer.py, src/config.py, configs/mvp_pretrained_siglip.yaml, experiments/exp_jepa_training.py, tests/test_smoke.py, tests/test_alignment_overfit.py

## CRITICAL ISSUES TO FIX (in order)

### Issue 1: JEPA Masking Broken with Pretrained Encoder

The TimmVisionEncoder ignores masking (del mask). This means JEPA MSE loss is trivially zero — context and target see identical images. The whole JEPA component is dead.

FIX: Implement proper patch-level masking INSIDE the timm forward pass.

The approach: Break open the timm ViT forward_features into its components:
1. Get patch embeddings via backbone.patch_embed(x)
2. Prepend CLS token
3. Add positional embedding
4. Apply mask: replace masked patches with a NEW learnable mask_token parameter
5. Run through backbone.blocks (transformer layers)
6. Apply backbone.norm (final LayerNorm)

Add a learnable mask_token parameter to TimmVisionEncoder.

IMPORTANT: Check the exact timm ViT architecture. Different timm versions expose different internals. Test with:
```python
import timm
m = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
print(dir(m))
print(m.patch_embed)
print(m.blocks)
print(m.pos_embed)
```

The mask tensor is (B, N) where N = num_patches, True = masked. CLS at index 0 is never masked.

### Issue 2: DistilBERT CLS Token Not a Sentence Embedding

DistilBERT's CLS token wasn't trained for sentence-level representation (unlike BERT's NSP objective). Using language_emb[:, 0, :] gives a weak signal.

FIX: Use mean pooling over non-padded tokens. In VL_JEPA.forward and get_joint_embedding:
```python
if attention_mask is not None:
    mask_float = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
    language_cls = (language_emb * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
else:
    language_cls = language_emb.mean(dim=1)
```

### Issue 3: EMA Projection Heads Dead Code

The model creates target_vision_proj, target_language_proj via EMA, but compute_jepa_loss uses student projections for contrastive loss. The EMA heads only feed target_patches for MSE.

FIX: Remove the dead EMA projection heads entirely (target_vision_proj, target_language_proj, target_language_encoder, target_vision_pred_head). Keep EMA only for the vision target encoder (which IS used for MSE targets). This simplifies the code and saves VRAM.

### Issue 4: VRAM Optimization

The RTX 3090 has ~10GB usable. Current setup (ViT-B + DistilBERT + predictor + multi-crop) likely exceeds this.

FIXES:
1. Add gradient checkpointing support: add a --gradient-checkpointing flag
2. Reduce default batch_size in mvp_pretrained_siglip.yaml from 16 to 8
3. Consider using vit_small_patch16_224 (~22M params) as fallback if ViT-B (86M) is too large
4. Add VRAM logging to the training loop

### Issue 5: Make JEPA MSE Meaningful with Frozen Encoders

With frozen encoders, the JEPA MSE tries to predict target from masked context. But if both encoders are frozen and identical, this is trivially solvable.

FIX: Add --unfreeze-after-epoch flag. When reached, set requires_grad=True on the last 2-4 transformer blocks of the vision encoder. This makes masking meaningful because the context encoder must learn to reconstruct from partial information.

### Issue 6: Proper Overfit Test

Update tests/test_alignment_overfit.py:
1. Update existing test to use SigLIP loss instead of InfoNCE
2. Add a test that verifies SigLIP sigmoid_contrastive_loss converges on synthetic data

### Issue 7: Config Updates

Update configs/mvp_pretrained_siglip.yaml:
- batch_size: 8 (safer for 10GB VRAM)
- gradient_checkpointing: true
- unfreeze_after_epoch: 5

## AFTER ALL FIXES

1. Run pytest tests/ -v — all must pass
2. Run 1-epoch test: python experiments/exp_jepa_training.py --config configs/mvp_pretrained_siglip.yaml --epochs 1 --fresh --no-tensorboard --batch-size 8
3. Verify: no NaN, MSE > 0 (not trivially zero), NCE@1 improving, VRAM < 10GB
4. git add -A && git commit -m 'feat: working JEPA masking, mean pooling, VRAM optimization' && git push

## IMPORTANT CONSTRAINTS
- All code changes must pass existing tests (38/38)
- Use the existing code structure, don't rewrite from scratch
- Keep backward compatibility with 'custom' encoders (the random-init path)
- Log VRAM usage in the training script
- The goal is a WORKING pipeline, not perfect results. We need to verify the architecture can learn before tuning hyperparameters.

## CONTEXT
This is R&D for reproducing VL-JEPA (arXiv:2512.10942) on commodity hardware. The paper uses massive pretrained models and data. We're finding new ways to achieve similar results on a single RTX 3090. The JEPA masking fix is the most critical — without it, we can't evaluate whether the JEPA approach works at all on commodity hardware.
