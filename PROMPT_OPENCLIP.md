Integrate OpenCLIP as an end-to-end backbone option for the VL-JEPA model. The current architecture uses separate CLIP ViT-B/16 + DistilBERT which plateaus at 25% R@1 on COCO. OpenCLIP gives aligned vision+text encoders from the start.

## Step 1: Install open_clip_torch
```bash
pip install open_clip_torch
```

## Step 2: Add OpenCLIP vision+text encoder classes in src/model.py

Create two new classes:

### OpenCLIPVisionEncoder(nn.Module)
- Wraps `open_clip.create_model_and_transforms()` vision encoder
- Should have the same interface as TimmVisionEncoder: `hidden_dim`, `num_patches`, `grid_size`, `mask_token`, `forward(x, mask=None)`
- The forward method should do patch-level masking like TimmVisionEncoder: patch_embed -> apply mask -> transformer blocks -> norm
- Access the model's internal patch embedding and transformer blocks directly (openclip models expose these)
- Support gradient checkpointing
- Freeze/unfreeze controlled by `freeze` parameter

### OpenCLIPLanguageEncoder(nn.Module)  
- Wraps open_clip's text encoder
- Interface: `hidden_dim`, `forward(input_ids, attention_mask=None)` -> (B, S, D)
- Return the full token sequence (not just CLS/pooler) — pooling happens downstream
- Support freeze/unfreeze

## Step 3: Update VL_JEPA class
In VL_JEPA.__init__, add a new branch for `vision_backbone == "openclip"`:
```python
if self.vision_backbone == "openclip":
    self.context_encoder = OpenCLIPVisionEncoder(...)
    self.language_encoder = OpenCLIPLanguageEncoder(...)
```
Also update the target_encoder (EMA copy) to work with OpenCLIP encoders.

## Step 4: Update config
Add a new config file `configs/openclip_vitb16.yaml`:
```yaml
model:
  hidden_dim: 768
  patch_size: 16
  image_size: 224
  mask_ratio: 0.75
  text_mask_ratio: 0.0
  predictor_layers: 4
  momentum_tau: 0.996
  vision_backbone: openclip
  text_backbone: openclip
  openclip_model: ViT-B-16
  openclip_pretrained: openai
  freeze_encoders: true
  projection_dim: 512
  contrastive_loss: siglip
  contrastive_on_global: true

training:
  epochs: 50
  batch_size: 128
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-4
  weight_decay: 0.1
  warmup_steps: 500
  unfreeze_after_epoch: 5
  unfreeze_vision_blocks: 4
  encoder_unfreeze_lr: 2.0e-5
  max_grad_norm: 1.0

loss:
  alpha: 0.1
  beta: 0.9
  gamma: 0.01
  label_smoothing: 0.05
```

## Step 5: Update config.py
Add `openclip_model` and `openclip_pretrained` to the config override mapping.

## Step 6: Update experiment script
Make sure exp_jepa_training.py can handle openclip backbones (the config loading and model creation should work with the new branches).

## Step 7: Update tests
- Add a test that creates a VL_JEPA model with openclip backbone and verifies forward pass works
- Add a test that verifies the openclip vision encoder produces correct output shapes
- Make sure existing tests still pass (don't break timm/DistilBERT paths)

## Step 8: Run tests and smoke test
```bash
python3.14 -m pytest tests/ -v
python3.14 experiments/exp_jepa_training.py --config configs/openclip_vitb16.yaml --epochs 1 --fresh --no-tensorboard --num-workers 4
```

## Step 9: Commit and push
Commit message: 'feat: add OpenCLIP end-to-end backbone option'
Push to origin/master.

## Important Notes
- open_clip's ViT exposes: `model.visual.conv1` (patch embed), `model.visual.transformer` (blocks), `model.visual.ln_post` (final norm), `model.visual.positional_embedding` (pos embed)
- The text encoder exposes: `model.token_embedding`, `model.transformer`, `model.ln_final`
- For patch masking, you need to: (1) get patch embeddings via conv1, (2) apply mask, (3) add position embeddings, (4) run through transformer blocks
- The openclip model returns (image_features, text_features, logit_scale) from forward, but we need intermediate representations — so access the encoders directly, not through the top-level forward
- Check `open_clip.list_pretrained()` to see available models and pretrained weights

## Constraints
- Do NOT break existing timm/DistilBERT code paths
- Keep all existing configs working
- Hardware: RTX 3090 (24GB VRAM)
- Commit message: descriptive, no mention of tools
- Push to origin/master after commit
