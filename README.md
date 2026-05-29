# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

PyTorch implementation of [VL-JEPA](https://arxiv.org/abs/2512.10942) — a JEPA-style vision-language model with masked patch prediction, EMA target encoders, and contrastive alignment. Optimized for commodity hardware (RTX 3090, 10GB VRAM).

**Paper authors**: Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung  
**Implementation**: Bruno Santos + TARS

---

## Overview

VL-JEPA learns joint vision-language representations by:

1. **Predicting** masked patch embeddings (I-JEPA style MSE against an EMA teacher).
2. **Aligning** image and text in a shared projection space (SigLIP sigmoid loss).
3. **Regularizing** embedding variance (VICReg-style) to prevent collapse.

This implementation uses **pretrained frozen encoders** (timm CLIP-pretrained ViT-B/16 + HuggingFace DistilBERT) as a starting point, with **phased training** that gradually introduces JEPA reconstruction after contrastive alignment is established. The vision backbone starts from multimodally-aligned CLIP features (pretrained on 400M image-text pairs) rather than reconstruction-pretrained MAE features, giving the model a strong vision-language prior from the outset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING FORWARD PASS                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Global + local crops (multi-crop)          Caption tokens (DistilBERT tokenizer)
           │                                              │
           ▼                                              ▼
  ┌────────────────────┐                       ┌────────────────────┐
  │  Context encoder   │  75% block-masked     │  Language encoder  │
  │ (timm CLIP ViT-B)  │  local/global views   │  (DistilBERT)      │
  │  frozen, 86M params│                       │  frozen, 66M params│
  └─────────┬──────────┘                       └─────────┬──────────┘
            │                                            │
            │         ┌────────────────────┐             │
            └────────►│  Predictor         │             │
                      │  4L transformer    │             │
                      └─────────┬──────────┘             │
                                │                        │
            MSE (masked patches)│                        │
                                ▼                        ▼
  ┌────────────────────┐              ┌──────────────────────────────────┐
  │  Target encoder    │◄── EMA τ ────  │  Projection heads (MLP)         │
  │  (frozen teacher)  │   0.996→1.0  │  768 → 512, L2-normalized       │
  │  full-image patches│              │  Global CLS → vision_proj       │
  └────────────────────┘              └───────────────┬──────────────────┘
                                                      │
                         ┌────────────────────────────┼────────────────────────────┐
                         │  SigLIP (FP32)             │                            │
                         │  • Pairwise sigmoid loss   │  Phase A: alignment only   │
                         │  • No softmax dependency   │  Phase B: add JEPA MSE     │
                         │  • Works at batch_size=8   │  Phase C: full training    │
                         └────────────────────────────┴────────────────────────────┘
                                                      │
                         L = α·MSE + β·SigLIP + γ·VarReg
```

### Components

| Component | Role | Details |
|-----------|------|---------|
| **Context encoder** | Student ViT; encodes masked/cropped images | timm CLIP ViT-B/16 (`openai`), frozen in phase A |
| **Target encoder** | EMA copy of context encoder; stop-gradient patch targets | Same weights, τ cosine 0.996→1.0 |
| **Predictor** | Lightweight transformer; predicts teacher patch embeddings | 4 layers |
| **Language encoder** | Text encoder | DistilBERT (frozen), mean pooling |
| **Projection heads** | 2-layer MLP: LayerNorm→Linear→GELU→Linear→L2 | 768→512, 10× base LR |
| **SigLIP loss** | Pairwise sigmoid contrastive loss | No softmax, works at small batches |
| **Variance reg** | VICReg-style std ≥ 1 on pre-norm projections | γ = 0.01 |

### Loss

```python
L = α · L_mse + β · L_siglip + γ · L_var

L_mse:    MSE(predicted_patches, target_patches) on masked positions only
L_siglip: pairwise sigmoid contrastive (no softmax dependency on batch size)
L_var:    VICReg variance on vision_proj_raw and language_proj_raw
```

### Phased Training

| Phase | Epochs | α (MSE) | β (SigLIP) | γ (Var) | What happens |
|-------|--------|---------|------------|---------|--------------|
| A | 1-5 | 0.0 | 1.0 | 0.01 | Alignment only, frozen encoders |
| B | 6-20 | 0.2 | 0.8 | 0.01 | Add JEPA MSE, unfreeze last 4 vision blocks (epoch 5, `encoder_unfreeze_lr=5e-5`) |
| C | 21+ | 0.3 | 0.7 | 0.01 | Full training |

---

## Results (CLIP backbone, COCO 2017, epoch 16)

Current metrics with the CLIP-pretrained ViT-B/16 backbone and gradient accumulation:

| Metric | Value | Notes |
|--------|-------|-------|
| **i2t R@1** | ~24% | Image→text retrieval (vs MAE baseline: 14.7% after 22 epochs) |
| **t2i R@1** | ~25% | Text→image retrieval |
| **NCE@1** | ~92% | In-batch retrieval accuracy |
| **Val Loss** | 0.77 | — |

The CLIP backbone reaches ~24-25% R@1 by epoch 16, well above the MAE baseline's 14.7% R@1 after 22 epochs — the multimodally-aligned starting features converge faster and higher.

### Early SigLIP baseline (RTX 3090, COCO 2017, 1 epoch)

| Metric | Value | Notes |
|--------|-------|-------|
| **Train NCE@1** | 78.16% | In-batch retrieval accuracy |
| **Val NCE@1** | 86.05% | In-batch retrieval accuracy |
| **Val Loss** | 1.04 | Started at 6.85 |
| **MSE** | 0.41 | Non-zero (JEPA masking working) |
| **VRAM** | 1.64 GB | Fits easily on RTX 3090 |
| **Skipped batches** | 0 | Zero NaN |
| **Training time** | 53 min | 1 epoch, batch_size=8 |

Comparison with previous approach (random encoders + InfoNCE):

| Approach | Val NCE@1 | VRAM | Notes |
|----------|-----------|------|-------|
| Random encoders + InfoNCE + 65K queue | ~3.2% | 7.3 GB | Stuck at random baseline for 25+ epochs |
| **Pretrained + SigLIP + mean pooling** | **86.05%** | **1.64 GB** | Works in 1 epoch |

---

## Setup

### Prerequisites

- Python 3.11+
- PyTorch 2.1+ with CUDA
- GPU with 2GB+ VRAM (tested on RTX 3090)

### Installation

```bash
git clone https://github.com/arnonbruno/vl-jepa.git
cd vl-jepa
pip install -r requirements.txt
```

---

## Training

### Quick start (pretrained SigLIP baseline)

```bash
python experiments/exp_jepa_training.py \
  --config configs/mvp_pretrained_siglip.yaml \
  --epochs 10 \
  --batch-size 8 \
  --fresh
```

### Full training with phased schedule

```bash
python experiments/exp_jepa_training.py \
  --config configs/mvp_pretrained_siglip.yaml \
  --epochs 30 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --phase-training \
  --gradient-checkpointing \
  --unfreeze-after-epoch 5 \
  --fresh
```

### CLI highlights

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML training config |
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 8 | Batch size (per accumulation step) |
| `--gradient-accumulation-steps` | 8 | Steps to accumulate before optimizer step; effective batch size = `batch_size × accumulation_steps` |
| `--lr` | 1e-4 | Peak learning rate |
| `--vision-backbone` | `vit_base_patch16_clip_224.openai` | timm vision model (CLIP-pretrained ViT-B/16) |
| `--text-backbone` | `distilbert-base-uncased` | HuggingFace text model |
| `--contrastive-loss` | `siglip` | Loss type: `siglip` or `infonce` |
| `--phase-training` | false | Use phased α/β/γ schedule |
| `--gradient-checkpointing` | false | Reduce VRAM at cost of speed |
| `--unfreeze-after-epoch` | 5 | Epoch to unfreeze the last 4 vision blocks (`encoder_unfreeze_lr=5e-5`) |
| `--resume` | — | Resume from checkpoint |
| `--fresh` | — | Start from scratch |

---

## Project structure

```
vl-jepa/
├── src/
│   ├── model.py          # VL-JEPA, TimmVisionEncoder, HFLanguageEncoder, SigLIP loss
│   ├── trainer.py        # Training loop, EMA, AMP, retrieval metrics
│   ├── dataset.py        # COCO 2017 + DistilBERT tokenizer
│   └── config.py         # YAML loader
├── experiments/
│   └── exp_jepa_training.py
├── configs/
│   ├── default.yaml
│   └── mvp_pretrained_siglip.yaml
├── tests/
│   ├── test_smoke.py
│   ├── test_dataset.py
│   └── test_alignment_overfit.py
├── INVESTIGATION.md      # Debugging history
├── SOTA_RESEARCH_GPT.md  # SOTA research findings
└── README.md
```

---

## Hardware

**Tested on**: NVIDIA RTX 3090 (24 GB), Fedora 43.

**VRAM usage**:
- Pretrained frozen (batch 8): ~1.6 GB
- Pretrained frozen (batch 16): ~3.2 GB
- With gradient checkpointing: ~1.2 GB
- Unfrozen encoders (batch 8): ~6-8 GB

---

## References

- [VL-JEPA paper](https://arxiv.org/abs/2512.10942)
- [I-JEPA](https://arxiv.org/abs/2301.08243)
- [V-JEPA](https://arxiv.org/abs/2404.08471)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [LiT](https://arxiv.org/abs/2111.07991)
- [MoCo](https://arxiv.org/abs/1911.05722)
- [VICReg](https://arxiv.org/abs/2105.04906)

---

## License

CC-BY 4.0 (matching the original paper).
