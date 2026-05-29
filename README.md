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

This implementation uses **pretrained frozen encoders** as a starting point, with **phased training** that gradually introduces JEPA reconstruction after contrastive alignment is established. Two encoder paths are supported:

- **Hybrid (timm CLIP ViT-B/16 + DistilBERT)** — the original path. The vision tower is multimodally-aligned, but the DistilBERT text CLS was never trained for sentence retrieval, which caps fine-grained ranking.
- **End-to-end OpenCLIP (`configs/openclip_vitb16.yaml`)** — both towers come from CLIP's 400M-pair pretraining, so vision and text are *already aligned*. The caption pipeline automatically switches to CLIP's BPE tokenizer (vocab 49408, SOT/EOT) and CLIP image normalization — feeding DistilBERT ids into the CLIP text tower would silently destroy the alignment.

### Breaking the ~25% R@1 ceiling

Two earlier runs (timm CLIP + DistilBERT) both plateaued at ~25% R@1 despite R@5≈52% / R@10≈66% and 78–92% in-batch NCE@1 — the right answer was in the top-10 but never ranked #1. That signature (in-batch accuracy saturated, global Recall@1 stuck) points at the *encoders* and the *negatives*, not optimization. The fixes:

1. **OpenCLIP end-to-end** — replaces the weak DistilBERT CLS with CLIP's aligned text tower (the largest structural lever). Enabled by the tokenizer fix above.
2. **Hard-negative mining** (`loss.hard_negative_weight`) — a VSE++ max-violation hinge that only penalizes the single hardest in-batch negative, directly sharpening top-1 once in-batch accuracy saturates.
3. **More data** (`src/image_text_dataset.py`, `experiments/download_cc3m.py`) — infrastructure to pretrain on CC3M/CC12M before fine-tuning on COCO (the highest long-term lever; COCO's 118K images are seen ~50× per run).
4. **Higher resolution** — set `model.image_size: 384`; the OpenCLIP vision tower interpolates its positional embeddings automatically.

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
L = α · L_mse + β · L_siglip + γ · L_var + δ · L_hardneg

L_mse:     MSE(predicted_patches, target_patches) on masked positions only
L_siglip:  pairwise sigmoid contrastive (no softmax dependency on batch size)
L_var:     VICReg variance on vision_proj_raw and language_proj_raw
L_hardneg: VSE++ max-violation hinge on the hardest in-batch negative (δ defaults
           to 0; set loss.hard_negative_weight > 0 to sharpen Recall@1)
```

### Phased Training

| Phase | Epochs | α (MSE) | β (SigLIP) | γ (Var) | What happens |
|-------|--------|---------|------------|---------|--------------|
| A | 1-3 | 0.0 | 1.0 | 0.01 | Alignment only, frozen encoders |
| B | 4+ | 0.1 | 0.9 | 0.01 | Add gentle JEPA MSE; unfreeze last 4 vision blocks (epoch 5, `encoder_unfreeze_lr=2e-5`) |

The contrastive head also uses SigLIP label smoothing (`loss.label_smoothing`, default
0.05) and the train image pipeline adds colour jitter / grayscale / random erasing to
curb overfitting. The training micro-batch is 128 (× 2 grad-accum = effective 256); a
larger *real* batch is what increases SigLIP in-batch negatives, since accumulation
averages independent per-micro-batch sigmoid losses.

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

### End-to-end OpenCLIP (recommended for breaking the ceiling)

```bash
python experiments/exp_jepa_training.py \
  --config configs/openclip_vitb16.yaml \
  --epochs 30 \
  --fresh
```

This uses aligned CLIP vision+text towers, the CLIP BPE tokenizer (selected
automatically from `text_backbone: openclip`), and hard-negative mining
(`hard_negative_weight: 0.2`). Add `--hard-negative-weight 0` to ablate it.

### Pretraining on CC3M before COCO

```bash
# 1. Fetch the caption/URL TSV (small) and print the img2dataset command (images)
python experiments/download_cc3m.py download-tsv --dataset cc3m --out data/cc3m
python experiments/download_cc3m.py make-img2dataset --tsv data/cc3m/cc3m.tsv --out data/cc3m/images
# 2. After img2dataset finishes, build the <image>\t<caption> manifest
python experiments/download_cc3m.py build-manifest --images data/cc3m/images --out data/cc3m/train.tsv
```

`src.image_text_dataset.ImageTextPairDataset` then trains on the manifest with
the same model/trainer code (shares the caption tokenizer + image transforms).

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
| `--unfreeze-after-epoch` | 5 | Epoch to unfreeze the last 4 vision blocks (`encoder_unfreeze_lr=2e-5`) |
| `--label-smoothing` | 0.05 | SigLIP target smoothing (anti-overfit) |
| `--hard-negative-weight` | 0.0 | VSE++ hardest-negative ranking weight (δ); sharpens Recall@1 |
| `--hard-negative-margin` | 0.2 | Margin for the hard-negative hinge |
| `--text-backbone openclip` | — | Use CLIP's aligned text tower + BPE tokenizer |
| `--resume` | — | Resume from checkpoint |
| `--fresh` | — | Start from scratch |

---

## Project structure

```
vl-jepa/
├── src/
│   ├── model.py              # VL-JEPA, Timm/OpenCLIP encoders, SigLIP + VSE++ losses
│   ├── trainer.py            # Training loop, EMA, AMP, retrieval metrics
│   ├── dataset.py            # COCO 2017 + DistilBERT/CLIP tokenizer (CaptionTokenizer)
│   ├── image_text_dataset.py # Generic CC3M/CC12M manifest dataset
│   ├── cached_dataset.py     # Precomputed frozen-encoder embeddings
│   └── config.py             # YAML loader
├── experiments/
│   ├── exp_jepa_training.py
│   └── download_cc3m.py      # CC3M/CC12M TSV + manifest preparation
├── configs/
│   ├── default.yaml
│   ├── mvp_pretrained_siglip.yaml
│   └── openclip_vitb16.yaml  # end-to-end OpenCLIP backbone
├── tests/
│   ├── test_smoke.py
│   ├── test_dataset.py
│   ├── test_image_text_dataset.py
│   ├── test_cached_dataset.py
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
