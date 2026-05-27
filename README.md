# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

PyTorch implementation of [VL-JEPA](https://arxiv.org/abs/2512.10942) — a JEPA-style vision-language model with masked patch prediction, EMA target encoders, and MoCo-style contrastive alignment.

**Paper authors**: Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung  
**Implementation**: TARS + Bruno Santos

---

## Overview

VL-JEPA learns joint vision-language representations by:

1. **Predicting** masked patch embeddings (I-JEPA style MSE against an EMA teacher).
2. **Aligning** image and text in a shared projection space (InfoNCE with a 65K memory bank).
3. **Regularizing** embedding variance (VICReg-style) to prevent collapse.

The contrastive stack was rebuilt through May 2026 after training was stable but NCE stayed at the random baseline (`ln(32) ≈ 3.47`). The current design uses **student-to-student** contrastive learning, **context-encoder CLS** queries, a **MoCo FIFO queue** of student language keys, **10× projection-head LR**, and **variance regularization**.

---

## Architecture (as of `8b6298d`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING FORWARD PASS                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Global + local crops (multi-crop)          Caption tokens (DistilBERT tokenizer)
           │                                              │
           ▼                                              ▼
  ┌────────────────────┐                       ┌────────────────────┐
  │  Context encoder   │  75% block-masked     │  Language encoder  │  12-layer Transformer
  │  (ViT-Large scale) │  local/global views   │  (BERT-base scale) │  vocab 30522, 768-dim
  │  12L, 768-dim      │                       └─────────┬──────────┘
  └─────────┬──────────┘                                 │
            │                                            │
            │         ┌────────────────────┐             │
            └────────►│  Predictor         │             │
                      │  (ViT-Small scale) │             │
                      │  6L transformer    │             │
                      └─────────┬──────────┘             │
                                │                        │
            MSE (masked patches)│                        │
                                ▼                        ▼
  ┌────────────────────┐              ┌──────────────────────────────────┐
  │  Target encoder    │◄── EMA τ ────│  vision_proj / language_proj     │
  │  (frozen teacher)  │   0.996→1.0  │  768 → 768, L2-normalized        │
  │  full-image patches│              │  Context CLS → vision_proj       │
  └────────────────────┘              └───────────────┬──────────────────┘
                                                      │
                         ┌────────────────────────────┼────────────────────────────┐
                         │  InfoNCE (FP32)            │                            │
                         │  • i2t: vision → lang keys │  Memory bank (MoCo)        │
                         │    (batch + 65K FIFO queue)│  65,536 student lang keys  │
                         │  • t2i: language → vision  │  (detached, normalized)    │
                         │    (in-batch only)         │                            │
                         └────────────────────────────┴────────────────────────────┘
                                                      │
                         L = α·MSE + β·InfoNCE + γ·VarReg
```

### Components

| Component | Role | Config default |
|-----------|------|----------------|
| **Context encoder** | Student ViT; encodes masked/cropped images | 12 layers, 768-dim, patch 16 |
| **Target encoder** | EMA copy of context encoder; stop-gradient patch targets | Same weights, `τ` cosine 0.996→1.0 |
| **Predictor** | ViT-Small-scale transformer; predicts teacher patch embeddings | 6 layers |
| **Language encoder** | Text transformer (DistilBERT tokenizer, custom weights) | 12 layers, 768-dim |
| **Projection heads** | `vision_proj`, `language_proj` → joint 768-d space | **10× base LR** |
| **Memory bank** | FIFO queue of student `language_proj` for i2t negatives | 65,536 keys |
| **Variance reg** | VICReg-style std ≥ 1 on pre-norm projections | `γ = 0.1` |

**~435M parameters** (~239M trainable) on COCO 2017 at 224×224, batch 32.

### Loss

```python
L = α · L_mse + β · L_nce + γ · L_var

L_mse:  MSE(predicted_patches, target_patches) on masked positions only
L_nce:  symmetric InfoNCE — i2t uses batch + queue keys; t2i uses in-batch vision
L_var:  VICReg variance on vision_proj_raw and language_proj_raw

α = 0.5    β = 0.5    γ = 0.1    (configs/default.yaml)
```

InfoNCE runs in **FP32** under AMP to avoid overflow. Gradients are clipped at **`max_grad_norm = 2.0`**.

### Key design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Student-to-student** contrastive (teacher projections detached for NCE) | Student–teacher NCE gave no gradient signal; EMA teacher too close to student |
| 2 | **Context encoder CLS** for vision queries (not predictor CLS) | Predictor CLS is reconstruction-focused, not aligned for retrieval |
| 3 | **Student `language_proj`** in memory bank (not teacher) | Teacher at `τ=0.996` is stale; queue of teacher keys collapsed to ~identical negatives |
| 4 | **Projection LR = 10×** base (predictor = 20×) | Contrastive heads need faster adaptation than encoders |
| 5 | **Variance regularization** | Prevents embedding collapse without removing contrastive signal |
| 6 | **Memory bank size 65,536** | Batch 32 alone is too few negatives for stable InfoNCE |

---

## Setup

### Prerequisites

- Python 3.11+
- PyTorch 2.1+ with CUDA 12.1
- 24 GB+ GPU VRAM (tested on RTX 3090)

### Installation

```bash
conda create -n vl-jepa python=3.11 -y
conda activate vl-jepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
git clone https://github.com/arnonbruno/vl-jepa.git
cd vl-jepa
```

---

## Training

### Smoke tests

```bash
python tests/test_smoke.py
```

### COCO 2017 (full pipeline)

```bash
# Default config (15 epochs, memory bank, multi-crop)
python experiments/exp_jepa_training.py

# Long run with explicit settings
python experiments/exp_jepa_training.py \
  --config configs/default.yaml \
  --epochs 50 \
  --batch-size 32 \
  --coco-root ~/.cache/torch/hub/checkpoints

# Resume from checkpoint
python experiments/exp_jepa_training.py --resume experiments/exp_jepa_768d_50ep/checkpoint_latest.pt
```

### CLI highlights

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML training config |
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Peak learning rate (encoders) |
| `--alpha` / `--beta` / `--gamma` | 0.5 / 0.5 / 0.1 | MSE / NCE / variance weights |
| `--resume` | — | Resume from checkpoint |
| `--nan-diagnostics-dir` | — | Dump tensors on NaN batches |

---

## Results (RTX 3090, COCO 2017)

### Reconstruction (MSE)

MSE on masked patches converges within the first 1–2 epochs (typically &lt; 0.01 train MSE).

### Contrastive (InfoNCE) — evolution

| Stage | Val NCE | Val NCE@1 | Notes |
|-------|---------|-----------|-------|
| Pre-fix (student–teacher NCE, B=32) | ~3.46 | ~3.18% | Stuck at `ln(32)` random baseline |
| + Student NCE + context CLS + var reg | ~3.46 | ~3.18% | Stable but still at in-batch random baseline |
| + Memory bank (65K), teacher keys (bug) | climbed to ~7.27 | degraded | Stale identical queue negatives |
| **+ Student queue keys (`8b6298d`)** | **~3.64 best** (ep 7) | ~3.18% | 50-epoch run `exp_jepa_768d_50ep` |

With a 65K queue, i2t NCE magnitude is **not** comparable to the old `ln(32)` baseline; monitor **NCE@1** and val loss trends instead of absolute NCE alone.

### Example: 19-epoch run (`experiments/exp_jepa_768d_50ep`)

| Epoch | Train loss | Val loss | Val NCE | Val NCE@1 | GPU |
|-------|------------|----------|---------|-----------|-----|
| 1 | 2.16 | 2.35 | 4.49 | 3.18% | 7.3 GB |
| 7 | 2.16 | **1.92** | **3.64** | 3.18% | 7.3 GB |
| 19 | 2.11 | 2.04 | 3.89 | 1.59% | 7.3 GB |

Training is **stable** (no NaN) with FP32 InfoNCE, capped GradScaler growth, and `max_grad_norm=2.0`. Contrastive alignment remains an active tuning area.

---

## Project structure

```
vl-jepa/
├── src/
│   ├── model.py          # VL-JEPA, MemoryBank, compute_jepa_loss
│   ├── trainer.py        # Training loop, EMA, AMP, memory bank enqueue
│   ├── dataset.py        # COCO 2017 + DistilBERT tokenizer
│   └── config.py         # YAML loader
├── experiments/
│   └── exp_jepa_training.py
├── configs/
│   └── default.yaml
├── tests/
│   ├── test_smoke.py
│   └── test_dataset.py
├── INVESTIGATION.md      # Contrastive debugging timeline
└── README.md
```

---

## Key commits (contrastive fixes)

| Commit | Summary |
|--------|---------|
| `c6caa1e` | Fix NaN from AMP GradScaler growth + EMA schedule |
| `93491ad` | InfoNCE in FP32; NaN diagnostics |
| `5243a7d` | Student embeddings for InfoNCE (not detached teacher) |
| `f88da89` | Configurable gradient clipping |
| `a712827` | `max_grad_norm` → 2.0 |
| `319dd72` | VICReg-style variance regularization (`γ=0.1`) |
| `adc126b` | Context encoder CLS for contrastive (not predictor CLS) |
| `bae0884` | Projection head LR 10× base |
| `08927de` | MoCo memory bank (65,536 language keys) |
| **`8b6298d`** | **Enqueue student `language_proj` in queue (fix stale teacher keys)** |

---

## Hardware

**Tested on**: NVIDIA RTX 3090 (24 GB), AMD Ryzen 9 9950X, 31 GB RAM, Fedora 43.

**Tips**:

- Batch 32 @ 224×224 + multi-crop: ~7–10 GB VRAM with AMP.
- Predictor LR 20×, projection LR 10×, encoder LR 1× (see `src/trainer.py`).
- EMA `τ` reaches 1.0 early (~epoch 4) via capped cosine schedule.

---

## References

- [VL-JEPA paper](https://arxiv.org/abs/2512.10942)
- [I-JEPA](https://arxiv.org/abs/2301.08243)
- [V-JEPA](https://arxiv.org/abs/2404.08471)
- [MoCo](https://arxiv.org/abs/1911.05722)
- [VICReg](https://arxiv.org/abs/2105.04906)

## License

CC-BY 4.0 (matching the original paper).
