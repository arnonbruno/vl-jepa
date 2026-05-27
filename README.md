# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

A PyTorch implementation of [VL-JEPA](https://arxiv.org/abs/2512.10942) — **properly fixed** with the correct JEPA training loop.

**Authors**: Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung
**Implementation by**: TARS + Bruno Santos

---

## Architecture (Fixed v2)

The original implementation had a critical bug: `vision_mask`/`language_mask` tensors were being used as **target labels** for CrossEntropyLoss instead of actual masks. The loss of ~19 was the **random guessing baseline** (`ln(8192) + ln(30522)`), not meaningful convergence.

### v2 — Proper JEPA Training Loop

```
Input Images (B,3,224,224)          Input Text (B,seq_len)
        ↓                                      ↓
  [75% Random Patch Mask]             [15% Token Mask (BERT-style)]
        ↓                                      ↓
  Context Encoder (student)           Language Encoder
  (processes masked image)            (processes masked text)
        ↓                                      ↓
  ┌──── Predictor (6-layer Transformer) ────┐
  │  Predicts target patch embeddings        │
  └────────────────┬────────────────────────┘
                   ↓
          L = α · MSE(predicted, target)     ← I-JEPA style
                + β · InfoNCE(vision, text)   ← VL-JEPA style

  Target Encoder (teacher): EMA copy of Context Encoder (τ = 0.996→1.0)
  Full image → Target Encoder → Target embeddings (stop-gradient)
```

### Key Components

| Component | Description | Params |
|-----------|-------------|--------|
| Context Encoder | 12-layer ViT, processes masked images | ~86M |
| Target Encoder | EMA copy of Context (no grad) | ~86M (shared weights) |
| Predictor | 6-layer Transformer, predicts target embeddings | ~43M |
| Language Encoder | 12-layer Transformer, processes text | ~86M |
| Projection Heads | vision_proj + language_proj for InfoNCE | ~1M |

### Loss Function

```python
L = α · L_mse + β · L_nce

L_mse:  MSE(predicted_patches, target_patches)   # only on masked positions
L_nce:  symmetric InfoNCE(vision_cls, text_cls)   # cross-modal alignment

α = 1.0   (MSE weight)
β = 0.5   (InfoNCE weight)
```

---

## Setup

### Prerequisites
- Python 3.11+
- PyTorch 2.1+ with CUDA 12.1
- 24GB+ GPU VRAM recommended (tested on RTX 3090)

### Installation

```bash
conda create -n vl-jepa python=3.11 -y
conda activate vl-jepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy matplotlib tensorboard scikit-learn pillow tqdm
git clone https://github.com/arnonbruno/vl-jepa.git
cd vl-jepa
```

---

## Running

### 1. Smoke Tests (Validation)

```bash
python tests/test_smoke.py
```

Expected output:
```
✅ Vision Encoder with Mask: PASSED
✅ Language Encoder: PASSED
✅ Predictor Module: PASSED
✅ VL-JEPA Forward + Loss: PASSED
✅ Momentum Update: PASSED
✅ GPU Forward + Backward: PASSED
✅ Joint Embedding: PASSED
✅ All tests passed!
```

### 2. Training (with GPU)

```bash
# Quick test (3 epochs, synthetic data)
python experiments/exp_jepa_training.py --epochs 3 --batch-size 32 --samples 1000

# Full training (50 epochs, synthetic data)
python experiments/exp_jepa_training.py --epochs 50 --batch-size 32 --samples 5000

# With real data
python experiments/exp_jepa_training.py --epochs 50 --batch-size 32 --data-dir /path/to/images
```

### 3. Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 15 | Training epochs |
| `--batch-size` | 32 | Batch size (RTX 3090 max: 64) |
| `--lr` | 3e-4 | Peak learning rate |
| `--samples` | 5000 | Training samples |
| `--hidden-dim` | 768 | Model dimension |
| `--data-dir` | None | Real image directory |
| `--alpha` | 1.0 | MSE loss weight |
| `--beta` | 0.5 | InfoNCE loss weight |
| `--predictor-layers` | 6 | Predictor depth |

---

## Experiment Results (v2 — RTX 3090)

### Quick Validation (3 epochs, 1K samples, B=32)

| Epoch | Train Loss | MSE (pred) | NCE (InfoNCE) | Val Loss | GPU Mem |
|-------|-----------|-----------|----------------|---------|---------|
| 1 | 1.7555 | 0.0073 | 3.4963 | 1.7357 | 10.26 GB |
| 2 | 1.7417 | 0.0044 | 3.4746 | 1.7351 | 10.26 GB |
| 3 | 1.7404 | 0.0042 | 3.4723 | 1.7351 | 10.27 GB |

**Key observations** (on synthetic data):
- **MSE decreasing** consistently (0.0073 → 0.0042): predictor learning to reconstruct patches ✓
- **NCE stable** at ~3.47: InfoNCE random baseline for B=32 is `ln(32) = 3.47` — expected on random data ✓
- **Gradient norm decreasing** (5.06 → 1.08): training stabilizing ✓
- **GPU utilization**: 10.27 GB / 25.3 GB (40%) — room to increase batch size ✓
- **324.8M params** total (239.0M trainable) ✓

**With real image-text data**, the NCE loss should decrease below `ln(batch_size)` as cross-modal alignment improves.

---

## Training Tips for RTX 3090

- **Batch size**: 32 with 224×224 images and 128 tokens uses ~10.3 GB. Can increase to 64.
- **AMP**: Automatic Mixed Precision (FP16) enabled by default — reduces memory by ~40%.
- **Gradient clipping**: `max_grad_norm=1.0` (configurable; stabilizes student-student contrastive grads).
- **Predictor LR**: 20× higher than encoder LR (I-JEPA practice).
- **EMA momentum**: Cosine schedule 0.996 → 1.0 over training.

---

## Project Structure

```
vl-jepa/
├── src/
│   ├── model.py          # VL-JEPA (fix v2): encoders, predictor, loss
│   ├── trainer.py        # Training loop with JEPA loss, EMA, AMP
│   └── __init__.py
├── tests/
│   └── test_smoke.py     # 7 tests: masking, forward, GPU, gradients
├── experiments/
│   └── exp_jepa_training.py  # Main training entry point
├── configs/
│   └── default.yaml      # (WIP)
└── README.md
```

---

## Changes from v1 (Bug Fixes)

| Issue | v1 (broken) | v2 (fixed) |
|-------|-------------|------------|
| Loss function | CrossEntropy on random targets | MSE + InfoNCE on masked positions |
| Masking | No actual masking | 75% random patch mask + 15% token mask |
| Target encoder | None | EMA copy of context encoder (stop-grad) |
| Predictor | None (FF layers only) | 6-layer transformer predictor |
| Joint projection | Defined but unused | Used in InfoNCE loss |
| Hardcoded paths | `/home/ulluboz/.openclaw/...` | Relative paths via `pathlib` |
| GPU utilization | 99% on RTX 5080 | 10.3 GB on RTX 3090 (batch 32) |

---

## Next Steps

1. **Real data**: Train on COCO / Flickr30K / CC12M for meaningful cross-modal alignment
2. **Long-form training**: 100-200 epochs with cosine LR decay
3. **Downstream eval**: Zero-shot retrieval, image-text matching
4. **Scaling**: Increase batch size to 64-128 for better InfoNCE
5. **Multi-GPU**: Distributed Data Parallel (DDP)

---

## Hardware

**Tested on**:
- GPU: NVIDIA GeForce RTX 3090 (24 GB VRAM)
- CPU: AMD Ryzen 9 9950X (16 cores)
- RAM: 31 GB
- OS: Linux Fedora 43

---

## References

- **Paper**: [VL-JEPA: Joint Embedding Predictive Architecture for Vision-language](https://arxiv.org/abs/2512.10942)
- **I-JEPA**: [Self-Supervised Learning from Images with JEPA](https://arxiv.org/abs/2301.08243)
- **V-JEPA**: [Revisiting JEPA for Visual Representations](https://arxiv.org/abs/2404.08471)
- **CLIP**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

---

## License

CC-BY 4.0 (matching original paper)