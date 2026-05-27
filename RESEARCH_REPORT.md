# VL-JEPA on Commodity Hardware: SOTA Research & Recommendations

**Date:** May 27, 2026  
**Target:** Single RTX 3090 (~10 GB usable VRAM), COCO 2017, PyTorch  
**Problem:** MSE reconstruction converges; InfoNCE / NCE@1 stuck at in-batch random baseline (~3.1%)

---

## Executive Summary

Your implementation correctly applies **I-JEPA mechanics** (block masking, EMA teacher, predictor, masked MSE) but diverges from SOTA vision–language JEPA/CLIP systems in three structural ways:

1. **Encoders are trained from scratch** — only the DistilBERT *tokenizer* is pretrained; `VisionEncoder` and `LanguageEncoder` are random 12-layer transformers (~435M params). Published JEPA and CLIP-style models almost always start from **MAE/DINO ViT** and **BERT-family** weights.
2. **Contrastive views are misaligned** — multi-crop uses a **local 96×96 masked** view for `vision_proj` (context CLS) but **global 224×224** for the teacher; alignment optimizes mismatched appearances.
3. **Metrics under-report progress** — with a 65K queue, **i2t NCE magnitude** shifts to `ln(32+65536)≈11.1`, while **NCE@1** still measures only in-batch diagonal accuracy (1/32 ≈ 3.12% random). A loss of ~4.0 vs random ~11.1 means the queue task is *partially* learned, but **pairwise image–caption matching** is not.

**Highest-impact fix:** Load pretrained ViT-B/L + DistilBERT (or SigLIP-B), freeze encoders for phase 1, train projections with **SigLIP** or CLIP-style **2-layer MLP heads**, and evaluate **full-dataset Recall@K** — not batch NCE@1 alone.

---

## 1. What SOTA VL-JEPA / JEPA / VLM Systems Do Differently

### 1.1 VL-JEPA (arXiv:2512.10942)

| Aspect | Paper / SOTA expectation | Your repo |
|--------|---------------------------|-----------|
| Vision backbone | Pretrained ViT (MAE / supervised), often **frozen** early | Custom ViT-L-scale, **random init** |
| Language | Pretrained transformer (BERT family) | Custom 12L transformer, **random init**; DistilBERT tokenizer only |
| JEPA loss | MSE in **representation space** on masked patches; EMA target | ✓ Implemented correctly |
| Alignment | Joint embedding + contrastive or sigmoid pairing | InfoNCE + MoCo queue |
| Scale | Multi-GPU, large effective batch or curated negatives | B=32, 65K FIFO queue |
| Predictor | Shallow ViT-S, **high LR** (≈20×) | ✓ 6 layers, 20× LR |
| Masking | Block mask ~75% (I-JEPA) | ✓ `block_patch_mask` |

The paper’s contribution is **combining predictive coding (JEPA) with vision–language alignment**. Reproduction without pretrained encoders on COCO alone is an order-of-magnitude harder problem than the paper’s intended setup.

**Reference:** [VL-JEPA](https://arxiv.org/abs/2512.10942) — Chen et al., Dec 2025.

### 1.2 I-JEPA (arXiv:2301.08243)

- **No contrastive loss** — only predicts masked patch representations from context.
- Vision encoder: **ViT-H/L pretrained** (MAE), often frozen; predictor trained.
- Mask ratio **0.75**, block masking, EMA momentum **0.996→1.0**.
- Trained on **ImageNet** (1.3M images), not caption pairs.
- Hardware: multi-GPU; ViT-H is heavy but encoder can stay frozen.

**Implication for you:** Your MSE path behaving like I-JEPA confirms the JEPA stack works. I-JEPA never had to solve cross-modal alignment from scratch.

**Reference:** [I-JEPA](https://arxiv.org/abs/2301.08243), [Meta code](https://github.com/facebookresearch/ijepa).

### 1.3 V-JEPA / V-JEPA 2 (arXiv:2404.08471, follow-ups)

- **Video** self-supervised prediction; ViT encoders at **1B+** params in v2.
- No language contrastive head in core V-JEPA.
- Training: thousands of GPU-hours on video.

**Implication:** V-JEPA is not a drop-in recipe for COCO caption alignment; use it only for **pretrained video ViT** ideas, not VL loss design.

**Reference:** [V-JEPA](https://arxiv.org/abs/2404.08471).

### 1.4 CLIP / SigLIP / ALIGN (contrastive VL baselines)

| Method | Loss | Batch / negatives | Pretraining |
|--------|------|-------------------|-------------|
| **CLIP** | Symmetric InfoNCE | **32K–65K** per step (distributed) | ViT + GPT/BERT on 400M–4B pairs |
| **SigLIP** | Sigmoid on all pairs | **Works well at B=256–4096**; no softmax over batch | Same scale data |
| **ALIGN** | InfoNCE | Large batch + dual encoders | ImageNet-scale web data |
| **MoCo v3** | InfoNCE | Queue + **momentum key encoder** | ImageNet |

**Implication:** InfoNCE at B=32 on COCO from scratch is the hardest point in the design space. SigLIP or pretrained + smaller aligned batch is the standard commodity-GPU path.

**References:**  
- [CLIP](https://arxiv.org/abs/2103.00020)  
- [SigLIP](https://arxiv.org/abs/2303.15343)  
- [MoCo v3](https://arxiv.org/abs/2004.05766)

### 1.5 Hardware requirements (literature vs your box)

| System | Typical hardware | Effective negatives | Pretrained? |
|--------|------------------|---------------------|-------------|
| CLIP ViT-L/14 | 256–1024× V100/A100, days | 32K+ | From scratch on web data |
| SigLIP-B/16 | TPU pods / multi-GPU | Moderate batch + sigmoid | JFT/WebLI-scale |
| I-JEPA ViT-H | 16–64 GPUs | N/A (no contrastive) | MAE init |
| **Your target** | **1× RTX 3090, ~10 GB** | 32 in-batch + 65K queue | **Must use init weights** |

---

## 2. Small-Batch Contrastive Learning (Commodity GPU)

### 2.1 Why batch 32 hurts InfoNCE

- Softmax over negatives is noisy with few in-batch negatives; gradient variance is high.
- **MoCo queue** helps **if** keys are diverse and representations are already semantic — you fixed stale-teacher keys (`8b6298d`), but **random-init** keys stay near-isotropic until encoders structure the space.
- **Gradient accumulation** increases effective batch for **t2i** only if you sync keys across micro-batches; queue already augments i2t.

### 2.2 Alternatives that work at small B

| Technique | Mechanism | Fits 10 GB? |
|-----------|-----------|-------------|
| **SigLIP** | Independent sigmoid per image–text pair; no global softmax | ✓ Best first try |
| **Memory bank (MoCo)** | FIFO negatives | ✓ You have this; needs good keys |
| **Cross-batch memory (DeCLIP, etc.)** | Store recent embeddings | ✓ Similar to your bank |
| **Grad accumulation** | B_eff = k×32 | ✓ Cheap |
| **Frozen pretrained encoders** | Smaller trainable head space | ✓ Critical |
| **Local loss + global queue** | DINO-style | Optional |
| **Hard negative mining** | Mine within queue | After warmup |

### 2.3 Is InfoNCE the right loss?

**For your constraints: prefer SigLIP-style sigmoid loss first**, then InfoNCE with queue once encoders are pretrained.

| Loss | Pros | Cons @ B=32, scratch |
|------|------|---------------------|
| **InfoNCE** | CLIP standard; well understood | Needs many clean negatives; your NCE@1 stuck |
| **SigLIP** | Stable at small/medium B; no partition function over 65K | Bias–variance tradeoff; needs bias init |
| **Triplet margin** | Simple | Hard mining fragile; slower convergence |
| **VICReg + alignment** | Anti-collapse | Does not replace semantic pairing |

**SigLIP sketch (replace cross-entropy block in `compute_jepa_loss`):**

```python
def siglip_loss(vision_proj, language_proj, logit_scale, logit_bias=0.0):
    # vision_proj, language_proj: (B, D) L2-normalized
    logits = vision_proj @ language_proj.T * logit_scale.exp() + logit_bias
    B = logits.size(0)
    labels = 2 * torch.eye(B, device=logits.device) - 1  # +1 on diag, -1 off
    return -F.logsigmoid(labels * logits).sum() / B
```

Add learnable `logit_bias` (SigLIP uses ~−10 init) per [SigLIP paper](https://arxiv.org/abs/2303.15343).

---

## 3. Root-Cause Analysis of Your Stuck NCE@1

### 3.1 What is actually working

From `INVESTIGATION.md` and `README.md`:

- MSE → 0: **predictor + EMA teacher path is correct**.
- Val NCE ~3.6–4.7 vs queue random ~11.1: model spreads language keys enough to beat **random among 65K queue entries**.
- Training stable: FP32 NCE, grad clip 2.0, student queue keys.

### 3.2 What is not working (and why)

| Issue | Evidence | Mechanism |
|-------|----------|-----------|
| **Random-init encoders** | No `from_pretrained` in repo | No semantic vision/text geometry to align |
| **View mismatch (multi-crop)** | `context_images=local`, NCE from `context_emb` CLS | Contrastive pairs **local masked** image vs caption of **global** scene |
| **MSE dominates early** | MSE → 0 by epoch 2–3 | Encoder capacity spent on patch prediction, not semantics |
| **NCE@1 metric** | Stuck at 3.12% = 1/32 | Does not measure queue retrieval; misleading with 65K bank |
| **Single linear proj** | `nn.Linear(768,768)` | Weaker than CLIP’s 2-layer MLP + norm |
| **t2i has no queue** | `compute_jepa_loss` | Asymmetric training signal |
| **Custom text encoder** | Not DistilBERT weights | Token IDs sit on random embedding table |

### 3.3 Code hotspots

**Multi-crop / NCE view mismatch** (`src/trainer.py` → `src/model.py`):

```python
# trainer: local → context, global → target_images
views = make_multicrop_views(...)
return self.model(..., context_images=views['local'], target_images=views['global'])

# model: contrastive uses CONTEXT (local) CLS
vision_cls = context_emb[:, 0, :]
vision_proj = F.normalize(self.vision_proj(vision_cls), ...)
```

**Fix:** For contrastive only, use **global unmasked** CLS (or EMA global CLS), keep local+masked for MSE.

**In-batch-only accuracy metric** (`src/model.py`):

```python
labels = torch.arange(batch_size, device=vision_proj.device)
logits_i2t = vision_proj @ lang_keys.T * scale  # lang_keys = batch + queue
nce_loss_i2t = F.cross_entropy(logits_i2t, labels)  # labels index batch cols only
nce_acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean()
```

**Fix:** Add `recall_at_k` on validation (encode full val set, cosine top-K).

---

## 4. Minimum Viable Setup (Single 3090, ~10 GB)

### 4.1 Goal definition

“Meaningful” on one GPU = **not** full CLIP-scale zero-shot, but:

- **COCO val image→text R@1 > 10%** (weak baseline; strong CLIP > 30%)
- **Monotonic drop** in SigLIP/InfoNCE on val
- **NCE@1 (in-batch) > 10%** within ~5 epochs after pretrain init

### 4.2 Recommended MVP architecture

| Component | Choice | VRAM (AMP, B=32) |
|-----------|--------|------------------|
| Vision | `timm vit_base_patch16_224.mae` or `vit_small` | ~2–4 GB frozen |
| Text | `distilbert-base-uncased` (HF, pooler/CLS) | ~1 GB frozen |
| Predictor | 4–6 layers, 384-d if ViT-S backbone | ~1 GB |
| Heads | 2-layer MLP 768→768→768 | negligible |
| Loss | Phase A: SigLIP only (5 ep) → Phase B: + JEPA MSE | — |
| Batch | 32–48, grad accum 2 if needed | ~8–10 GB |

### 4.3 Acceptable compromises

| Compromise | Accept? |
|------------|---------|
| ViT-B instead of ViT-L | ✓ Yes |
| Frozen encoders + train heads/predictor | ✓ Yes |
| COCO only (no CC12M) | ✓ For reproduction; cap R@K expectations |
| SigLIP instead of exact paper InfoNCE | ✓ For hardware |
| Shorter schedule (15–30 ep) | ✓ With pretrained init |
| Skip video / multi-image | ✓ |

### 4.4 Not acceptable if goal is paper-faithful SOTA

- Training 400M+ pairs
- ViT-H from scratch on COCO
- Batch 32 InfoNCE without pretrain

---

## 5. Lightweight Backbones

| Backbone | Params | MAE/IN1K init | COCO 1×3090 | Notes |
|----------|--------|---------------|-------------|-------|
| ViT-S/16 | ~22M | ✓ timm | ✓ Fast iteration | Best for debugging alignment |
| ViT-B/16 | ~86M | ✓ timm | ✓ Frozen + JEPA | **Recommended MVP** |
| ViT-L/16 | ~307M | ✓ | Tight if unfrozen | Your current width; freeze most layers |
| DeiT-S/B | ~22–86M | ✓ | ✓ | Good CLIP distillation targets |
| EfficientNet-B0 | ~5M | ✓ | ✓ | Weaker for JEPA patch prediction |
| DistilBERT | ~66M | ✓ HF | ✓ | Replace custom `LanguageEncoder` |
| MiniLM / TinyBERT | <30M | ✓ | ✓ | If VRAM tight |

**JEPA note:** Predictor should stay **narrower/shallower** than context encoder (you already use 6L predictor vs 12L context — good).

---

## 6. Language Encoder Role

| Setup | Convergence | Your status |
|-------|-------------|-------------|
| Random 12L transformer | Very poor cross-modal | **Current** |
| DistilBERT pretrained, frozen | Fast alignment | **Recommended** |
| BERT-base | Slightly better, +VRAM | If DistilBERT plateaus |
| CLIP text tower | Strong but couples to CLIP image space | Transfer learning path |

**DistilBERT vs BERT-base:** Marginal if both pretrained; DistilBERT wins on **speed and VRAM**. What matters is **pretrained weights**, not tokenizer alone.

**Action:** Replace `LanguageEncoder` with HuggingFace `DistilBertModel` and take `last_hidden_state[:, 0]` or mean pool.

---

## 7. Pretrained Weights — Standard Approach

### 7.1 Typical staging

```
Stage 0 (optional): MAE/DINO ViT on ImageNet  →  already in timm checkpoint
Stage 1: Freeze ViT + text, train MLP projections with SigLIP/CLIP loss (5–10 ep)
Stage 2: Unfreeze last K ViT blocks + JEPA MSE with α=0.3, β=0.7 (10–20 ep)
Stage 3 (optional): Unfreeze full encoder, low LR (1e-5)
```

### 7.2 Weight sources

| Module | Source |
|--------|--------|
| ViT | `timm.create_model('vit_base_patch16_224.mae', pretrained=True)` |
| Text | `transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')` |
| Strong baseline | OpenCLIP / SigLIP weights (sanity check pipeline) |

### 7.3 I-JEPA checkpoint transfer

If you obtain Meta I-JEPA ViT weights, load into `context_encoder` (patch embed + blocks), copy to `target_encoder`, **reset predictor**.

---

## 8. Prioritized Change List (Most Impactful First)

### P0 — Do first (likely unlocks learning)

1. **Load pretrained ViT-B/16 (MAE) + DistilBERT**; freeze both for 5 epochs.  
2. **Fix contrastive view:** `vision_proj` from **global 224 CLS**, unmasked; keep local+mask for MSE only.  
3. **Replace InfoNCE with SigLIP** (or add `logit_bias` + sigmoid); keep queue optional after baseline works.  
4. **CLIP-style projection MLP** (Linear → GELU → Linear, optional LayerNorm).  
5. **Add val Recall@1/5/10** over full COCO val (~5K); stop relying on batch NCE@1 alone.

### P1 — High value, low cost

6. **Curriculum:** epochs 1–5 `α=0, β=1`; then `α=0.3, β=0.7`.  
7. **Symmetric queue:** enqueue both `vision_proj` and `language_proj` for t2i.  
8. **Gradient accumulation** (2×) for contrastive symmetry.  
9. **2-layer text pooling:** mean pool last 4 tokens, not only index 0 (captions are short).  
10. **Learning rate:** encoders `1e-5`, heads `1e-4`, predictor `2e-4` after unfreeze.

### P2 — Tuning once signal exists

11. Tune `γ` (variance reg) down (0.01) once SigLIP prevents collapse.  
12. Memory bank 16K–65K ablation (may hurt early training).  
13. `text_mask_ratio=0.15` BERT MLM auxiliary (paper-dependent).  
14. EMA τ schedule: hold τ=0.99 until contrastive R@1 > 5%.  
15. Hard negative mining from queue.

### P3 — Scale / polish

16. ViT-L last-4-blocks unfreeze.  
17. CC3M subset or COCO+Conceptual captions.  
18. torch.compile / fused AdamW for throughput.

---

## 9. Concrete Architecture / Code Changes

### 9.1 Contrastive view fix (minimal diff)

In `VL_JEPA.forward`, branch CLS for alignment:

```python
# After context_emb and target_global_emb are computed:
# JEPA: masked local context
context_emb = self.context_encoder(context_images, patch_mask)

# Contrastive: global, unmasked (student)
with torch.set_grad_enabled(self.training):
    global_emb = self.context_encoder(target_images, mask=None)
vision_cls_for_nce = global_emb[:, 0, :]
vision_proj = F.normalize(self.vision_proj(vision_cls_for_nce), p=2, dim=-1, eps=1e-6)
```

### 9.2 Pretrained vision wrapper (timm)

```python
import timm

class TimmVisionEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224.mae", hidden_dim=768):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.hidden_dim = self.backbone.num_features  # 768 for base

    def forward(self, x, mask=None):
        # Map block_patch_mask to timm patch mask or use your patch embed path
        return self.backbone.forward_features(x)  # (B, 1+N, D)
```

Start with **frozen** `backbone` + your existing predictor on patch tokens if shapes match.

### 9.3 Pretrained language (transformers)

```python
from transformers import DistilBertModel

class HFLanguageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.hidden_dim = self.bert.config.dim  # 768

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
```

### 9.4 Projection MLP (CLIP-style)

```python
class ProjectionHead(nn.Module):
    def __init__(self, dim=768, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1, eps=1e-6)
```

Lower `out_dim` (256) saves VRAM and often helps contrastive optimization.

### 9.5 Config sketch for MVP (`configs/mvp_pretrained.yaml`)

```yaml
model:
  vision_backbone: vit_base_patch16_224.mae
  text_backbone: distilbert-base-uncased
  freeze_encoders: true
  proj_dim: 256
  mask_ratio: 0.75
  predictor_layers: 4

training:
  batch_size: 32
  grad_accum_steps: 2
  learning_rate: 1.0e-4
  encoder_lr: 0.0        # phase 1
  use_multi_crop: true
  contrastive_on_global: true
  memory_bank_size: 0     # disable until SigLIP works in-batch

loss:
  type: siglip            # or clip_infonce
  alpha: 0.0              # phase 1: alignment only
  beta: 1.0
  gamma: 0.01
```

### 9.6 Evaluation snippet (full-val R@K)

```python
@torch.no_grad()
def coco_retrieval_recall(model, val_loader, device, k_list=(1, 5, 10)):
    model.eval()
    image_feats, text_feats = [], []
    for images, input_ids, attn in val_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        v, t = model.get_joint_embedding(images, input_ids)
        image_feats.append(v.cpu())
        text_feats.append(t.cpu())
    V = torch.cat(image_feats)  # (N, D)
    T = torch.cat(text_feats)
    sim = V @ T.T
    ranks = sim.argsort(dim=1, descending=True)
    gt = torch.arange(V.size(0))
  i2t_r1 = (ranks[:, :1] == gt.unsqueeze(1)).any(dim=1).float().mean().item()
    # ... R@5, R@10, t2i symmetric
    return {"i2t_r1": i2t_r1, ...}
```

---

## 10. References & Repositories

### Papers

| ID | Title | Link |
|----|-------|------|
| VL-JEPA | Vision-Language JEPA (target paper) | https://arxiv.org/abs/2512.10942 |
| I-JEPA | Self-supervised vision JEPA | https://arxiv.org/abs/2301.08243 |
| V-JEPA | Video JEPA | https://arxiv.org/abs/2404.08471 |
| CLIP | Contrastive VL | https://arxiv.org/abs/2103.00020 |
| SigLIP | Sigmoid loss for VL | https://arxiv.org/abs/2303.15343 |
| MAE | Masked autoencoder ViT init | https://arxiv.org/abs/2111.06377 |
| MoCo v3 | Momentum contrast | https://arxiv.org/abs/2004.05766 |
| VICReg | Variance regularization | https://arxiv.org/abs/2105.04906 |

### Code

| Project | URL | Use |
|---------|-----|-----|
| This repo | `vl-jepa/` | Your JEPA + contrastive stack |
| I-JEPA | https://github.com/facebookresearch/ijepa | Masking, EMA, predictor LR |
| OpenCLIP | https://github.com/mlfoundations/open_clip | SigLIP/CLIP losses, pretrained weights |
| timm | https://github.com/huggingface/pytorch-image-models | MAE ViT checkpoints |
| HuggingFace transformers | https://github.com/huggingface/transformers | DistilBERT |
| Lightly (MoCo/SimCLR) | https://github.com/lightly-ai/lightly | Memory bank patterns |

---

## 11. Decision Matrix: InfoNCE vs SigLIP vs JEPA-only

| If your goal is… | Use |
|------------------|-----|
| Reproduce **paper** on 1 GPU | Pretrained encoders + paper loss weights + global CLS for NCE |
| **Debug** contrastive quickly | SigLIP, frozen ViT-B + DistilBERT, no queue |
| **Max R@K** on COCO val | OpenCLIP/SigLIP baseline, then add JEPA MSE fine-tune |
| Prove JEPA stack only | I-JEPA on ImageNet subset (no language) |

---

## 12. Summary Table: Your Setup vs Recommended MVP

| Item | Current | Recommended MVP |
|------|---------|-----------------|
| Vision init | Random | timm MAE ViT-B/16 |
| Text init | Random | HF DistilBERT |
| Contrastive loss | InfoNCE + 65K queue | SigLIP → then optional InfoNCE |
| NCE view | Local masked CLS | Global unmasked CLS |
| Projection | Linear 768→768 | MLP 768→256 |
| Metric | Batch NCE@1 | Full val R@1/5/10 |
| Phase 1 loss | α=0.5, β=0.5 | α=0, β=1 (5 ep) |
| Memory bank | 65K student lang keys | Off until in-batch works |

---

## Appendix A: Why queue helped NCE magnitude but not NCE@1

- **i2t cross-entropy** with 65K negatives: easy to beat uniform random (your ~4 vs ~11).  
- **Positive column** is still index `i` in the **first 32 columns** — model must beat 31 in-batch hard positives + 65K distractors. Random-init vision/text are not co-linear with caption semantics → argmax rarely hits `i`.  
- **NCE@1** only checks argmax among all keys; with untrained alignment, chance stays **1/32** for the in-batch positive block dominating argmax among semantically similar random features.

Once pretrained, in-batch positives share structure with captions; **NCE@1 should rise within epochs** without needing 65K queue.

---

## Appendix B: Suggested experiment protocol (reproducible)

1. **Baseline A:** Frozen ViT-B + DistilBERT, SigLIP only, global CLS, 10 epochs → log val R@1.  
2. **Baseline B:** Same + OpenCLIP ViT-B/16 weights (sanity: pipeline should hit known R@K ballpark).  
3. **+JEPA:** Enable MSE α=0.3 after R@1 > 5%.  
4. **Ablate:** InfoNCE vs SigLIP, queue on/off, local vs global CLS.  
5. Document in `INVESTIGATION.md` with val R@K table, not only NCE loss.

---

*Report generated for the vl-jepa reproduction effort. Cross-check hyperparameters against the official VL-JEPA paper appendix when available; arXiv:2512.10942 was not fetched in-session due to tool restrictions.*
