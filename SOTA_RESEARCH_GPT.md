# SOTA Analysis for VL-JEPA Reproduction on RTX 3090

Date: 2026-05-27

Target hardware: single RTX 3090 with a conservative 10 GB usable VRAM budget.

Scope: `src/model.py`, `src/trainer.py`, `src/dataset.py` (`src/data.py` is absent), `src/config.py`, `configs/default.yaml`, `experiments/exp_jepa_training.py`, `INVESTIGATION.md`, `README.md`, and `tests/`.

## 1. Executive Summary

The current code has a stable JEPA-style masked-patch training loop, but it is not a faithful or practical reproduction path for SOTA VL-JEPA. The biggest gap is not the memory bank: the model is trying to align a randomly initialized ViT-like image encoder and a randomly initialized BERT-like text encoder on COCO with batch 32. Published VL-JEPA, V-JEPA, I-JEPA, CLIP, SigLIP, and LiT-style systems rely on pretrained visual and/or text encoders, much larger data, much larger effective batches, or sigmoid losses that reduce the global-softmax dependency.

Top recommendations:

1. Replace random encoders with pretrained backbones and freeze them first.
   - Use `timm` MAE/DINO/DeiT ViT-B or ViT-S for vision and HuggingFace `DistilBertModel` for text.
   - Expected NCE@1 impact: from random baseline around 3.1% to roughly 8-20% in-batch within a few epochs if the retrieval pipeline is correct.
   - VRAM: lower than current full training when frozen; viable at batch 16-32 under 10 GB.

2. Fix the contrastive view: do not use local masked context CLS for image-text alignment.
   - Current multi-crop path aligns a 96x96 local masked image representation to the caption of the full image.
   - Compute `vision_proj` from a global unmasked student view; keep local masked views for JEPA MSE.
   - Expected NCE@1 impact: moderate to high, likely +2-8 points once encoders are pretrained.
   - VRAM: extra global student forward costs memory if trainable; cheap if frozen or wrapped in gradient checkpointing.

3. Add SigLIP loss and run it before queue-based InfoNCE.
   - SigLIP's pairwise sigmoid loss is specifically stronger than softmax CLIP at smaller batch sizes below about 16k.
   - Start with in-batch SigLIP, no queue, `alpha=0`, `beta=1`, frozen encoders.
   - Expected NCE@1 impact: high for commodity hardware, likely the fastest path above random.
   - VRAM: lower than 65K softmax queue logits; no large queue needed for the first successful baseline.

4. Add full COCO retrieval Recall@K and a tiny overfit test.
   - Batch NCE@1 is not enough, especially with a 65K queue and ambiguous COCO captions.
   - Full-val image-to-text/text-to-image Recall@1/5/10 and "overfit 128 pairs" should become gating metrics.
   - Expected NCE@1 impact: indirect, but prevents tuning the wrong signal.
   - VRAM: evaluation can run on CPU-stored embeddings; negligible.

5. Stage training instead of optimizing JEPA and alignment from scratch at the same time.
   - Phase A: frozen encoders, global CLS, SigLIP only.
   - Phase B: add projection MLP, unfreeze last 2-4 blocks, keep encoder LR tiny.
   - Phase C: re-enable JEPA MSE with a smaller weight and optionally reintroduce memory queues.
   - Expected NCE@1 impact: high because it removes the early MSE-vs-alignment conflict.
   - VRAM: controllable via frozen encoders, smaller projection dimension, and batch 16-32.

## 2. What Is Wrong in the Current Implementation

### 2.1 The model is random-init at SOTA scale

`src/model.py` defines custom `VisionEncoder` and `LanguageEncoder` classes. Both use randomly initialized transformer stacks. The dataset uses a pretrained DistilBERT tokenizer, but the actual language model is a random `nn.Embedding` plus 12 random transformer layers.

That makes the contrastive task far harder than CLIP-style pretraining on COCO should be. COCO has about 118k training images, which is tiny compared with the data scale used by CLIP, SigLIP, LiT, and VL-JEPA. With random image and text encoders, a batch size of 32 has too little signal to create semantic geometry before the model finds easy non-semantic shortcuts.

Exact change:

File: `src/model.py`

Replace or wrap `LanguageEncoder` with a pretrained HuggingFace model:

```python
from transformers import DistilBertModel


class HFLanguageEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.hidden_dim = self.bert.config.dim

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state
```

Add a pretrained vision option. If adding `timm` is acceptable:

```python
import timm


class TimmVisionEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224.mae"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.hidden_dim = self.backbone.num_features
        self.patch_size = 16
        self.image_size = 224
        self.grid_size = 14
        self.num_patches = 196

    def forward(self, x, mask=None):
        # First baseline: ignore JEPA mask for contrastive global features.
        # Keep the existing custom VisionEncoder for masked JEPA until this is integrated.
        feats = self.backbone.forward_features(x)
        return feats
```

Dependency note: add `timm` to `requirements.txt` or keep this path behind an optional import until the pretrained-backbone baseline is implemented.

Why it helps:

Pretrained encoders already place images and words in semantically useful latent spaces. Contrastive learning then tunes the bridge instead of discovering vision, language, and alignment simultaneously.

Expected NCE@1 impact:

High. If the rest of the pipeline is correct, batch NCE@1 should move above 3.1% quickly. A realistic first target is 8-20% in-batch NCE@1 and COCO val i2t R@1 above 5-10%, not SOTA CLIP numbers.

VRAM:

Frozen DistilBERT plus frozen ViT-B should fit under 10 GB at batch 16-32 with AMP. If unfreezing, use batch 8-16, gradient checkpointing, and encoder LR around `1e-5`.

### 2.2 The contrastive view is wrong under multi-crop

`configs/default.yaml` enables multi-crop:

```yaml
training:
  use_multi_crop: true
  global_crop_size: 224
  local_crop_size: 96
```

`src/trainer.py` sends the local crop as `context_images` and the global crop as `target_images`:

```python
return self.model(
    views['global'],
    input_ids,
    attention_mask,
    context_images=views['local'],
    target_images=views['global'],
    mask_seed=mask_seed,
)
```

Then `src/model.py` uses the masked local context CLS for image-text contrastive alignment:

```python
vision_cls = context_emb[:, 0, :]
vision_proj_raw = self.vision_proj(vision_cls)
```

This means the text caption for the full COCO image is matched against a 96x96 local masked crop, often missing the captioned object. That is a strong explanation for random NCE@1 even after queue fixes.

Exact change:

File: `src/model.py`, function `VL_JEPA.forward`.

Return a separate global student representation for NCE:

```python
# Existing: local masked context for JEPA MSE.
context_emb = self.context_encoder(context_images, patch_mask)

# Existing: teacher targets.
with torch.no_grad(), _temporarily_eval(*teacher_modules):
    target_emb = self.target_encoder(context_images, mask=None)
    target_global_emb = self.target_encoder(target_images, mask=None)

# New: global unmasked student view for contrastive alignment.
# If encoders are frozen in phase A, this is cheap and stable.
global_student_emb = self.context_encoder(target_images, mask=None)

predicted = self.predictor(context_emb)

# Use global full-image CLS for NCE, not local masked CLS.
vision_cls = global_student_emb[:, 0, :]
language_cls = language_emb[:, 0, :]
```

Optional lower-VRAM variant for phase A:

```python
# If context_encoder is frozen during contrastive warmup:
with torch.no_grad():
    global_student_emb = self.context_encoder(target_images, mask=None)
vision_cls = global_student_emb[:, 0, :].detach()
```

Why it helps:

Captions describe the full image, not an arbitrary local crop. SOTA multi-crop self-distillation uses local crops for view consistency, but image-text contrastive alignment usually uses global image features or treats local crops carefully as extra positives.

Expected NCE@1 impact:

Moderate to high. Alone, with random encoders, it may not rescue training. With pretrained encoders, it should be one of the highest-impact fixes.

VRAM:

One extra global vision forward. If frozen, negligible activation memory. If trainable, reduce batch size or checkpoint the context encoder.

### 2.3 InfoNCE is the wrong first loss for batch 32 from scratch

The current `compute_jepa_loss()` uses symmetric CLIP-style cross entropy plus a 65K language queue for image-to-text. This is valid in principle, but it is not the most robust path on a single 10 GB GPU.

SOTA evidence:

- CLIP-style InfoNCE depends heavily on large effective batches or many clean negatives.
- SigLIP replaces the softmax with independent pairwise sigmoid terms and is reported to outperform softmax CLIP at smaller batch sizes below about 16k.
- OpenCLIP recommends gradient accumulation for larger effective batches, but that is slower and requires careful feature caching if the loss should see accumulated negatives.

Exact change:

File: `src/model.py`

Add a learnable bias to `VL_JEPA.__init__`:

```python
self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
self.logit_bias = nn.Parameter(torch.tensor(-10.0))
```

Add a SigLIP helper:

```python
def sigmoid_contrastive_loss(
    vision_proj: torch.Tensor,
    language_proj: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = vision_proj.size(0)
    scale = logit_scale.float().clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()
    logits = vision_proj.float() @ language_proj.float().T * scale + logit_bias.float()

    labels = -torch.ones_like(logits)
    labels.fill_(-1.0)
    labels.diagonal().fill_(1.0)

    loss = -F.logsigmoid(labels * logits).sum() / bsz
    acc = (logits.argmax(dim=1) == torch.arange(bsz, device=logits.device)).float().mean()
    return loss, acc.detach()
```

Modify `compute_jepa_loss()`:

```python
loss_type = outputs.get("contrastive_loss_type", "infonce")

if loss_type == "siglip":
    nce_loss, nce_acc = sigmoid_contrastive_loss(
        vision_proj,
        language_proj,
        outputs["logit_scale"],
        outputs["logit_bias"],
    )
else:
    # Existing InfoNCE path.
    ...
```

Return `logit_bias` from `VL_JEPA.forward`:

```python
"logit_bias": self.logit_bias,
```

Why it helps:

The sigmoid objective does not force every update through a noisy batch softmax partition function. It is simpler, less dependent on huge batches, and a better first reproduction target for a single GPU.

Expected NCE@1 impact:

High for small hardware. The metric is still argmax accuracy, so it should rise if alignment improves. Use Recall@K as the main metric.

VRAM:

For batch 32 the matrix is tiny. You can disable the 65K queue at first, saving about 200 MB for the queue plus queue logit compute.

### 2.4 The projection heads are too weak

Current projections are single linear layers:

```python
self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
self.language_proj = nn.Linear(hidden_dim, hidden_dim)
```

SOTA CLIP/SigLIP-style training commonly benefits from projection heads, lower output dimensions, normalization, and sometimes LayerNorm. A linear head is not necessarily wrong, but it leaves no capacity to reconcile JEPA patch-prediction features with cross-modal retrieval geometry.

Exact change:

File: `src/model.py`

```python
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1, eps=1e-6)
```

Then:

```python
self.vision_proj = ProjectionHead(hidden_dim, out_dim=256)
self.language_proj = ProjectionHead(hidden_dim, out_dim=256)
```

Adjust `MemoryBank` dimension to projection dimension instead of `model.hidden_dim`:

```python
projection_dim = getattr(model, "projection_dim", model.hidden_dim)
self.memory_bank = MemoryBank(memory_bank_size, projection_dim, device)
```

Why it helps:

It separates JEPA representation learning from retrieval-space geometry. Lower projection dimension also reduces queue memory and logit compute.

Expected NCE@1 impact:

Low to moderate alone, moderate when paired with pretrained encoders and SigLIP.

VRAM:

Negligible. A 65K queue at 256 dims is about 67 MB in fp32 instead of about 201 MB at 768 dims.

### 2.5 The training objective needs a curriculum

Current defaults:

```yaml
loss:
  alpha: 0.5
  beta: 0.5
  gamma: 0.1
```

The investigation shows MSE collapses to almost zero quickly while NCE@1 remains random. This suggests the model is solving the easy EMA patch prediction task while failing the harder semantic matching task.

Exact change:

File: `experiments/exp_jepa_training.py` or `src/trainer.py`

Implement phase-dependent loss weights:

```python
def loss_weights_for_epoch(epoch: int) -> tuple[float, float, float]:
    # Epoch is 1-based.
    if epoch <= 5:
        return 0.0, 1.0, 0.01  # alignment warmup
    if epoch <= 20:
        return 0.2, 0.8, 0.01  # add JEPA gently
    return 0.3, 0.7, 0.01
```

Inside the epoch loop before training batches:

```python
trainer.alpha, trainer.beta, trainer.gamma = loss_weights_for_epoch(epoch + 1)
```

Why it helps:

The contrastive geometry must exist before JEPA reconstruction can preserve or refine it. Otherwise the model can spend most early optimization capacity on patch-prediction shortcuts.

Expected NCE@1 impact:

Moderate to high, especially after pretrained initialization.

VRAM:

No extra memory.

### 2.6 The memory bank should not be the first rescue tool

The current memory bank is technically useful, but it cannot fix random features. It also changes the interpretation of NCE loss: random i2t CE with a full 65K queue is around `log(65568) ~= 11.1`, while t2i random is around `log(32) ~= 3.47`; their average is around 7.3. A reported NCE around 3.6-4.7 is not directly comparable to the old in-batch baseline.

Problems to address:

- Queue only contains language keys, so t2i remains in-batch only.
- COCO contains semantically similar captions and multiple captions per image; queues can introduce false negatives.
- Queue keys are stale by design; staleness is tolerable only after representation geometry is stable.

Exact change:

File: `src/trainer.py`

Disable memory bank in phase A:

```yaml
training:
  memory_bank_size: 0
```

Change `VL_JEPA_Trainer.__init__` to allow no queue:

```python
self.memory_bank = (
    MemoryBank(memory_bank_size, model.projection_dim, device)
    if memory_bank_size and memory_bank_size > 0
    else None
)
```

Guard enqueue:

```python
if self.memory_bank is not None:
    queue_keys = outputs.get("language_proj")
    if queue_keys is not None:
        self.memory_bank.enqueue(queue_keys)
```

Later, add symmetric queues:

```python
self.language_bank = MemoryBank(memory_bank_size, projection_dim, device)
self.vision_bank = MemoryBank(memory_bank_size, projection_dim, device)
```

Why it helps:

You first need in-batch positives to become recognizable. After that, queues add useful negatives. Starting with a 65K queue can make the task dominated by stale or false negatives before the model has semantic structure.

Expected NCE@1 impact:

Low in phase A if SigLIP is used, moderate later for retrieval.

VRAM:

Disabling the 768-dim fp32 queue saves about 200 MB plus logit compute. A 256-dim queue reduces this to about 67 MB.

### 2.7 Metrics are insufficient

Tests cover shapes, finite values, checkpoint round-trip, masking, and gradient flow. They do not test whether image-text alignment can improve on real or synthetic paired data. Current training logs rely heavily on batch NCE@1.

Exact change:

File: `src/trainer.py` or a new `src/eval.py`

Add full-validation retrieval:

```python
@torch.no_grad()
def retrieval_recall(model, loader, device, k_list=(1, 5, 10)):
    model.eval()
    image_feats = []
    text_feats = []

    for images, input_ids, attention_mask in loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        v, t = model.get_joint_embedding(images, input_ids)
        image_feats.append(v.float().cpu())
        text_feats.append(t.float().cpu())

    image_feats = F.normalize(torch.cat(image_feats), dim=-1)
    text_feats = F.normalize(torch.cat(text_feats), dim=-1)
    sim = image_feats @ text_feats.T
    target = torch.arange(sim.size(0))

    out = {}
    i2t = sim.argsort(dim=1, descending=True)
    t2i = sim.T.argsort(dim=1, descending=True)
    for k in k_list:
        out[f"i2t_r{k}"] = (i2t[:, :k] == target[:, None]).any(dim=1).float().mean().item()
        out[f"t2i_r{k}"] = (t2i[:, :k] == target[:, None]).any(dim=1).float().mean().item()
    return out
```

Add an overfit test:

File: `tests/test_alignment_overfit.py`

```python
def test_alignment_overfits_tiny_batch():
    # Build 32-128 deterministic synthetic image/text pairs.
    # Train only projection heads for 200-500 steps.
    # Assert in-batch retrieval rises far above random.
    assert metrics["nce_acc"] > 0.50
```

Why it helps:

If a model cannot overfit 128 fixed pairs, the issue is implementation-level. If it can overfit but fails COCO, the issue is data/scale/hyperparameters.

Expected NCE@1 impact:

Indirect but critical. This prevents losing days to metrics that cannot distinguish progress.

VRAM:

Full-val COCO embeddings are small: 5000 x 256 fp32 is about 5 MB per modality.

## 3. Recommended Priority Order

### P0: Establish a working contrastive baseline

1. Add pretrained DistilBERT text encoder.
2. Add pretrained ViT-S/ViT-B global image encoder.
3. Freeze both encoders.
4. Use global unmasked image CLS for contrastive alignment.
5. Train projection heads with SigLIP only (`alpha=0`, `beta=1`, `gamma=0.01`, queue off).
6. Add full-val Recall@K and tiny overfit test.

Success criteria:

- Tiny overfit reaches in-batch retrieval above 50%.
- COCO val i2t/t2i R@1 is above random.
- Batch NCE@1 moves above 3.1% within 1-3 epochs.

### P1: Add JEPA back without destroying alignment

1. Add JEPA MSE with `alpha=0.1-0.3`, keep `beta=0.7-0.9`.
2. Use existing local masked context for MSE, global image CLS for contrastive.
3. Unfreeze last 2-4 vision blocks only if Recall@K is stable.
4. Use encoder LR `1e-5`, projection LR `1e-4`, predictor LR `2e-4`.

Success criteria:

- MSE improves without Recall@K collapse.
- NCE@1 and Recall@K continue improving or stay stable.

### P2: Reintroduce queues and larger effective batches

1. Add 4K or 16K queue first, not 65K.
2. Use projection dim 256.
3. Add symmetric vision/text queues if staying with InfoNCE.
4. Consider feature-cache gradient accumulation so the loss sees a larger effective batch.
5. Filter obvious false negatives if multiple captions per image are added.

Success criteria:

- Queue improves Recall@5/10 without reducing R@1.
- NCE loss is interpreted against the correct random baseline.

### P3: Scale data and model

1. Add CC3M/CC12M subset or curated web-caption data.
2. Try ViT-B full unfreeze, then ViT-L partial unfreeze.
3. Consider OpenCLIP/SigLIP initialization as a sanity baseline.
4. Add video only after image-text retrieval works.

## 4. RTX 3090 / 10 GB VRAM Recipe

Recommended first command-line profile:

```bash
python experiments/exp_jepa_training.py \
  --config configs/mvp_pretrained_siglip.yaml \
  --epochs 10 \
  --batch-size 16 \
  --lr 1e-4 \
  --alpha 0.0 \
  --beta 1.0 \
  --gamma 0.01 \
  --fresh
```

Recommended `configs/mvp_pretrained_siglip.yaml`:

```yaml
model:
  hidden_dim: 768
  patch_size: 16
  image_size: 224
  mask_ratio: 0.75
  text_mask_ratio: 0.0
  predictor_layers: 4
  momentum_tau: 0.996
  vision_backbone: vit_base_patch16_224.mae
  text_backbone: distilbert-base-uncased
  freeze_encoders: true
  projection_dim: 256
  contrastive_loss: siglip
  contrastive_on_global: true

training:
  epochs: 10
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 0.05
  warmup_steps: 500
  max_steps: null
  use_multi_crop: true
  global_crop_size: 224
  local_crop_size: 96
  max_grad_norm: 1.0
  memory_bank_size: 0

loss:
  alpha: 0.0
  beta: 1.0
  gamma: 0.01

data:
  coco_root: ~/.cache/torch/hub/checkpoints
  image_size: 224
  batch_size: 16
  num_workers: 4
  max_caption_length: 64

output:
  output_dir: experiments
  log_interval: 10
  checkpoint_interval: 5
```

VRAM notes:

- Current random 12L vision + random 12L language + predictor is too expensive for broad ablation under 10 GB, even if a single run reports around 7-10 GB.
- Frozen pretrained encoders reduce gradient and optimizer memory substantially.
- Batch 16 is the safe baseline. Try batch 24 or 32 only after measuring peak memory.
- Projection dim 256 reduces queue memory and often improves optimization.
- If unfreezing encoders, use gradient checkpointing or unfreeze only the last 2-4 blocks.

## 5. SOTA Context and References

VL-JEPA:

- Paper: https://arxiv.org/abs/2512.10942
- OpenReview: https://openreview.net/forum?id=tjimrqc2BU
- Key point from the paper page: VL-JEPA uses a pretrained/frozen-style X-Encoder and a pretrained Y-Encoder such as EmbeddingGemma-300M, with large-scale pretraining on datasets like Datacomp/YFCC/Action100M and SFT on millions of VQA/caption/classification samples. This is not comparable to random-init COCO-only training.

I-JEPA and V-JEPA:

- I-JEPA code: https://github.com/facebookresearch/ijepa
- V-JEPA code: https://github.com/facebookresearch/jepa
- V-JEPA 2 paper page: https://arxiv.org/html/2506.09985v1
- Key point: published JEPA recipes use large models, long schedules, large batches, EMA teachers, and extensive data. The core JEPA MSE path in this repo is plausible, but it does not solve cross-modal alignment by itself.

SigLIP / small-batch contrastive:

- SigLIP paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.pdf
- HuggingFace docs: https://huggingface.co/docs/transformers/model_doc/siglip
- Key point: sigmoid loss removes the global softmax normalization dependency and performs better than softmax CLIP at smaller batch sizes.

MoCo / memory queues:

- MoCo paper: https://arxiv.org/abs/1911.05722
- VISSL MoCo loss reference: https://github.com/facebookresearch/vissl/blob/main/vissl/losses/moco_loss.py
- Key point: queues help when keys are consistent and semantically meaningful. They cannot create alignment from random image/text encoders on small data.

DINO / EMA teacher stability:

- DINO code: https://github.com/facebookresearch/dino
- Key point: EMA teachers, centering, sharpening, and multi-crop are useful stabilization tools, but DINO does not imply local masked crops should be directly paired with global image captions.

LiT / frozen encoders:

- LiT paper: https://arxiv.org/abs/2111.07991
- Google blog: https://research.google/blog/locked-image-tuning-adding-language-understanding-to-image-models/
- Key point: locked pretrained image towers are data- and compute-efficient, often better than fully training both towers, and well suited to a single GPU.

OpenCLIP practical training:

- OpenCLIP: https://github.com/mlfoundations/open_clip
- Key point: use gradient accumulation for larger effective batch, but if the loss should see accumulated negatives, cache/recompute features carefully. Plain optimizer gradient accumulation does not automatically add contrastive negatives across micro-batches.

## 6. Final Diagnosis

The current NCE@1 plateau is expected for the present recipe. The implementation has fixed several real bugs, but the remaining issue is structural: random dual encoders, COCO-only data, local masked contrastive views, single linear heads, and CLIP-style InfoNCE at batch 32 are not enough.

The most realistic reproduction path on a 10 GB single GPU is not "train VL-JEPA from scratch." It is:

1. reproduce a frozen-pretrained SigLIP/CLIP-style image-text alignment baseline,
2. verify full-val retrieval,
3. add JEPA masked prediction as an auxiliary objective,
4. only then tune queues, EMA, and multi-crop details.

