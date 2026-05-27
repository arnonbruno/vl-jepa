"""
VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language
arXiv:2512.10942v2

Fixed implementation with proper JEPA principles:
  - Random patch/token masking before encoding
  - Target encoder (EMA of context encoder) with stop-gradient
  - Predictor network that predicts target representations from context
  - MSE loss on masked positions (I-JEPA style)
  - InfoNCE contrastive loss for cross-modal alignment (VL-JEPA style)
"""

import copy
import contextlib
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGIT_SCALE_MIN = math.log(1.0 / 100.0)
LOGIT_SCALE_MAX = math.log(100.0)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def _infer_patch_grid(num_patches: int) -> Tuple[int, int]:
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}")
    grid = int(math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"num_patches must be a square grid, got {num_patches}")
    return grid, grid


@contextlib.contextmanager
def _temporarily_eval(*modules: nn.Module):
    """Run teacher modules deterministically without leaking mode changes."""
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, was_training in zip(modules, modes):
            module.train(was_training)


@torch.no_grad()
def block_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.75,
    device: torch.device = torch.device('cpu'),
    min_block_ratio: float = 0.15,
    max_block_ratio: float = 0.50,
    aspect_ratio: Tuple[float, float] = (0.3, 3.0),
    generator: Optional[torch.Generator] = None,
    min_visible_patches: Optional[int] = None,
) -> torch.Tensor:
    """I-JEPA-style rectangular block mask (True = masked).

    The function samples random aspect-ratio blocks until the requested mask
    budget is reached, then trims/fills to keep the exact masked patch count.
    On very small grids, the mask budget is capped so every sample keeps at
    least a minimal visible context.
    """
    grid_h, grid_w = _infer_patch_grid(num_patches)
    if min_visible_patches is None:
        min_visible_patches = max(1, min(num_patches, math.ceil(num_patches * 0.25)))
    min_visible_patches = max(0, min(num_patches, int(min_visible_patches)))
    max_masked = max(0, num_patches - min_visible_patches)
    raw_target = int(num_patches * mask_ratio)
    target = min(max(1, raw_target), max_masked) if max_masked > 0 else 0
    min_area = max(1, int(num_patches * min_block_ratio))
    max_area = max(min_area, int(num_patches * max_block_ratio))
    log_ar_min, log_ar_max = math.log(aspect_ratio[0]), math.log(aspect_ratio[1])
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    if target == 0:
        return mask

    for i in range(batch_size):
        attempts = 0
        view = mask[i].view(grid_h, grid_w)
        while int(mask[i].sum().item()) < target and attempts < 64:
            attempts += 1
            area = int(torch.randint(
                min_area, max_area + 1, (1,), device=device, generator=generator,
            ).item())
            ratio = math.exp(torch.empty(
                (), device=device,
            ).uniform_(log_ar_min, log_ar_max, generator=generator).item())
            block_h = max(1, min(grid_h, int(round(math.sqrt(area / ratio)))))
            block_w = max(1, min(grid_w, int(round(math.sqrt(area * ratio)))))
            top = int(torch.randint(
                0, grid_h - block_h + 1, (1,), device=device, generator=generator,
            ).item())
            left = int(torch.randint(
                0, grid_w - block_w + 1, (1,), device=device, generator=generator,
            ).item())
            view[top:top + block_h, left:left + block_w] = True

        current = int(mask[i].sum().item())
        if current < target:
            unmasked = (~mask[i]).nonzero(as_tuple=False).flatten()
            fill = unmasked[torch.randperm(
                unmasked.numel(), device=device, generator=generator,
            )[:target - current]]
            mask[i, fill] = True
        elif current > target:
            masked = mask[i].nonzero(as_tuple=False).flatten()
            keep = masked[torch.randperm(
                masked.numel(), device=device, generator=generator,
            )[:target]]
            mask[i].zero_()
            mask[i, keep] = True

    return mask


@torch.no_grad()
def random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.75,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Backward-compatible alias for the block masking policy."""
    return block_patch_mask(batch_size, num_patches, mask_ratio, device)


def _random_resized_crop_batch(
    images: torch.Tensor,
    size: int,
    scale: Tuple[float, float],
) -> torch.Tensor:
    """Torch-only random resized crop for already batched tensors."""
    bsz, _, height, width = images.shape
    crops = []
    area = height * width
    for i in range(bsz):
        crop_h = height
        crop_w = width
        for _ in range(10):
            target_area = area * torch.empty(
                (), device=images.device,
            ).uniform_(scale[0], scale[1]).item()
            aspect = math.exp(torch.empty(
                (), device=images.device,
            ).uniform_(math.log(0.75), math.log(1.3333)).item())
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_h <= height and 0 < crop_w <= width:
                break
        crop_h = max(1, min(crop_h, height))
        crop_w = max(1, min(crop_w, width))
        top = int(torch.randint(0, height - crop_h + 1, (1,), device=images.device).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,), device=images.device).item())
        crop = images[i:i + 1, :, top:top + crop_h, left:left + crop_w]
        crops.append(F.interpolate(crop, size=(size, size), mode='bilinear', align_corners=False))
    return torch.cat(crops, dim=0)


def _center_crop_resize_batch(images: torch.Tensor, size: int) -> torch.Tensor:
    _, _, height, width = images.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    crop = images[:, :, top:top + side, left:left + side]
    return F.interpolate(crop, size=(size, size), mode='bilinear', align_corners=False)


@torch.no_grad()
def make_multicrop_views(
    images: torch.Tensor,
    global_size: int = 224,
    local_size: int = 96,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    """Return one global target view and one local context view per image."""
    if training:
        global_view = _random_resized_crop_batch(images, global_size, scale=(0.60, 1.00))
        local_view = _random_resized_crop_batch(images, local_size, scale=(0.25, 0.55))
    else:
        global_view = _center_crop_resize_batch(images, global_size)
        local_view = _center_crop_resize_batch(images, local_size)
    return {'global': global_view, 'local': local_view}


@torch.no_grad()
def random_token_mask(batch_size: int, seq_len: int,
                      mask_ratio: float = 0.15,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """Return boolean mask (True = masked) for language tokens."""
    num_masked = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(seq_len, device=mask.device)[:num_masked]
        mask[i, idx] = True
    return mask


@torch.no_grad()
def apply_bert_token_mask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int = 103,
    vocab_size: int = 30522,
) -> torch.Tensor:
    """Apply BERT-style masking and return a new tensor (no in-place mutation).

    80% [MASK], 10% random token, 10% unchanged.
    """
    masked_ids = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    masked_ids = torch.where(
        mask & (rand < 0.8),
        torch.full_like(masked_ids, mask_token_id),
        masked_ids,
    )
    masked_ids = torch.where(
        mask & (rand >= 0.8) & (rand < 0.9),
        torch.randint(0, vocab_size, input_ids.shape, device=input_ids.device),
        masked_ids,
    )
    return masked_ids


# ---------------------------------------------------------------------------
# Vision Encoder (context) — processes masked images
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Vision encoder — processes (possibly masked) images into patch embeddings."""

    def __init__(self, hidden_dim: int = 768, patch_size: int = 16, image_size: int = 224):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        num_patches = self.num_patches
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Learnable [MASK] patch embedding (replaces masked patches before transformer)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _pos_embed_for_grid(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Interpolate positional embeddings for multi-crop/local resolutions."""
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"Patch grid must be positive, got {(grid_h, grid_w)}")
        if grid_h == self.grid_size and grid_w == self.grid_size:
            return self.pos_embed[:, :grid_h * grid_w + 1, :]

        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        patch_pos = patch_pos.reshape(1, self.grid_size, self.grid_size, self.hidden_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(grid_h, grid_w), mode='bicubic', align_corners=False,
        )
        patch_pos = torch.nan_to_num(patch_pos, nan=0.0, posinf=0.0, neginf=0.0)
        patch_pos = patch_pos.to(dtype=cls_pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.hidden_dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2 or 'bias' in name or 'norm' in name.lower():
                continue
            nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
            mask: (B, num_patches) boolean, True = patch is masked
        Returns:
            (B, num_patches+1, hidden_dim) patch + [CLS] embeddings
        """
        B, C, H, W = x.shape
        if H <= 0 or W <= 0:
            raise ValueError(f"Image size must be positive, got {(H, W)}")

        # Patchify
        p = self.patch_size
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
            H += pad_h
            W += pad_w
        grid_h, grid_w = H // p, W // p
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, C * p * p)

        # Embed patches
        x = self.patch_embed(x)                              # (B, N, D)

        # Replace masked patches with mask_token
        if mask is not None:
            if mask.shape != (B, x.size(1)):
                raise ValueError(f"Expected mask shape {(B, x.size(1))}, got {tuple(mask.shape)}")
            mask_token = self.mask_token.expand(B, x.size(1), -1)
            x = torch.where(mask.unsqueeze(-1), mask_token, x)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                # (B, N+1, D)

        # Add positional embeddings
        x = x + self._pos_embed_for_grid(grid_h, grid_w)

        # Transformer
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Language Encoder — processes text tokens
# ---------------------------------------------------------------------------

class LanguageEncoder(nn.Module):
    """Language encoder — processes tokenized text into embeddings."""

    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 768,
                 max_seq_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2 or 'bias' in name or 'norm' in name.lower():
                continue
            nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) token indices
            attention_mask: (B, seq_len) optional — 1 = keep, 0 = pad
        Returns:
            (B, seq_len, hidden_dim) token embeddings
        """
        seq_len = input_ids.size(1)

        # Token + positional embeddings
        x = self.token_embed(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)

        # Transformer with optional padding mask
        src_key_padding_mask = None
        if attention_mask is not None:
            # PyTorch TransformerEncoder expects True for positions to IGNORE
            src_key_padding_mask = ~attention_mask.bool()

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Predictor — predicts target embeddings from context embeddings
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """Lightweight transformer that predicts target patch embeddings.

    Takes context encoder output (with both visible and [MASK] tokens) and
    predicts the corresponding target encoder output for masked positions.
    """

    def __init__(self, hidden_dim: int = 768, num_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() < 2 or 'bias' in name or 'norm' in name.lower():
                continue
            nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D) context encoder output (including [CLS])
        Returns:
            (B, N+1, D) predicted target embeddings
        """
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# VL-JEPA: Joint Embedding Predictive Architecture
# ---------------------------------------------------------------------------

class VL_JEPA(nn.Module):
    """
    VL-JEPA with proper JEPA training loop.

    Components:
      - context_encoder: VisionEncoder that processes masked images
      - target_encoder: EMA copy of context_encoder (frozen, no grad)
      - predictor: Predictor that maps context -> target embeddings
      - language_encoder: LanguageEncoder for text modality
      - vision_proj / language_proj: project to joint embedding space
      - temperature: scaling factor for InfoNCE
    """

    def __init__(self, hidden_dim: int = 768, patch_size: int = 16,
                 image_size: int = 224, mask_ratio: float = 0.75,
                 predictor_layers: int = 6, momentum_tau: float = 0.996,
                 text_mask_ratio: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.momentum_tau = momentum_tau
        self.text_mask_ratio = text_mask_ratio

        # Context encoder (student) — trained with gradients
        self.context_encoder = VisionEncoder(hidden_dim, patch_size, image_size)

        # Target encoder (teacher) — EMA of context_encoder, no gradients
        self.target_encoder = VisionEncoder(hidden_dim, patch_size, image_size)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = Predictor(hidden_dim, num_layers=predictor_layers)

        # Language encoder
        self.language_encoder = LanguageEncoder(hidden_dim=hidden_dim)

        # Projections to joint space
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        self.language_proj = nn.Linear(hidden_dim, hidden_dim)

        # Student prediction head and EMA teacher head define the MSE loss space.
        # Applying a head on only one side makes the predictor/teacher spaces asymmetric.
        self.vision_pred_head = nn.Linear(hidden_dim, hidden_dim)

        # Learnable temperature for InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)  # ~ln(1/0.07)

        self._init_weights()

        # EMA teachers for language and projection heads stabilize cross-modal NCE.
        self.target_language_encoder = copy.deepcopy(self.language_encoder)
        self.target_vision_proj = copy.deepcopy(self.vision_proj)
        self.target_language_proj = copy.deepcopy(self.language_proj)
        self.target_vision_pred_head = copy.deepcopy(self.vision_pred_head)
        for module in (
            self.target_language_encoder,
            self.target_vision_proj,
            self.target_language_proj,
            self.target_vision_pred_head,
        ):
            for p in module.parameters():
                p.requires_grad = False

    def _init_weights(self):
        # projection heads with smaller init
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.language_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.vision_pred_head.weight, gain=0.1)

    @torch.no_grad()
    def momentum_update(self):
        """EMA update: target = tau * target + (1 - tau) * student."""
        tau = self.momentum_tau
        pairs = (
            (self.context_encoder, self.target_encoder),
            (self.language_encoder, self.target_language_encoder),
            (self.vision_proj, self.target_vision_proj),
            (self.language_proj, self.target_language_proj),
            (self.vision_pred_head, self.target_vision_pred_head),
        )
        for student, teacher in pairs:
            for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
                teacher_p.copy_(
                    tau * teacher_p.detach() + (1 - tau) * student_p.detach(),
                )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        context_images: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        mask_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass for training.

        Returns dict with:
          - predicted_patches: (B, N+1, D) predictor output
          - target_patches: (B, N+1, D) target encoder output (detached)
          - patch_mask: (B, N) which patches were masked in vision
          - vision_cls: (B, D) normalized vision [CLS] embedding
          - language_cls: (B, D) normalized language [CLS] embedding
          - vision_proj: (B, D) normalized vision projection
          - language_proj: (B, D) normalized language projection
        """
        B = images.size(0)
        device = images.device
        context_images = images if context_images is None else context_images
        target_images = images if target_images is None else target_images
        p = self.context_encoder.patch_size
        num_patches = (
            math.ceil(context_images.size(2) / p)
            * math.ceil(context_images.size(3) / p)
        )
        generator = None
        if mask_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(mask_seed)

        # ---- Generate masks if not provided ----
        if patch_mask is None:
            patch_mask = block_patch_mask(
                B, num_patches, self.mask_ratio, device, generator=generator,
            )

        # ---- 1. Context encoder (with masking) ----
        context_emb = self.context_encoder(context_images, patch_mask)  # (B, N+1, D)

        # ---- 2. Target encoders (full local target for MSE, global target for NCE) ----
        teacher_modules = (
            self.target_encoder,
            self.target_language_encoder,
            self.target_vision_proj,
            self.target_language_proj,
            self.target_vision_pred_head,
        )
        with torch.no_grad(), _temporarily_eval(*teacher_modules):
            target_emb = self.target_encoder(context_images, mask=None)  # (B, N+1, D)
            target_global_emb = self.target_encoder(target_images, mask=None)

        # ---- 3. Predictor ----
        predicted = self.predictor(context_emb)  # (B, N+1, D)

        # ---- 4. Language encoding ----
        clean_input_ids = input_ids
        if token_mask is None and self.training and self.text_mask_ratio > 0:
            token_mask = random_token_mask(
                B, input_ids.size(1), mask_ratio=self.text_mask_ratio, device=device,
            )
        if token_mask is not None and token_mask.any():
            input_ids = apply_bert_token_mask(input_ids, token_mask)

        language_emb = self.language_encoder(input_ids, attention_mask)  # (B, S, D)
        with torch.no_grad(), _temporarily_eval(*teacher_modules):
            target_language_emb = self.target_language_encoder(clean_input_ids, attention_mask)

        # ---- 5. Joint projections ----
        # Use context encoder CLS for contrastive alignment (student vision representation).
        vision_cls = context_emb[:, 0, :]          # (B, D)
        language_cls = language_emb[:, 0, :]        # (B, D) — use [CLS] equivalent (first token)

        vision_proj_raw = self.vision_proj(vision_cls)
        language_proj_raw = self.language_proj(language_cls)
        vision_proj = F.normalize(vision_proj_raw, p=2, dim=-1, eps=1e-6)
        language_proj = F.normalize(language_proj_raw, p=2, dim=-1, eps=1e-6)
        with torch.no_grad(), _temporarily_eval(*teacher_modules):
            target_vision_proj = F.normalize(
                self.target_vision_proj(target_global_emb[:, 0, :]), p=2, dim=-1, eps=1e-6,
            )
            target_language_proj = F.normalize(
                self.target_language_proj(target_language_emb[:, 0, :]), p=2, dim=-1, eps=1e-6,
            )
            target_patches = self.target_vision_pred_head(target_emb).detach()

        # Predictor output mapped to the EMA teacher-head space.
        predicted_patches = self.vision_pred_head(predicted)

        return {
            'predicted_patches': predicted_patches,   # (B, N+1, D)
            'target_patches': target_patches,         # (B, N+1, D), detached
            'patch_mask': patch_mask,                 # (B, N)
            'vision_cls': vision_cls,
            'language_cls': language_cls,
            'vision_proj': vision_proj,               # (B, D), normalized
            'language_proj': language_proj,            # (B, D), normalized
            'vision_proj_raw': vision_proj_raw,       # (B, D), pre-normalize (for variance loss)
            'language_proj_raw': language_proj_raw,
            'target_vision_proj': target_vision_proj,  # (B, D), normalized + detached
            'target_language_proj': target_language_proj,
            'logit_scale': self.logit_scale,           # scalar
        }

    @torch.no_grad()
    def get_joint_embedding(self, images: torch.Tensor,
                            input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized joint embeddings for retrieval (inference)."""
        vision_emb = self.context_encoder(images)
        language_emb = self.language_encoder(input_ids)

        vision_cls = vision_emb[:, 0, :]
        language_cls = language_emb[:, 0, :]

        vision_proj = F.normalize(self.vision_proj(vision_cls), p=2, dim=-1, eps=1e-6)
        language_proj = F.normalize(self.language_proj(language_cls), p=2, dim=-1, eps=1e-6)

        return vision_proj, language_proj


# ---------------------------------------------------------------------------
# Loss computation (standalone function for clarity)
# ---------------------------------------------------------------------------

def variance_loss(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance regularization: encourage std >= 1 per feature dim."""
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(1.0 - std))


def compute_jepa_loss(
    outputs: Dict[str, torch.Tensor],
    alpha: float = 1.0,       # weight for MSE prediction loss
    beta: float = 0.5,        # weight for InfoNCE contrastive loss
    gamma: float = 0.1,       # weight for variance regularization (anti-collapse)
) -> Dict[str, torch.Tensor]:
    """
    Compute VL-JEPA loss:

      L = α * L_mse + β * L_nce + γ * L_var

    L_mse: MSE between predicted and target patch embeddings, averaged over
           masked positions only (skip [CLS] token at index 0).
    L_nce: InfoNCE loss aligning vision and language [CLS] projections.
    L_var: VICReg-style variance loss on pre-normalized projections (prevents collapse).
    """
    predicted = outputs['predicted_patches']   # (B, N+1, D)
    target = outputs['target_patches']          # (B, N+1, D), detached
    patch_mask = outputs['patch_mask']          # (B, N)

    vision_proj = outputs['vision_proj'].float()        # (B, D) normalized
    language_proj = outputs['language_proj'].float()    # (B, D) normalized

    # ---- MSE on masked patches (skip [CLS] at index 0) ----
    # predicted[:, 1:] and target[:, 1:] -> (B, N, D)
    # FP32 MSE avoids AMP overflow when predictor/teacher magnitudes grow.
    pred_patches = predicted[:, 1:, :].float()
    tgt_patches = target[:, 1:, :].float()

    mse_all = F.mse_loss(pred_patches, tgt_patches, reduction='none')  # (B, N, D)
    mse_all = mse_all.mean(dim=-1)  # (B, N) — average over feature dim

    # Mask out unmasked positions. Average per sample first so a zero-mask
    # sample contributes 0 instead of changing the whole-batch denominator.
    patch_mask_expanded = patch_mask.to(dtype=mse_all.dtype)  # (B, N) — 1 = masked
    masked_counts = patch_mask_expanded.sum(dim=1)
    masked_sums = (mse_all * patch_mask_expanded).sum(dim=1)
    mse_per_sample = torch.where(
        masked_counts > 0,
        masked_sums / masked_counts.clamp(min=1),
        torch.zeros_like(masked_sums),
    )
    mse_masked = mse_per_sample.mean()

    # ---- InfoNCE contrastive loss (FP32 for AMP stability) ----
    batch_size = vision_proj.size(0)
    logit_scale = outputs.get('logit_scale', torch.tensor(2.659, device=vision_proj.device))
    scale = logit_scale.float().clamp(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX).exp()

    # Student-to-student contrastive: both projections receive gradients.
    logits_i2t = vision_proj @ language_proj.T * scale
    logits_t2i = logits_i2t.T
    labels = torch.arange(batch_size, device=vision_proj.device)

    # Symmetric NCE (both directions)
    nce_loss_i2t = F.cross_entropy(logits_i2t, labels)
    nce_loss_t2i = F.cross_entropy(logits_t2i, labels)
    nce_loss = (nce_loss_i2t + nce_loss_t2i) / 2
    nce_acc_i2t = (logits_i2t.argmax(dim=1) == labels).float().mean()
    nce_acc_t2i = (logits_t2i.argmax(dim=1) == labels).float().mean()

    # ---- Variance regularization (pre-normalized projections) ----
    vision_raw = outputs['vision_proj_raw'].float()
    language_raw = outputs['language_proj_raw'].float()
    var_loss = (variance_loss(vision_raw) + variance_loss(language_raw)) / 2

    # ---- Total ----
    total_loss = alpha * mse_masked + beta * nce_loss + gamma * var_loss

    return {
        'mse_loss': mse_masked,
        'nce_loss': nce_loss,
        'var_loss': var_loss,
        'total_loss': total_loss,
        'logit_scale': scale.detach(),
        'nce_acc': ((nce_acc_i2t + nce_acc_t2i) / 2).detach(),
    }
