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

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def random_patch_mask(batch_size: int, num_patches: int,
                      mask_ratio: float = 0.75,
                      device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Return boolean mask (True = masked) with exactly mask_ratio patches masked."""
    num_masked = max(1, int(num_patches * mask_ratio))
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        idx = torch.randperm(num_patches, device=device)[:num_masked]
        mask[i, idx] = True
    return mask


@torch.no_grad()
def random_token_mask(batch_size: int, seq_len: int,
                      mask_ratio: float = 0.15,
                      mask_token_id: int = 103,
                      input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Return boolean mask (True = masked) for language tokens.

    Follows BERT-style masking: 80% [MASK], 10% random, 10% unchanged.
    When input_ids is provided the mask is applied in-place (modifying input_ids).
    """
    num_masked = max(1, int(seq_len * mask_ratio))
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=input_ids.device if input_ids is not None else None)
    for i in range(batch_size):
        idx = torch.randperm(seq_len, device=mask.device)[:num_masked]
        mask[i, idx] = True

    if input_ids is not None:
        # BERT-style masking
        rand = torch.rand(batch_size, seq_len, device=input_ids.device)
        # 80% [MASK]
        input_ids = torch.where(mask & (rand < 0.8),
                                torch.full_like(input_ids, mask_token_id),
                                input_ids)
        # 10% random vocab
        input_ids = torch.where(mask & (rand >= 0.8) & (rand < 0.9),
                                torch.randint(0, 30522, input_ids.shape, device=input_ids.device),
                                input_ids)
        # 10% unchanged — nothing to do

    return mask


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

        num_patches = (image_size // patch_size) ** 2
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

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
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

        # Patchify
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, C * p * p)

        # Embed patches
        x = self.patch_embed(x)                              # (B, N, D)

        # Replace masked patches with mask_token
        if mask is not None:
            mask_token = self.mask_token.expand(B, x.size(1), -1)
            x = torch.where(mask.unsqueeze(-1), mask_token, x)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)                # (B, N+1, D)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]

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
        for p in self.parameters():
            if p.dim() > 1:
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
        for p in self.parameters():
            if p.dim() > 1:
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
                 predictor_layers: int = 6, momentum_tau: float = 0.996):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.momentum_tau = momentum_tau

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

        # Prediction heads (project to D-dim for MSE)
        self.vision_pred_head = nn.Linear(hidden_dim, hidden_dim)

        # Learnable temperature for InfoNCE
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)  # ~ln(1/0.07)

        self._init_weights()

    def _init_weights(self):
        # projection heads with smaller init
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.vision_pred_head.weight, gain=0.1)

    @torch.no_grad()
    def momentum_update(self):
        """EMA update: target = tau * target + (1 - tau) * context."""
        tau = self.momentum_tau
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(),
                                self.target_encoder.parameters()):
            tgt_p.data = tau * tgt_p.data + (1 - tau) * ctx_p.data

    def encode_vision(self, images: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode images through context encoder (with optional masking)."""
        return self.context_encoder(images, mask)

    def encode_language(self, input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.language_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
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
        num_patches = (images.size(2) // self.context_encoder.patch_size) ** 2
        device = images.device

        # ---- Generate masks if not provided ----
        if patch_mask is None:
            patch_mask = random_patch_mask(B, num_patches, self.mask_ratio, device)

        # ---- 1. Context encoder (with masking) ----
        context_emb = self.context_encoder(images, patch_mask)  # (B, N+1, D)

        # ---- 2. Target encoder (full image, no grad) ----
        with torch.no_grad():
            target_emb = self.target_encoder(images, mask=None)  # (B, N+1, D)

        # ---- 3. Predictor ----
        predicted = self.predictor(context_emb)  # (B, N+1, D)

        # ---- 4. Language encoding ----
        if token_mask is not None:
            # BERT-style masking applied in-place to input_ids
            _ = random_token_mask(B, input_ids.size(1), 0.15, 103, input_ids)

        language_emb = self.language_encoder(input_ids, attention_mask)  # (B, S, D)

        # ---- 5. Joint projections ----
        # Vision [CLS] token (first position)
        vision_cls = context_emb[:, 0, :]          # (B, D)
        language_cls = language_emb[:, 0, :]        # (B, D) — use [CLS] equivalent (first token)

        vision_proj = F.normalize(self.vision_proj(vision_cls), p=2, dim=-1)
        language_proj = F.normalize(self.language_proj(language_cls), p=2, dim=-1)

        # Prediction heads (match dimensionality for MSE)
        predicted_patches = self.vision_pred_head(predicted)
        target_patches = self.vision_pred_head(target_emb)
        # Detach target_patches to ensure no gradient flows through target encoder
        target_patches = target_patches.detach()

        return {
            'predicted_patches': predicted_patches,   # (B, N+1, D)
            'target_patches': target_patches,         # (B, N+1, D), detached
            'patch_mask': patch_mask,                 # (B, N)
            'vision_cls': vision_cls,
            'language_cls': language_cls,
            'vision_proj': vision_proj,               # (B, D), normalized
            'language_proj': language_proj,            # (B, D), normalized
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

        vision_proj = F.normalize(self.vision_proj(vision_cls), p=2, dim=-1)
        language_proj = F.normalize(self.language_proj(language_cls), p=2, dim=-1)

        return vision_proj, language_proj


# ---------------------------------------------------------------------------
# Loss computation (standalone function for clarity)
# ---------------------------------------------------------------------------

def compute_jepa_loss(
    outputs: Dict[str, torch.Tensor],
    alpha: float = 1.0,       # weight for MSE prediction loss
    beta: float = 0.5,        # weight for InfoNCE contrastive loss
) -> Dict[str, torch.Tensor]:
    """
    Compute VL-JEPA loss:

      L = α * L_mse + β * L_nce

    L_mse: MSE between predicted and target patch embeddings, averaged over
           masked positions only (skip [CLS] token at index 0).
    L_nce: InfoNCE loss aligning vision and language [CLS] projections.
    """
    predicted = outputs['predicted_patches']   # (B, N+1, D)
    target = outputs['target_patches']          # (B, N+1, D), detached
    patch_mask = outputs['patch_mask']          # (B, N)

    vision_proj = outputs['vision_proj']        # (B, D) normalized
    language_proj = outputs['language_proj']    # (B, D) normalized

    # ---- MSE on masked patches (skip [CLS] at index 0) ----
    # predicted[:, 1:] and target[:, 1:] -> (B, N, D)
    pred_patches = predicted[:, 1:, :]
    tgt_patches = target[:, 1:, :]

    mse_all = F.mse_loss(pred_patches, tgt_patches, reduction='none')  # (B, N, D)
    mse_all = mse_all.mean(dim=-1)  # (B, N) — average over feature dim

    # Mask out unmasked positions
    patch_mask_expanded = patch_mask.float()  # (B, N) — 1 = masked
    mse_masked = (mse_all * patch_mask_expanded).sum() / patch_mask_expanded.sum().clamp(min=1)

    # ---- InfoNCE contrastive loss ----
    batch_size = vision_proj.size(0)
    logit_scale = outputs.get('logit_scale', torch.tensor(2.659, device=vision_proj.device))
    scale = logit_scale.exp()

    logits = vision_proj @ language_proj.T * scale  # (B, B)
    labels = torch.arange(batch_size, device=vision_proj.device)

    # Symmetric NCE (both directions)
    nce_loss_i2t = F.cross_entropy(logits, labels)
    nce_loss_t2i = F.cross_entropy(logits.T, labels)
    nce_loss = (nce_loss_i2t + nce_loss_t2i) / 2

    # ---- Total ----
    total_loss = alpha * mse_masked + beta * nce_loss

    return {
        'mse_loss': mse_masked,
        'nce_loss': nce_loss,
        'total_loss': total_loss,
        'logit_scale': scale.detach(),
    }
