"""
VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language
arXiv:2512.10942v2

Core model implementation. Vision and language encoders produce joint embeddings
that can be used for downstream multimodal tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class VisionEncoder(nn.Module):
    """Vision encoder - processes images into embeddings."""
    
    def __init__(self, hidden_dim: int = 768, patch_size: int = 16, image_size: int = 224):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            (B, num_patches+1, hidden_dim) embeddings
        """
        B, C, H, W = x.shape
        
        # Patchify
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, -1, C * self.patch_size * self.patch_size)
        
        # Embed patches
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Apply transformer
        x = self.transformer(x)
        return x


class LanguageEncoder(nn.Module):
    """Language encoder - processes text into embeddings."""
    
    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 768, max_seq_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len) token indices
            attention_mask: (B, seq_len) optional attention mask
        Returns:
            (B, seq_len, hidden_dim) embeddings
        """
        seq_len = input_ids.size(1)
        
        # Token + positional embeddings
        x = self.token_embed(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)
        
        # Apply transformer
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        return x


class PredictionHead(nn.Module):
    """Prediction head for masked patch/token prediction."""
    
    def __init__(self, hidden_dim: int = 768, vocab_size_vision: int = 8192):
        super().__init__()
        self.vision_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size_vision)
        )
        self.language_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 30522)  # BERT vocab
        )
    
    def forward(self, vision_emb: torch.Tensor, language_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict masked vision patches and language tokens."""
        vision_logits = self.vision_head(vision_emb)
        language_logits = self.language_head(language_emb)
        return vision_logits, language_logits


class VL_JEPA(nn.Module):
    """
    VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language.
    
    Jointly learns vision and language representations by predicting masked
    regions in both modalities.
    """
    
    def __init__(self, hidden_dim: int = 768, patch_size: int = 16, image_size: int = 224):
        super().__init__()
        
        self.vision_encoder = VisionEncoder(hidden_dim, patch_size, image_size)
        self.language_encoder = LanguageEncoder(hidden_dim=hidden_dim)
        self.prediction_head = PredictionHead(hidden_dim)
        
        # Joint projection to shared space
        self.vision_proj = nn.Linear(hidden_dim, hidden_dim)
        self.language_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.hidden_dim = hidden_dim
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings."""
        return self.vision_encoder(images)
    
    def encode_language(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to embeddings."""
        return self.language_encoder(input_ids, attention_mask)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            vision_mask: (B, num_patches) binary mask for vision
            language_mask: (B, seq_len) binary mask for language
        
        Returns:
            dict with logits and embeddings
        """
        # Encode modalities
        vision_emb = self.encode_vision(images)
        language_emb = self.encode_language(input_ids, attention_mask)
        
        # Project to shared space
        vision_proj = self.vision_proj(vision_emb)
        language_proj = self.language_proj(language_emb)
        
        # Get predictions
        vision_logits, language_logits = self.prediction_head(vision_emb, language_emb)
        
        return {
            'vision_emb': vision_emb,
            'language_emb': language_emb,
            'vision_proj': vision_proj,
            'language_proj': language_proj,
            'vision_logits': vision_logits,
            'language_logits': language_logits,
        }
    
    def get_joint_embedding(self, images: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get joint embeddings for image-text pair (CLS tokens)."""
        vision_emb = self.encode_vision(images)
        language_emb = self.encode_language(input_ids)
        
        # Use CLS token (first token)
        vision_cls = vision_emb[:, 0, :]  # (B, hidden_dim)
        language_cls = language_emb[:, 0, :]  # (B, hidden_dim)
        
        vision_proj = F.normalize(self.vision_proj(vision_cls), p=2, dim=-1)
        language_proj = F.normalize(self.language_proj(language_cls), p=2, dim=-1)
        
        return vision_proj, language_proj
