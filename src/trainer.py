"""Training utilities for VL-JEPA."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from .model import VL_JEPA


class VL_JEPA_Trainer:
    """Trainer for VL-JEPA model."""
    
    def __init__(
        self,
        model: VL_JEPA,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        self.criterion_vision = nn.CrossEntropyLoss()
        self.criterion_language = nn.CrossEntropyLoss()
    
    def compute_loss(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss."""
        
        output = self.model(images, input_ids)
        
        vision_logits = output['vision_logits']  # (B, num_patches, vocab_vision)
        language_logits = output['language_logits']  # (B, seq_len, vocab_language)
        
        # Vision loss (skip CLS token)
        if vision_mask is not None:
            vision_loss = self.criterion_vision(
                vision_logits[:, 1:].reshape(-1, vision_logits.size(-1)),
                vision_mask.reshape(-1)
            )
        else:
            vision_loss = 0
        
        # Language loss
        if language_mask is not None:
            language_loss = self.criterion_language(
                language_logits.reshape(-1, language_logits.size(-1)),
                language_mask.reshape(-1)
            )
        else:
            language_loss = 0
        
        total_loss = vision_loss + language_loss if isinstance(vision_loss, torch.Tensor) else language_loss
        
        metrics = {
            'vision_loss': float(vision_loss) if isinstance(vision_loss, torch.Tensor) else 0,
            'language_loss': float(language_loss) if isinstance(language_loss, torch.Tensor) else 0,
            'total_loss': float(total_loss),
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if vision_mask is not None:
            vision_mask = vision_mask.to(self.device)
        if language_mask is not None:
            language_mask = language_mask.to(self.device)
        
        # Forward + backward
        loss, metrics = self.compute_loss(images, input_ids, vision_mask, language_mask)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return metrics
    
    @torch.no_grad()
    def eval_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single evaluation step."""
        
        self.model.eval()
        
        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if vision_mask is not None:
            vision_mask = vision_mask.to(self.device)
        if language_mask is not None:
            language_mask = language_mask.to(self.device)
        
        loss, metrics = self.compute_loss(images, input_ids, vision_mask, language_mask)
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
