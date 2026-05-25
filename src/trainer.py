"""
Training utilities for VL-JEPA.
Fixed with proper JEPA training loop:
  - JEPA loss: MSE on masked patches + InfoNCE cross-modal alignment
  - Momentum update of target encoder (EMA)
  - AMP + gradient clipping for GPU efficiency
  - Validation split with eval loop
  - TensorBoard logging
"""

import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from typing import Dict, Tuple, Optional, Iterator, Any
from pathlib import Path

from .model import (
    LOGIT_SCALE_MAX,
    LOGIT_SCALE_MIN,
    VL_JEPA,
    compute_jepa_loss,
    make_multicrop_views,
)


def _amp_autocast(device: torch.device):
    """AMP autocast with fallback for PyTorch < 2.4."""
    if device.type != 'cuda':
        return contextlib.nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()


def _make_grad_scaler(device: torch.device):
    """GradScaler with fallback for PyTorch < 2.4."""
    if device.type != 'cuda':
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', init_scale=1024.0, growth_interval=2000)
    return torch.cuda.amp.GradScaler(init_scale=1024.0, growth_interval=2000)


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack (images, input_ids) or (images, input_ids, attention_mask) batches."""
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) < 2:
        raise ValueError(f"Batch must have at least 2 elements, got {len(batch)}")
    images, input_ids = batch[0], batch[1]
    attention_mask = batch[2] if len(batch) > 2 else None
    return images, input_ids, attention_mask


class VL_JEPA_Trainer:
    """Trainer for VL-JEPA with proper JEPA loss."""

    def __init__(
        self,
        model: VL_JEPA,
        device: torch.device,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
        max_steps: int = 100_000,
        alpha: float = 1.0,        # MSE weight
        beta: float = 0.5,         # InfoNCE weight
        momentum_tau: float = 0.996,
        momentum_tau_end: float = 1.0,
        use_multi_crop: bool = False,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        eval_mask_seed: int = 17_029,
    ):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.use_multi_crop = use_multi_crop
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.eval_mask_seed = eval_mask_seed

        # Set momentum schedule
        self.model.momentum_tau = momentum_tau
        self.momentum_tau_end = momentum_tau_end
        self._momentum_schedule = self._cosine_schedule(momentum_tau, momentum_tau_end, max_steps)

        # Optimizer with parameter groups (different LR for predictor)
        # Following I-JEPA: predictor gets 20x higher learning rate
        decay_params = []
        no_decay_params = []
        predictor_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'predictor' in name:
                predictor_params.append(p)
            elif 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                no_decay_params.append(p)
            elif 'logit_scale' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': predictor_params, 'weight_decay': weight_decay, 'lr': learning_rate * 20.0},
        ]

        self.optimizer = optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
        self.scaler = _make_grad_scaler(device)

        # Warmup + cosine scheduler
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step = 0

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Loss tracking
        self.running_loss = 0.0
        self.running_steps = 0

    @torch.no_grad()
    def _clamp_stability_params(self) -> None:
        """Keep scalar stability parameters inside their valid training range."""
        self.model.logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)

    @staticmethod
    def _assert_finite_loss(loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor]) -> None:
        if torch.isfinite(loss):
            return
        parts = []
        for key in ('mse_loss', 'nce_loss', 'total_loss'):
            value = loss_dict.get(key)
            if isinstance(value, torch.Tensor):
                parts.append(f"{key}={value.detach().float().item()}")
        raise FloatingPointError(f"Non-finite JEPA loss detected ({', '.join(parts)})")

    def _forward_model(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        training: bool,
        mask_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_multi_crop:
            return self.model(
                images, input_ids, attention_mask, mask_seed=mask_seed,
            )

        views = make_multicrop_views(
            images,
            global_size=self.global_crop_size,
            local_size=self.local_crop_size,
            training=training,
        )
        return self.model(
            views['global'],
            input_ids,
            attention_mask,
            context_images=views['local'],
            target_images=views['global'],
            mask_seed=mask_seed,
        )

    @staticmethod
    def _cosine_schedule(start: float, end: float, steps: int) -> torch.Tensor:
        """Cosine schedule from start to end over `steps`."""
        t = torch.linspace(0, 1, steps)
        return end + (start - end) * (1 + torch.cos(t * math.pi)) / 2

    def _maybe_sync_batchnorm(self):
        """Sync batch norm if using DataParallel (no-op for now)."""
        pass

    def train_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step with proper JEPA loss."""
        self.model.train()

        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Forward pass (with AMP)
        if self.scaler is not None:
            with _amp_autocast(self.device):
                outputs = self._forward_model(images, input_ids, attention_mask, training=True)
                loss_dict = compute_jepa_loss(outputs, self.alpha, self.beta)
                loss = loss_dict['total_loss']
                self._assert_finite_loss(loss, loss_dict)

            # Backward with AMP
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 3.0, error_if_nonfinite=True,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._clamp_stability_params()
            # GradScaler does not set _opt_called on LambdaLR-wrapped optimizers
            self.optimizer._opt_called = True
            self.scheduler.step()
            self._step += 1
        else:
            outputs = self._forward_model(images, input_ids, attention_mask, training=True)
            loss_dict = compute_jepa_loss(outputs, self.alpha, self.beta)
            loss = loss_dict['total_loss']
            self._assert_finite_loss(loss, loss_dict)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 3.0, error_if_nonfinite=True,
            )
            self.optimizer.step()
            self._clamp_stability_params()
            self.scheduler.step()
            self._step += 1

        # Momentum update (EMA)
        tau = self._momentum_schedule[self._step - 1].item() if (self._step - 1) < len(self._momentum_schedule) else self.momentum_tau_end
        self.model.momentum_tau = tau
        self.model.momentum_update()

        # Running average
        self.running_loss += loss.item()
        self.running_steps += 1

        metrics = {
            'mse_loss': loss_dict['mse_loss'].item(),
            'nce_loss': loss_dict['nce_loss'].item(),
            'total_loss': loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0),
            'logit_scale': loss_dict.get('logit_scale', torch.tensor(0.0)).item(),
            'nce_acc': loss_dict.get('nce_acc', torch.tensor(0.0)).item(),
            'momentum_tau': tau,
            'lr': self.optimizer.param_groups[0]['lr'],
        }

        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Single evaluation step (no gradients, no EMA update)."""
        self.model.eval()

        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with _amp_autocast(self.device):
            outputs = self._forward_model(
                images, input_ids, attention_mask, training=False, mask_seed=mask_seed,
            )
            loss_dict = compute_jepa_loss(outputs, self.alpha, self.beta)
            self._assert_finite_loss(loss_dict['total_loss'], loss_dict)

        return {
            'mse_loss': loss_dict['mse_loss'].item(),
            'nce_loss': loss_dict['nce_loss'].item(),
            'total_loss': loss_dict['total_loss'].item(),
            'nce_acc': loss_dict.get('nce_acc', torch.tensor(0.0)).item(),
        }

    def train_epoch(
        self,
        train_loader: Iterator,
        val_loader: Optional[Iterator] = None,
        epoch: int = 0,
        log_interval: int = 10,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train for one epoch over the data loader."""
        self.model.train()
        epoch_start = time.time()
        epoch_metrics = {'mse_loss': [], 'nce_loss': [], 'total_loss': [], 'nce_acc': []}

        for batch_idx, batch in enumerate(train_loader):
            if num_batches and batch_idx >= num_batches:
                break

            images, input_ids, attention_mask = _unpack_batch(batch)
            metrics = self.train_step(images, input_ids, attention_mask)

            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = sum(epoch_metrics['total_loss'][-log_interval:]) / log_interval
                print(f"  Batch {batch_idx+1:4d} | Loss: {avg_loss:.4f} | "
                      f"MSE: {metrics['mse_loss']:.4f} | NCE: {metrics['nce_loss']:.4f} | "
                      f"GN: {metrics['grad_norm']:.2f} | LR: {metrics['lr']:.2e}")

        # Validation
        val_metrics = None
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)

        elapsed = time.time() - epoch_start

        summary = {
            'epoch': epoch,
            'avg_mse': sum(epoch_metrics['mse_loss']) / len(epoch_metrics['mse_loss']),
            'avg_nce': sum(epoch_metrics['nce_loss']) / len(epoch_metrics['nce_loss']),
            'avg_loss': sum(epoch_metrics['total_loss']) / len(epoch_metrics['total_loss']),
            'avg_nce_acc': sum(epoch_metrics['nce_acc']) / len(epoch_metrics['nce_acc']),
            'elapsed': elapsed,
            'steps_per_sec': len(epoch_metrics['total_loss']) / elapsed if elapsed > 0 else 0,
        }

        if val_metrics:
            summary['val_loss'] = val_metrics['total_loss']
            summary['val_mse'] = val_metrics['mse_loss']
            summary['val_nce'] = val_metrics['nce_loss']
            summary['val_nce_acc'] = val_metrics['nce_acc']

        return summary

    @torch.no_grad()
    def evaluate(self, val_loader: Iterator, num_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        metrics_sum = {'mse_loss': 0.0, 'nce_loss': 0.0, 'total_loss': 0.0, 'nce_acc': 0.0}
        count = 0

        for batch_idx, batch in enumerate(val_loader):
            if num_batches and batch_idx >= num_batches:
                break

            images, input_ids, attention_mask = _unpack_batch(batch)
            metrics = self.eval_step(
                images,
                input_ids,
                attention_mask,
                mask_seed=self.eval_mask_seed + batch_idx,
            )
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
            count += 1

        if count == 0:
            raise ValueError("Cannot evaluate an empty validation loader")
        return {k: v / count for k, v in metrics_sum.items()}

    def save_checkpoint(self, path: str, extra: Optional[Dict] = None):
        """Save full training state."""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'step': self._step,
            'running_loss': self.running_loss,
            'running_steps': self.running_steps,
        }
        if extra:
            state.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load training state (tolerant of missing/extra keys in older checkpoints)."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except (ValueError, KeyError):
                pass
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (ValueError, KeyError):
                    pass
            if self.scaler and checkpoint.get('scaler_state_dict'):
                try:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except (ValueError, KeyError):
                    pass
            self._step = checkpoint.get('step', 0)
            self.running_loss = checkpoint.get('running_loss', 0.0)
            self.running_steps = checkpoint.get('running_steps', 0)