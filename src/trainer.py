"""
Training utilities for VL-JEPA.
Fixed with proper JEPA training loop:
  - JEPA loss: MSE on masked patches + InfoNCE cross-modal alignment
  - Momentum update of target encoder (EMA)
  - AMP + configurable global gradient clipping (default max_grad_norm=2.0)
  - Validation split with eval loop
  - TensorBoard logging
"""

import contextlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, Iterator, Any, Union
from pathlib import Path

from .model import (
    LOGIT_SCALE_MAX,
    LOGIT_SCALE_MIN,
    MemoryBank,
    VL_JEPA,
    compute_jepa_loss,
    make_multicrop_views,
)

# Prevent GradScaler from growing init_scale=1024 into ~500K+ over long runs,
# which can overflow FP16 backward passes on consumer GPUs.
# PyTorch requires growth_factor > 1; a huge interval effectively freezes growth.
_GRAD_SCALER_GROWTH_INTERVAL = 2**31 - 1
_MAX_GRAD_SCALER_SCALE = 8192.0

_CHECKPOINT_STATE_KEYS = frozenset({
    'model_state_dict',
    'optimizer_state_dict',
    'scheduler_state_dict',
    'scaler_state_dict',
    'memory_bank_state_dict',
    'model_ema_state_dict',
    'model_eval_state',
    'step',
    'running_loss',
    'running_steps',
    'skipped_batches',
})

# Forward outputs checked before loss (catches NaN activations with finite-looking loss).
_OUTPUT_FINITE_KEYS = (
    'predicted_patches',
    'target_patches',
    'vision_proj',
    'language_proj',
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is invalid or contains non-finite tensors."""


@torch.no_grad()
def tensor_summary(t: torch.Tensor) -> Dict[str, Any]:
    """Compact finite/min/max/mean stats for NaN diagnostics."""
    if not isinstance(t, torch.Tensor):
        return {'error': f'not a tensor: {type(t).__name__}'}
    finite_mask = torch.isfinite(t)
    finite_frac = float(finite_mask.float().mean().item()) if t.numel() else 1.0
    summary: Dict[str, Any] = {
        'shape': list(t.shape),
        'dtype': str(t.dtype),
        'finite_frac': finite_frac,
    }
    if t.numel() == 0:
        summary.update({'min': None, 'max': None, 'mean': None, 'std': None})
        return summary
    if finite_mask.any():
        vals = t[finite_mask].float()
        summary.update({
            'min': float(vals.min().item()),
            'max': float(vals.max().item()),
            'mean': float(vals.mean().item()),
            'std': float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0,
        })
    else:
        summary.update({'min': None, 'max': None, 'mean': None, 'std': None})
    return summary


def find_first_nonfinite_output(outputs: Dict[str, torch.Tensor]) -> Optional[str]:
    """Return name of first forward output tensor containing NaN/Inf."""
    for key in _OUTPUT_FINITE_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0 and not torch.isfinite(value).all():
            return key
    return None


def find_first_nonfinite_input(
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Optional[str]:
    """Return name of first batch input tensor containing NaN/Inf."""
    if images.numel() > 0 and not torch.isfinite(images).all():
        return 'images'
    if input_ids.numel() > 0 and not torch.isfinite(input_ids.float()).all():
        return 'input_ids'
    if attention_mask is not None and attention_mask.numel() > 0:
        if not torch.isfinite(attention_mask.float()).all():
            return 'attention_mask'
    return None


def summarize_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Per-key tensor summaries for diagnostic dumps."""
    out: Dict[str, Any] = {}
    for key in _OUTPUT_FINITE_KEYS:
        value = outputs.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = tensor_summary(value)
    patch_mask = outputs.get('patch_mask')
    if isinstance(patch_mask, torch.Tensor):
        out['patch_mask'] = {
            'shape': list(patch_mask.shape),
            'masked_frac': float(patch_mask.float().mean().item()),
        }
    logit_scale = outputs.get('logit_scale')
    if isinstance(logit_scale, torch.Tensor):
        out['logit_scale'] = float(logit_scale.detach().float().item())
    return out


def _find_nonfinite_tensor(obj: Any, *, prefix: str = "") -> Optional[str]:
    """Return location of first non-finite tensor in nested state, or None if clean."""
    if isinstance(obj, torch.Tensor):
        if obj.numel() > 0 and not torch.isfinite(obj).all():
            return prefix or "<tensor>"
        return None
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found = _find_nonfinite_tensor(value, prefix=child_prefix)
            if found is not None:
                return found
        return None
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            child_prefix = f"{prefix}[{index}]"
            found = _find_nonfinite_tensor(value, prefix=child_prefix)
            if found is not None:
                return found
        return None
    return None


def validate_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    path: Optional[Union[str, Path]] = None,
    check_optimizer: bool = True,
    check_scaler: bool = True,
) -> None:
    """Validate checkpoint tensors before resume. Raises CheckpointError if corrupt."""
    location = f" ({path})" if path else ""
    if not isinstance(checkpoint, dict):
        raise CheckpointError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}{location}")

    model_state = checkpoint.get('model_state_dict')
    if model_state is None:
        raise CheckpointError(
            f"Checkpoint missing 'model_state_dict'{location}. "
            "Use --fresh to train from scratch or pass a valid --resume PATH."
        )

    bad = _find_nonfinite_tensor(model_state, prefix='model')
    if bad is not None:
        raise CheckpointError(
            f"Corrupted checkpoint{location}: non-finite values in model weights at {bad}. "
            "Use --fresh to start from scratch or --resume PATH with a healthy checkpoint."
        )

    if check_optimizer and checkpoint.get('optimizer_state_dict') is not None:
        bad = _find_nonfinite_tensor(checkpoint['optimizer_state_dict'], prefix='optimizer')
        if bad is not None:
            raise CheckpointError(
                f"Corrupted checkpoint{location}: non-finite values in optimizer state at {bad}. "
                "Use --fresh or --resume PATH with optimizer state omitted (weights-only load)."
            )

    if check_scaler and checkpoint.get('scaler_state_dict') is not None:
        bad = _find_nonfinite_tensor(checkpoint['scaler_state_dict'], prefix='scaler')
        if bad is not None:
            raise CheckpointError(
                f"Corrupted checkpoint{location}: non-finite values in AMP GradScaler state at {bad}. "
                "Use --fresh or resume from a checkpoint saved before AMP overflow."
            )


def _amp_autocast(device: torch.device):
    """AMP autocast with fallback for PyTorch < 2.4."""
    if device.type != 'cuda':
        return contextlib.nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast('cuda')
    return torch.cuda.amp.autocast()


def _make_grad_scaler(device: torch.device):
    """GradScaler with growth interval capped so scale stays near init_scale."""
    if device.type != 'cuda':
        return None
    kwargs = dict(
        init_scale=1024.0,
        backoff_factor=0.5,
        growth_interval=_GRAD_SCALER_GROWTH_INTERVAL,
    )
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', **kwargs)
    return torch.cuda.amp.GradScaler(**kwargs)


def _clamp_grad_scaler_scale(scaler) -> None:
    """Hard cap AMP scale after update (belt-and-suspenders vs runaway growth)."""
    if scaler is None:
        return
    scale = scaler.get_scale()
    if scale <= _MAX_GRAD_SCALER_SCALE:
        return
    device = scaler._scale.device if hasattr(scaler, '_scale') else 'cuda'
    scaler._scale.copy_(torch.tensor(_MAX_GRAD_SCALER_SCALE, device=device))


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack (images, input_ids) or (images, input_ids, attention_mask) batches."""
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch tuple/list, got {type(batch)}")
    if len(batch) < 2:
        raise ValueError(f"Batch must have at least 2 elements, got {len(batch)}")
    images, input_ids = batch[0], batch[1]
    attention_mask = batch[2] if len(batch) > 2 else None
    return images, input_ids, attention_mask


def _unpack_cached_batch(
    batch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack cached embedding batches (local, global, language_emb, attention_mask)."""
    if not isinstance(batch, (list, tuple)) or len(batch) != 4:
        raise ValueError(
            f"Cached batch must be (local, global, language_emb, attention_mask), "
            f"got {type(batch)} len={len(batch) if isinstance(batch, (list, tuple)) else 'n/a'}",
        )
    return batch[0], batch[1], batch[2], batch[3]


def loss_weights_for_epoch(epoch: int) -> Tuple[float, float, float]:
    """Return phase-training (alpha, beta, gamma) weights for a 1-based epoch.

    Phase A (warmup): contrastive-only so the projection heads stabilize.
    Phase B onward: a gentle JEPA prediction weight is mixed in. The MSE term
    regularizes the (partially unfrozen) vision encoder toward spatially
    predictable features instead of pure contrastive shortcuts, which helps
    curb the val-loss overfitting seen once vision unfreezes.
    """
    if epoch <= 3:
        return 0.0, 1.0, 0.01
    return 0.1, 0.9, 0.01


@torch.no_grad()
def retrieval_recall(
    model: nn.Module,
    loader: Iterator,
    device: torch.device,
    k_list: Tuple[int, ...] = (1, 5, 10),
    *,
    use_cached_embeddings: bool = False,
) -> Dict[str, float]:
    """Compute full-loader image/text retrieval Recall@K."""
    model.eval()
    image_feats = []
    text_feats = []

    for batch in loader:
        if use_cached_embeddings:
            local, global_emb, language_emb, attention_mask = _unpack_cached_batch(batch)
            global_emb = global_emb.to(device)
            language_emb = language_emb.to(device)
            attention_mask = attention_mask.to(device)
            vision_proj, language_proj = model.get_joint_embedding_from_encoder_outputs(
                global_emb.float(),
                language_emb.float(),
                attention_mask,
            )
        else:
            images, input_ids, attention_mask = _unpack_batch(batch)
            images = images.to(device)
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            vision_proj, language_proj = model.get_joint_embedding(
                images,
                input_ids,
                attention_mask,
            )
        image_feats.append(vision_proj.float().cpu())
        text_feats.append(language_proj.float().cpu())

    if not image_feats:
        raise ValueError("Cannot compute retrieval recall on an empty loader")

    image_feats = F.normalize(torch.cat(image_feats, dim=0), dim=-1)
    text_feats = F.normalize(torch.cat(text_feats, dim=0), dim=-1)
    sim = image_feats @ text_feats.T
    target = torch.arange(sim.size(0))

    out: Dict[str, float] = {}
    i2t_rank = sim.argsort(dim=1, descending=True)
    t2i_rank = sim.T.argsort(dim=1, descending=True)
    for k in k_list:
        k_eff = min(k, sim.size(0))
        out[f"i2t_r{k}"] = (
            i2t_rank[:, :k_eff] == target[:, None]
        ).any(dim=1).float().mean().item()
        out[f"t2i_r{k}"] = (
            t2i_rank[:, :k_eff] == target[:, None]
        ).any(dim=1).float().mean().item()
    return out


class ModelEMA:
    """Exponential moving average of *all* model tensors (params + buffers).

    Fine-tuning a zero-shot CLIP model on a small dataset (COCO, 118K pairs)
    drifts the weights away from the robust pretrained solution: retrieval R@1
    peaks after a couple of epochs and then steadily declines even as the
    contrastive training loss keeps falling (classic robustness loss, see
    Wortsman et al. "Robust fine-tuning of zero-shot models", CVPR 2022).

    Averaging the trajectory of weights (Polyak/EMA averaging) keeps a smoothed
    set of parameters that does not chase the late-epoch overfitting, and is the
    standard cheap remedy for the "peak-then-decline" pattern. We EMA every
    floating-point tensor; integer buffers (e.g. token ids) are copied as-is.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.decay
        for name, tensor in model.state_dict().items():
            shadow = self.shadow.get(name)
            if shadow is None:
                self.shadow[name] = tensor.detach().clone()
                continue
            if tensor.is_floating_point():
                shadow.mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(tensor.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor.to(self.shadow[name].device))


def wise_ft_interpolate(
    zeroshot: Dict[str, torch.Tensor],
    finetuned: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """WiSE-FT weight-space ensemble: ``(1 - alpha) * zeroshot + alpha * finetuned``.

    ``alpha=1`` returns the fine-tuned weights, ``alpha=0`` the zero-shot model.
    Intermediate values recover the best-of-both-worlds robustness reported by
    Wortsman et al. — they beat *every* individual fine-tuning checkpoint on the
    target distribution at no extra training or inference cost. Only
    floating-point tensors are interpolated; others are taken from ``finetuned``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"WiSE-FT alpha must be in [0, 1], got {alpha}")
    # Interpolate entirely on CPU, one parameter at a time. This is an
    # eval-time op (it can be slow), but for large backbones like ViT-L/14
    # (~600M params) doing it on the GPU would hold three full copies of every
    # tensor in VRAM at once and OOM on top of the live training memory.
    out: Dict[str, torch.Tensor] = {}
    for name, ft_tensor in finetuned.items():
        zs_tensor = zeroshot.get(name)
        if (
            zs_tensor is not None
            and ft_tensor.is_floating_point()
            and zs_tensor.shape == ft_tensor.shape
        ):
            ft_cpu = ft_tensor.detach().to("cpu", ft_tensor.dtype)
            zs_cpu = zs_tensor.detach().to("cpu", ft_tensor.dtype)
            out[name] = (1.0 - alpha) * zs_cpu + alpha * ft_cpu
        else:
            out[name] = ft_tensor.detach().to("cpu").clone()
    return out


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
        alpha: float = 0.5,        # MSE weight
        beta: float = 0.5,         # InfoNCE weight
        gamma: float = 0.1,         # Variance regularization weight
        label_smoothing: float = 0.0,  # SigLIP target smoothing (anti-overfit)
        hard_negative_weight: float = 0.0,  # delta: VSE++ hardest-negative ranking
        hard_negative_margin: float = 0.2,
        momentum_tau: float = 0.996,
        momentum_tau_end: float = 1.0,
        momentum_schedule_steps: int = 15_000,
        use_multi_crop: bool = False,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        eval_mask_seed: int = 17_029,
        max_grad_norm: float = 2.0,
        memory_bank_size: int = 65536,
        unfreeze_after_epoch: Optional[int] = None,
        unfreeze_vision_blocks: int = 2,
        unfreeze_text_blocks: int = 0,
        encoder_unfreeze_lr: float = 1e-5,
        check_finite: bool = True,
        nan_diagnostics_dir: Optional[Union[str, Path]] = None,
        use_cached_embeddings: bool = False,
        use_model_ema: bool = False,
        model_ema_decay: float = 0.999,
        wise_ft_alpha: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_cached_embeddings = use_cached_embeddings

        # ---- Robust fine-tuning state (model-weight EMA + WiSE-FT) ----
        # These directly target the "R@1 peaks at epoch 3 then declines while
        # the contrastive loss keeps falling" pattern: EMA smooths the weight
        # trajectory and WiSE-FT interpolates back toward the robust zero-shot
        # init. Both are no-ops at eval time unless explicitly enabled.
        self.use_model_ema = bool(use_model_ema)
        self.model_ema = (
            ModelEMA(self.model, model_ema_decay) if self.use_model_ema else None
        )
        self.wise_ft_alpha = float(wise_ft_alpha)
        # Snapshot the zero-shot init (CLIP-seeded weights) on CPU for WiSE-FT.
        # Only needed when we will actually interpolate (alpha < 1).
        self._zeroshot_state: Optional[Dict[str, torch.Tensor]] = None
        if self.wise_ft_alpha < 1.0:
            self._zeroshot_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }
        self._holding_eval_weights = False
        self._eval_live_snapshot: Optional[Dict[str, torch.Tensor]] = None
        projection_dim = getattr(model, "projection_dim", model.hidden_dim)
        self.memory_bank = (
            MemoryBank(memory_bank_size, projection_dim, device)
            if memory_bank_size and memory_bank_size > 0
            else None
        )
        self.check_finite = check_finite
        self.nan_diagnostics_dir = (
            Path(nan_diagnostics_dir) if nan_diagnostics_dir else None
        )
        self._weights_corrupted = False
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = float(label_smoothing)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_negative_margin = float(hard_negative_margin)
        self.use_multi_crop = use_multi_crop
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.eval_mask_seed = eval_mask_seed
        self.max_grad_norm = max_grad_norm
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.unfreeze_vision_blocks = max(1, int(unfreeze_vision_blocks))
        self.unfreeze_text_blocks = max(0, int(unfreeze_text_blocks))
        self.encoder_unfreeze_lr = float(encoder_unfreeze_lr)
        self._vision_unfrozen = False

        # EMA cosine schedule capped so tau reaches 1.0 in early training (~epoch 4),
        # not stretched across the full LR cosine horizon (100K+ steps).
        self.model.momentum_tau = momentum_tau
        self.momentum_tau_end = momentum_tau_end
        schedule_steps = max(1, min(max_steps, momentum_schedule_steps))
        self._momentum_schedule = self._cosine_schedule(
            momentum_tau, momentum_tau_end, schedule_steps,
        )

        # Optimizer with parameter groups (different LR for predictor)
        # Following I-JEPA: predictor gets 20x higher learning rate
        decay_params = []
        no_decay_params = []
        predictor_params = []
        proj_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'predictor' in name:
                predictor_params.append(p)
            elif 'vision_proj' in name or 'language_proj' in name:
                proj_params.append(p)
            elif 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                no_decay_params.append(p)
            elif 'logit_scale' in name or 'logit_bias' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        # Randomly-initialized MLP projection heads need a higher LR (10x) to
        # learn alignment from scratch. CLIP-seeded linear projections already
        # encode the pretrained alignment, so they fine-tune at the base LR —
        # a 10x LR would blow the pretrained matrices away in the first steps.
        proj_lr_scale = (
            1.0
            if getattr(model, 'projection_type', 'mlp') in ('clip', 'clip_residual')
            else 10.0
        )

        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': predictor_params, 'weight_decay': weight_decay, 'lr': learning_rate * 20.0},
            # Projection heads (LR scaled by backbone alignment, see above)
            {'params': proj_params, 'weight_decay': weight_decay, 'lr': learning_rate * proj_lr_scale},
        ]

        self.optimizer = optim.AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95))
        self.scaler = _make_grad_scaler(device)

        # Warmup + cosine scheduler (per param group; encoder group uses constant scale)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step = 0
        self._skipped_batches = 0

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_lambdas_for_optimizer(),
        )

        # Loss tracking
        self.running_loss = 0.0
        self.running_steps = 0

        # Gradient accumulation state. Gradients accumulate across micro-batches
        # until step_optimizer=True triggers an optimizer update.
        self._accum_counter = 0
        self._last_grad_norm = 0.0

    def _main_lr_lambda(self, current_step: int) -> float:
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _encoder_lr_lambda(_current_step: int) -> float:
        """Keep encoder unfreeze LR at its configured base (no warmup/cosine)."""
        return 1.0

    def _lr_lambdas_for_optimizer(self) -> list:
        n_groups = len(self.optimizer.param_groups)
        lambdas: list = [self._main_lr_lambda] * n_groups
        # Unfrozen encoder params are appended via add_param_group (5th+ group).
        if n_groups > 4:
            lambdas[-1] = self._encoder_lr_lambda
        return lambdas

    def _rebuild_scheduler(self) -> None:
        # Fresh scheduler: new param groups lack initial_lr required for last_epoch >= 0.
        steps_done = self._step
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_lambdas_for_optimizer(),
        )
        for _ in range(steps_done):
            self.scheduler.step()

    def _maybe_unfreeze_vision(self, epoch: Optional[int]) -> None:
        if self._vision_unfrozen:
            return
        if self.unfreeze_after_epoch is None or epoch is None:
            return
        if epoch < int(self.unfreeze_after_epoch):
            return

        vision_toggled = self.model.unfreeze_vision_last_blocks(self.unfreeze_vision_blocks)
        text_toggled = 0
        if self.unfreeze_text_blocks > 0:
            text_toggled = self.model.unfreeze_text_last_blocks(self.unfreeze_text_blocks)
        if vision_toggled <= 0 and text_toggled <= 0:
            self._vision_unfrozen = True
            return

        existing = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        new_params = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in existing
        ]
        if new_params:
            self.optimizer.add_param_group({
                'params': new_params,
                'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 0.0),
                'lr': self.encoder_unfreeze_lr,
            })
            self._rebuild_scheduler()
            text_msg = (
                f" + {self.unfreeze_text_blocks} text blocks" if text_toggled > 0 else ""
            )
            print(
                f"  -> Unfroze last {self.unfreeze_vision_blocks} vision blocks{text_msg} "
                f"at epoch {epoch} ({len(new_params)} tensors, lr={self.encoder_unfreeze_lr:.2e})"
            )
        self._vision_unfrozen = True

    @torch.no_grad()
    def _clamp_stability_params(self) -> None:
        """Keep scalar stability parameters inside their valid training range."""
        self.model.logit_scale.clamp_(LOGIT_SCALE_MIN, LOGIT_SCALE_MAX)

    @staticmethod
    def _loss_is_finite(loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor]) -> bool:
        if not torch.isfinite(loss):
            return False
        for key in ('mse_loss', 'nce_loss', 'var_loss', 'total_loss'):
            value = loss_dict.get(key)
            if isinstance(value, torch.Tensor) and not torch.isfinite(value):
                return False
        return True

    def _first_nonfinite_weight(self) -> Optional[str]:
        return _find_nonfinite_tensor(self.model.state_dict(), prefix='model')

    def _save_nan_diagnostic(
        self,
        reason: str,
        *,
        batch_index: Optional[int] = None,
        epoch: Optional[int] = None,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None,
        grad_norm: Optional[float] = None,
        nonfinite_location: Optional[str] = None,
    ) -> Optional[Path]:
        """Write a JSON diagnostic bundle when NaN/Inf is detected (if dir configured)."""
        if self.nan_diagnostics_dir is None:
            return None

        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        tag_parts = [reason]
        if epoch is not None:
            tag_parts.append(f'e{epoch}')
        if batch_index is not None:
            tag_parts.append(f'b{batch_index}')
        out_dir = self.nan_diagnostics_dir / f"nan_{stamp}_{'_'.join(tag_parts)}"
        out_dir.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            'reason': reason,
            'epoch': epoch,
            'batch_index': batch_index,
            'step': self._step,
            'skipped_batches': self._skipped_batches,
            'weights_corrupted': self._weights_corrupted,
            'nonfinite_location': nonfinite_location,
            'amp_scale': float(self.scaler.get_scale()) if self.scaler else None,
            'lr': self.optimizer.param_groups[0]['lr'],
            'momentum_tau': self.model.momentum_tau,
        }
        if images is not None:
            payload['images'] = tensor_summary(images)
        if input_ids is not None:
            payload['input_ids'] = tensor_summary(input_ids.float())
        if attention_mask is not None:
            payload['attention_mask'] = tensor_summary(attention_mask.float())
        if outputs is not None:
            payload['outputs'] = summarize_outputs(outputs)
        if loss_dict is not None:
            payload['loss'] = {
                key: float(value.detach().float().item())
                if isinstance(value, torch.Tensor) else value
                for key, value in loss_dict.items()
            }
        if grad_norm is not None:
            payload['grad_norm'] = grad_norm

        bad_weight = self._first_nonfinite_weight()
        if bad_weight is not None:
            payload['first_nonfinite_weight'] = bad_weight

        diag_path = out_dir / 'diagnostic.json'
        with open(diag_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        if images is not None:
            torch.save(
                {
                    'images': images.detach().cpu(),
                    'input_ids': input_ids.detach().cpu() if input_ids is not None else None,
                    'attention_mask': (
                        attention_mask.detach().cpu()
                        if attention_mask is not None else None
                    ),
                },
                out_dir / 'batch_inputs.pt',
            )
        return out_dir

    def _skipped_metrics(
        self,
        reason: str = 'non_finite',
        *,
        batch_index: Optional[int] = None,
        epoch: Optional[int] = None,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        loss_dict: Optional[Dict[str, torch.Tensor]] = None,
        grad_norm: Optional[float] = None,
        nonfinite_location: Optional[str] = None,
    ) -> Dict[str, float]:
        """Metrics for batches skipped due to NaN/Inf loss, inputs, outputs, or weights."""
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_counter = 0
        self._skipped_batches += 1
        self._save_nan_diagnostic(
            reason,
            batch_index=batch_index,
            epoch=epoch,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            outputs=outputs,
            loss_dict=loss_dict,
            grad_norm=grad_norm,
            nonfinite_location=nonfinite_location,
        )
        tau = self._current_momentum_tau()
        return {
            'mse_loss': float('nan'),
            'nce_loss': float('nan'),
            'total_loss': float('nan'),
            'grad_norm': grad_norm if grad_norm is not None else 0.0,
            'logit_scale': 0.0,
            'nce_acc': 0.0,
            'momentum_tau': tau,
            'lr': self.optimizer.param_groups[0]['lr'],
            'skipped': True,
            'skip_reason': reason,
            'weights_corrupted': self._weights_corrupted,
        }

    def _current_momentum_tau(self) -> float:
        idx = self._step - 1
        if 0 <= idx < len(self._momentum_schedule):
            return self._momentum_schedule[idx].item()
        return self.momentum_tau_end

    def _forward_model(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        training: bool,
        mask_seed: Optional[int] = None,
        local_patch_emb: Optional[torch.Tensor] = None,
        global_patch_emb: Optional[torch.Tensor] = None,
        language_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_cached_embeddings:
            if local_patch_emb is None or global_patch_emb is None or language_emb is None:
                raise ValueError("Cached forward requires local/global/language embeddings")
            return self.model.forward_from_cache(
                local_patch_emb.float(),
                global_patch_emb.float(),
                language_emb.float(),
                attention_mask,
                mask_seed=mask_seed,
                compute_jepa=self.alpha != 0,
            )

        compute_jepa = self.alpha != 0
        if not self.use_multi_crop:
            return self.model(
                images, input_ids, attention_mask, mask_seed=mask_seed,
                compute_jepa=compute_jepa,
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
            compute_jepa=compute_jepa,
        )

    @staticmethod
    def _cosine_schedule(start: float, end: float, steps: int) -> torch.Tensor:
        """Cosine schedule from start to end over `steps`."""
        t = torch.linspace(0, 1, steps)
        return end + (start - end) * (1 + torch.cos(t * math.pi)) / 2

    def _maybe_sync_batchnorm(self):
        """Sync batch norm if using DataParallel (no-op for now)."""
        pass

    def _backward_and_step(
        self,
        loss: torch.Tensor,
        accumulation_steps: int = 1,
        step_optimizer: bool = True,
    ) -> Tuple[bool, float]:
        """Backward (accumulating), then clip + optimizer step when requested.

        Gradients accumulate across micro-batches. ``zero_grad`` runs only when a
        new accumulation window starts; the optimizer/scheduler/EMA updates run
        only when ``step_optimizer`` is True. The micro-batch loss is scaled by
        ``1 / accumulation_steps`` so a *full* window matches a full-batch mean
        gradient. A short final window (``len % accum != 0``) is rescaled by
        ``accumulation_steps / actual_microbatches`` before the clip/step so the
        partial window is not under-weighted.

        Returns (success, grad_norm). For pure-accumulation calls (no optimizer
        step) success reflects only that backward ran and grad_norm is the last
        computed value.
        """
        accumulation_steps = max(1, int(accumulation_steps))

        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        scaled_loss = loss / accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        self._accum_counter += 1

        if not step_optimizer:
            # Still inside the accumulation window; defer the optimizer update.
            return True, self._last_grad_norm

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        actual = self._accum_counter
        if actual > 0 and actual != accumulation_steps:
            scale = accumulation_steps / float(actual)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        if self.scaler is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=False,
            )
            grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not math.isfinite(grad_norm_val):
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_counter = 0
                self.scaler.update()
                _clamp_grad_scaler_scale(self.scaler)
                self._last_grad_norm = grad_norm_val
                return False, grad_norm_val

            self.scaler.step(self.optimizer)
            self.scaler.update()
            _clamp_grad_scaler_scale(self.scaler)
            self.optimizer._opt_called = True
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=False,
            )
            grad_norm_val = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            if not math.isfinite(grad_norm_val):
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_counter = 0
                self._last_grad_norm = grad_norm_val
                return False, grad_norm_val

            self.optimizer.step()

        self._accum_counter = 0
        self._clamp_stability_params()
        self.scheduler.step()
        self._step += 1

        tau = self._current_momentum_tau()
        self.model.momentum_tau = tau
        if not self.use_cached_embeddings:
            self.model.momentum_update()

        if self.model_ema is not None:
            self.model_ema.update(self.model)

        self._last_grad_norm = grad_norm_val
        return True, grad_norm_val

    def train_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        batch_index: Optional[int] = None,
        epoch: Optional[int] = None,
        local_patch_emb: Optional[torch.Tensor] = None,
        global_patch_emb: Optional[torch.Tensor] = None,
        language_emb: Optional[torch.Tensor] = None,
        accumulation_steps: int = 1,
        step_optimizer: bool = True,
    ) -> Dict[str, float]:
        """Single training step with proper JEPA loss.

        With gradient accumulation, set ``accumulation_steps`` to the window size
        and ``step_optimizer=True`` only on the final micro-batch of each window
        (or the last batch of the epoch). Gradients accumulate across micro-batch
        calls and the optimizer/scheduler/EMA update only fires on the step call.
        """
        self.model.train()
        self._maybe_unfreeze_vision(epoch)

        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if local_patch_emb is not None:
            local_patch_emb = local_patch_emb.to(self.device, non_blocking=True)
        if global_patch_emb is not None:
            global_patch_emb = global_patch_emb.to(self.device, non_blocking=True)
        if language_emb is not None:
            language_emb = language_emb.to(self.device, non_blocking=True)

        skip_kw = dict(
            batch_index=batch_index,
            epoch=epoch,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if self.check_finite:
            bad_weight = self._first_nonfinite_weight()
            if bad_weight is not None:
                self._weights_corrupted = True
                return self._skipped_metrics(
                    'corrupt_weights_pre_forward',
                    nonfinite_location=bad_weight,
                    **skip_kw,
                )

            if self.use_cached_embeddings:
                for name, tensor in (
                    ('local_patch_emb', local_patch_emb),
                    ('global_patch_emb', global_patch_emb),
                    ('language_emb', language_emb),
                ):
                    if tensor is not None and tensor.numel() > 0 and not torch.isfinite(tensor).all():
                        return self._skipped_metrics(
                            'non_finite_input',
                            nonfinite_location=name,
                            **skip_kw,
                        )
            else:
                bad_input = find_first_nonfinite_input(images, input_ids, attention_mask)
                if bad_input is not None:
                    return self._skipped_metrics(
                        'non_finite_input',
                        nonfinite_location=bad_input,
                        **skip_kw,
                    )

        with _amp_autocast(self.device):
            outputs = self._forward_model(
                images,
                input_ids,
                attention_mask,
                training=True,
                local_patch_emb=local_patch_emb,
                global_patch_emb=global_patch_emb,
                language_emb=language_emb,
            )

            if self.check_finite:
                bad_output = find_first_nonfinite_output(outputs)
                if bad_output is not None:
                    return self._skipped_metrics(
                        'non_finite_output',
                        outputs=outputs,
                        nonfinite_location=bad_output,
                        **skip_kw,
                    )

            loss_dict = compute_jepa_loss(
                outputs, self.alpha, self.beta, self.gamma, memory_bank=self.memory_bank,
                label_smoothing=self.label_smoothing,
                hard_negative_weight=self.hard_negative_weight,
                hard_negative_margin=self.hard_negative_margin,
            )
            loss = loss_dict['total_loss']

        if not self._loss_is_finite(loss, loss_dict):
            return self._skipped_metrics(
                'non_finite_loss',
                outputs=outputs,
                loss_dict=loss_dict,
                **skip_kw,
            )

        ok, grad_norm = self._backward_and_step(
            loss, accumulation_steps=accumulation_steps, step_optimizer=step_optimizer,
        )
        if not ok:
            return self._skipped_metrics(
                'non_finite_grad',
                outputs=outputs,
                loss_dict=loss_dict,
                grad_norm=grad_norm,
                **skip_kw,
            )

        queue_keys = outputs.get('language_proj')
        if self.memory_bank is not None and queue_keys is not None:
            self.memory_bank.enqueue(queue_keys)

        if self.check_finite:
            bad_weight = self._first_nonfinite_weight()
            if bad_weight is not None:
                self._weights_corrupted = True
                return self._skipped_metrics(
                    'corrupt_weights_post_step',
                    outputs=outputs,
                    loss_dict=loss_dict,
                    grad_norm=grad_norm,
                    nonfinite_location=bad_weight,
                    **skip_kw,
                )

        self.running_loss += loss.item()
        self.running_steps += 1

        metrics = {
            'mse_loss': loss_dict['mse_loss'].item(),
            'nce_loss': loss_dict['nce_loss'].item(),
            'var_loss': loss_dict['var_loss'].item(),
            'total_loss': loss.item(),
            'grad_norm': grad_norm,
            'logit_scale': loss_dict.get('logit_scale', torch.tensor(0.0)).item(),
            'nce_acc': loss_dict.get('nce_acc', torch.tensor(0.0)).item(),
            'momentum_tau': self.model.momentum_tau,
            'lr': self.optimizer.param_groups[0]['lr'],
            'skipped': False,
            'weights_corrupted': False,
        }
        if self.device.type == 'cuda':
            metrics['vram_alloc_gb'] = torch.cuda.memory_allocated(self.device) / 1e9
            metrics['vram_reserved_gb'] = torch.cuda.memory_reserved(self.device) / 1e9

        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_seed: Optional[int] = None,
        *,
        local_patch_emb: Optional[torch.Tensor] = None,
        global_patch_emb: Optional[torch.Tensor] = None,
        language_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single evaluation step (no gradients, no EMA update)."""
        self.model.eval()

        images = images.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if local_patch_emb is not None:
            local_patch_emb = local_patch_emb.to(self.device, non_blocking=True)
        if global_patch_emb is not None:
            global_patch_emb = global_patch_emb.to(self.device, non_blocking=True)
        if language_emb is not None:
            language_emb = language_emb.to(self.device, non_blocking=True)

        with _amp_autocast(self.device):
            outputs = self._forward_model(
                images,
                input_ids,
                attention_mask,
                training=False,
                mask_seed=mask_seed,
                local_patch_emb=local_patch_emb,
                global_patch_emb=global_patch_emb,
                language_emb=language_emb,
            )
            loss_dict = compute_jepa_loss(
                outputs, self.alpha, self.beta, self.gamma, memory_bank=None,
                label_smoothing=self.label_smoothing,
                hard_negative_weight=self.hard_negative_weight,
                hard_negative_margin=self.hard_negative_margin,
            )
            if not self._loss_is_finite(loss_dict['total_loss'], loss_dict):
                return {
                    'mse_loss': float('nan'),
                    'nce_loss': float('nan'),
                    'var_loss': float('nan'),
                    'total_loss': float('nan'),
                    'nce_acc': 0.0,
                    'skipped': True,
                }

        return {
            'mse_loss': loss_dict['mse_loss'].item(),
            'nce_loss': loss_dict['nce_loss'].item(),
            'var_loss': loss_dict['var_loss'].item(),
            'total_loss': loss_dict['total_loss'].item(),
            'nce_acc': loss_dict.get('nce_acc', torch.tensor(0.0)).item(),
            'skipped': False,
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
        skipped = 0

        for batch_idx, batch in enumerate(train_loader):
            if num_batches and batch_idx >= num_batches:
                break

            if self.use_cached_embeddings:
                local, global_emb, language_emb, attention_mask = _unpack_cached_batch(batch)
                metrics = self.train_step(
                    images=torch.empty(0),
                    input_ids=torch.empty(0),
                    attention_mask=attention_mask,
                    batch_index=batch_idx,
                    epoch=epoch + 1,
                    local_patch_emb=local,
                    global_patch_emb=global_emb,
                    language_emb=language_emb,
                )
            else:
                images, input_ids, attention_mask = _unpack_batch(batch)
                metrics = self.train_step(
                    images, input_ids, attention_mask, batch_index=batch_idx, epoch=epoch + 1,
                )

            if metrics.get('skipped'):
                skipped += 1
                continue

            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])

            if (batch_idx + 1) % log_interval == 0 and epoch_metrics['total_loss']:
                avg_loss = sum(epoch_metrics['total_loss'][-log_interval:]) / min(
                    log_interval, len(epoch_metrics['total_loss']),
                )
                print(f"  Batch {batch_idx+1:4d} | Loss: {avg_loss:.4f} | "
                      f"MSE: {metrics['mse_loss']:.4f} | NCE: {metrics['nce_loss']:.4f} | "
                      f"GN: {metrics['grad_norm']:.2f} | LR: {metrics['lr']:.2e}")

        # Validation
        val_metrics = None
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)

        elapsed = time.time() - epoch_start
        n = len(epoch_metrics['total_loss'])

        summary = {
            'epoch': epoch,
            'avg_mse': sum(epoch_metrics['mse_loss']) / n if n else float('nan'),
            'avg_nce': sum(epoch_metrics['nce_loss']) / n if n else float('nan'),
            'avg_loss': sum(epoch_metrics['total_loss']) / n if n else float('nan'),
            'avg_nce_acc': sum(epoch_metrics['nce_acc']) / n if n else 0.0,
            'elapsed': elapsed,
            'steps_per_sec': n / elapsed if elapsed > 0 else 0,
            'skipped_batches': skipped,
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

            if self.use_cached_embeddings:
                local, global_emb, language_emb, attention_mask = _unpack_cached_batch(batch)
                metrics = self.eval_step(
                    torch.empty(0),
                    torch.empty(0),
                    attention_mask,
                    mask_seed=self.eval_mask_seed + batch_idx,
                    local_patch_emb=local,
                    global_patch_emb=global_emb,
                    language_emb=language_emb,
                )
            else:
                images, input_ids, attention_mask = _unpack_batch(batch)
                metrics = self.eval_step(
                    images,
                    input_ids,
                    attention_mask,
                    mask_seed=self.eval_mask_seed + batch_idx,
                )
            if metrics.get('skipped'):
                continue
            for k in metrics_sum:
                metrics_sum[k] += metrics[k]
            count += 1

        if count == 0:
            raise ValueError("Cannot evaluate an empty validation loader")
        return {k: v / count for k, v in metrics_sum.items()}

    def build_eval_state_dict(
        self,
        live_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the state dict to evaluate with (EMA + WiSE-FT applied).

        Precedence: start from the EMA weights if model EMA is enabled, else the
        live student (``live_state_dict`` when provided, otherwise
        ``model.state_dict()``); then, if ``wise_ft_alpha < 1``, interpolate
        toward the zero-shot init. Passing the live snapshot is required when
        the module currently holds eval weights so WiSE-FT is not applied twice.
        """
        # Build the eval state on CPU. The interpolation and any large-backbone
        # weight juggling happens off the GPU so it cannot OOM on top of the
        # live training allocation; eval_weights() loads it back to the GPU
        # parameter-by-parameter at the end.
        if self.model_ema is not None:
            src = self.model_ema.state_dict()
        elif live_state_dict is not None:
            src = live_state_dict
        elif self._eval_live_snapshot is not None:
            src = self._eval_live_snapshot
        else:
            src = self.model.state_dict()
        base = {k: v.detach().to("cpu") for k, v in src.items()}
        if self._zeroshot_state is not None and self.wise_ft_alpha < 1.0:
            base = wise_ft_interpolate(
                self._zeroshot_state, base, self.wise_ft_alpha,
            )
        return base

    def snapshot_live_state(self) -> Dict[str, torch.Tensor]:
        """CPU clone of the live student weights (for best-checkpoint saves)."""
        return {
            k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()
        }

    @staticmethod
    def _cpu_clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.detach().to("cpu").clone()
            else:
                out[key] = value
        return out

    @contextlib.contextmanager
    def eval_weights(self):
        """Temporarily load the robust (EMA/WiSE-FT) weights for evaluation.

        Restores the live training weights on exit so training is unaffected.
        A no-op when neither model EMA nor WiSE-FT is active.
        """
        if self.model_ema is None and (
            self._zeroshot_state is None or self.wise_ft_alpha >= 1.0
        ):
            yield
            return
        # Snapshot the live weights on CPU so we don't keep a second full copy
        # of the model in VRAM while evaluating. load_state_dict copies each
        # parameter back in place, so the CPU->GPU transfer is one tensor at a
        # time rather than a full GPU duplicate.
        live = self.snapshot_live_state()
        self._eval_live_snapshot = live
        self._holding_eval_weights = True
        try:
            self.model.load_state_dict(
                self.build_eval_state_dict(live_state_dict=live), strict=False,
            )
            yield
        finally:
            self.model.load_state_dict(live, strict=False)
            self._holding_eval_weights = False
            self._eval_live_snapshot = None

    def save_checkpoint(
        self,
        path: str,
        extra: Optional[Dict] = None,
        live_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Save full training state.

        ``model_state_dict`` is always the live student. Pass ``live_state_dict``
        (or rely on the snapshot taken by ``eval_weights``) when saving inside
        that context so the swapped eval tensors are not written as the student.
        ``model_eval_state`` is the EMA/WiSE-FT weights that produced the
        retrieval score — a single interpolation, never a second WiSE-FT apply
        on already-interpolated tensors.
        """
        holding = self._holding_eval_weights
        if live_state_dict is not None:
            model_state = self._cpu_clone_state(live_state_dict)
        elif holding and self._eval_live_snapshot is not None:
            model_state = self._cpu_clone_state(self._eval_live_snapshot)
        else:
            model_state = self.model.state_dict()

        if holding:
            eval_state = self._cpu_clone_state(self.model.state_dict())
        else:
            eval_state = self.build_eval_state_dict(live_state_dict=model_state)

        state = {
            'model_state_dict': model_state,
            'model_eval_state': eval_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'memory_bank_state_dict': (
                self.memory_bank.state_dict() if self.memory_bank is not None else None
            ),
            'model_ema_state_dict': (
                self.model_ema.state_dict() if self.model_ema is not None else None
            ),
            'step': self._step,
            'running_loss': self.running_loss,
            'running_steps': self.running_steps,
            'skipped_batches': self._skipped_batches,
        }
        if extra:
            state.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        *,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Load training state (tolerant of missing/extra keys in older checkpoints).

        Returns checkpoint metadata (epoch, val_loss, config, etc.) excluding state dicts.
        Raises CheckpointError if validate=True and tensors contain NaN/Inf.
        """
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)

        if validate:
            validate_checkpoint(
                checkpoint,
                path=ckpt_path,
                check_optimizer=load_optimizer,
                check_scaler=load_optimizer,
            )

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self.memory_bank is not None and checkpoint.get('memory_bank_state_dict') is not None:
            self.memory_bank.load_state_dict(checkpoint['memory_bank_state_dict'])
        if self.model_ema is not None and checkpoint.get('model_ema_state_dict') is not None:
            self.model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
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
            self._skipped_batches = checkpoint.get('skipped_batches', 0)

        return {
            key: value
            for key, value in checkpoint.items()
            if key not in _CHECKPOINT_STATE_KEYS
        }
