"""
VL-JEPA Experiment: Proper JEPA Training
-----------------------------------------
Trains the fixed VL-JEPA model with:

  L = α * MSE(predicted_patches, target_patches) + β * InfoNCE(visual_cls, text_cls)

Features:
  - Real masking (75% vision patches, 15% language tokens)
  - EMA target encoder (τ = 0.996 → 1.0 cosine schedule)
  - Predictor (6-layer transformer)
  - AMP + gradient clipping for GPU efficiency
  - COCO 2017 train/val caption dataloaders
  - Checkpointing + TensorBoard logging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]
    _HAS_TENSORBOARD = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer
from src.config import load_config, overrides_from_cli, print_config
from src.dataset import (
    COCOCaptionDataset,
    create_dataloaders,
    expected_split_length,
)


class _GraphTraceWrapper(torch.nn.Module):
    """Adapter so TensorBoard add_graph can trace VL-JEPA (dict outputs -> tensor)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(images, input_ids, attention_mask)
        return outputs['predicted_patches'].sum()


def _expand_coco_root(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.expanduser(path)


def _resolve_max_steps(
    train_cfg: Dict[str, Any],
    train_batches: int,
) -> int:
    """Derive total optimizer steps from epochs × train batches when unset."""
    max_steps = train_cfg.get("max_steps")
    if max_steps is not None:
        return int(max_steps)
    epochs = int(train_cfg.get("epochs", 1))
    return epochs * max(1, train_batches)


def _build_parser(base_cfg: dict) -> argparse.ArgumentParser:
    model = base_cfg.get("model", {})
    training = base_cfg.get("training", {})
    loss = base_cfg.get("loss", {})
    data = base_cfg.get("data", {})
    output = base_cfg.get("output", {})

    parser = argparse.ArgumentParser(description='VL-JEPA Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML config (default: configs/default.yaml)',
    )
    parser.add_argument('--epochs', type=int, default=training.get('epochs', 15),
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=training.get('batch_size', 32),
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=training.get('learning_rate', 3e-4),
                        help='Peak learning rate')
    parser.add_argument('--weight-decay', type=float, default=training.get('weight_decay', 0.05),
                        help='AdamW weight decay')
    parser.add_argument('--warmup', type=int, default=training.get('warmup_steps', 500),
                        help='Warmup steps')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Total training steps (default: epochs × train batches)')
    parser.add_argument('--multi-crop', action=argparse.BooleanOptionalAction,
                        default=training.get('use_multi_crop', False),
                        help='Use global target crops and local masked context crops')
    parser.add_argument('--global-crop-size', type=int,
                        default=training.get('global_crop_size', 224),
                        help='Global target crop size')
    parser.add_argument('--local-crop-size', type=int,
                        default=training.get('local_crop_size', 96),
                        help='Local context crop size')
    parser.add_argument('--hidden-dim', type=int, default=model.get('hidden_dim', 768),
                        help='Hidden dimension')
    parser.add_argument('--patch-size', type=int, default=model.get('patch_size', 16),
                        help='Vision patch size')
    parser.add_argument('--image-size', type=int, default=model.get('image_size', 224),
                        help='Input image size')
    parser.add_argument('--mask-ratio', type=float, default=model.get('mask_ratio', 0.75),
                        help='Fraction of vision patches to mask')
    parser.add_argument('--text-mask-ratio', type=float, default=model.get('text_mask_ratio', 0.0),
                        help='Fraction of language tokens to mask (default: 0, no MLM loss)')
    parser.add_argument('--predictor-layers', type=int, default=model.get('predictor_layers', 6),
                        help='Number of predictor transformer layers')
    parser.add_argument('--momentum-tau', type=float, default=model.get('momentum_tau', 0.996),
                        help='EMA momentum coefficient (start)')
    parser.add_argument(
        '--coco-root',
        type=str,
        default=data.get('coco_root', '~/.cache/torch/hub/checkpoints'),
        help='COCO 2017 root (train2017/, val2017/, annotations/)',
    )
    parser.add_argument(
        '--download',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Download missing COCO archives (default: on)',
    )
    parser.add_argument(
        '--max-caption-length',
        type=int,
        default=data.get('max_caption_length', 64),
        help='Max tokenized caption length',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=data.get('num_workers', 4),
        help='DataLoader worker processes',
    )
    parser.add_argument('--output-dir', type=str, default=output.get('output_dir', 'experiments'),
                        help='Output directory for metrics/checkpoints')
    parser.add_argument('--alpha', type=float, default=loss.get('alpha', 1.0),
                        help='MSE loss weight')
    parser.add_argument('--beta', type=float, default=loss.get('beta', 0.5),
                        help='InfoNCE loss weight')
    parser.add_argument('--log-interval', type=int, default=output.get('log_interval', 10),
                        help='Log every N batches')
    parser.add_argument('--checkpoint-interval', type=int,
                        default=output.get('checkpoint_interval', 5),
                        help='Save periodic checkpoint every N epochs')
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable TensorBoard logging (default: on)')
    parser.add_argument('--tensorboard-dir', type=str, default=None,
                        help='TensorBoard log directory (default: <exp_dir>/tensorboard)')
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML config',
    )
    pre_args, remaining = pre_parser.parse_known_args()
    base_cfg = load_config(pre_args.config)

    parser = _build_parser(base_cfg)
    args = parser.parse_args(remaining)

    cfg = load_config(
        args.config,
        overrides=overrides_from_cli(
            hidden_dim=args.hidden_dim,
            patch_size=args.patch_size,
            image_size=args.image_size,
            mask_ratio=args.mask_ratio,
            text_mask_ratio=args.text_mask_ratio,
            predictor_layers=args.predictor_layers,
            momentum_tau=args.momentum_tau,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup,
            max_steps=args.max_steps,
            use_multi_crop=args.multi_crop,
            global_crop_size=args.global_crop_size,
            local_crop_size=args.local_crop_size,
            alpha=args.alpha,
            beta=args.beta,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
        ),
    )

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]
    out_cfg = cfg["output"]

    coco_root = _expand_coco_root(args.coco_root)
    image_size = model_cfg["image_size"]
    batch_size = train_cfg["batch_size"]

    print("=" * 70)
    print("VL-JEPA v2 — Proper JEPA Training")
    print("=" * 70)
    print(f"\nConfiguration (config: {args.config}):")
    print_config(cfg)
    print(f"  COCO root: {coco_root}")
    print(f"  Download missing archives: {args.download}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    model = VL_JEPA(
        hidden_dim=model_cfg["hidden_dim"],
        patch_size=model_cfg["patch_size"],
        image_size=model_cfg["image_size"],
        mask_ratio=model_cfg["mask_ratio"],
        predictor_layers=model_cfg["predictor_layers"],
        momentum_tau=model_cfg["momentum_tau"],
        text_mask_ratio=model_cfg.get("text_mask_ratio", 0.0),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params/1e6:.1f}M params ({trainable_params/1e6:.1f}M trainable)")

    print("\nLoading COCO 2017 caption dataloaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=args.num_workers,
        coco_root=coco_root,
        image_size=image_size,
        max_caption_length=args.max_caption_length,
        download=args.download,
    )

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)

    max_steps = _resolve_max_steps(train_cfg, train_batches)
    train_cfg["max_steps"] = max_steps

    print(f"\nData: COCO 2017 captions")
    print(f"  Train: {train_samples} images "
          f"(expected {expected_split_length('train')}), "
          f"{train_batches} batches (shuffle=True)")
    print(f"  Val:   {val_samples} images "
          f"(expected {expected_split_length('val')}), "
          f"{val_batches} batches (shuffle=False)")
    print(f"  Max steps: {max_steps}")

    trainer = VL_JEPA_Trainer(
        model=model,
        device=device,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=max_steps,
        alpha=loss_cfg["alpha"],
        beta=loss_cfg["beta"],
        use_multi_crop=train_cfg.get("use_multi_crop", False),
        global_crop_size=train_cfg.get("global_crop_size", model_cfg["image_size"]),
        local_crop_size=train_cfg.get("local_crop_size", 96),
    )

    output_dir = Path(out_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"exp_jepa_{model_cfg['hidden_dim']}d_{train_cfg['epochs']}ep"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.tensorboard and _HAS_TENSORBOARD:
        tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else exp_dir / 'tensorboard'
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"\nTensorBoard: {tb_dir}")
    elif args.tensorboard and not _HAS_TENSORBOARD:
        print("\n  ⚠️  TensorBoard requested but torch.utils.tensorboard is unavailable")

    print(f"\n{'=' * 70}")
    print(f"Training for {train_cfg['epochs']} epochs...")
    print(f"{'=' * 70}")

    all_metrics: list[Dict[str, Any]] = []
    best_val_loss = float('inf')
    total_start = time.time()
    graph_logged = False
    last_train_metrics: Dict[str, float] = {}

    try:
        for epoch in range(train_cfg["epochs"]):
            train_ds = train_loader.dataset
            if isinstance(train_ds, COCOCaptionDataset):
                train_ds.set_epoch(epoch)

            epoch_start = time.time()
            epoch_metrics: Dict[str, list[float]] = {
                'mse_loss': [], 'nce_loss': [], 'total_loss': [], 'nce_acc': [],
            }

            for batch_idx, batch in enumerate(train_loader):
                images, input_ids, attention_mask = batch
                if writer is not None and not graph_logged:
                    try:
                        trace_model = _GraphTraceWrapper(model).to(device)
                        trace_images = images.to(device)
                        trace_ids = input_ids.to(device)
                        trace_mask = attention_mask.to(device)
                        writer.add_graph(trace_model, (trace_images, trace_ids, trace_mask))
                        graph_logged = True
                    except Exception as graph_err:
                        print(f"\n  ⚠️  TensorBoard graph trace skipped: {graph_err}")
                        graph_logged = True

                metrics = trainer.train_step(images, input_ids, attention_mask)
                last_train_metrics = metrics

                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])

                if (batch_idx + 1) % out_cfg["log_interval"] == 0:
                    window = epoch_metrics['total_loss'][-out_cfg["log_interval"]:]
                    avg_loss = sum(window) / len(window)
                    lr = metrics['lr']
                    gn = metrics['grad_norm']
                    print(f"  E{epoch+1:2d} B{batch_idx+1:5d}/{train_batches} | "
                          f"Loss: {avg_loss:.4f} | MSE: {metrics['mse_loss']:.4f} | "
                          f"NCE: {metrics['nce_loss']:.4f} | "
                          f"NCE@1: {metrics['nce_acc']:.2%} | GN: {gn:.2f} | LR: {lr:.2e}",
                          end='\r')

            avg_mse = sum(epoch_metrics['mse_loss']) / len(epoch_metrics['mse_loss'])
            avg_nce = sum(epoch_metrics['nce_loss']) / len(epoch_metrics['nce_loss'])
            avg_loss = sum(epoch_metrics['total_loss']) / len(epoch_metrics['total_loss'])
            avg_nce_acc = sum(epoch_metrics['nce_acc']) / len(epoch_metrics['nce_acc'])
            epoch_time = time.time() - epoch_start

            val_metrics = trainer.evaluate(val_loader)
            val_mse = val_metrics['mse_loss']
            val_nce = val_metrics['nce_loss']
            val_loss = val_metrics['total_loss']
            val_nce_acc = val_metrics['nce_acc']

            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            print(f"\nEpoch {epoch+1:2d}/{train_cfg['epochs']} | "
                  f"Train: {avg_loss:.4f} (MSE: {avg_mse:.4f}, NCE: {avg_nce:.4f}, "
                  f"NCE@1: {avg_nce_acc:.2%}) | "
                  f"Val: {val_loss:.4f} (MSE: {val_mse:.4f}, NCE: {val_nce:.4f}, "
                  f"NCE@1: {val_nce_acc:.2%}) | "
                  f"{epoch_time:.1f}s | GPU: {gpu_mem:.2f}GB | "
                  f"τ: {trainer.model.momentum_tau:.3f}")

            epoch_record = {
                'epoch': epoch + 1,
                'train_mse': avg_mse,
                'train_nce': avg_nce,
                'train_nce_acc': avg_nce_acc,
                'train_loss': avg_loss,
                'val_mse': val_mse,
                'val_nce': val_nce,
                'val_nce_acc': val_nce_acc,
                'val_loss': val_loss,
                'time': epoch_time,
                'gpu_mem_gb': gpu_mem,
            }
            all_metrics.append(epoch_record)

            if writer is not None:
                step = epoch + 1
                writer.add_scalar('train/loss', avg_loss, step)
                writer.add_scalar('train/mse', avg_mse, step)
                writer.add_scalar('train/nce', avg_nce, step)
                writer.add_scalar('train/nce_acc', avg_nce_acc, step)
                writer.add_scalar('val/loss', val_loss, step)
                writer.add_scalar('val/mse', val_mse, step)
                writer.add_scalar('val/nce', val_nce, step)
                writer.add_scalar('val/nce_acc', val_nce_acc, step)
                if last_train_metrics:
                    writer.add_scalar('train/lr', last_train_metrics['lr'], step)
                    writer.add_scalar('train/grad_norm', last_train_metrics['grad_norm'], step)
                    writer.add_scalar(
                        'train/momentum_tau', last_train_metrics['momentum_tau'], step,
                    )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = exp_dir / 'checkpoint_best.pt'
                trainer.save_checkpoint(str(ckpt_path), {
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'config': cfg,
                })
                print(f"  → Best model saved ({val_loss:.4f})")

            if (epoch + 1) % out_cfg["checkpoint_interval"] == 0:
                ckpt_path = exp_dir / f'checkpoint_epoch{epoch+1}.pt'
                trainer.save_checkpoint(str(ckpt_path), {'epoch': epoch + 1})
                print(f"  → Checkpoint saved: epoch {epoch+1}")

            with open(exp_dir / 'metrics.json', 'w') as f:
                json.dump(all_metrics, f, indent=2)

    finally:
        if writer is not None:
            writer.close()

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"Training completed in {total_time:.1f}s")
    print(f"Results saved to: {exp_dir}")
    print(f"{'=' * 70}")

    if len(all_metrics) > 1:
        first, last = all_metrics[0], all_metrics[-1]
        print(f"\nTraining Summary:")
        print(f"  Initial loss: {first['train_loss']:.4f}")
        print(f"  Final loss:   {last['train_loss']:.4f}")
        print(f"  Improvement:  {first['train_loss'] - last['train_loss']:.4f} "
              f"({(1 - last['train_loss']/first['train_loss'])*100:.1f}%)")
        print(f"  Best val loss: {best_val_loss:.4f}")
        avg_gpu = sum(m.get('gpu_mem_gb', 0) for m in all_metrics) / len(all_metrics)
        print(f"  Avg GPU mem:  {avg_gpu:.2f}GB")
        avg_speed = sum(m.get('time', 0) for m in all_metrics) / len(all_metrics)
        print(f"  Avg epoch:    {avg_speed:.1f}s")
        print(f"  Steps/sec:    {train_batches / (total_time/train_cfg['epochs']):.1f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
