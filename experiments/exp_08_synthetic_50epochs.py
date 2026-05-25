"""
Experiment 08: Large-Scale Synthetic Training (50 Epochs)
Full validation run: synthetic data, real convergence, production pipeline.
"""

import warnings

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


class SyntheticDataset:
    """Synthetic dataset for large-scale training."""
    
    def __init__(self, num_samples=5000, batch_size=32, num_patches=196, seq_len=128):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_patches = num_patches
        self.seq_len = seq_len
        self.num_batches = num_samples // batch_size
    
    def __iter__(self):
        for _ in range(self.num_batches):
            images = torch.randn(self.batch_size, 3, 224, 224)
            input_ids = torch.randint(0, 30522, (self.batch_size, self.seq_len))
            vision_mask = torch.randint(0, 8192, (self.batch_size, self.num_patches))
            language_mask = torch.randint(0, 30522, (self.batch_size, self.seq_len))
            
            yield images, input_ids, vision_mask, language_mask


def main():
    warnings.warn(
        "experiments/exp_08_synthetic_50epochs.py is deprecated and uses an outdated API. "
        "Use experiments/exp_jepa_training.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print("=" * 70)
    print("VL-JEPA Experiment 08: Synthetic Training (50 Epochs)")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Dataset
    dataset = SyntheticDataset(num_samples=5000, batch_size=32)
    print(f"Dataset: {dataset.num_samples} synthetic samples, {dataset.num_batches} batches/epoch")
    
    # Training
    num_epochs = 50
    all_metrics = []
    checkpoint_interval = 10
    
    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_metrics = {'vision_loss': [], 'language_loss': [], 'total_loss': []}
        
        for batch_idx, (images, input_ids, vision_mask, language_mask) in enumerate(dataset):
            metrics = trainer.train_step(images, input_ids, vision_mask, language_mask)
            
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
            
            # Progress every 25% of batches
            if (batch_idx + 1) % max(1, (dataset.num_batches // 4)) == 0:
                avg_loss = sum(epoch_metrics['total_loss'][-10:]) / min(10, len(epoch_metrics['total_loss']))
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{dataset.num_batches} | Avg Loss: {avg_loss:.4f}")
        
        # Summary
        avg_vision_loss = sum(epoch_metrics['vision_loss']) / len(epoch_metrics['vision_loss'])
        avg_language_loss = sum(epoch_metrics['language_loss']) / len(epoch_metrics['language_loss'])
        avg_epoch_loss = sum(epoch_metrics['total_loss']) / len(epoch_metrics['total_loss'])
        
        elapsed_min = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1}/{num_epochs} | V: {avg_vision_loss:.4f} | L: {avg_language_loss:.4f} | Total: {avg_epoch_loss:.4f} | {elapsed_min:.1f}min")
        
        all_metrics.append({
            'epoch': epoch + 1,
            'avg_vision_loss': avg_vision_loss,
            'avg_language_loss': avg_language_loss,
            'avg_total_loss': avg_epoch_loss,
        })
        
        # Save metrics every epoch
        metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_08_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments') / f'exp_08_checkpoint_epoch{epoch+1}.pt'
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  → Checkpoint saved: epoch {epoch+1}")
    
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining completed in {elapsed:.1f}s ({hours:.1f}h) ({elapsed/num_epochs:.1f}s per epoch)")
    
    # Save metrics
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_08_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save final checkpoint
    ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_08_checkpoint_final.pt')
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Final checkpoint saved: {ckpt_path}")
    
    print("\n" + "=" * 70)
    print("✅ Experiment 08 completed successfully!")
    print("=" * 70)
    
    # Summary
    print(f"\nTraining Summary (50 epochs):")
    print(f"  Initial loss: {all_metrics[0]['avg_total_loss']:.4f}")
    print(f"  Final loss: {all_metrics[-1]['avg_total_loss']:.4f}")
    print(f"  Total improvement: {all_metrics[0]['avg_total_loss'] - all_metrics[-1]['avg_total_loss']:.4f}")
    print(f"  Total time: {hours:.1f}h")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
