"""
Experiment 03: Large-Scale Training
Increased batch size, dataset size, and epochs to stress-test GPU and track convergence.
"""

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, '/home/ulluboz/.openclaw/workspace/vl-jepa')

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


class SyntheticDataset:
    """Synthetic dataset for training."""
    
    def __init__(self, num_samples=5000, batch_size=64, num_patches=196, seq_len=128):
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
    print("=" * 70)
    print("VL-JEPA Experiment 03: Large-Scale Training")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Dataset
    dataset = SyntheticDataset(num_samples=5000, batch_size=32)
    print(f"Dataset: {dataset.num_samples} samples, {dataset.num_batches} batches per epoch")
    
    # Training
    num_epochs = 15
    all_metrics = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_metrics = {'vision_loss': [], 'language_loss': [], 'total_loss': []}
        
        for batch_idx, (images, input_ids, vision_mask, language_mask) in enumerate(dataset):
            metrics = trainer.train_step(images, input_ids, vision_mask, language_mask)
            
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = sum(epoch_metrics['total_loss'][-20:]) / 20
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{dataset.num_batches} | Avg Loss: {avg_loss:.4f}")
        
        # Summary
        avg_vision_loss = sum(epoch_metrics['vision_loss']) / len(epoch_metrics['vision_loss'])
        avg_language_loss = sum(epoch_metrics['language_loss']) / len(epoch_metrics['language_loss'])
        avg_epoch_loss = sum(epoch_metrics['total_loss']) / len(epoch_metrics['total_loss'])
        
        print(f"Epoch {epoch+1}/{num_epochs} | V-Loss: {avg_vision_loss:.4f} | L-Loss: {avg_language_loss:.4f} | Total: {avg_epoch_loss:.4f}")
        
        all_metrics.append({
            'epoch': epoch + 1,
            'avg_vision_loss': avg_vision_loss,
            'avg_language_loss': avg_language_loss,
            'avg_total_loss': avg_epoch_loss,
        })
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/num_epochs:.1f}s per epoch)")
    
    # Save metrics
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_03_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save checkpoint
    ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_03_checkpoint.pt')
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Checkpoint saved: {ckpt_path}")
    
    print("\n" + "=" * 70)
    print("✅ Experiment 03 completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
