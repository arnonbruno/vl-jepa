"""
Experiment 01: Basic Training Loop
Tests end-to-end training with synthetic data.
"""

import torch
import sys
sys.path.insert(0, '/home/ulluboz/.openclaw/workspace/vl-jepa')

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


def create_synthetic_batch(batch_size=4, num_patches=196, seq_len=128):
    """Create synthetic batch for testing."""
    images = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    
    # Random masks for prediction tasks
    vision_mask = torch.randint(0, 8192, (batch_size, num_patches))
    language_mask = torch.randint(0, 30522, (batch_size, seq_len))
    
    return images, input_ids, vision_mask, language_mask


def main():
    print("=" * 70)
    print("VL-JEPA Experiment 01: Basic Training Loop")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Training loop
    num_steps = 5
    print(f"\nTraining for {num_steps} steps...")
    
    for step in range(num_steps):
        images, input_ids, vision_mask, language_mask = create_synthetic_batch()
        
        metrics = trainer.train_step(images, input_ids, vision_mask, language_mask)
        
        print(f"Step {step+1}: loss={metrics['total_loss']:.4f} | "
              f"vision_loss={metrics['vision_loss']:.4f} | "
              f"lang_loss={metrics['language_loss']:.4f}")
    
    # Eval step
    print(f"\nEvaluation...")
    images, input_ids, vision_mask, language_mask = create_synthetic_batch()
    metrics = trainer.eval_step(images, input_ids, vision_mask, language_mask)
    print(f"Eval loss: {metrics['total_loss']:.4f}")
    
    # Test checkpoint saving
    print(f"\nTesting checkpoint save/load...")
    ckpt_path = '/tmp/vl_jepa_test.pt'
    trainer.save_checkpoint(ckpt_path)
    print(f"  ✓ Checkpoint saved: {ckpt_path}")
    
    # Create new trainer and load
    model2 = VL_JEPA(hidden_dim=768)
    trainer2 = VL_JEPA_Trainer(model2, device)
    trainer2.load_checkpoint(ckpt_path)
    print(f"  ✓ Checkpoint loaded")
    
    print("\n" + "=" * 70)
    print("✅ Experiment 01 completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
