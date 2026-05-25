"""
Experiment 04: Real Data Training with Flickr30K
50 epochs on real image-text pairs for production-grade validation.
"""

import warnings

import torch
import torch.nn as nn
import json
import time
from pathlib import Path
import sys
import os
from PIL import Image
from torchvision import transforms
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


class Flickr30KDataset:
    """Real Flickr30K dataset loader."""
    
    def __init__(self, root_dir, batch_size=32, num_workers=4):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Image directory and captions file
        self.images_dir = self.root_dir / 'flickr30k_images'
        self.captions_file = self.root_dir / 'results.csv'
        
        # Load captions mapping: image_id -> list of 5 captions
        self.captions_map = self._load_captions()
        self.image_ids = list(self.captions_map.keys())
        self.num_samples = len(self.image_ids)
        self.num_batches = max(1, self.num_samples // self.batch_size)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Simple tokenizer (word-based with BERT vocab)
        self.vocab_size = 30522
        self.seq_len = 128
    
    def _load_captions(self):
        """Load captions from results.csv (Flickr30K format)."""
        captions_map = {}
        
        if not self.captions_file.exists():
            print(f"⚠️  Captions file not found: {self.captions_file}")
            return captions_map
        
        try:
            with open(self.captions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        image_id = parts[0].strip().replace('.jpg', '')
                        caption = parts[1].strip()
                        if image_id not in captions_map:
                            captions_map[image_id] = []
                        captions_map[image_id].append(caption)
        except Exception as e:
            print(f"Error loading captions: {e}")
        
        return captions_map
    
    def _simple_tokenize(self, text):
        """Simple word-based tokenization."""
        tokens = text.lower().split()[:self.seq_len]
        # Hash tokens to vocabulary range
        token_ids = [hash(t) % self.vocab_size for t in tokens]
        # Pad to seq_len
        while len(token_ids) < self.seq_len:
            token_ids.append(0)
        return torch.tensor(token_ids[:self.seq_len])
    
    def _load_image(self, image_path):
        """Load and preprocess image."""
        try:
            img = Image.open(image_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return random tensor if load fails
            return torch.randn(3, 224, 224)
    
    def __iter__(self):
        """Iterate over batches."""
        random.shuffle(self.image_ids)
        
        for batch_idx in range(self.num_batches):
            batch_images = []
            batch_input_ids = []
            batch_vision_mask = []
            batch_language_mask = []
            
            for i in range(self.batch_size):
                idx = batch_idx * self.batch_size + i
                if idx >= len(self.image_ids):
                    break
                
                image_id = self.image_ids[idx]
                image_path = self.images_dir / f"{image_id}.jpg"
                
                # Load image
                image = self._load_image(image_path)
                
                # Get random caption
                captions = self.captions_map.get(image_id, [""])
                caption = random.choice(captions) if captions else ""
                
                # Tokenize
                input_ids = self._simple_tokenize(caption)
                
                # Create masks (random masking)
                vision_mask = torch.randint(0, 2, (196,))
                language_mask = torch.randint(0, 2, (128,))
                
                batch_images.append(image)
                batch_input_ids.append(input_ids)
                batch_vision_mask.append(vision_mask)
                batch_language_mask.append(language_mask)
            
            if batch_images:
                yield (
                    torch.stack(batch_images),
                    torch.stack(batch_input_ids),
                    torch.stack(batch_vision_mask),
                    torch.stack(batch_language_mask)
                )


def main():
    warnings.warn(
        "experiments/exp_04_flickr30k.py is deprecated and uses an outdated API. "
        "Use experiments/exp_jepa_training.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print("=" * 70)
    print("VL-JEPA Experiment 04: Real Data Training (Flickr30K)")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Try to detect Flickr30K
    flickr_root = Path.home() / 'flickr30k'
    if not flickr_root.exists():
        print(f"\n⚠️  Flickr30K not found at {flickr_root}")
        print("Download from: https://www.kaggle.com/datasets/hsankesara/flickr-30k-images")
        print("Or use: https://github.com/BryanPlummer/flickr30k_entities")
        print("\nUsing synthetic data fallback for 50 epochs...")
        print("Replace with real data when Flickr30K is available.")
        use_synthetic = True
    else:
        use_synthetic = False
    
    # Model
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Dataset
    if use_synthetic:
        from experiments.exp_03_large_scale import SyntheticDataset
        dataset = SyntheticDataset(num_samples=5000, batch_size=32)
        print(f"Using synthetic data: {dataset.num_samples} samples")
    else:
        dataset = Flickr30KDataset(flickr_root, batch_size=32)
        print(f"Flickr30K dataset: {dataset.num_samples} samples, {dataset.num_batches} batches/epoch")
    
    # Training
    num_epochs = 50
    all_metrics = []
    checkpoint_interval = 5
    
    print(f"\nTraining for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_metrics = {'vision_loss': [], 'language_loss': [], 'total_loss': []}
        
        for batch_idx, (images, input_ids, vision_mask, language_mask) in enumerate(dataset):
            metrics = trainer.train_step(images, input_ids, vision_mask, language_mask)
            
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
            
            if (batch_idx + 1) % max(1, (dataset.num_batches // 4)) == 0:
                avg_loss = sum(epoch_metrics['total_loss'][-10:]) / min(10, len(epoch_metrics['total_loss']))
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
        
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments') / f'exp_04_checkpoint_epoch{epoch+1}.pt'
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  → Checkpoint saved: epoch {epoch+1}")
        
        # Save metrics every epoch (rolling save)
        metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_04_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/num_epochs:.1f}s per epoch)")
    
    # Save metrics
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_04_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save final checkpoint
    ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_04_checkpoint_final.pt')
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Final checkpoint saved: {ckpt_path}")
    
    print("\n" + "=" * 70)
    print("✅ Experiment 04 completed successfully!")
    print("=" * 70)
    
    # Summary
    print(f"\nTraining Summary (50 epochs):")
    print(f"  Initial loss: {all_metrics[0]['avg_total_loss']:.4f}")
    print(f"  Final loss: {all_metrics[-1]['avg_total_loss']:.4f}")
    print(f"  Total improvement: {all_metrics[0]['avg_total_loss'] - all_metrics[-1]['avg_total_loss']:.4f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
