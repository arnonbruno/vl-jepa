"""
Experiment 06: Real Data Training with Unsplash Lite
50 epochs on 25K diverse real images with auto-generated captions.
"""

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


class UnsplashLiteDataset:
    """Unsplash Lite dataset (25K images with captions)."""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        
        print("Loading Unsplash Lite dataset from HuggingFace...")
        try:
            from datasets import load_dataset
            # Unsplash Lite is available on HF
            self.dataset = load_dataset('keremberke/unsplash-lite', num_proc=4)
            print(f"✓ Loaded Unsplash Lite: {len(self.dataset['train'])} images")
            self.data = self.dataset['train']
            self.num_samples = len(self.data)
            self.num_batches = max(1, self.num_samples // self.batch_size)
        except Exception as e:
            print(f"✗ Failed to load Unsplash Lite: {e}")
            raise
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Tokenizer params
        self.vocab_size = 30522
        self.seq_len = 128
        
        # Shuffle indices once
        self.indices = list(range(self.num_samples))
        random.shuffle(self.indices)
    
    def _simple_tokenize(self, text):
        """Simple word-based tokenization."""
        if not isinstance(text, str):
            text = str(text)
        
        tokens = text.lower().split()[:self.seq_len]
        token_ids = [hash(t) % self.vocab_size for t in tokens]
        while len(token_ids) < self.seq_len:
            token_ids.append(0)
        return torch.tensor(token_ids[:self.seq_len], dtype=torch.long)
    
    def _load_image(self, image_data):
        """Load and preprocess image from dataset."""
        try:
            # Handle different image formats (PIL, tensor, or path)
            if isinstance(image_data, Image.Image):
                img = image_data.convert('RGB')
            elif isinstance(image_data, str):
                img = Image.open(image_data).convert('RGB')
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                from io import BytesIO
                img = Image.open(BytesIO(image_data['bytes'])).convert('RGB')
            else:
                # Fallback: return black tensor
                return torch.zeros(3, 224, 224)
            
            return self.transform(img)
        except Exception as e:
            # Return black tensor if load fails
            return torch.zeros(3, 224, 224)
    
    def __iter__(self):
        """Iterate over batches."""
        for batch_idx in range(self.num_batches):
            batch_images = []
            batch_input_ids = []
            batch_vision_mask = []
            batch_language_mask = []
            
            for i in range(self.batch_size):
                idx = batch_idx * self.batch_size + i
                if idx >= len(self.indices):
                    break
                
                sample_idx = self.indices[idx]
                sample = self.data[sample_idx]
                
                # Load image
                if 'image' in sample:
                    image = self._load_image(sample['image'])
                else:
                    image = torch.zeros(3, 224, 224)
                
                # Get caption (or use title/tags as fallback)
                caption = ""
                if 'caption' in sample:
                    caption = sample['caption']
                elif 'title' in sample:
                    caption = sample['title']
                elif 'tags' in sample:
                    tags = sample['tags']
                    if isinstance(tags, (list, tuple)):
                        caption = ' '.join(tags[:10])
                    else:
                        caption = str(tags)
                
                # Tokenize
                input_ids = self._simple_tokenize(caption)
                
                # Random masking (15% of patches/tokens)
                vision_mask = torch.rand(196) > 0.15
                language_mask = torch.rand(128) > 0.15
                
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
    print("=" * 70)
    print("VL-JEPA Experiment 06: Real Data Training (Unsplash Lite)")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Dataset
    print("\nLoading Unsplash Lite...")
    dataset = UnsplashLiteDataset(batch_size=32)
    print(f"Unsplash Lite: {dataset.num_samples} real images, {dataset.num_batches} batches/epoch")
    
    # Training
    num_epochs = 50
    all_metrics = []
    checkpoint_interval = 10
    
    print(f"\nTraining for {num_epochs} epochs on real data...")
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
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{dataset.num_batches} | Loss: {avg_loss:.4f}")
        
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
        metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_06_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments') / f'exp_06_checkpoint_epoch{epoch+1}.pt'
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  → Checkpoint saved: epoch {epoch+1}")
    
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining completed in {elapsed:.1f}s ({hours:.1f}h, {elapsed/num_epochs:.1f}s/epoch)")
    
    # Save metrics
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_06_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save checkpoint
    ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_06_checkpoint_final.pt')
    trainer.save_checkpoint(str(ckpt_path))
    
    print("\n" + "=" * 70)
    print("✅ Experiment 06 completed successfully!")
    print("=" * 70)
    
    # Summary
    print(f"\nTraining Summary (50 epochs on 25K Unsplash images):")
    print(f"  Initial loss: {all_metrics[0]['avg_total_loss']:.4f}")
    print(f"  Final loss: {all_metrics[-1]['avg_total_loss']:.4f}")
    print(f"  Improvement: {all_metrics[0]['avg_total_loss'] - all_metrics[-1]['avg_total_loss']:.4f}")
    print(f"  Training time: {hours:.1f}h")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
