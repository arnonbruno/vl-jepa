"""
Experiment 05: Real Data Training with COCO 2014
50 epochs on 330K real image-text pairs from Microsoft COCO dataset.
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
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import VL_JEPA
from src.trainer import VL_JEPA_Trainer


class COCODataset:
    """Microsoft COCO 2014 dataset loader."""
    
    def __init__(self, root_dir, split='train', batch_size=32):
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.split = split
        
        # COCO directories
        self.images_dir = self.root_dir / 'val2014' if split == 'val' else self.root_dir / 'train2014'
        self.annotations_file = self.root_dir / f'annotations/captions_{split}2014.json'
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")
        
        # Load COCO
        print(f"Loading COCO {split}2014 annotations...")
        self.coco = COCO(str(self.annotations_file))
        self.img_ids = list(self.coco.imgs.keys())
        self.num_samples = len(self.img_ids)
        self.num_batches = max(1, self.num_samples // self.batch_size)
        
        print(f"COCO {split}2014: {self.num_samples} images, {len(self.coco.anns)} captions")
        
        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Tokenizer params
        self.vocab_size = 30522
        self.seq_len = 128
    
    def _simple_tokenize(self, text):
        """Simple word-based tokenization."""
        tokens = text.lower().split()[:self.seq_len]
        token_ids = [hash(t) % self.vocab_size for t in tokens]
        while len(token_ids) < self.seq_len:
            token_ids.append(0)
        return torch.tensor(token_ids[:self.seq_len], dtype=torch.long)
    
    def _load_image(self, img_id):
        """Load and preprocess image from COCO."""
        try:
            img_info = self.coco.imgs[img_id]
            img_path = self.images_dir / img_info['file_name']
            
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            # Return black tensor if load fails
            return torch.zeros(3, 224, 224)
    
    def __iter__(self):
        """Iterate over batches."""
        random.shuffle(self.img_ids)
        
        for batch_idx in range(self.num_batches):
            batch_images = []
            batch_input_ids = []
            batch_vision_mask = []
            batch_language_mask = []
            
            for i in range(self.batch_size):
                idx = batch_idx * self.batch_size + i
                if idx >= len(self.img_ids):
                    break
                
                img_id = self.img_ids[idx]
                
                # Load image
                image = self._load_image(img_id)
                
                # Get random caption for this image
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                if ann_ids:
                    ann_id = random.choice(ann_ids)
                    caption = self.coco.anns[ann_id]['caption']
                else:
                    caption = ""
                
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
    warnings.warn(
        "experiments/exp_05_coco.py is deprecated and uses an outdated API. "
        "Use experiments/exp_jepa_training.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print("=" * 70)
    print("VL-JEPA Experiment 05: Real Data Training (COCO 2014)")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset path (assuming torchvision downloads to ~/COCO)
    coco_root = Path.home() / 'COCO'
    
    if not coco_root.exists():
        print(f"\n📥 COCO dataset not found at {coco_root}")
        print("Attempting to download COCO 2014 train split (~13GB)...")
        
        try:
            from torchvision.datasets import CocoCaptions
            # This will auto-download
            coco_dataset = CocoCaptions(
                root=str(coco_root / 'train2014'),
                annFile=str(coco_root / 'annotations/captions_train2014.json'),
                download=True
            )
            print("✓ COCO downloaded successfully")
        except Exception as e:
            print(f"✗ Auto-download failed: {e}")
            print("\nManual download:")
            print("1. Visit: http://cocodataset.org/")
            print("2. Download: train2014.zip and captions_train2014.zip")
            print("3. Extract to ~/COCO/")
            sys.exit(1)
    
    # Model
    model = VL_JEPA(hidden_dim=768, patch_size=16, image_size=224)
    trainer = VL_JEPA_Trainer(model, device, learning_rate=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Dataset
    try:
        dataset = COCODataset(coco_root, split='train', batch_size=32)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("COCO setup failed. Please download manually and retry.")
        sys.exit(1)
    
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
            
            # Progress updates every 25% of batches
            if (batch_idx + 1) % max(1, (dataset.num_batches // 4)) == 0:
                avg_loss = sum(epoch_metrics['total_loss'][-10:]) / min(10, len(epoch_metrics['total_loss']))
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{dataset.num_batches} | Avg Loss: {avg_loss:.4f}")
        
        # Summary
        avg_vision_loss = sum(epoch_metrics['vision_loss']) / len(epoch_metrics['vision_loss'])
        avg_language_loss = sum(epoch_metrics['language_loss']) / len(epoch_metrics['language_loss'])
        avg_epoch_loss = sum(epoch_metrics['total_loss']) / len(epoch_metrics['total_loss'])
        
        elapsed_min = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1}/{num_epochs} | V-Loss: {avg_vision_loss:.4f} | L-Loss: {avg_language_loss:.4f} | Total: {avg_epoch_loss:.4f} | Time: {elapsed_min:.1f}min")
        
        all_metrics.append({
            'epoch': epoch + 1,
            'avg_vision_loss': avg_vision_loss,
            'avg_language_loss': avg_language_loss,
            'avg_total_loss': avg_epoch_loss,
        })
        
        # Save metrics every epoch
        metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_05_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments') / f'exp_05_checkpoint_epoch{epoch+1}.pt'
            trainer.save_checkpoint(str(ckpt_path))
            print(f"  → Checkpoint saved: epoch {epoch+1}")
    
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f"\nTraining completed in {elapsed:.1f}s ({hours:.1f}h) ({elapsed/num_epochs:.1f}s per epoch)")
    
    # Save final metrics
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_05_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    # Save final checkpoint
    ckpt_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_05_checkpoint_final.pt')
    trainer.save_checkpoint(str(ckpt_path))
    print(f"Final checkpoint saved: {ckpt_path}")
    
    print("\n" + "=" * 70)
    print("✅ Experiment 05 completed successfully!")
    print("=" * 70)
    
    # Summary
    print(f"\nTraining Summary (50 epochs on {dataset.num_samples} real images):")
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
