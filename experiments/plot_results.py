"""
Plot training results from VL-JEPA experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_exp02_results():
    """Plot exp_02 training metrics."""
    
    metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_02_metrics.json')
    
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    # Extract data
    epochs = [m['epoch'] for m in metrics]
    vision_losses = [m['avg_vision_loss'] for m in metrics]
    language_losses = [m['avg_language_loss'] for m in metrics]
    total_losses = [m['avg_total_loss'] for m in metrics]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Vision loss
    axes[0].plot(epochs, vision_losses, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Vision Loss', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs)
    
    # Plot 2: Language loss
    axes[1].plot(epochs, language_losses, 'o-', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Language Loss', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs)
    
    # Plot 3: Total loss
    axes[2].plot(epochs, total_losses, 'o-', linewidth=2, markersize=8, color='#F18F01')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Loss', fontsize=11)
    axes[2].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(epochs)
    
    plt.tight_layout()
    
    # Save
    plot_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_02_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    
    return plot_path


if __name__ == '__main__':
    plot_exp02_results()
