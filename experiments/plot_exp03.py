"""
Plot Exp 03 training results: Loss vs Epochs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load metrics
metrics_path = Path('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_03_metrics.json')
with open(metrics_path) as f:
    metrics = json.load(f)

# Extract data
epochs = [m['epoch'] for m in metrics]
total_loss = [m['avg_total_loss'] for m in metrics]
vision_loss = [m['avg_vision_loss'] for m in metrics]
language_loss = [m['avg_language_loss'] for m in metrics]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot losses
ax.plot(epochs, total_loss, 'o-', linewidth=2.5, markersize=6, label='Total Loss', color='#1f77b4')
ax.plot(epochs, vision_loss, 's-', linewidth=2, markersize=5, label='Vision Loss', color='#ff7f0e', alpha=0.8)
ax.plot(epochs, language_loss, '^-', linewidth=2, markersize=5, label='Language Loss', color='#2ca02c', alpha=0.8)

# Styling
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('VL-JEPA Experiment 03: Loss Convergence (15 Epochs)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='upper right')
ax.set_xticks(range(1, 16, 1))

# Add final loss annotation
final_loss = total_loss[-1]
ax.annotate(f'Final: {final_loss:.4f}', 
            xy=(15, final_loss), 
            xytext=(13, final_loss + 0.02),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_03_loss_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved: /home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_03_loss_plot.png")
print(f"Total loss: {total_loss[0]:.4f} → {total_loss[-1]:.4f} (Δ {total_loss[0] - total_loss[-1]:.4f})")
