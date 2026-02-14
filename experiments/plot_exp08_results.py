"""
Plot Exp 08 results based on monitored checkpoints and cron reports.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data collected from cron reports during training
epochs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 37]
losses = [19.4576, 19.4013, 19.3918, 19.3863, 19.3818, 19.3713, 19.3600, 19.3525, 19.3509, 19.3508, 19.3491, 19.3460]

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot loss curve
ax.plot(epochs, losses, 'o-', linewidth=2.5, markersize=8, label='Total Loss', color='#1f77b4')

# Mark checkpoint epochs
checkpoint_epochs = [10, 20, 30]
for ep in checkpoint_epochs:
    idx = epochs.index(ep)
    ax.axvline(x=ep, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(ep, losses[idx] + 0.003, f'ckpt', ha='center', fontsize=9, color='gray')

# Styling
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax.set_title('VL-JEPA Experiment 08: Convergence (37 Epochs, Synthetic Data)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right')

# Add annotations
initial_loss = losses[0]
final_loss = losses[-1]
improvement = initial_loss - final_loss
ax.annotate(f'Start: {initial_loss:.4f}', 
            xy=(epochs[0], initial_loss), 
            xytext=(5, initial_loss - 0.01),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.7))
ax.annotate(f'Final: {final_loss:.4f}\n(↓{improvement:.4f})', 
            xy=(epochs[-1], final_loss), 
            xytext=(epochs[-1] - 8, final_loss + 0.005),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.tight_layout()
plt.savefig('/home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_08_loss_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved: /home/ulluboz/.openclaw/workspace/vl-jepa/experiments/exp_08_loss_plot.png")
print(f"\nExp 08 Summary:")
print(f"  Epochs trained: 37 / 50 (74%)")
print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Final loss: {final_loss:.4f}")
print(f"  Total improvement: {improvement:.4f} ({improvement/initial_loss*100:.2f}%)")
print(f"  Training time: ~30 minutes")
