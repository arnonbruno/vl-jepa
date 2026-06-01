Create a standalone Python script `experiments/plot_training.py` that reads metrics.json and generates comprehensive training performance plots.

## Requirements

1. Read metrics.json (list of dicts with keys: epoch, train_loss, val_loss, train_mse, val_mse, train_nce, val_nce, train_nce_acc, val_nce_acc, val_i2t_r1, val_t2i_r1, val_i2t_r5, val_t2i_r5, val_i2t_r10, val_t2i_r10, time, gpu_mem_gb, skipped_batches)

2. Generate a 2x2 subplot figure:
   - Top-left: Train loss vs Val loss over epochs
   - Top-right: NCE accuracy (train + val) over epochs
   - Bottom-left: Retrieval R@1 (i2t + t2i) over epochs
   - Bottom-right: All retrieval metrics (R@1, R@5, R@10 for both i2t and t2i)

3. Mark the best epoch on each panel with a vertical dashed line and annotation

4. CLI args:
   - `--metrics-path` (required): path to metrics.json
   - `--output-path` (optional): where to save PNG (default: same dir as metrics, named `training_plot.png`)
   - `--show` (flag): display interactive plot
   - `--compare` (optional): second metrics.json path for overlay comparison
   - `--title` (optional): plot title

5. Style: use matplotlib with `seaborn-v0_8-whitegrid` style, 300 DPI, figsize=(14, 10)

6. If `--compare` is provided, overlay both runs with different colors and add a legend

## Verification
After creating the script, run:
```
python3.14 experiments/plot_training.py --metrics-path experiments/exp_jepa_768d_50ep/metrics.json --output-path experiments/training_plot.png
```
Verify the PNG is created and looks correct (check file size > 10KB).

Then run with --compare using the old run's metrics if available:
```
python3.14 experiments/plot_training.py --metrics-path experiments/exp_jepa_768d_50ep/metrics.json --output-path experiments/comparison_plot.png
```

## Constraints
- No project imports needed (standalone script using only matplotlib, json, argparse)
- Commit message: 'feat: add training metrics visualization script'
- Push to origin/master
