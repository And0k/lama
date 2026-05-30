# Monitoring Training Output for NetCDF Pipeline

This document describes the visual and metric monitoring generated during
LaMa training on NetCDF oceanographic data.

## Sample Images

During training, `DirectoryVisualizer` saves JPEG snapshots to disk at
configurable intervals. These provide a visual sanity check on inpainting
quality without launching TensorBoard.

### Output Directory Structure

```
<workdir>/samples/
├── epoch0000_train/
│   ├── batch0000000.jpg
│   ├── batch0000100.jpg
│   └── ...
├── epoch0000_val/
│   ├── batch0000000.jpg
│   └── ...
├── epoch0001_train/
│   └── ...
└── ...
```

The directory name is `epoch{NNNN}{suffix}` where suffix is `_train`, `_val`,
`_test`, or `_extra_val_{key}`. Inside, each file is
`batch{BBBBBBB}[_r{rank}].jpg` where `rank` is the DDP process rank (omitted
for single-GPU).

### What Each JPEG Contains

Each JPEG is a horizontal strip of images concatenated side-by-side, one row
per sample in the batch (up to `max_items_in_batch`, default 10). Red boundary
lines mark the mask contour.

The columns (left to right) are determined by the `key_order` config:

```yaml
# configs/training/visualizer/directory.yaml
key_order:
  - image              # Ground truth (original input)
  - predicted_image    # Generator output before discriminator
  - discr_output_fake  # Discriminator score map (rescaled to [0,1])
  - discr_output_real  # Discriminator score map (rescaled to [0,1])
  - inpainted          # Final inpainted result
```

The default layout for batch_size=2:

```
┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐
│   image[0]  │ predicted.. │ discr_fake  │ discr_real  │  inpainted  │
│  (H × W)    │  (H × W)    │  (H × W)    │  (H × W)    │  (H × W)    │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│   image[1]  │ predicted.. │ discr_fake  │ discr_real  │  inpainted  │
│  (H × W)    │  (H × W)    │  (H × W)    │  (H × W)    │  (H × W)    │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

The final column (`inpainted`) has no red mask boundary when
`last_without_mask: true` (default).

### Rescaling

`discr_output_fake` and `discr_output_real` are rescaled to [0, 1] per-image
before saving (raw discriminator outputs can be outside this range). This is
configured by `rescale_keys`:

```yaml
rescale_keys:
  - discr_output_fake
  - discr_output_real
```

### Configuring Output Resolution

The JPEG resolution is the **native spatial resolution** of your data — no
resizing is applied. For CMEMS data with shape `(298, 390)`, each column is
298×390 pixels and the full strip for 5 columns is 298×1950 pixels.

To save disk space or reduce file sizes, reduce `visualize_each_iters`:

```yaml
training_model:
  visualize_each_iters: 1000  # Save every 1000 steps (default in NC config)
```

Or via Hydra CLI:

```bash
python bin/train.py -cn nc/lama_small_nc \
  training_model.visualize_each_iters=500
```

To disable sample image saving entirely (useful in Colab or disk-constrained
environments):

```bash
python bin/train.py -cn nc/lama_small_nc \
  training_model.visualize_each_iters=999999 \
  training_model.store_discr_outputs_for_vis=false
```

Setting `store_discr_outputs_for_vis: false` also skips the discriminator
forward pass for visualization, saving GPU memory.

## TensorBoard Metrics

All scalar metrics are logged to TensorBoard via PyTorch Lightning's
`TensorBoardLogger`. The logs are written to:

```
<workdir>/tb_logs/nc_training/<version>/events.out.tfevents.*
```

In `train_nc_colab.py`, `version` is fixed (`TB_VERSION`) so that multiple
reruns append to the same x-axis.

### What Is Logged

**Training step metrics** (logged every step with `on_step=True`):

| Metric | Description |
|--------|-------------|
| `train_l1_gen` | Generator L1 loss |
| `train_l1_gen_known` | L1 loss on known (non-masked) pixels |
| `train_feat_match` | Feature matching loss |
| `train_adversarial_gen` | Adversarial generator loss |
| `train_discr` | Discriminator loss |
| `train_resnet_pl` | Perceptual loss (ResNet-based) |

**Validation epoch metrics** (averaged over validation set):

| Metric | Description |
|--------|-------------|
| `val_ssim/mean` | Structural Similarity Index |
| `val_rmse/mean` | Root Mean Square Error |
| `val_correlation/mean` | Pearson correlation coefficient |

These come from `configs/nc/evaluator/physical.yaml` which registers
`RMSEScore` and `CorrelationScore` as `PairwiseScore` subclasses.

### Launching TensorBoard

```bash
# During training (live updates)
tensorboard --logdir <workdir>/tb_logs --port 6006

# In Colab
%tensorboard --logdir <workdir>/tb_logs
```

### After Training (Offline Analysis)

```bash
# Export TensorBoard data to CSV for pandas/matplotlib analysis
tensorboard --logdir <workdir>/tb_logs --port 6007
# Then in Python:
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("<workdir>/tb_logs/nc_training")
ea.Reload()
scalars = ea.Tags()["scalars"]
# ea.Scalars("train_l1_gen") returns list of WallTime, Step, Value
```

## PyTorch Lightning Checkpoints

PL saves checkpoints to the directory configured by `ModelCheckpoint`:

```yaml
# In training config
trainer:
  checkpoint_kwargs:
    dirpath: <workdir>/checkpoints
    save_last: true
    save_top_k: 1
    monitor: val_ssim/mean
    mode: max
```

Checkpoint files:

```
<workdir>/checkpoints/
├── last.ckpt          # Latest checkpoint (for resume)
└── epoch=0005-step=01200.ckpt  # Best checkpoint (by monitored metric)
```

### Resuming Training

```bash
# Resume from last checkpoint
python bin/train.py -cn nc/lama_small_nc \
  +resume_from=<workdir>/checkpoints/last.ckpt

# In train_nc_colab.py, auto-detects last.ckpt in checkpoints dir
```

### Loading for Inference

```python
import torch
ckpt = torch.load("<workdir>/checkpoints/last.ckpt")
model.load_state_dict(ckpt["state_dict"])
```

## Summary of Monitoring Commands

```bash
# Live TensorBoard
tensorboard --logdir <workdir>/tb_logs --port 6006

# Check sample images
ls <workdir>/samples/epoch*_train/
# Open a specific one:
eog <workdir>/samples/epoch0000_train/batch0000000.jpg

# List checkpoints
ls <workdir>/checkpoints/

# Resume training from checkpoint
python bin/train.py -cn nc/lama_small_nc +resume_from=<workdir>/checkpoints/last.ckpt

# TensorBoard to CSV (offline)
python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
ea = EventAccumulator('<workdir>/tb_logs/nc_training')
ea.Reload()
data = {tag: [(s.step, s.value) for s in ea.Scalars(tag)] for tag in ea.Tags()['scalars']}
# Convert to DataFrame, save to CSV, etc.
"
```
