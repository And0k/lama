# Monitoring Hydro T+S Training

This document describes the visual and metric monitoring generated during
Hydro T+S training.

## Visualization Outputs

The `HydroVisualizationCallback` saves PNG plots to `<outdir>/viz/` at
configurable intervals (`viz_every` parameter, default 5 epochs).

### Output Files

```
<outdir>/viz/
├── input_ep0000.png            # Input channels (val sample)
├── input_train_ep0000.png      # Input channels (training sample, varies)
├── oi_T_S_ep0000.png           # OI T+S reconstruction (val)
├── oi_T_S_train_ep0000.png     # OI T+S reconstruction (train)
├── lama_T_S_ep0000.png         # LaMa T+S reconstruction (val)
├── lama_T_S_train_ep0000.png   # LaMa T+S reconstruction (train)
├── oi_error_ep0000.png         # OI error heatmaps (val)
├── oi_error_train_ep0000.png   # OI error heatmaps (train)
├── lama_error_ep0000.png       # LaMa error heatmaps (val)
├── lama_error_train_ep0000.png # LaMa error heatmaps (train)
└── ...
```

### Two Visualization Types

1. **Val sample** (fixed, `lazy=False`): same input every epoch — good for
   tracking model progress on a consistent baseline.
2. **Training sample** (varies, `lazy=True`): different bathymetry, T/S fields,
   and mode each epoch — shows scene diversity.

### What Each Plot Shows

- **input**: 2×2 grid of u, T, v, S with bathymetry fill, CTD lines, CMEMS dots
- **oi_T_S / lama_T_S**: 2-row T+S field reconstruction with colorbars
- **oi_error / lama_error**: 2-row error heatmaps (ΔT, ΔS) with obs-point overlay

## TensorBoard Metrics

All scalar metrics are logged to TensorBoard via PyTorch Lightning's
`TensorBoardLogger`. Logs are written to:

```
<outdir>/tb_logs/hydro_T_S/<version>/events.out.tfevents.*
```

### What Is Logged

**Training step metrics** (logged every step):

| Metric | Description |
|--------|-------------|
| `train_loss` | Total weighted loss (uncertainty-weighted) |
| `loss/0` – `loss/5` | Individual loss terms (obs_l1, domain_l1, inv_var, stability, bbl, geostrophic) |
| `sigma/0` – `sigma/5` | Learned uncertainty weights (exp(0.5 * log_var)) |

**Validation epoch metrics**:

| Metric | Description |
|--------|-------------|
| `val_loss/dataloader_idx_0` | Val loss (standard mode) |
| `val_loss/dataloader_idx_1` | OOD val loss (stress mode) |

### Launching TensorBoard

```bash
# During training (live updates)
tensorboard --logdir <outdir>/tb_logs --port 6006

# In Colab
%tensorboard --logdir <outdir>/tb_logs
```

### Offline Analysis

```bash
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("<outdir>/tb_logs/hydro_T_S")
ea.Reload()
scalars = ea.Tags()["scalars"]
# ea.Scalars("train_loss") returns list of WallTime, Step, Value
```

## Checkpoints

PL saves checkpoints to `<outdir>/checkpoints/`:

```
<outdir>/checkpoints/
├── hydro-00-val_loss=2.8502.ckpt
├── hydro-01-val_loss=1.9234.ckpt
└── ...
```

### Auto-Resume

The training script auto-detects the best checkpoint (lowest val_loss) and
resumes from it. To start fresh, delete the checkpoint directory:

```bash
rm -rf <outdir>/checkpoints/*
```

### Loading for Inference

```python
import torch
ckpt = torch.load("<outdir>/checkpoints/hydro-XX-val_loss=X.XXXX.ckpt")
model.load_state_dict(ckpt["state_dict"])
```

## Log Messages

Training logs use Python logging with the `hydro_train` logger:

```
2026-05-31 00:27:19,362 [INFO] hydro_train: Device: cpu
2026-05-31 00:27:19,362 [INFO] hydro_train: Generating train dataset: 400 samples, lazy=True, augment=True
2026-05-31 00:27:19,614 [INFO] hydro_train: HydroGenerator: 254.9K params (base_ch=32, n_blocks=6, ratio_g=0.50)
2026-05-31 00:27:21,988 [INFO] hydro_train:   [viz ep=0 val] T LaMa RMSE=0.3899 OI RMSE=0.3227
```

Key things to watch:
- `train_loss` decreasing steadily (not oscillating)
- `sigma/0` – `sigma/5` stabilizing after ~10 epochs
- `val_loss` tracking `train_loss` (not diverging)
- `ood_val_loss` within 40% of `val_loss`
- Viz RMSE: LaMa should beat OI after ~20 epochs

## Summary of Commands

```bash
# Live TensorBoard
tensorboard --logdir <outdir>/tb_logs --port 6006

# Check visualization PNGs
ls <outdir>/viz/

# List checkpoints
ls <outdir>/checkpoints/

# Resume training (auto-detected)
python notebooks/train_hydro_colab.py

# Start fresh
rm -rf <outdir>/checkpoints/*
```
