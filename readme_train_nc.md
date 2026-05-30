# Training LaMa on NetCDF Oceanographic Data

## Overview

Train the LaMa inpainting model on CMEMS (Copernicus Marine Service) NetCDF
oceanographic data. The pipeline uses **vertical slices** (depth × longitude)
with **domain-specific losses** for physical ocean fields.

Runs on PyTorch Lightning 2.6 with **manual optimization** (required for
separate generator / discriminator optimizers).

## Key Differences from Image Training

| Aspect          | Image Training                  | NetCDF Training                           |
| --------------- | ------------------------------- | ----------------------------------------- |
| Data format     | RGB images (C,H,W)              | NetCDF 4D (T,D,Lat,Lon) → 2D slices       |
| Slice mode      | N/A                             | `["time", "latitude"]` → depth × lon      |
| Mask type       | Irregular (MixedMaskGenerator)  | Vertical lines (profile gaps)             |
| Loss function   | L1 + perceptual (VGG)           | L1 + L2 + adversarial + domain losses     |
| Perceptual loss | Enabled (VGG/ResNet)            | Disabled (not useful for physical fields) |
| Augmentation    | Random flip, crop, color jitter | Flip, noise, depth dropout                |
| Channels        | 3 (RGB)                         | 3 (V, thetao, so)                         |
| Optimisation    | Automatic (PL)                  | Manual (PL 2.6, two optimizers)           |

## Loss Functions

### L1 Loss (Masked)
Pixel-wise L1 with separate weights for known and masked regions.

```yaml
l1:
  weight_missing: 1.0   # Masked (predicted) regions
  weight_known: 10.0     # Known (unmasked) regions — higher preserves originals
```

### L2 Loss (Masked)
MSE with the same weighting. Penalizes large errors more than L1.

```yaml
l2:
  weight_missing: 0.5
  weight_known: 5.0
```

### Domain-Specific Losses

Implemented in `saicinpainting/training/losses/physical.py`:

#### Smoothness Loss (total variation)
Penalizes spatial gradients — encourages physically smooth reconstructions.

```yaml
domain:
  smoothness:
    weight: 1.0
```

#### Physical Bounds Loss
Penalizes predictions outside valid ranges. MSE of the violation per channel.

```yaml
domain:
  bounds:
    weight: 0.5
    bounds:          # Per-channel [vmin, vmax] in [0,1] scaled space
      - [0.0, 1.0]   # |V| (velocity magnitude, always ≥ 0)
      - [0.0, 1.0]   # thetao
      - [0.0, 1.0]   # so
```

#### Gradient Consistency Loss
Laplacian-based: compares second-order spatial structure of pred vs target.

```yaml
domain:
  gradient:
    weight: 0.1
```

### Why No Perceptual Loss?

VGG/ResNet perceptual losses are trained on ImageNet. Oceanographic fields
have fundamentally different spatial statistics — perceptual loss introduces
artifacts and degrades physical consistency.

## Data Augmentation

Applied via `netcdf.data.augment.get_netcdf_transforms`:

| Transform | Description | Physics Safety |
|-----------|-------------|----------------|
| `RandomFlipLatLon` | Random lat flip | Adjusts sign for velocity |
| `GaussianNoise` | Measurement noise | Safe for all variables |
| `DepthDropout` | Zero random depth levels | Simulates missing data |

```yaml
transform:
  _target_: netcdf.data.augment.get_netcdf_transforms
  transform_variant: "light"
  flip_lat: true
  flip_lon: false      # Disabled: would need uo sign adjustment
  noise_sigma: 0.01
  depth_dropout: 0.1
```

## Configuration Files

### Training wrapper
`configs/training/cmems_vertical_l1_l2_domain.yaml`

Composes all sub-configs via Hydra defaults:

```yaml
defaults:
  - location: docker
  - /data: cmems_vertical          # → configs/training/nc/data/cmems_vertical.yaml
  - generator: ffc_resnet_075      # FFC-ResNet, 75% global ratio
  - discriminator: pix2pixhd_nlayer
  - optimizers: optimizer_nc       # AdamW, lr=1e-4
  - trainer: pl26_compat           # PL 2.6, manual optimization
  - hydra: overrides               # Output dir template
```

### Data config
`configs/training/nc/data/cmems_vertical.yaml`

- Source: CMEMS Baltic inflow (62 time × 27 depth × 298 lat × 390 lon)
- Subset: `lat[48..125]` (54.82–56.11N) × `lon[141..364]` (14.43–20.63E)
- Slice: `["time", "latitude"]` → each sample is 27 × 224 (depth × lon)
- Samples: 62 × 78 = **4,836**
- `fill_ratio_threshold: 1.0` (accept all — NaN is handled via masking)

### Trainer config
`configs/training/trainer/pl26_compat.yaml`

- `max_epochs: 40`, `limit_train_batches: 25000`
- `val_check_interval: 25000` (validate once per epoch at ~25k steps)
- `precision: 32`, `num_sanity_val_steps: 8`

### Optimizer config
`configs/training/optimizers/optimizer_nc.yaml`

AdamW for both generator and discriminator: `lr=1e-4, betas=[0.0, 0.999]`.

## Output Channels

| Ch | Variable | Meaning | Scaling (vmin, vmax) |
|----|----------|---------|---------------------|
| 0 | \|V\| | Velocity magnitude (sqrt(uo²+vo²)) | [0.0, 2.5] m/s |
| 1 | thetao | Temperature | [-2.0, 22.0] °C |
| 2 | so | Salinity | [29.0, 36.0] PSU |

Input has 4 channels (3 output channels + 1 mask). `|V|` is computed at
runtime from `uo` and `vo` via `computed_variables`.

## Vertical Slice Mask Generation

```yaml
mask_generator:
  type: vertical
  n_lines: 15        # Vertical gap lines per sample
  x_num: 20          # Observation points along x (depth)
  y_num: 24          # Observation points along y (longitude)
  y_law: log          # Log-distributed (denser near surface)
  seed: 42
```

Simulates gaps in vertical ocean profiles — the primary inpainting task.

## Training Commands

### Smoke Test (2 batches, 1 epoch)

```bash
source .venv/bin/activate
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.max_epochs=1 \
  trainer.kwargs.limit_train_batches=2 \
  trainer.kwargs.val_check_interval=2 \
  trainer.kwargs.num_sanity_val_steps=0 \
  data.nc.data.batch_size=1 \
  data.nc.data.dataloader_kwargs.batch_size=1 \
  data.nc.data.val.batch_size=1
```

### First Real Run (5 epochs, full dataset)

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.max_epochs=5 \
  trainer.kwargs.limit_train_batches=4836 \
  trainer.kwargs.val_check_interval=4836
```

### Full Training (40 epochs)

```bash
.venv/bin/python bin/train.py --config-name=cmems_vertical_l1_l2_domain
```

With 4,836 samples and batch_size=2, one epoch = 2,418 steps. Override to
align epoch counter with actual data passes:

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.limit_train_batches=2418 \
  trainer.kwargs.val_check_interval=2418
```

### Resume from Checkpoint

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.resume_from_checkpoint=/path/to/checkpoint.ckpt
```

### Override Data Path

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  'data.nc.data.dataset.filepaths=["/path/to/other.nc"]'
```

### GPU Training

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.accelerator=gpu \
  trainer.kwargs.devices=1 \
  trainer.kwargs.precision=16 \
  data.nc.data.batch_size=4 \
  data.nc.data.dataloader_kwargs.batch_size=4
```

## Key Metrics to Watch

| Metric | Where | What It Tells You |
|--------|-------|-------------------|
| `train_gen_l1` | TB / stdout | L1 reconstruction loss (should decrease steadily) |
| `train_gen_l2` | TB | L2 loss (should decrease; faster = more large errors fixed) |
| `train_gen_adv` | TB | Adversarial loss (should oscillate, not collapse to 0) |
| `train_gen_smooth` | TB | Smoothness penalty (should decrease then stabilize) |
| `train_gen_bounds` | TB | Bounds violation (should approach 0 quickly) |
| `train_gen_grad` | TB | Gradient consistency (should decrease) |
| `train_discr_adv` | TB | Discriminator loss (should stabilize, not diverge) |
| `val_ssim` | TB / checkpoint | Structural similarity on validation (should increase) |
| `val_rmse` | TB | RMSE on validation (should decrease) |
| `val_correlation` | TB | Pearson correlation on validation (should increase) |

### Red Flags

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `gen_l1` not decreasing | Learning rate too low, or bug | Check data loading, increase lr |
| `gen_adv` → 0 instantly | Discriminator too weak | Increase discr capacity or lr |
| `discr_adv` → 0 instantly | Discriminator too strong | Decrease discr lr or weight |
| NaN in losses | Numerical instability | Lower lr, check scaling ranges |
| `gen_smooth` dominates | Smoothness weight too high | Decrease `domain.smoothness.weight` |
| Validation SSIM = 0.1 | Model outputs constant | Check data pipeline, augmentation |

## Testing

```bash
.venv/bin/python -m pytest tests/test_nc_losses.py -v       # Domain losses
.venv/bin/python -m pytest tests/test_netcdf_package.py -v   # NetCDF package
.venv/bin/python -m pytest tests/test_nc_advanced.py -v      # Preprocessing, multi-file, coordinates
```

## Hydro T+S Pipeline

Train a dual-head generator (T+S) with physical losses on synthetic Baltic data.
Ported from `bin/todo/hydro_attention.py`.

### Key differences from standard pipeline

| Aspect | Standard | Hydro T+S |
|--------|----------|-----------|
| Input channels | 3 (\|V\|, thetao, so) + 1 mask | 8 (T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathy, sigma) |
| Output channels | 3 | 2 (T, S) |
| Generator | FFCResNet | HydroGenerator (530K params) |
| Data source | NetCDF files | Synthetic Baltic (no files needed) |
| Losses | L1 + L2 + adversarial | L1 + L2 + adversarial + stability + BBL + geostrophic + inv-var MSE |

### Training commands

```bash
# Smoke test (1 epoch, batch_size=1)
.venv/bin/python bin/train.py --config-name=nc/training/hydro_T_S \
  data=/nc/data/synthetic_baltic \
  data.nc.data.batch_size=1 \
  data.nc.data.val_batch_size=1 \
  trainer.kwargs.max_epochs=1 \
  trainer.kwargs.limit_train_batches=2 \
  trainer.kwargs.val_check_interval=2 \
  trainer.kwargs.num_sanity_val_steps=0

# Full synthetic training (400 samples, 40 epochs)
.venv/bin/python bin/train.py --config-name=nc/training/hydro_T_S \
  data=/nc/data/synthetic_baltic
```

### Batch format

The `HydroSyntheticDataset` returns batches with extra keys consumed by
the physical loss functions:

| Key | Shape | Used by |
|-----|-------|---------|
| `image` | (B, 8, H, W) | Generator input |
| `target` | (B, 2, H, W) | L1/L2/adversarial losses |
| `mask` | (B, 1, H, W) | Below-bottom mask |
| `sigma_obs` | (B, 1, H, W) | `inverse_variance_mse` |
| `obs_mask` | (B, 1, H, W) | `inverse_variance_mse` |
| `u_input` | (B, 1, H, W) | `geostrophic_balance_loss` |
| `bathy_indices` | (B, W) | `bottom_boundary_layer_loss` |

## File Structure

```
configs/training/
├── cmems_vertical_l1_l2_domain.yaml    # Main training wrapper
├── nc/data/cmems_vertical.yaml         # Data config (lat/lon subset, scaling)
├── generator/ffc_resnet_075.yaml       # FFC-ResNet generator
├── discriminator/pix2pixhd_nlayer.yaml # PatchGAN discriminator
├── optimizers/optimizer_nc.yaml        # AdamW settings
├── trainer/pl26_compat.yaml            # PL 2.6 trainer params
├── hydra/overrides.yaml                # Output dir template
└── data/cmems_vertical.yaml            # → nc/data/cmems_vertical (redirect)

saicinpainting/training/losses/
├── physical.py                         # Domain losses: smoothness, bounds, gradient
├── feature_matching.py                 # L1/L2 masked losses + feature matching
├── adversarial.py                      # NonSaturatingWithR1 GAN loss

saicinpainting/training/trainers/
├── base.py                             # Manual opt, domain loss init, NC dataloaders, PL 2.6 compat
└── default.py                          # L1/L2 + domain losses, input padding (div 8)

netcdf/data/
├── dataset.py                          # NetCDFDataset: slicing, transforms, lon subsetting
├── slicer.py                           # resolve_indices, _resolve_axis_count, subset_xarray
├── augment.py                          # Oceanography-safe augmentations

tests/
├── test_nc_losses.py                   # Domain losses tests
├── test_netcdf_package.py              # NetCDF package tests
└── test_nc_advanced.py                 # Preprocessing, multi-file, coordinates tests
```
