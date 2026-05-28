# Training LaMa on NetCDF Oceanographic Data

## Overview

Train the LaMa inpainting model on CMEMS (Copernicus Marine Service) NetCDF
oceanographic data. The pipeline uses **vertical slices** (depth × longitude)
with **domain-specific losses** for physical ocean fields.

Runs on PyTorch Lightning 2.6 with **manual optimization** (required for
separate generator / discriminator optimizers).

## Key Differences from Image Training

| Aspect | Image Training | NetCDF Training |
|--------|---------------|-----------------|
| Data format | RGB images (C,H,W) | NetCDF 4D (T,D,Lat,Lon) → 2D slices |
| Slice mode | N/A | `["time", "latitude"]` → depth × lon |
| Mask type | Irregular (MixedMaskGenerator) | Vertical lines (profile gaps) |
| Loss function | L1 + perceptual (VGG) | L1 + L2 + adversarial + domain losses |
| Perceptual loss | Enabled (VGG/ResNet) | Disabled (not useful for physical fields) |
| Augmentation | Random flip, crop, color jitter | Flip, noise, depth dropout |
| Channels | 3 (RGB) | 3 (|V|, thetao, so) |
| Optimisation | Automatic (PL) | Manual (PL 2.6, two optimizers) |

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
    bounds:           # Per-channel [vmin, vmax] in [0,1] scaled space
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

## OmegaConf Resolvers

Index specs use custom resolvers registered in `bin/train.py`:

```yaml
lat_indices: ${slice:48,126}       # → slice(48, 126)
lon_indices: ${slice:141,365,2}    # → slice(141, 365, 2)
some_list: ${indices:1,5,8,21}    # → [1, 5, 8, 21]
```

No `eval()` or string parsing — OmegaConf resolves these at config load time.

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

### Full Training

```bash
.venv/bin/python bin/train.py --config-name=cmems_vertical_l1_l2_domain
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

### Fine-tune from Pretrained big-lama

The pretrained generator (`input_nc=4, output_nc=3`) matches the NC config
channel layout. See `readme_train_nc_next.md` for the loading procedure.

## Testing

```bash
.venv/bin/python -m pytest tests/test_nc_losses.py -v       # Domain losses (22 tests)
.venv/bin/python -m pytest tests/test_netcdf_package.py -v   # NetCDF package (34 tests)
```

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

configs/nc/data/
├── cmems_vertical.yaml                 # Standalone data config (for demo/inference)
└── default.yaml                        # Base NC data config template

saicinpainting/training/losses/
├── physical.py                         # Domain losses: smoothness, bounds, gradient
├── feature_matching.py                 # L1/L2 masked losses + feature matching
├── adversarial.py                      # NonSaturatingWithR1 GAN loss
└── ...

saicinpainting/training/trainers/
├── base.py                             # Manual opt, domain loss init, NC dataloaders, PL 2.6 compat
└── default.py                          # L1/L2 + domain losses, input padding (div 8)

netcdf/data/
├── dataset.py                          # NetCDFDataset: slicing, transforms, lon subsetting
├── slicer.py                           # resolve_indices, _resolve_axis_count, subset_xarray
├── augment.py                          # Oceanography-safe augmentations
└── ...

tests/
├── test_nc_losses.py                   # 22 tests for domain losses
└── test_netcdf_package.py              # 34 tests for NetCDF package
```
