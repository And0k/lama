# Configuration Files

This directory contains Hydra configuration files for LaMa training and inference.

## NetCDF Configuration

### `nc/data/default.yaml`

Default NetCDF dataset configuration:

```yaml
batch_size: 4
val_batch_size: 2
num_workers: 0  # Use 0 for netCDF3 files; >0 requires netCDF4 format

dataset:
  _target_: netcdf.data.dataset.NetCDFDataset
  filepaths: []
  variables: ["thetao", "so", "uo"]
  scaling:
    thetao: [-3.0, 40.0]
    so: [0.0, 40.0]
    uo: [-2.0, 2.0]
  time_indices: null
  depth_indices: null
  lat_indices: null
  lon_indices: null
  slice_mode: null
  mask_variable: null
  mask_generator:
    _target_: saicinpainting.training.data.masks.MixedMaskGenerator
    irregular_proba: 1.0
    segm_proba: 0.0
  coverage_threshold: 0.0
  fill_ratio_threshold: 1.0
```

**Key parameters:**
- `filepaths`: List of NetCDF file paths (required)
- `variables`: List of variable names to use as image channels (must be 4D: time, depth, lat, lon)
  - Recommended: `thetao` (temperature), `so` (salinity), `uo` (eastward velocity)
  - Avoid: `bottomT` (3D only, no depth dimension)
- `slice_mode`: Slicing mode for 4D data (time, depth, lat, lon)
  - `["time", "depth"]`: Returns (lat, lon) horizontal fields
  - `["time", "latitude"]`: Returns (depth, lon) vertical slices
  - `["time", "longitude"]`: Returns (depth, lat) vertical slices
  - `["coordinate"]`: Returns depth profiles at specific lat/lon points
  - `null` or omit: Original behavior (extract along all longitudes/latitudes)
- `depth_indices`: Depth levels to extract (e.g., `[0,1,2]` or `slice(0,10)`)
- `lat_indices`: Latitude indices (used with `slice_mode: ["time", "latitude"]`)
- `lon_indices`: Longitude indices (used with `slice_mode: ["time", "longitude"]`)
- `time_indices`: Time steps to use (e.g., `slice(0,10)` or `0`)
- `fill_ratio_threshold`: Maximum allowed ratio of `_FillValue`/NaN cells in a slice (0.0–1.0). Slices exceeding this threshold are excluded from the dataset at init time. Default `1.0` means no filtering.
- `coverage_threshold`: Minimum mask coverage ratio for a sample to be considered valid (metadata only, does not skip samples)
- `mask_generator`: Mask generation config for training (set `segm_proba: 0.0` if detectron2 is not installed)

### `nc/demo/default.yaml`

Demo configuration with vertical mask support:
```yaml
# Mask generator config
mask_generator:
  _target_: saicinpainting.training.data.masks.MixedMaskGenerator
  irregular_proba: 1.0

# Vertical mask config for vertical slice inpainting
vertical_mask:
  n_lines: 15
  x_num: 20
  y_num: 24
  y_law: log
  seed: 42

# Projections to run
projections:
  time_depth:
    slice_mode: ["time", "depth"]
    dim_labels: ["latitude", "longitude"]
    mask_type: horizontal
  time_latitude:
    slice_mode: ["time", "latitude"]
    dim_labels: ["depth", "longitude"]
    mask_type: vertical
    extra_kwargs:
      lat_indices: 3  # Fix latitude to index 3
  time_longitude:
    slice_mode: ["time", "longitude"]
    dim_labels: ["depth", "latitude"]
    mask_type: vertical
    extra_kwargs:
      lon_indices: 3  # Fix longitude to index 3

# Hyper-parameters
time_indices: [0]
depth_indices: { _type_: slice, start: 0, stop: 3 }
fill_ratio_threshold: 1.0
```

Use vertical masks in slice_mode `["time", "latitude"]` or `["time", "longitude"]`:
```bash
python bin/demo_nc.py
```

### `nc/evaluator/physical.yaml`

Physical metrics evaluator for NetCDF training (RMSE, Correlation, SSIM):

```yaml
# @package _group_
kind: default
ssim: true
lpips: false
fid: false
rmse: true
correlation: true
integral_kind: null
```

Replaces LPIPS/FID with physics-relevant metrics aligned to L1/L2 training losses.
Used by default in NC training configs (`lama_small_nc.yaml`, `cmems_vertical_l1_l2_domain.yaml`).

### `nc/training/optimizers/default.yaml`

ADAM optimizer settings (betas [0.0, 0.999]).

## NetCDF Configuration

### `nc/training/lama_small_nc.yaml`

Main training config for NetCDF oceanographic inpainting:

```yaml
run_title: "lama_netcdf_small"

training_model:
  kind: default
  visualize_each_iters: 1000
  concat_mask: true
  store_discr_outputs_for_vis: true

losses:
  l1:
    weight_missing: 0
    weight_known: 10
  perceptual:
    weight: 0
  adversarial:
    kind: r1
    weight: 10
    gp_coef: 0.001
    mask_as_fake_target: true
    allow_scale_mask: true
  feature_matching:
    weight: 100
  resnet_pl:
    weight: 30
    weights_path: ${env:TORCH_HOME}

defaults:
  - location: docker
  - _self_
  - data: default
  - generator: ffc_resnet_075
  - discriminator: pix2pixhd_nlayer
  - optimizers: default
  - visualizer: directory
  - evaluator: /nc/evaluator/physical
  - trainer: any_gpu_large_ssim_ddp_final
  - hydra: overrides
```

## Model Configuration

### `nc/model/lama_small_nc.yaml`

```yaml
_target_: models.factory.build_model
in_channels: 3
num_classes: 3
generator_cfg:
  _target_: saicinpainting.training.modules.ffc.FFCResNetGenerator
  input_nc: ${in_channels}
  output_nc: ${num_classes}
  ngf: 64
  n_downsampling: 3
  n_blocks: 9
```

## Prediction Configuration

### `nc/prediction/default.yaml`

```yaml
# @package _group_
indir: no       # override in CLI
outdir: no      # override in CLI

model:
  path: no      # override in CLI
  checkpoint: last.ckpt

dataset:
  kind: default
  slice_mode: ["time", "depth"]
  time_index: 0
  depth_indices: null  # e.g., [0,1,2] or slice(0,10)

device: cuda
out_key: inpainted
batch_size: 1
num_workers: 0
max_samples: null

refine: False
refiner:
  gpu_ids: 0,1
  modulo: 8
  n_iters: 15
  lr: 0.002
  min_side: 256
  max_scales: 3
  px_budget: 1800000
```

## Error Analysis

### `bin/analyze_errors_nc.py`

Analyze NetCDF inpainting prediction quality using physics-aware metrics:

```bash
python bin/analyze_errors_nc.py /path/to/predictions_dir --variables thetao so uo --top-n 10
```

Computes RMSE, Correlation, SSIM, StabilityViolation, and BBLGradient scores.
Saves `analysis_results.csv` and visualization grids of best/worst samples.

## Standard LaMa Configuration

See the main `README.md` for the original LaMa configuration documentation.

## Overriding Parameters

Use Hydra's command-line syntax to override:
```bash
python bin/train.py -cn nc/lama_small_nc \
  nc.data.dataset.filepaths=["/path/to/file.nc"] \
  nc.data.dataset.variables=["thetao","so","uo"] \
  nc.data.dataset.slice_mode=["time","latitude"] \
  nc.data.dataset.lat_indices="[100, 200]"
```

For inference:
```bash
python bin/predict_nc.py \
  dataset.filepaths=["/path/to/file.nc"] \
  dataset.slice_mode=["time","depth"] \
  dataset.depth_indices=[0,1,2,3,4]
```

To skip slices with too many fill values:
```bash
python bin/demo_nc.py \
  nc.fill_ratio_threshold=0.5  # Skip slices where >50% of cells are _FillValue
```

## OmegaConf Resolvers

netcdf/resolvers.py — single registration point for slice, indices, env resolvers,
so index specs are parsed via OmegaConf resolvers in this way:

```yaml
lat_indices: ${slice:48,126}       # → slice(48, 126)
lon_indices: ${slice:141,365,2}    # → slice(141, 365, 2)
some_list: ${indices:1,5,8,21}    # → [1, 5, 8, 21]
```


## Advanced Features

### `nc/data/variable_transforms.yaml`

Variable-specific preprocessing transforms applied before [0,1] scaling:

```yaml
# @package _group_
variable_transforms:
  so:                        # Log transform for salinity
    type: log
    offset: 1.0
  thetao:                    # Square-root transform for temperature
    type: sqrt
  wo:                        # Power transform for vertical velocity
    type: power
    exponent: 0.3
  bottomT:                   # Depth-dependent scaling
    type: depth_dependent
    surface_factor: 1.2
    deep_factor: 0.8
    depth_threshold: 500.0
```

**Supported transform types:**
- `log`: Logarithmic transform. Params: `base` ("natural", "10", "2"), `offset` (default 1.0)
- `sqrt`: Square-root transform. No params.
- `power`: General power transform. Params: `exponent` (default 0.5)
- `depth_dependent`: Depth-dependent scaling. Params: `surface_factor`, `deep_factor`, `depth_threshold`

**Usage in dataset config:**
```yaml
dataset:
  _target_: netcdf.data.dataset.NetCDFDataset
  variable_transforms:
    so:
      type: log
      offset: 1.0
```

**Usage in Hydra CLI:**
```bash
python bin/train.py -cn nc/lama_small_nc \
  nc.data.dataset.variable_transforms.so.type=log \
  nc.data.dataset.variable_transforms.so.offset=1.0
```

### `nc/data/multi_file.yaml`

Multi-file time-series aggregation:

```yaml
# @package _group_
dataset:
  _target_: netcdf.data.multi_file.MultiFileNetCDFDataset
  filepaths: ["/path/to/file1.nc", "/path/to/file2.nc"]
  variables: ["thetao", "so"]
  time_axis: "time"
  deduplicate: true           # Remove duplicate time steps across files
  sequence_length: 3          # Create temporal sequences of 3 consecutive frames
```

**Key parameters:**
- `deduplicate`: When files overlap in time, keep only the first occurrence
- `sequence_length`: Number of consecutive time steps per sample. Set to 1 for single frames.
- `time_axis`: Name of the time dimension (default "time")

**Usage:**
```python
from netcdf.data.multi_file import MultiFileNetCDFDataset
dataset = MultiFileNetCDFDataset(
    filepaths=["file1.nc", "file2.nc", "file3.nc"],
    variables=["thetao", "so"],
    scaling={"thetao": (-3.0, 40.0), "so": (0.0, 40.0)},
    deduplicate=True,
    sequence_length=3,
)
```

### `nc/data/coordinates.yaml`

Advanced coordinate system support:

```yaml
# @package _group_
regrid:
  enabled: true
  method: bilinear           # "bilinear" or "nearest"
  target_lat_size: 180
  target_lon_size: 360
```

**Functions:**
- `detect_coordinate_system(filepath)`: Detect regular, rotated pole, or curvilinear grids
- `rotated_to_regular(lat, lon, pole_lat, pole_lon)`: Convert rotated coordinates
- `regrid_dataset(in, out, ...)`: Regrid a NetCDF file to regular lat/lon

**Usage:**
```python
from netcdf.data.coordinates import detect_coordinate_system, regrid_dataset

# Detect coordinate system
cs = detect_coordinate_system("ocean_model_output.nc")
print(cs["type"])  # "rotated_pole" or "curvilinear"

# Regrid to regular lat/lon
regrid_dataset("ocean_model_output.nc", "regular_latlon.nc",
               target_lat_size=180, target_lon_size=360)
```

### `nc/visualization/default.yaml`

Fixed colorbar limits for all visualization functions. When a value is
`null`, the limit is auto-scaled from the data.

```yaml
# @package _group_
colorbar:
  T:         {vmin: null, vmax: null}   # Temperature [°C]
  S:         {vmin: null, vmax: null}   # Salinity [PSU]
  u:         {vmin: null, vmax: null}   # Zonal velocity [m/s]
  v:         {vmin: null, vmax: null}   # Meridional velocity [m/s]
  error:     {emax: null}               # Symmetric error limit (±emax) °C or PSU
```

**Override examples:**
```bash
# Fix T colorbar to 0–20 °C
python bin/demo_nc.py colorbar.T.vmin=0 colorbar.T.vmax=20

# Fix error display to ±1.5
python notebooks/train_hydro_colab.py colorbar.error.emax=1.5
```

Functions that accept `clim`: `plot_slice`, `plot_comparison`,
`plot_input_channels`, `plot_netcdf_inference`, `plot_two_field_rows`,
`plot_fields_result`, `plot_fields_error`.  Callers in `bin/demo_nc.py`,
`bin/demo_nc_attention.py`, `bin/predict_nc.py`, and
`notebooks/train_hydro_colab.py` read `colorbar` from the Hydra config
automatically.

## Hydro T+S Pipeline

### Overview

The hydro pipeline trains a dual-head generator (T+S) with physical losses
on synthetic Baltic Sea data. Ported from `bin/todo/hydro_attention.py`.

| Config | Purpose |
|--------|---------|
| `nc/training/hydro_train.yaml` | Training config (epochs, lr, model capacity, augmentation) |
| `nc/training/generator/hydro.yaml` | HydroGenerator (8ch→2ch, configurable capacity) |
| `nc/training/discriminator/hydro.yaml` | PatchGAN discriminator (2ch input) |
| `nc/data/synthetic_baltic.yaml` | Baltic synthetic dataset (no NetCDF files needed) |

### Architecture: HydroGenerator (LaMa-lite with FFCResBlock)

- 8 input channels: `[T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathymetry, sigma_obs]`
- 2 output channels: `[T_pred, S_pred]`
- Encoder → N × FFCResBlock (local 3×3 + FFT global) → dual T/S heads → TSCrossAttention → Sigmoid
- `FFCResBlock`: explicit local/global channel split via `ratio_g`, Fourier mixing without activation in freq domain
- `TSCrossAttention`: channel-wise cross-attention (T gates S, S gates T)
- Configurable via `base_ch` (default 32), `n_blocks` (default 6), `ratio_g` (default 0.5)
- 254.9K params (base_ch=32, n_blocks=6) — 2× smaller than SimpleResBlock version

### Generalization Features

| Feature | Description |
|---------|-------------|
| **Uncertainty weighting** | Learnable `log_vars` (Kendall & Gal 2017) — 6 loss terms auto-balanced |
| **OneCycleLR** | Cosine annealing with 10% warmup, replaces bare Adam |
| **FFCResBlock** | Local 3×3 + Fourier global branch (ratio_g=0.5), no Dropout2d needed |
| **Physical augmentation** | Horizontal flip, amplitude scaling ±20%, random CTD dropout |
| **Curriculum learning** | n_ctd ∈ {2..7}, linearly reduced over training |
| **OOD validation** | Separate stress-mode val set, logged as `ood_val_loss` |
| **Gradient-dependent noise** | σ scaled by local gradient magnitude (sharper thermocline → more noise) |
| **Model capacity** | Configurable `base_ch`/`n_blocks` (default 255K params) |

### Physical Losses (with uncertainty weighting)

Loss weights are learned automatically via `log_vars`. Initial sigma = 1.0 for all terms.
Smoothness (TV) stays outside uncertainty weighting with fixed weight.

| Loss | Initial σ | Formula |
|------|-----------|---------|
| `obs_l1` | 1.0 | L1 at observation points |
| `domain_l1` | 1.0 | L1 in water domain |
| `inverse_variance` | 1.0 | L1 weighted by 1/σ² |
| `stability` | 1.0 | ReLU(-∂ρ(T,S)/∂z) with full EOS |
| `bbl` | 1.0 | (∂T/∂z)² + (∂S/∂z)² at seabed |
| `geostrophic` | 1.0 | ∂ρ/∂x + ∂u/∂z anti-correlation |
| `isopycnal` | 1.0 | ‖∇T,∇S along isopycnal‖² (W=0.2) |
| `smoothness` | fixed | Total variation regularization |

### Hydro Training Config (`hydro_train.yaml`)

```yaml
# Training
epochs: 40                        # total training epochs
batch_size: 4                     # training micro-batch size
val_batch_size: 2                 # validation micro-batch size
lr: 2e-4                          # peak learning rate (OneCycleLR)
seed: 42                          # master RNG seed

# Model
base_ch: 32                       # base channel count (reduce for small datasets)
n_blocks: 6                       # number of FFCResBlocks
ratio_g: 0.5                      # fraction of channels for global (Fourier) branch

# Data — training dataset
n_train: 400                      # training dataset length (scenes when lazy=True)
n_val: 80                         # validation dataset length
nx: 64                            # horizontal grid points per scene
nz: 64                            # vertical grid points per scene
augment: true                     # physical augmentation (flip, amplitude scale)
lazy: true                        # on-the-fly scene generation (infinite diversity)
n_ctd: 5                          # CTD profiles per sample (max for curriculum)
n_cmems_x: 8                      # CMEMS sparse columns per sample
n_cmems_z: 10                     # CMEMS depth levels per column
noise_ctd: 0.05                   # CTD measurement noise σ
noise_cmems: 0.10                 # CMEMS measurement noise σ
mode_mix:                         # scene mode mixture (must sum to 1)
  realistic: 0.1                  # climatological Baltic ranges
  extended: 0.2                   # ×1.5 wider ranges, stronger slopes
  stress: 0.7                     # extreme / rare physics
n_variants: 1                     # scene variants per sample (val/OOD pre-generated)
num_workers: 0                    # DataLoader workers (0 = main process)

# Val dataset (inherits train params, separate seed)
val_seed_offset: 100              # val seed = seed + val_seed_offset

# OOD val dataset (stress-heavy, fewer CTDs)
ood_n_ratio: 0.5                  # ood n_val = int(n_val * ood_n_ratio)
ood_seed_offset: 200              # ood seed = seed + ood_seed_offset
ood_n_ctd: 2                      # fewer CTD profiles (harder inpainting)
ood_noise_ctd: 0.08               # higher CTD noise σ
ood_noise_cmems: 0.15             # higher CMEMS noise σ
ood_mode_mix:                     # stress-dominant mode mixture
  realistic: 0.0
  extended: 0.3
  stress: 0.7

# Visualization
viz_every: 5                      # generate plots every N epochs
viz_n_ctd: 3                      # CTD profiles in fixed visualization sample
```

### Training Commands

```bash
# Smoke test (synthetic data, 1 epoch, batch_size=1)
source .venv/bin/activate
python notebooks/train_hydro_colab.py epochs=1 batch_size=1 n_train=8 n_val=4

# Small model (for small datasets, ~28K params)
python notebooks/train_hydro_colab.py base_ch=16 n_blocks=2

# Full training with defaults (lazy mode, infinite diversity)
python notebooks/train_hydro_colab.py epochs=40 batch_size=4

# Stage training: auto-resumes from best checkpoint
python notebooks/train_hydro_colab.py epochs=60

# Disable augmentation
python notebooks/train_hydro_colab.py augment=false
```
