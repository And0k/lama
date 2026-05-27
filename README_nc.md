# NetCDF Oceanographic Data Support for LaMa Inpainting

This document describes the NetCDF integration for LaMa inpainting, designed to work with CMEMS (Copernicus Marine Service) oceanographic data and similar NetCDF files.

## Overview

The `netcdf/` subpackage provides components for:
- Loading NetCDF oceanographic data with arbitrary variables
- Unified slicing system for extracting 2D slices from 4D data (time, depth, lat, lon)
- Coordinate-based profile extraction at specific geographic locations
- Oceanography-aware data augmentations for training
- Generating masks for training
- Visualizing results

## Unified Slicing System

The netcdf.data.slicer module provides a unified interface for all slicing operations, supporting both index-based and coordinate-based extraction:

### Index-based Slicing (`slice_nc`)
For working with pre-loaded numpy arrays:

```python
from netcdf.data.slicer import slice_nc
import numpy as np

arr = np.ndarray(shape=(T, D, Y, X))  # Temperature
# Extract slice with axes in the order specified:
slice_nc(arr, ("lat", "lon"), (t, d))    # H×W output: horizontal field
slice_nc(arr, ("depth", "lon"), (t, lat)) # D×W output: vertical slice along longitude
slice_nc(arr, ("depth", "lat"), (t, lon)) # D×H output: vertical slice along latitude
```

### xarray-based Subsetting (`subset_xarray`)
Used internally by NetCDFDataset for lazy loading:

```python
from netcdf.data.slicer import subset_xarray
import xarray as xr

ds = xr.open_dataset("file.nc")
# Subset over time and depth dimensions, keeping lat and lon
subset = subset_xarray(ds, slice_mode=["time", "depth"], 
                      time_indices=slice(0, 10), depth_indices=[0, 5, 10])
# Returns sliced xarray Dataset with lazy loading
```

### Coordinate-based Extraction (`subset_by_coords`)
For extracting profiles at specific geographic coordinates:

```python
from netcdf.data.slicer import subset_by_coords
import xarray as xr

ds = xr.open_dataset("file.nc")
# Extract temperature and salinity profiles at specific coordinates
profiles = subset_by_coords(ds, variables=["thetao", "so"],
                           time_indices=[0],  # First time step
                           depth_indices=slice(0, 20),  # Top 20 depth levels
                           lat_coords=[54.5, 55.0, 56.0],  # Specific latitudes
                           lon_coords=[12.0, 15.0, 20.0],  # Specific longitudes
                           method="nearest")
# Returns dict: {"thetao": (depth_array, values_array), "so": (depth_array, values_array)}
```

### Axis Selection
| axes argument | idx indices | Output Shape | Use Case |
|---------------|-------------|--------------|----------|
| `("lat", "lon")` | `(t, d)` | `(Y, X)` | Horizontal field at time t, depth d |
| `("depth", "lon")` | `(t, lat)` | `(D, X)` | Vertical slice along longitude |
| `("depth", "lat")` | `(t, lon)` | `(D, Y)` | Vertical slice along latitude |

## Supported Variable Types

### 4D Variables (Recommended)
Variables with shape `(time, depth, latitude, longitude)`:
- `thetao` - Sea water temperature
- `so` - Sea water salinity
- `uo` - Sea water velocity (eastward)
- `vo` - Sea water velocity (northward)
- `wo` - Sea water velocity (vertical)
- `sob` - Sea water salinity (bottom)

These can be sliced along longitude or latitude axis to produce vertical profiles (depth × spatial dimension).

### 3D Variables (Limited Support)
Variables with shape `(time, latitude, longitude)`:
- `bottomT` - Sea water temperature at the bottom

**⚠️ Limitation**: These variables lack a depth dimension. When used with `NetCDFDataset`, a dummy depth dimension of size 1 is added. This produces single-row slices `(1, latitude)` or `(1, longitude)` which are **not suitable for meaningful inpainting** of vertical profiles.

**If you attempt to use `slice_axis="lon"` or `"lat"` with only 3D variables, the dataset will:**
1. Add a dummy depth dimension (size 1)
2. Produce slices with shape `(1, latitude)` or `(1, longitude)`
3. **Raise an error early** if you request depth slicing with only 3D variables available

## Installation Requirements

```bash
pip install xarray netCDF4 numpy matplotlib torch
```

For mask generation during training (optional, defaults to irregular-only masks):
```bash
pip install saicinpainting  # Required for Segm masks (also needs detectron2)
```

Without `detectron2`, only irregular masks are supported. Configuration examples use `segm_proba: 0.0` to avoid this requirement.

## Quick Start

### 1. Basic Loading and Slicing

```python
from netcdf.data.loader import load_netcdf, get_available_variables
from netcdf.data.slicer import slice_nc

# Check available variables
variables = get_available_variables("/path/to/file.nc")
print(f"Available: {variables}")

# Load data
data = load_netcdf(
    "/path/to/file.nc",
    variables=["thetao", "so"],
    time_indices=slice(0, 2),
    depth_indices=slice(0, 10),
    replace_fill_value=True,
    normalize=True,
)

# Extract 2D slices using slice_nc
arr = data["thetao"]
horiz = slice_nc(arr, ("lat", "lon"), (0, 5))  # Horizontal field at t=0, d=5
print(f"Horizontal field shape: {horiz.shape}")
```

### 2. Using NetCDFDataset with slice_mode

```python
from netcdf.data.dataset import NetCDFDataset
from torch.utils.data import DataLoader

dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao", "so"],  # Multi-channel: T and S
    slice_mode=["time", "depth"],  # Returns lat × lon fields (default behavior)
    # slice_mode=["time", "latitude"]: Returns depth × lon fields (vertical slices)
    # slice_mode=["time", "longitude"]: Returns depth × lat fields (vertical slices)
    # slice_mode=["coordinate"]: Returns depth profiles at specific coordinates
    time_indices=slice(0, 10),  # Use first 10 time steps
    depth_indices=slice(0, 20),   # Use first 20 depth levels
    lat_indices=None,  # Used for time_lat mode
    lon_indices=None,  # Used for time_lon mode
    lat_coords=[54.5, 55.0, 56.0],  # Used for coordinate mode
    lon_coords=[12.0, 15.0, 20.0],  # Used for coordinate mode
    mask_generator_params={
        'type': 'mixed',
        'params': {
            'irregular_proba': 1.0,
            'box_proba': 0.0,
            'segm_proba': 0.0,  # Disabled: requires detectron2
        }
    }
)

# For shape (T=62, D=27, H=298, W=390):
# slice_mode=["time", "depth"] produces (H=298, W=390) lat×lon fields
# Each time step yields one sample with selected depth
# Total samples = number of time indices selected
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 3. Using NetCDFDataset with Coordinate Extraction

```python
from netcdf.data.dataset import NetCDFDataset
from torch.utils.data import DataLoader

dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao", "so", "uo"],  # Temperature, Salinity, Zonal velocity
    slice_mode=["coordinate"],  # Extract profiles at specific coordinates
    time_indices=slice(0, 5),   # First 5 time steps
    depth_indices=slice(0, 30), # Top 30 depth levels
    lat_coords=[54.0, 54.5, 55.0, 55.5, 56.0],  # 5 latitude positions
    lon_coords=[10.0, 12.0, 14.0, 16.0, 18.0],  # 5 longitude positions
    mask_generator_params={
        'type': 'mixed',
        'params': {
            'irregular_proba': 1.0,
            'box_proba': 0.0,
            'segm_proba': 0.0,
        }
    }
)

# Each sample is a depth profile: (C=3, D=30) for each (time, lat, lon) combination
# Total samples = 5 time × 5 lat × 5 lon = 125 samples
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### 4. Using Augmentations for Training

```python
from netcdf.data.dataset import NetCDFDataset
from netcdf.data.augment import get_netcdf_transforms
from torch.utils.data import DataLoader

# Get oceanography-safe augmentations
transform = get_netcdf_transforms(
    transform_variant="light",  # "none", "light", or "full"
    out_size=None,  # No resizing for lat×lon fields; set [256,256] for fixed models
    flip_lat=True,  # Random latitude flip (safe for scalars; flips sign for uo/vo)
    flip_lon=False, # Avoid longitude flip for velocity fields
    noise_sigma=0.01,  # Gaussian noise for measurement uncertainty
    depth_dropout=0.1  # Randomly zero out 10% of depth levels
)

dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao", "so"],
    slice_mode=["time", "depth"],
    time_indices=slice(0, 10),
    depth_indices=slice(0, 20),
    transform=transform,  # Apply augmentations to each sample
    mask_generator_params={
        'type': 'mixed',
        'params': {
            'irregular_proba': 1.0,
            'box_proba': 0.0,
            'segm_proba': 0.0,
        }
    }
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 5. Using Computed Variables

The `NetCDFDataset` supports computed variables — variables derived from raw NetCDF
fields at runtime. Currently implemented: velocity magnitude `|V| = sqrt(uo² + vo²)`.

```python
from netcdf.data.dataset import NetCDFDataset

dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["|V|", "thetao", "so"],   # 3 output channels
    computed_variables={
        "|V|": {
            "inputs": ["uo", "vo"],
            "operation": "velocity_magnitude",
        }
    },
    scaling={
        "|V|": (0.0, 2.5),     # auto-derived from uo/vo if omitted
        "thetao": (-3.0, 40.0),
        "so": (0.0, 40.0),
        "uo": (-2.0, 2.0),
        "vo": (-2.0, 2.0),
    },
    slice_mode=["time", "depth"],
    time_indices=[0],
    depth_indices=slice(0, 3),
)

sample = dataset[0]
sample["image"].shape      # (3, H, W)
sample["fill_mask"].shape  # (3, H, W) — per-channel fill-value mask, NaN where _FillValue
```

The `fill_mask` output key is backward-compatible — existing code that unpacks
`image`, `mask`, `meta` is unaffected.

### 6. Running the Demo

```bash
python bin/demo_nc.py
```

The demo is now Hydra-driven and configured via `configs/nc/demo/default.yaml`:
- **3-channel inference**: `|V|`, `thetao`, `so` as 3 image channels
- **NaN masking**: Fill-value regions are displayed as white via `cmap.set_bad(color='white')`
- **Shared colorbars**: Each row has its own colorbar with physical units
- **SSIM**: Computed on the first channel (`|V|`)
- **Output**: 3×3 grid figures (3 variables × Original/Masked/Inpainted)

### 7. Running Inference (predict_nc.py)

```bash
python bin/predict_nc.py \
  --netcdf /path/to/file.nc \
  --variable thetao \
  --slice_mode=["time","depth"] \
  --time_index 0 \
  --depth_indices 0 1 2 3 4 \
  --checkpoint /path/to/lama-big.ckpt \
  --out_dir ./results \
  --batch_size 4
```

**Note on `--depth_indices`**: This accepts multiple integers to select specific depth levels.
Alternatively, use `--depth_end 30` to process all depths up to index 30.

## Configuration via YAML

### Training Configuration (`configs/nc/training/lama_small_nc.yaml`)

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
  - evaluator: default_inpainted
  - trainer: any_gpu_large_ssim_ddp_final
  - hydra: overrides
```

### Dataset Configuration (`configs/data/dataset_nc.yaml`)

```yaml
# NetCDF dataset configuration for oceanographic field inpainting
# Each sample = full 2D lat×lon field at a single time step, one channel per variable.

# Uses the restructured netcdf package under configs/nc/data/default.yaml
# For backward compatibility, this file includes the nc config group
# To override values, create a new config in configs/data/ and reference it

defaults:
  - /nc/data/default.yaml

# Override specific values here if needed
# batch_size: 4
# num_workers: 0
# variables: ["thetao", "so", "uo"]
# scaling:
#   thetao: [-3.0, 40.0]
#   so: [0.0, 40.0]
#   uo: [-2.0, 2.0]
#   vo: [-2.0, 2.0]
#   wo: [-0.5, 0.5]
#   bottomT: [-3.0, 40.0]
#   sob: [0.0, 40.0]
#   time_indices: null
#   depth_indices: null
#   lat_indices: null    # Used for time_lat mode
#   lon_indices: null    # Used for time_lon mode
#   lat_coords: null     # Used for coordinate mode
#   lon_coords: null     # Used for coordinate mode
#   slice_mode: null     # Options: null (lat×lon), ["time", "depth"], ["time", "latitude"], 
#                        #          ["time", "longitude"], ["coordinate"]
#   mask_variable: null  # For inference with pre-existing mask variable
#   mask_generator:  # For training with random masks
#     _target_: saicinpainting.training.data.masks.MixedMaskGenerator
#     irregular_proba: 1.0
#     box_proba: 0.0
#     segm_proba: 0.0       # Disabled: requires detectron2
#     squares_proba: 0.0
#     superres_proba: 0.0
#     outpainting_proba: 0.0
#   coverage_threshold: 0.0
#   transform_variant: "light"  # "none", "light", "full" for augmentations
#   transform_out_size: null    # [256, 256] for fixed-size models, None for native size
```

### Augmentation Configuration (`configs/data/transforms/transforms_nc.yaml`)

```yaml
# Data transforms for NetCDF oceanographic slices
# Applied after extracting slices from the 4D NetCDF data

_target_: netcdf.data.augment.get_netcdf_transforms
transform_variant: "light"  # "none", "light", or "full"
out_size: null  # [256, 256] for fixed models, None for native lat×lon size

# Augmentation options
flip_lat: true  # Random latitude flip (safe for scalars; flips sign for uo/vo)
flip_lon: false # Avoid longitude flip for velocity fields
noise_sigma: 0.01  # Gaussian noise for measurement uncertainty
depth_dropout: 0.1  # Randomly zero out depth levels (simulates missing obs)
resize_enabled: false  # Set true to enable resizing
resize_size: [256, 256]  # Target size if resize_enabled: true
```

### Model Configuration (`configs/nc/model/lama_small_nc.yaml`)

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

### Optimizer Configuration (`configs/nc/training/optimizers/default.yaml`)

```yaml
# @package _group_
# Optimizer settings for NetCDF pipeline

optimizers:
  generator:
    lr: 0.0001
    betas: [0.0, 0.999]
    eps: 0.00000001
    weight_decay: 0
  discriminator:
    lr: 0.0001
    betas: [0.0, 0.999]
    eps: 0.00000001
    weight_decay: 0
```

### Prediction Configuration (`configs/nc/prediction/default.yaml`)

```yaml
# @package _group_
# Prediction configuration for NetCDF oceanographic data

indir: no  # to be overriden in CLI
outdir: no  # to be overriden in CLI

model:
  path: no  # to be overriden in CLI
  checkpoint: last.ckpt

dataset:
  kind: default
  slice_axis: lon
  time_index: 0
  depth_indices: null  # e.g., [0,1,2] or slice(0,10)

device: cuda
out_key: inpainted

refine: False  # refiner will only run if this is True
refiner:
  gpu_ids: 0,1  # the GPU ids of the machine to use. If only single GPU, use: "0,"
  modulo: 8
  n_iters: 15   # number of iterations of refinement for each scale
  lr: 0.002     # learning rate
  min_side: 256 # all sides of image on all scales should be >= min_side / sqrt(2)
  max_scales: 3 # max number of downscaling scales for the image-mask pyramid
  px_budget: 1800000  # pixels budget. Any image will be resized to satisfy height*width <= px_budget
```

## What's Implemented

| Component | File | Status |
|-----------|------|--------|
| NetCDF Loader | `netcdf/data/loader.py` | ✅ Complete |
| Unified Slicing System | `netcdf/data/slicer.py` | ✅ Complete (`slice_nc`, `subset_xarray`, `subset_by_coords`) |
| Mask Generator | `netcdf/mask_generator.py` | ✅ Complete |
| PyTorch Dataset | `netcdf/data/dataset.py` | ✅ Complete |
| Visualization | `netcdf/visualization.py` | ✅ Complete |
| Data Augmentations | `netcdf/data/augment.py` | ✅ Complete |
| Inference Script | `bin/predict_nc.py` | ✅ Complete |
| Demo Script | `bin/demo_nc.py` | ✅ Complete (Hydra-driven, multi-variable) |
| Computed Variables | `netcdf/data/dataset.py` | ✅ `|V| = sqrt(uo² + vo²)` |
| Fill Mask Tracking | `netcdf/data/dataset.py` | ✅ Per-channel NaN mask in sample dict |
| Tests | `tests/test_netcdf_package.py` | ✅ 24 tests passing |
| Pipeline Tests | `tests/pipeline/` | ✅ 52 tests covering full inference pipeline |

## Limitations and Restrictions

### Running Pipeline Tests

The `tests/pipeline/` directory contains a sequential test suite that validates the complete NetCDF inpainting pipeline. Run all 52 pipeline tests:

```bash
python -m pytest tests/pipeline/ -v
```

Pipeline stages tested (each must pass for the next to be valid):

| Step | File | What it tests |
|------|------|---------------|
| 01 | `test_01_loader.py` | NetCDF file loading, variable access, fill value handling |
| 02 | `test_02_slicer.py` | 4D→2D slicing, axis ordering, sample count computation |
| 03 | `test_03_mask_generator.py` | Binary mask generation, batch masks, shape/dtype/value correctness |
| 04 | `test_04_dataset.py` | PyTorch Dataset (all slice modes), DataLoader batching, 3D variables |
| 05 | `test_05_model.py` | Model build, forward pass, single/multi-channel, eval mode |
| 06 | `test_06_inference.py` | End-to-end: NetCDF→Dataset→DataLoader→Model→output + visualization |

All tests use a synthetic NetCDF fixture (no external data files required).

### Current Limitations

1. **Only 4D variables supported for meaningful vertical slicing**
   - 3D variables (like `bottomT`) get dummy depth=1 dimension
   - This produces single-row slices unsuitable for vertical profile inpainting

2. **Slicing is primarily along spatial axes** (for 2D field output)
   - `slice_mode=["time", "depth"]`: produces slices of shape `(latitude, longitude)`
   - `slice_mode=["time", "latitude"]`: produces slices of shape `(depth, longitude)`
   - `slice_mode=["time", "longitude"]`: produces slices of shape `(depth, latitude)`
   - `slice_mode=["coordinate"]`: produces depth profiles of shape `(depth,)` per variable

3. **All channels must have same dimensionality**
   - Cannot mix 4D and 3D variables in same sample
   - All requested variables are stacked along channel dimension

4. **No segmentation masks without detectron2**
   - Set `segm_proba: 0.0` in mask generator params
   - Defaults to irregular-only masks

5. **num_workers > 0 may fail with netCDF3 files**
   - Use `num_workers: 0` or netCDF4 format
   - See AGENT.md notes on netCDF and worker_init_fn

6. **Computed variables (e.g. `|V|`) require input variables in the file**
   - Velocity magnitude needs both `uo` and `vo` present in the NetCDF file
   - `_FillValue` propagation: if any input has a fill value, output is also masked

## Next Steps: Advanced Features

For enhancing the NetCDF pipeline:

### 1. ~~Velocity Magnitude Computation~~ (Done)
- ✅ Compute velocity magnitude: `|V| = sqrt(uo² + vo²)` via `computed_variables`
- Configured in YAML as `{"|V|": {"inputs": ["uo", "vo"], "operation": "velocity_magnitude"}}`
- Scaling auto-derived from uo/vo ranges when not explicitly set
- Fill-value propagation: masked regions in uo or vo produce NaN in `|V|`

### 2. Variable-specific Preprocessing
- Log transform for salinity or other variables requiring normalization
- Depth-dependent scaling factors

### 3. Multi-file Time-series Aggregation
- Aggregate multiple NetCDF files for temporal training sequences
- Handle time coordinate alignment across files

### 4. Advanced Coordinate Systems
- Support for different coordinate systems (e.g., rotated poles)
- Horizontal regridding capabilities

### Implementation Priority

1. ✅ Core loading, unified slicing, and inference (done)
2. ✅ Coordinate-based profile extraction mode (done)
3. ✅ Oceanography-aware data augmentations (done)
4. ✅ Velocity magnitude computation (`computed_variables`)
5. ⬜ Variable-specific preprocessing (e.g., log transform for salinity)
6. ⬜ Multi-file time-series aggregation for training

## File Structure

```
netcdf/                              # 🆕 NetCDF-extension (autonomous package)
├── data/
│   ├── __init__.py        # Package exports
│   ├── dataset.py         # PyTorch Dataset with slicing and transforms
│   ├── loader.py          # NetCDF loading with fill value handling
│   ├── slicer.py          # Unified 4D slicing system (index and coordinate-based)
│   ├── augment.py         # Oceanography-safe data augmentations
│   └── visualization_nc.py # Matplotlib utilities
├── mask_generator.py      # Mask generator wrapper
└── visualization.py       # Visualization utilities
```

## Example: Real Data

Using the CMEMS file at `/mnt/b/WorkData/CMEMS/231220_inflow/_source_NetCDF/...`:
- Shape: `(62, 27, 298, 390)` for thetao (time, depth, lat, lon)
- Available variables: `thetao`, `so`, `uo`, `vo`, `wo`, `bottomT`, `sob`
- Valid fill value: `-999.0f` (replaced with NaN)

The dataset correctly handles this file and produces:
- `slice_mode=["time", "depth"]`: slices of shape `(298, 390)` (latitude × longitude)
- `slice_mode=["time", "latitude"]`: slices of shape `(27, 390)` (depth × longitude)
- `slice_mode=["time", "longitude"]`: slices of shape `(27, 298)` (depth × latitude)
- `slice_mode=["coordinate"]`: depth profiles of shape `(27,)` at each (time, lat, lon)

For single-time inference:
```python
dataset = NetCDFDataset(
    filepaths=[nc_file],
    variables=["thetao", "so"],
    slice_mode=["time", "depth"],
    time_indices=[0],
    depth_indices=slice(0, 10),
)
# Produces 1 time × 10 depth × 390 longitude = 3900 samples
# Each sample: (2 channels, 298 latitude, 390 longitude)
```