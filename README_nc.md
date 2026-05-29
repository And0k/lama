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

## Installation Requirements

```bash
pip install xarray netCDF4 numpy matplotlib torch
```

For mask generation during training (optional, defaults to irregular-only masks):
```bash
pip install saicinpainting  # Required for Segm masks (also needs detectron2)
```

Without `detectron2`, only irregular masks are supported. Configuration examples use `segm_proba: 0.0` to avoid this requirement.

## Basic Loading and Slicing

### Axis Selection
| axes argument      | idx indices | Output Shape | Use Case                            |
| ------------------ | ----------- | ------------ | ----------------------------------- |
| `("lat", "lon")`   | `(t, d)`    | `(Y, X)`     | Horizontal field at time t, depth d |
| `("depth", "lon")` | `(t, lat)`  | `(D, X)`     | Vertical slice along longitude      |
| `("depth", "lat")` | `(t, lon)`  | `(D, Y)`     | Vertical slice along latitude       |
### Supported Variable Types

#### 4D Variables (Recommended)
Variables with shape `(time, depth, latitude, longitude)`:
- `thetao` - Sea water temperature
- `so` - Sea water salinity
- `uo` - Sea water velocity (eastward)
- `vo` - Sea water velocity (northward)
- `wo` - Sea water velocity (vertical)
- `sob` - Sea water salinity (bottom)

These can be sliced along longitude or latitude axis to produce vertical profiles (depth × spatial dimension).

#### 3D Variables (Limited Support)
Variables with shape `(time, latitude, longitude)`:
- `bottomT` - Sea water temperature at the bottom

**⚠️ Limitation**: These variables lack a depth dimension. When used with `NetCDFDataset`, a dummy depth dimension of size 1 is added. This produces single-row slices `(1, latitude)` or `(1, longitude)` which are **not suitable for meaningful inpainting** of vertical profiles.

**If you attempt to use `slice_axis="lon"` or `"lat"` with only 3D variables, the dataset will:**
1. Add a dummy depth dimension (size 1)
2. Produce slices with shape `(1, latitude)` or `(1, longitude)`
3. **Raise an error early** if you request depth slicing with only 3D variables available

### Loading and Slicing

There are two levels of API for reading NetCDF data, plus a low-level slicer:

**`load_netcdf(filepath, ...)`** — opens a file, selects time/depth indices,
returns a `dict[str, np.ndarray]` of raw float32 arrays. Simple standalone call:
give a path, get numpy. Only supports time and depth selection (not lat/lon).
Best for quick inspection or one-off analysis.

**`NetCDFDataset(filepaths, ...)`** — a PyTorch `Dataset` that internally opens
each file per `__getitem__` call, subsets via `subset_xarray()`, applies scaling
to [0,1], generates masks, and returns `(image, mask, meta)` dicts. Supports all
slice modes. Use for training/inference pipelines with `DataLoader`.

Under the hood, `NetCDFDataset` uses `subset_xarray()` from `netcdf.data.slicer`,
which is the internal building block that subsets an already-open xarray Dataset
along any combination of dimensions.

#### Index-based Slicing (`slice_nc`)
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

#### Raw loading (`load_netcdf`)

```python
from netcdf.data.loader import load_netcdf, get_available_variables

variables = get_available_variables("/path/to/file.nc")
data = load_netcdf(
    "/path/to/file.nc",
    variables=["thetao", "so"],
    time_indices=slice(0, 2),
    depth_indices=slice(0, 10),
    replace_fill_value=True,
)
arr = data["thetao"]  # raw float32, NaN where _FillValue, shape (2, 10, Y, X)
```

#### xarray-based Subsetting (`subset_xarray`)
Internal building block used by `NetCDFDataset` for lazy loading:

```python
from netcdf.data.slicer import subset_xarray
import xarray as xr

ds = xr.open_dataset("file.nc")
subset = subset_xarray(ds, slice_mode=["time", "depth"],
                      time_indices=slice(0, 10), depth_indices=[0, 5, 10])
# Returns sliced xarray Dataset — still lazy, no data loaded yet
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

## Dataset construction: NetCDFDataset

### Slice extraction

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
    lat_indices=None,  # Used with slice_mode=["time", "latitude"]
    lon_indices=None,  # Used with slice_mode=["time", "longitude"]
    lat_coords=[54.5, 55.0, 56.0],  # Used for coordinate mode
    lon_coords=[12.0, 15.0, 20.0],  # Used for coordinate mode
    fill_ratio_threshold=0.5,  # Skip slices where >50% of cells are _FillValue/NaN
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
# Total samples = number of time indices selected (after fill_ratio filtering)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```
### Coordinate Extraction

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

### Mask Generation

The dataset generates random binary masks (1 = to-inpaint, 0 = valid data) for training. Masks are configured via `mask_generator_params` in the dataset config.

#### Original Mask Types

| Type | Config | Dependencies | Description |
|------|--------|-------------|-------------|
| **irregular** | `irregular_proba: 1.0` | `saicinpainting` | Random irregular blobs (default) |
| **box** | `box_proba: 1.0` | `saicinpainting` | Random rectangular boxes |
| **segmentation** | `segm_proba: 1.0` | `saicinpainting` + `detectron2` | Semantic segmentation masks (e.g., buildings, sky) |
| **squares** | `squares_proba: 1.0` | `saicinpainting` | Grid of random squares |
| **superres** | `superres_proba: 1.0` | `saicinpainting` | Super-resolution style masks |
| **outpainting** | `outpainting_proba: 1.0` | `saicinpainting` | Outpainting masks (keep center region) |
| **vertical** | `type: "vertical"` | None (built-in) | Vertical line masks for depth slices |

##### Irregular-only Masks (Default)

"Irregular-only masks" means the dataset uses `MixedMaskGenerator` with only
`irregular_proba: 1.0` set and all other mask types at 0. This requires
`saicinpainting` but does **not** require `detectron2`. The irregular generator
produces random blob-shaped masks that simulate missing data regions.

```yaml
mask_generator:
  _target_: saicinpainting.training.data.masks.MixedMaskGenerator
  irregular_proba: 1.0
  box_proba: 0.0
  segm_proba: 0.0       # Disabled: requires detectron2
  squares_proba: 0.0
  superres_proba: 0.0
  outpainting_proba: 0.0
```

##### Mixed Masks

Combine multiple mask types by setting their probabilities. The generator
samples one type per sample based on the probability weights:

```yaml
mask_generator:
  _target_: saicinpainting.training.data.masks.MixedMaskGenerator
  irregular_proba: 0.5
  box_proba: 0.3
  segm_proba: 0.2    # Requires detectron2
```

#### Vertical masks that simulate real data in the required area: sparse CTD profiles and low resolution of reanalisys models

For vertical slice inpainting (depth × longitude or depth × latitude), we use the built-in vertical mask generator:
```python
dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao"],
    slice_mode=["time", "latitude"],
    mask_generator_params={
        'type': 'vertical',
        'n_lines': 15,      # Number of vertical lines to mask
        'x_num': 20,        # Grid points in x direction
        'y_num': 24,        # Grid points in y direction
        'y_law': 'log',     # Spacing law: 'log' or 'linear'
        'seed': 42,
    }
)
```

#### No Masks

For inference or when masks come from a file variable:

```python
dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao"],
    mask_variable="mask_var",   # Read mask from this NetCDF variable
    mask_generator_params=None, # Or omit entirely
)
```

### Augmentations for Training

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

### Computed Variables

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

### Fill Ratio Filtering

Skip slices where too many cells are `_FillValue` or NaN. Useful for coastal
or ice-covered regions where large portions of the grid are masked.

```python
dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao", "so"],
    slice_mode=["time", "depth"],
    time_indices=[0],
    depth_indices=slice(0, 3),
    fill_ratio_threshold=0.5,  # Exclude slices where >50% of cells are fill/NaN
)
```

At init time, each candidate slice is checked. Slices whose maximum per-variable
fill ratio exceeds the threshold are excluded. The metadata dict includes
`fill_ratio` for each returned sample.

Via Hydra CLI:
```bash
python bin/demo_nc.py nc.fill_ratio_threshold=0.5
```

### Variable-specific Preprocessing

Apply per-variable transforms before [0,1] scaling. Useful for log-transforming
salinity or applying depth-dependent scaling factors.

```python
from netcdf.data.dataset import NetCDFDataset

dataset = NetCDFDataset(
    filepaths=["/path/to/file.nc"],
    variables=["thetao", "so", "uo"],
    scaling={"thetao": (-3.0, 40.0), "so": (0.0, 40.0), "uo": (-2.0, 2.0)},
    variable_transforms={
        "so": {"type": "log", "offset": 1.0},  # log(salinity + 1)
        "thetao": {"type": "sqrt"},              # sqrt(temperature)
    },
    slice_mode=["time", "depth"],
    time_indices=slice(0, 10),
    depth_indices=slice(0, 20),
)
```

**Supported transforms:**
- `log`: Logarithmic transform. Params: `base` ("natural", "10", "2"), `offset`
- `sqrt`: Square-root transform (no params)
- `power`: General power transform. Params: `exponent`
- `depth_dependent`: Depth-dependent scaling. Params: `surface_factor`, `deep_factor`, `depth_threshold`

Scaling ranges are automatically transformed to match the preprocessing. For example,
if `so` has scaling `(0.0, 40.0)` and a log transform with offset 1.0, the effective
scaling becomes `(log(1.0), log(41.0))`.

### Multi-file Time-series Aggregation

Combine multiple NetCDF files into a unified time series with automatic time coordinate alignment and deduplication.

```python
from netcdf.data.multi_file import MultiFileNetCDFDataset

dataset = MultiFileNetCDFDataset(
    filepaths=["file1.nc", "file2.nc", "file3.nc"],
    variables=["thetao", "so"],
    scaling={"thetao": (-3.0, 40.0), "so": (0.0, 40.0)},
    deduplicate=True,        # Remove duplicate time steps across files
    sequence_length=3,       # Create temporal sequences of 3 consecutive frames
)
```

Each sample is a tensor of shape `(sequence_length, C, H, W)` when `sequence_length > 1`. The metadata includes `"sequence": True` and sample-level metadata for each frame.

### Advanced Coordinate Systems

Detect and transform between coordinate systems, and regrid data to regular
lat/lon grids.

```python
from netcdf.data.coordinates import detect_coordinate_system, regrid_dataset

# Detect coordinate system
cs = detect_coordinate_system("ocean_model_output.nc")
print(cs["type"])  # "regular", "rotated_pole", "curvilinear", or "unknown"

# Regrid to regular lat/lon
regrid_dataset(
    "ocean_model_output.nc",
    "regular_latlon.nc",
    target_lat_size=180,
    target_lon_size=360,
    method="bilinear",  # or "nearest"
)
```

For rotated pole coordinates (e.g., NEMO model output):
```python
from netcdf.data.coordinates import rotated_to_regular
lat_reg, lon_reg = rotated_to_regular(
    lat_rot, lon_rot,
    pole_lat=39.25,
    pole_lon=-162.0,
)
```

## Running

See [`README_monitor_nc.md`](README_monitor_nc.md) for details on sample images,
TensorBoard metrics, checkpoints, and offline analysis.

### Demo

```bash
python bin/demo_nc.py
```

The demo is now Hydra-driven and configured via `configs/nc/demo/default.yaml`:
- **3-channel inference**: `|V|`, `thetao`, `so` as 3 image channels
- **Distinct mask colors**: Fill-value regions in light gray, generated masks in white
- **Shared colorbars**: Each row has its own colorbar with physical units
- **SSIM**: Computed on the first channel (`|V|`)
- **Output**: 3×3 grid figures (3 variables × Original/Masked/Inpainted)

### Inference (predict_nc.py)

```bash
python bin/predict_nc.py \
  dataset.filepaths=["/path/to/file.nc"] \
  dataset.variables=["thetao"] \
  dataset.slice_mode=["time","depth"] \
  dataset.time_indices=0 \
  dataset.depth_indices=[0,1,2,3,4] \
  checkpoint=/path/to/lama-big.ckpt \
  out_dir=./results \
  batch_size=4
```

Set `dataset.filepaths` and `checkpoint` as they have no defaults. Other parameters come from `configs/nc/prediction/default.yaml`.

## Configuration

All YAML configs live under `configs/nc/` and are applied via Hydra defaults in
the training/inference entry points. See [`configs/README.md`](configs/README.md)
for full documentation of every config file, parameter reference, and CLI override
examples.

### Disabling Irrelevant Pipeline Features

The LaMa training pipeline was designed for natural images. Several components
are not relevant to the oceanographic pipeline and can be turned off to save
memory and speed up computation:

```yaml
training_model:
  visualize_each_iters: 999999  # Disable periodic sample JPG saving
  store_discr_outputs_for_vis: false  # Skip extra discriminator forward pass for vis

losses:
  perceptual:
    weight: 0       # VGG perceptual loss: trained on ImageNet, not useful for physical fields
  resnet_pl:
    weight: 0       # ResNet perceptual loss: same reason
  feature_matching:
    weight: 100     # Keep: uses discriminator features already computed in adversarial step
```

Setting `store_discr_outputs_for_vis: false` saves meaningful GPU memory by
skipping two extra discriminator forward passes (for `discr_output_fake` and
`discr_output_real` tensors) that are only used in visualization JPEGs. The
discriminator itself is still used for adversarial and feature-matching losses.

Other pipeline features already disabled by default in NC configs: `fake_fakes`,
`rescale_scheduler`, `const_area_crop`.

Quick summary of config groups:

| Config group | File | Purpose |
|---|---|---|
| `nc/data/default` | `configs/nc/data/default.yaml` | Dataset, variables, scaling, mask generator |
| `nc/data/variable_transforms` | `configs/nc/data/variable_transforms.yaml` | Pre-scaling transforms (log, sqrt, depth-dependent) |
| `nc/data/multi_file` | `configs/nc/data/multi_file.yaml` | Multi-file aggregation |
| `nc/data/coordinates` | `configs/nc/data/coordinates.yaml` | Coordinate systems, regridding |
| `nc/evaluator/physical` | `configs/nc/evaluator/physical.yaml` | SSIM + RMSE + Correlation metrics |
| `nc/training/optimizers` | `configs/nc/training/optimizers/default.yaml` | Adam optimizer settings |
| `nc/model/lama_small_nc` | `configs/nc/model/lama_small_nc.yaml` | FFCResNet generator architecture |
| `nc/prediction/default` | `configs/nc/prediction/default.yaml` | Inference settings |
| `nc/demo/default` | `configs/nc/demo/default.yaml` | Demo with vertical masks |
| `visualizer/directory` | `configs/training/visualizer/directory.yaml` | Sample image saving |

## netcdf package utilities
### Evaluation (`netcdf/evaluation.py`)

Physical metrics for oceanographic inpainting evaluation, registered as `PairwiseScore` subclasses:

```python
from netcdf.evaluation import RMSEScore, CorrelationScore, compute_ssim, resolve_vel_scaling

# Per-sample RMSE (used during training evaluation)
rmse_score = RMSEScore()
rmse_score.forward(pred_batch, target_batch)
values = rmse_score.individual_values  # per-sample RMSE

# Per-sample Pearson correlation
corr_score = CorrelationScore()
corr_score.forward(pred_batch, target_batch)
values = corr_score.individual_values  # per-sample correlation

# Standalone SSIM (single pair of numpy arrays)
ssim_val = compute_ssim(original_array, inpainted_array)

# Resolve velocity magnitude scaling from uo/vo ranges
scaling = {"uo": (-2.0, 2.0), "vo": (-2.0, 2.0)}
vmin, vmax = resolve_vel_scaling(scaling)
```

**Training evaluation config** (`configs/nc/evaluator/physical.yaml`):
- SSIM + RMSE + Correlation (LPIPS and FID disabled)
- No integral metric (`integral_kind: null`) — monitor individual metrics
- Metrics appear in training logs as `ssim/mean`, `rmse/mean`, `correlation/mean`

### Tensor Utilities (`netcdf/utils.py`)

```python
from netcdf.utils import pad_to_modulo, next_modulo

# Pad tensor to multiple of 8 (required by LaMa architecture)
padded, (orig_H, orig_W) = pad_to_modulo(tensor, mod=8)

# Round up to nearest multiple
next_mult8 = next_modulo(17, 8)  # Returns 24
```

### Input Variables Metadata  (`netcdf/data/loader.py`)

```python
from netcdf.data.loader import read_units, resolve_output_channels

# Read variable units from NetCDF file
units = read_units("file.nc", ["thetao", "so", "|V|"])
# Returns: {"thetao": "degC", "so": "PSU", "|V|": "m s-1"}

# Resolve output channels with fallback for missing variables
channels = resolve_output_channels("file.nc", ["|V|", "thetao", "so"])
# Returns: [{"name": "thetao", "scaling_key": "thetao"}, ...]
```

### Visualization (`netcdf/visualization.py`)

```python
from netcdf.visualization import plot_netcdf_inference

# Plot multi-channel inference with distinct fill-value and mask colors
fig = plot_netcdf_inference(
    channels=original_list,
    inpainted=inpainted_list,
    fill_masks=fill_mask_list,
    generated_mask=mask_array,
    var_names=["thetao", "so", "|V|"],
    dim_x="longitude",
    dim_y="latitude",
    orig_shape=(H, W),
    ssim_val=0.85,
    units={"thetao": "degC", "so": "PSU", "|V|": "m s-1"},
)
```

## What's Implemented

| Component                          | File                            | Status                                                                     |
| ---------------------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| NetCDF Loader                      | `netcdf/data/loader.py`         | ✅                                                                          |
| Unified Slicing System             | `netcdf/data/slicer.py`         | ✅ (`slice_nc`, `subset_xarray`, `subset_by_coords`)                        |
| Mask Generator                     | `netcdf/mask_generator.py`      | ✅                                                                          |
| Vertical Mask Generator            | `netcdf/data/vertical_masks.py` | ✅                                                                          |
| PyTorch Dataset                    | `netcdf/data/dataset.py`        | ✅                                                                          |
| Evaluation Metrics                 | `netcdf/evaluation.py`          | ✅ (`compute_ssim`, `resolve_vel_scaling`, `RMSEScore`, `CorrelationScore`) |
| Tensor Utilities                   | `netcdf/utils.py`               | ✅ (`pad_to_modulo`, `next_modulo`)                                         |
| Visualization                      | `netcdf/visualization.py`       | ✅                                                                          |
| Data Augmentations                 | `netcdf/data/augment.py`        | ✅                                                                          |
| Inference Script                   | `bin/predict_nc.py`             | ✅                                                                          |
| Demo Script                        | `bin/demo_nc.py`                | ✅ (Hydra-driven, multi-variable)                                           |
| Computed Variables                 | `netcdf/data/dataset.py`        | ✅ `\|V\| = sqrt(uo² + vo²)`                                                |
| Fill Mask Tracking                 | `netcdf/data/dataset.py`        | ✅ Per-channel NaN mask in sample dict                                      |
| Fill Ratio Filtering               | `netcdf/data/dataset.py`        | ✅ Skip slices exceeding `_FillValue` ratio threshold                       |
| Unit Reading                       | `netcdf/data/loader.py`         | ✅ `read_units()` for variable metadata                                     |
| Output Channel Resolution          | `netcdf/data/loader.py`         | ✅ `resolve_output_channels()` with fallback                                |
| Variable-specific Preprocessing    | `netcdf/data/preprocessing.py`  | ✅ Log, sqrt, power, depth-dependent transforms                             |
| Multi-file Time-series Aggregation | `netcdf/data/multi_file.py`     | ✅ Time alignment, deduplication, temporal sequences                        |
| Advanced Coordinate Systems        | `netcdf/data/coordinates.py`    | ✅ Rotated pole, regridding, coordinate detection                           |
| Tests                              | `tests/test_netcdf_package.py`  | ✅ 24 tests passing                                                         |
| Pipeline Tests                     | `tests/pipeline/`               | ✅ 52 tests covering full inference pipeline                                |
| Advanced Feature Tests             | `tests/test_nc_advanced.py`     | ✅ Preprocessing, multi-file, coordinates tests                             |

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

1. **Only 4D variables supported for meaningful vertical slicing**. 3D variables (like `bottomT`) get dummy depth=1 dimension, and this produces single-row slices unsuitable for vertical profile inpainting

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
   - `_FillValue` propagation: if any input has a fill value, output is also masked

## File Structure

```
netcdf/                              # NetCDF-extension (autonomous package)
├── data/
│   ├── __init__.py        # Package exports
│   ├── dataset.py         # PyTorch Dataset with slicing and transforms
│   ├── loader.py          # NetCDF loading, fill value handling, unit reading
│   ├── slicer.py          # Unified 4D slicing system (index and coordinate-based)
│   ├── augment.py         # Oceanography-safe data augmentations
│   ├── vertical_masks.py  # Vertical mask generation for depth slices
│   ├── preprocessing.py   # Variable-specific transforms (log, sqrt, power, depth-dependent)
│   ├── multi_file.py      # Multi-file time-series aggregation with time alignment
│   └── coordinates.py     # Coordinate system support (rotated pole, regridding)
├── evaluation.py          # SSIM computation, velocity scaling resolution
├── mask_generator.py      # Mask generator wrapper
├── utils.py               # Tensor utilities (padding, modulo operations)
└── visualization.py       # Visualization utilities (inference grids, comparisons)
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