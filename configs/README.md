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

### `nc/training/optimizers/default.yaml`

Optimizer settings (ADAM with betas [0.0, 0.999]).

### `nc/data/transforms/transforms_nc.yaml`

Data augmentation and normalization for NetCDF slices.

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
  --config-name nc/prediction/default \
  nc.dataset.filepaths=["/path/to/file.nc"] \
  nc.dataset.slice_mode=["time","depth"] \
  nc.dataset.depth_indices="[0,1,2,3,4]"
```

To skip slices with too many fill values:
```bash
python bin/demo_nc.py \
  nc.fill_ratio_threshold=0.5  # Skip slices where >50% of cells are _FillValue
```
