# Configuration Files

This directory contains Hydra configuration files for LaMa training and inference.

## NetCDF Configuration

### `data/dataset_nc.yaml`

Configuration for NetCDF oceanographic data loading.
This file includes the nc config group and can be overridden.

**Key parameters:**
- `filepaths`: List of NetCDF file paths (required)
- `variables`: List of variable names to use as image channels (must be 4D: time, depth, lat, lon)
  - Recommended: `thetao` (temperature), `so` (salinity), `uo` (eastward velocity)
  - Avoid: `bottomT` (3D only, no depth dimension)
- `slice_mode`: Slicing mode for 4D data (time, depth, lat, lon)
  - `time_depth`: Returns (lat, lon) horizontal fields - process each depth level
  - `time_lat`: Returns (depth, lon) vertical slices - process along latitude
  - `time_lon`: Returns (depth, lat) vertical slices - process along longitude
  - `null` or omit: Original behavior (extract along all longitudes/latitudes)
- `depth_indices`: Depth levels to extract (e.g., `[0,1,2]` or `slice(0,10)`)
- `lat_indices`: Latitude indices (used with `slice_mode: ["time", "latitude"]`)
- `lon_indices`: Longitude indices (used with `slice_mode: time_lon`)
- `time_indices`: Time steps to use (e.g., `slice(0,10)` or `0`)
- `mask_generator_params.type`: `"mixed"` for irregular/random masks
- `segm_proba`: Set to `0.0` if detectron2 is not installed

### `nc/data/default.yaml`

Default NetCDF data configuration:
```yaml
filepaths: []  # Populated via CLI or overrides
variables: ["thetao"]
scaling:
  thetao: [-3.0, 40.0]  # Scale values to [0,1]
time_indices: 0            # Single time step or slice
depth_indices: [0, 1, 2, 3, 4]  # Depth levels
slice_mode: ["time", "depth"]     # time_depth, time_lat, time_lon, or null
```

### `training/nc/lama_small_nc.yaml`

Main training configuration using `nc/data/default.yaml` defaults.

**Key settings:**
- Uses FFC ResNet generator (0.75 width)
- R1 adversarial loss
- Feature matching loss
- Can be run with: `python bin/train.py -cn nc/lama_small_nc`

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
  nc.data.filepaths=["/path/to/file.nc"] \
  nc.data.variables=["thetao","so","uo"] \
  nc.data.slice_mode=["time","latitude"] \
  nc.data.lat_indices="[100, 200]"
```

For inference:
```bash
python bin/predict_nc.py \
  --config-name nc/prediction/default \
  nc.data.filepaths=["/path/to/file.nc"] \
  nc.data.slice_mode=["time","depth"] \
  nc.data.depth_indices="[0,1,2,3,4]"
```
