"""PyTorch Dataset for NetCDF oceanographic data.

Each sample is a full 2D spatial field (lat × lon) at a single time step
with one channel per requested variable.  Scaling to [0, 1] is performed
using configurable physical ranges per variable.

No slicing into depth-latitude or depth-longitude planes is performed here —
the dataset returns the full horizontal field so that the model operates on
the native lat × lon grid.  Depth-aware vertical profile inpainting should
be handled by a separate specialized dataset.
"""

import os
from functools import reduce
from operator import mul
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

from ..mask_generator import generate_mask
from .slicer import _resolve_axis_count, compute_sample_count, subset_xarray

import logging

logger = logging.getLogger(__name__)



class NetCDFDataset(Dataset):
    """
    PyTorch Dataset for loading NetCDF oceanographic data as 2D spatial fields.

    Each sample consists of:
    - image: FloatTensor of shape (C, H, W) where C = len(variables),
      H = latitude count, W = longitude count.  Values are in [0, 1] after
      per-channel scaling using configurable physical ranges.
    - mask: FloatTensor of shape (1, H, W) for inpainting.
    - meta: dict with metadata.
    """

    def __init__(
        self,
        filepaths: List[str],
        variables: List[str],
        scaling: Dict[str, Tuple[float, float]],
        time_indices: Optional[Union[int, List[int], slice]] = None,
        depth_indices: Optional[Union[int, List[int], slice]] = None,
        lat_indices: Optional[Union[int, List[int], slice]] = None,
        lon_indices: Optional[Union[int, List[int], slice]] = None,
        mask_variable: Optional[str] = None,
        mask_generator: Optional[Dict] = None,
        coverage_threshold: float = 0.0,
        transform: Optional[Callable] = None,
        slice_mode: Optional[Sequence[str]] = None,
        lat_coords: Optional[List[float]] = None,
        lon_coords: Optional[List[float]] = None,
        computed_variables: Optional[Dict[str, Dict]] = None,
    ):
        self.filepaths = filepaths
        self.variables = variables
        self.scaling = scaling
        self.computed_variables = computed_variables or {}
        self.time_indices = time_indices
        self.depth_indices = depth_indices
        self.lat_indices = lat_indices
        self.lon_indices = lon_indices
        self.mask_variable = mask_variable
        self.mask_generator_params = mask_generator
        self.coverage_threshold = coverage_threshold
        self.transform = transform
        self.slice_mode = list(slice_mode) if slice_mode else None
        self.lat_coords = list(lat_coords) if lat_coords else None
        self.lon_coords = list(lon_coords) if lon_coords else None

        self._raw_vars_needed = []
        for var in self.variables:
            if var in self.computed_variables:
                self._raw_vars_needed.extend(self.computed_variables[var]["inputs"])
            else:
                self._raw_vars_needed.append(var)
        self._raw_vars_needed = list(dict.fromkeys(self._raw_vars_needed))

        self._cumulative_length = [0]
        self._file_meta = []

        for filepath in filepaths:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            n = self._scan_file(filepath)
            self._cumulative_length.append(self._cumulative_length[-1] + n)

        self.total_length = self._cumulative_length[-1]
        if self.total_length == 0:
            logger.warning("Dataset has zero length. Check file paths and indices.")

    def _scan_file(self, filepath: str) -> int:
        ds = xr.open_dataset(filepath)
        first_var = self._raw_vars_needed[0]
        if first_var not in ds.data_vars:
            ds.close()
            raise ValueError(
                f"Requested variable '{first_var}' not found in {filepath}. "
                f"Available: {list(ds.data_vars.keys())}"
            )

        var = ds[first_var]
        shape = var.shape
        dims = var.dims

        # Compute number of samples based on slice_mode
        n = self._compute_sample_count(shape, dims)

        ds.close()

        self._file_meta.append({
            "filepath": filepath,
            "shape": shape,
            "dims": dims,
        })
        return n

    def _compute_sample_count(self, shape, dims) -> int:
        """Compute expected number of samples by reducing over slice_mode axes.

        self.slice_mode should be defined as a sequence of dimension names,
        e.g. ["time", "latitude"] instead of "time_lat". This eliminates
        branching entirely—each axis contributes multiplicatively via a
        single uniform resolution path.
        
        For slice_mode=["coordinate"], samples are combinations of:
        - time_indices × lat_coords × lon_coords
        """
        if self.slice_mode is None:
            # Original behavior: one sample per time step (returns full lat×lon field)
            dims_list = list(dims)
            return shape[dims_list.index("time")] if "time" in dims_list else shape[0]

        if self.slice_mode == ["coordinate"]:
            # Coordinate mode: combinations of time, lat, lon coordinates
            time_count = _resolve_axis_count(shape, list(dims), "time", self.time_indices)
            lat_count = len(self.lat_coords) if self.lat_coords else 0
            lon_count = len(self.lon_coords) if self.lon_coords else 0
            return time_count * lat_count * lon_count

        index_map: dict[str, int | slice | Sequence[int] | None] = {
            "time": self.time_indices,
            "depth": self.depth_indices,
            "latitude": self.lat_indices,
            "longitude": self.lon_indices,
        }

        counts: tuple[int, ...] = tuple(
            _resolve_axis_count(shape, list(dims), axis, index_map.get(axis))
            for axis in self.slice_mode
        )

        return reduce(mul, counts, 1)

    def _compute_valid_times(self, ds, shape, dims):
        time_dim_size = self._get_time_size(shape, dims)
        if self.time_indices is None:
            return list(range(time_dim_size))
        if isinstance(self.time_indices, int):
            return [self.time_indices] if 0 <= self.time_indices < time_dim_size else []
        if isinstance(self.time_indices, slice):
            start, stop, step = self.time_indices.indices(time_dim_size)
            return list(range(start, stop, step))
        return [i for i in self.time_indices if 0 <= i < time_dim_size]

    @staticmethod
    def _get_time_size(shape, dims):
        if "time" in dims:
            return shape[list(dims).index("time")]
        return shape[0]

    def __len__(self) -> int:
        return self.total_length

    @staticmethod
    def scale_channel(x: np.ndarray, vmin: float, vmax: float) -> torch.Tensor:
        t = torch.from_numpy(x).float()
        t = torch.nan_to_num(t, nan=vmin)
        t = (t - vmin) / (vmax - vmin)
        return torch.clamp(t, 0.0, 1.0)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_length}")

        file_idx = 0
        while idx >= self._cumulative_length[file_idx + 1]:
            file_idx += 1
        local_idx = idx - self._cumulative_length[file_idx]

        meta = self._file_meta[file_idx]
        filepath = meta["filepath"]

        try:
            return self._getitem_with_slicer(filepath, local_idx)
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            return self._get_dummy_sample()


    def _getitem_with_slicer(self, filepath: str, local_idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Extract slices using xarray isel for optimal lazy loading."""
        ds = xr.open_dataset(filepath)
        dims = list(ds.dims)
        sizes = list(ds.sizes.values())

        # Handle coordinate mode
        if self.slice_mode == ["coordinate"]:
            return self._getitem_with_coords(filepath, local_idx, ds, dims, sizes)

        # When slice_mode is None, default to ["time", "depth"] behavior
        # (one lat×lon field per time-step, iterating over depth levels)
        effective_mode = self.slice_mode if self.slice_mode is not None else ["time", "depth"]

        index_map = {
            "time": self.time_indices,
            "depth": self.depth_indices,
            "latitude": self.lat_indices,
            "longitude": self.lon_indices,
        }

        counts = [_resolve_axis_count(sizes, dims, axis, index_map.get(axis)) for axis in effective_mode]

        indices = []
        remaining = local_idx
        for count in reversed(counts):
            indices.append(remaining % count)
            remaining //= count
        indices = list(reversed(indices))
        fixed_indices = dict(zip(effective_mode, indices))

        sel_kw = {dim: idx for dim, idx in fixed_indices.items() if dim in ds.dims}
        selection = ds.isel(sel_kw) if sel_kw else ds

        result = self._extract_fields(
            filepath, selection, ds,
            time_index=fixed_indices.get("time", local_idx),
            depth_idx=fixed_indices.get("depth", 0),
            lat_idx=fixed_indices.get("latitude", 0),
            lon_idx=fixed_indices.get("longitude", 0),
        )
        ds.close()
        return result

    def _extract_fields(self, filepath, selection, ds, *, time_index, depth_idx, lat_idx, lon_idx):
        """Build image + mask tensors from an xarray selection."""
        raw_data = {}
        raw_fill_masks = {}
        mask_arr = None
        for var_name in self._raw_vars_needed:
            if var_name not in ds.data_vars:
                logger.error(f"Variable {var_name} not found in {filepath}")
                return self._get_dummy_sample()
            arr = selection[var_name].values.astype(np.float32)
            fill_mask = np.zeros(arr.shape, dtype=bool)
            if "_FillValue" in selection[var_name].attrs:
                fv = selection[var_name].attrs["_FillValue"]
                fill_mask = (arr == fv)
                arr = np.where(fill_mask, np.nan, arr)
            raw_data[var_name] = arr
            raw_fill_masks[var_name] = fill_mask

        H, W = raw_data[self._raw_vars_needed[0]].shape

        if self.mask_variable is not None and self.mask_variable in ds.data_vars:
            try:
                m = selection[self.mask_variable].values.astype(np.float32)
                while m.ndim < 2:
                    m = m[np.newaxis]
                if m.ndim > 2:
                    m = m.reshape(-1, *m.shape[-2:])[-1]
                mask_arr = m
            except Exception as e:
                logger.warning(f"Failed to load mask from {filepath}: {e}")

        channels = []
        fill_masks_list = []
        for var_name in self.variables:
            if var_name in self.computed_variables:
                cfg = self.computed_variables[var_name]
                if cfg["operation"] == "velocity_magnitude":
                    u = raw_data[cfg["inputs"][0]]
                    v = raw_data[cfg["inputs"][1]]
                    arr = np.sqrt(u**2 + v**2)
                    fm = raw_fill_masks[cfg["inputs"][0]] | raw_fill_masks[cfg["inputs"][1]]
                else:
                    raise ValueError(f"Unknown computed variable operation: {cfg['operation']}")
            else:
                arr = raw_data[var_name]
                fm = raw_fill_masks[var_name]

            if var_name in self.scaling:
                ch = self.scale_channel(arr, *self.scaling[var_name])
            else:
                ch = self.scale_channel(arr, float(np.nanmin(arr)), float(np.nanmax(arr)))
            channels.append(ch)
            fill_masks_list.append(torch.from_numpy(fm.astype(np.uint8)).float())

        image = torch.stack(channels, dim=0)
        fill_mask_tensor = torch.stack(fill_masks_list, dim=0)

        if mask_arr is not None:
            mask_np = (np.nan_to_num(mask_arr, nan=0.0) > 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_np[None].astype(np.float32))
        elif self.mask_generator_params is not None:
            mask_np = generate_mask(shape=(H, W), mask_params=self.mask_generator_params)
            if mask_np.ndim == 3:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[..., 0]
            mask_tensor = torch.from_numpy(mask_np[None].astype(np.float32))
        else:
            mask_tensor = torch.zeros(1, H, W, dtype=torch.float32)

        coverage = float(mask_tensor.mean())
        coverage_ok = coverage >= self.coverage_threshold

        return {
            "image": image,
            "mask": mask_tensor,
            "fill_mask": fill_mask_tensor,
            "meta": {
                "filepath": filepath,
                "time_index": time_index,
                "depth_index": depth_idx,
                "lat_index": lat_idx,
                "lon_index": lon_idx,
                "slice_mode": self.slice_mode,
                "coverage": coverage,
                "coverage_ok": coverage_ok,
                "variables": self.variables,
            },
        }

    def _getitem_with_coords(self, filepath: str, local_idx: int, ds: xr.Dataset, dims: List[str], sizes: List[int]) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Extract coordinate-based samples (profiles at specific lat/lon points)."""
        # Calculate indices for time, lat, lon coordinates
        time_count = _resolve_axis_count(sizes, dims, "time", self.time_indices)
        lat_count = len(self.lat_coords) if self.lat_coords else 0
        lon_count = len(self.lon_coords) if self.lon_coords else 0
        
        if lat_count == 0 or lon_count == 0:
            ds.close()
            return self._get_dummy_sample()
            
        # Calculate which combination this index corresponds to
        lat_lon_count = lat_count * lon_count
        time_idx_in_file = local_idx // lat_lon_count
        lon_lat_idx_in_file = local_idx % lat_lon_count
        lat_idx_in_file = lon_lat_idx_in_file // lon_count
        lon_idx_in_file = lon_lat_idx_in_file % lon_count
        
        # Get actual coordinate values
        lat_coord = self.lat_coords[lat_idx_in_file] if self.lat_coords else 0
        lon_coord = self.lon_coords[lon_idx_in_file] if self.lon_coords else 0
        
        # Build selection for xarray
        sel_kw = {}
        if self.time_indices is not None:
            # We need to select the specific time index
            time_vals = ds["time"].isel(time=self.time_indices).values if hasattr(self.time_indices, '__iter__') and not isinstance(self.time_indices, slice) else \
                      [ds["time"].isel(time=self.time_indices).values] if isinstance(self.time_indices, int) else \
                      ds["time"].isel(time=self.time_indices).values
            if hasattr(time_vals, '__len__') and len(time_vals) > time_idx_in_file:
                sel_kw["time"] = time_vals[time_idx_in_file]
            else:
                sel_kw["time"] = ds["time"].isel(time=self.time_indices)[time_idx_in_file] if hasattr(ds["time"].isel(time=self.time_indices), '__len__') else \
                                ds["time"].isel(time=self.time_indices)
        else:
            sel_kw["time"] = ds["time"][time_idx_in_file] if time_idx_in_file < len(ds["time"]) else 0
            
        # For depth, use all or specified indices
        if self.depth_indices is not None:
            sel_kw["depth"] = self.depth_indices
            
        # Select the nearest latitude and longitude points
        sel_kw["latitude"] = lat_coord
        sel_kw["longitude"] = lon_coord
        
        # Apply selection
        try:
            selection = ds.isel(sel_kw) if any(k in ["time", "depth"] for k in sel_kw.keys()) else ds
            selection = selection.sel(latitude=lat_coord, longitude=lon_coord, method="nearest")
        except Exception as e:
            logger.warning(f"Failed to select coordinates lat={lat_coord}, lon={lon_coord}: {e}")
            ds.close()
            return self._get_dummy_sample()
        
        sliced_data = {}
        mask_arr = None
        for var_name in self.variables:
            if var_name not in ds.data_vars:
                logger.error(f"Variable {var_name} not found in {filepath}")
                ds.close()
                return self._get_dummy_sample()
            arr = selection[var_name].values.astype(np.float32)
            if "_FillValue" in selection[var_name].attrs:
                arr = np.where(arr == selection[var_name].attrs["_FillValue"], np.nan, arr)
            sliced_data[var_name] = arr

        # For coordinate mode, we expect depth profiles (1D arrays)
        # Shape should be (D,) for each variable
        D = sliced_data[self.variables[0]].shape[0] if len(sliced_data[self.variables[0]].shape) > 0 else 1
        H, W = 1, 1  # For metadata compatibility

        time_index = time_idx_in_file if self.time_indices is not None else time_idx_in_file
        depth_idx = -1  # Indicates profile data in metadata
        lat_idx = lat_idx_in_file
        lon_idx = lon_idx_in_file

        if self.mask_variable is not None and self.mask_variable in ds.data_vars:
            try:
                m = selection[self.mask_variable].values.astype(np.float32)
                # For mask, we might want to expand to match depth dimension
                while m.ndim < 1:
                    m = m[np.newaxis]
                if m.ndim > 1:
                    # If mask has spatial dimensions, take the value at the selected point
                    if m.ndim >= 2:
                        m = m.flat[0] if m.size > 0 else np.array([0.0])
                mask_arr = m
            except Exception as e:
                logger.warning(f"Failed to load mask from {filepath}: {e}")

        ds.close()

        channels = []
        for var_name in self.variables:
            arr = sliced_data[var_name]
            # Ensure we have a 1D array for depth profile
            if arr.ndim == 0:
                arr = np.array([arr.item()])
            elif arr.ndim > 1:
                # Flatten to 1D if needed (shouldn't happen for proper coordinate selection)
                arr = arr.flatten()
                
            if var_name in self.scaling:
                ch = self.scale_channel(arr, *self.scaling[var_name])
            else:
                ch = self.scale_channel(arr, float(np.nanmin(arr)), float(np.nanmax(arr)))
            channels.append(ch)

        # Stack channels to get (C, D) shape for profile data
        image = torch.stack(channels, dim=0) if len(channels) > 1 else channels[0].unsqueeze(0)

        # Get mask tensor
        if mask_arr is not None:
            # Ensure mask is at least 1D
            if np.isscalar(mask_arr):
                mask_arr = np.array([mask_arr])
            mask_np = (np.nan_to_num(mask_arr, nan=0.0) > 0).astype(np.uint8)
            # Match depth dimension if needed
            if len(mask_np) != image.shape[-1]:  # Assuming last dimension is depth
                if len(mask_np) == 1:
                    mask_np = np.repeat(mask_np, image.shape[-1])
                else:
                    mask_np = np.zeros(image.shape[-1], dtype=np.uint8)
            mask_tensor = torch.from_numpy(mask_np[None].astype(np.float32))
        elif self.mask_generator_params is not None:
            # Generate mask matching depth dimension
            mask_depth = image.shape[-1] if image.ndim > 1 else len(channels[0]) if channels else 1
            mask_np = generate_mask(shape=(mask_depth,), mask_params=self.mask_generator_params)
            if mask_np.ndim == 2:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[..., 0]
            mask_tensor = torch.from_numpy(mask_np[None].astype(np.float32))
        else:
            mask_depth = image.shape[-1] if image.ndim > 1 else len(channels[0]) if channels else 1
            mask_tensor = torch.zeros(1, mask_depth, dtype=torch.float32)

        coverage = float(mask_tensor.mean())
        coverage_ok = coverage >= self.coverage_threshold

        sample = {
            "image": image,
            "mask": mask_tensor,
            "fill_mask": torch.zeros_like(image),
            "meta": {
                "filepath": filepath,
                "time_index": time_index,
                "depth_index": depth_idx,  # -1 indicates profile data
                "lat_index": lat_idx,
                "lon_index": lon_idx,
                "lat_coord": self.lat_coords[lat_idx_in_file] if self.lat_coords else None,
                "lon_coord": self.lon_coords[lon_idx_in_file] if self.lon_coords else None,
                "slice_mode": self.slice_mode,
                "coverage": coverage,
                "coverage_ok": coverage_ok,
                "variables": self.variables,
            },
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_dummy_sample(self) -> Dict[str, Union[torch.Tensor, Dict]]:
        C = len(self.variables)
        H, W = 64, 64
        if self._file_meta:
            meta = self._file_meta[0]
            shape, dims = meta["shape"], meta["dims"]
            dims_list = list(dims)
            if "latitude" in dims_list and "longitude" in dims_list:
                H = shape[dims_list.index("latitude")]
                W = shape[dims_list.index("longitude")]
            elif len(shape) >= 2:
                H, W = shape[-2], shape[-1]
        return {
            "image": torch.zeros(C, H, W, dtype=torch.float32),
            "mask": torch.zeros(1, H, W, dtype=torch.float32),
            "fill_mask": torch.zeros(C, H, W, dtype=torch.float32),
            "meta": {"error": True, "variables": self.variables},
        }

    def get_shape(self) -> Tuple[int, int, int]:
        C = len(self.variables)
        if self._file_meta:
            shape, dims = self._file_meta[0]["shape"], self._file_meta[0]["dims"]
            dims_list = list(dims)
            if self.slice_mode is None:
                if "latitude" in dims_list and "longitude" in dims_list:
                    H = shape[dims_list.index("latitude")]
                    W = shape[dims_list.index("longitude")]
                    return C, H, W
                elif len(shape) >= 2:
                    return C, shape[-2], shape[-1]
            elif self.slice_mode == ["coordinate"]:
                # For coordinate mode, we return (C, D, 1) where D is depth dimension
                # The third dimension is kept as 1 for compatibility with 2D expectation
                if "depth" in dims_list:
                    D = shape[dims_list.index("depth")]
                    return C, D, 1
                elif len(shape) >= 1:
                    return C, shape[-1], 1
                else:
                    return C, 1, 1
            else:
                # Regular slicing modes
                output_dims = [d for d in dims_list if d not in self.slice_mode]
                if len(output_dims) >= 2:
                    H = shape[dims_list.index(output_dims[0])]
                    W = shape[dims_list.index(output_dims[1])]
                    return C, H, W
                elif len(shape) >= 2:
                    return C, shape[-2], shape[-1]
        return C, 64, 64