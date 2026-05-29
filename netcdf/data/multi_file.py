"""Multi-file time-series aggregation for NetCDF datasets.

Handles multiple NetCDF files that together form a time series, with
automatic time coordinate alignment, deduplication, and optional
temporal sequence creation for training.

Usage::

    dataset = MultiFileNetCDFDataset(
        filepaths=["file1.nc", "file2.nc", "file3.nc"],
        variables=["thetao", "so"],
        scaling={"thetao": (-3.0, 40.0), "so": (0.0, 40.0)},
        time_axis="time",
        deduplicate=True,
        sequence_length=1,  # 1 = single frames, >1 = temporal pairs
    )
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from ..mask_generator import generate_mask
from .slicer import _resolve_axis_count, resolve_indices

logger = logging.getLogger(__name__)


def scan_time_coordinates(
    filepaths: List[str],
    time_axis: str = "time",
) -> List[Dict]:
    """Scan time coordinates across multiple NetCDF files.

    Args:
        filepaths: List of NetCDF file paths.
        time_axis: Name of the time dimension.

    Returns:
        List of dicts with keys: filepath, time_values, n_times.
    """
    file_info = []
    for fp in filepaths:
        if not os.path.exists(fp):
            logger.warning("File not found: %s", fp)
            continue
        ds = xr.open_dataset(fp)
        if time_axis not in ds.dims and time_axis not in ds.coords:
            logger.warning("Time axis '%s' not found in %s", time_axis, fp)
            ds.close()
            continue
        time_vals = ds[time_axis].values if time_axis in ds.coords else np.arange(ds.sizes[time_axis])
        file_info.append({
            "filepath": fp,
            "time_values": time_vals,
            "n_times": ds.sizes[time_axis],
        })
        ds.close()
    return file_info


def align_time_coordinates(
    file_info: List[Dict],
    deduplicate: bool = True,
    time_axis: str = "time",
) -> List[Dict]:
    """Align and optionally deduplicate time coordinates across files.

    Args:
        file_info: Output of scan_time_coordinates.
        deduplicate: If True, remove duplicate time values across files.
                     When duplicates exist, the file with the earlier index wins.
        time_axis: Name of the time dimension.

    Returns:
        List of dicts with aligned time info:
        - filepath: file path
        - time_values: time values in this file after alignment
        - global_time_indices: mapping from local valid index to global index
        - local_time_indices: local indices that are valid (not duplicates)
        - n_times: number of valid time steps
    """
    if not file_info:
        return []

    seen = {}
    global_idx = 0
    file_results = []

    for info in file_info:
        local_valid = []
        global_indices = []
        time_vals_valid = []
        for local_idx, t in enumerate(info["time_values"]):
            t_key = str(t)
            if deduplicate and t_key in seen:
                logger.debug(
                    "Duplicate time %s found in %s (already seen)",
                    t, info["filepath"],
                )
                continue
            seen[t_key] = global_idx
            local_valid.append(local_idx)
            global_indices.append(global_idx)
            time_vals_valid.append(t)
            global_idx += 1

        file_results.append({
            "filepath": info["filepath"],
            "time_values": np.array(time_vals_valid) if time_vals_valid else info["time_values"],
            "global_time_indices": global_indices,
            "local_time_indices": local_valid,
            "n_times": len(local_valid),
        })

    return file_results


class MultiFileNetCDFDataset(Dataset):
    """PyTorch Dataset aggregating multiple NetCDF files into a unified time series.

    Supports:
    - Automatic time coordinate alignment across files
    - Time coordinate deduplication
    - Temporal sequence creation (consecutive time step pairs)
    - Per-variable scaling and preprocessing

    When slice_mode is None, defaults to extracting lat×lon fields
    at a single depth level (first available or as specified by depth_indices).
    """

    def __init__(
        self,
        filepaths: List[str],
        variables: List[str],
        scaling: Dict[str, Tuple[float, float]],
        time_axis: str = "time",
        deduplicate: bool = True,
        sequence_length: int = 1,
        time_indices: Optional[Union[int, List[int], slice]] = None,
        depth_indices: Optional[Union[int, List[int], slice]] = None,
        lat_indices: Optional[Union[int, List[int], slice]] = None,
        lon_indices: Optional[Union[int, List[int], slice]] = None,
        slice_mode: Optional[Sequence[str]] = None,
        mask_variable: Optional[str] = None,
        mask_generator: Optional[Dict] = None,
        coverage_threshold: float = 0.0,
        fill_ratio_threshold: float = 1.0,
        transform: Optional[Callable] = None,
        variable_transforms: Optional[Dict] = None,
    ):
        self.filepaths = filepaths
        self.variables = variables
        self.scaling = scaling
        self.time_axis = time_axis
        self.deduplicate = deduplicate
        self.sequence_length = max(1, sequence_length)
        self.time_indices = resolve_indices(time_indices)
        self.depth_indices = resolve_indices(depth_indices)
        self.lat_indices = resolve_indices(lat_indices)
        self.lon_indices = resolve_indices(lon_indices)
        self.slice_mode = list(slice_mode) if slice_mode else None
        self.mask_variable = mask_variable
        self.mask_generator_params = mask_generator
        self.coverage_threshold = coverage_threshold
        self.fill_ratio_threshold = fill_ratio_threshold
        self.transform = transform
        self._preprocessing = variable_transforms

        file_info = scan_time_coordinates(filepaths, time_axis)
        self._aligned = align_time_coordinates(file_info, deduplicate, time_axis)

        self._global_sorted = sorted(
            [(gi, t, info) for info in self._aligned
             for gi, t in zip(info["global_time_indices"], info["time_values"])],
            key=lambda x: x[0],
        )

        self._file_meta = []
        self._cumulative_length = [0]
        for info in self._aligned:
            if info["n_times"] == 0:
                continue
            n = info["n_times"]
            self._file_meta.append({
                "filepath": info["filepath"],
                "global_time_indices": info["global_time_indices"],
                "local_time_indices": info["local_time_indices"],
                "n_times": info["n_times"],
            })
            self._cumulative_length.append(self._cumulative_length[-1] + n)

        self.total_length = self._cumulative_length[-1]

        if self.sequence_length > 1:
            valid_sequences = list(range(max(0, self.total_length - self.sequence_length + 1)))
            self._sequence_starts = valid_sequences
            self.total_length = len(valid_sequences)

        if self.total_length == 0:
            logger.warning("MultiFileNetCDFDataset has zero length.")

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        if self.sequence_length > 1:
            return self._get_sequence(idx)
        return self._get_single(idx)

    def _get_single(self, idx: int) -> Dict[str, Union[ torch.Tensor, Dict]]:
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_length}")

        file_idx = 0
        while idx >= self._cumulative_length[file_idx + 1]:
            file_idx += 1
        local_idx = idx - self._cumulative_length[file_idx]
        meta = self._file_meta[file_idx]

        ds = xr.open_dataset(meta["filepath"])
        time_idx = meta["local_time_indices"][local_idx % meta["n_times"]]

        sel = {self.time_axis: time_idx}

        effective_mode = self.slice_mode if self.slice_mode is not None else ["time", "depth"]

        if "depth" in effective_mode:
            if self.depth_indices is not None:
                sel["depth"] = self.depth_indices
            else:
                sel["depth"] = 0
        elif self.depth_indices is not None:
            sel["depth"] = self.depth_indices
        elif "depth" in ds.dims:
            sel["depth"] = 0

        if self.lat_indices is not None:
            sel["latitude"] = self.lat_indices
        if self.lon_indices is not None:
            sel["longitude"] = self.lon_indices

        selection = ds.isel(sel)
        result = self._extract_fields(meta["filepath"], selection, ds, time_index=time_idx)
        ds.close()

        if self.transform:
            result = self.transform(result)
        return result

    def _get_sequence(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        start = self._sequence_starts[idx]
        samples = [self._get_single(start + i) for i in range(self.sequence_length)]
        images = torch.stack([s["image"] for s in samples], dim=0)
        masks = torch.stack([s["mask"] for s in samples], dim=0)
        fill_masks = torch.stack([s["fill_mask"] for s in samples], dim=0)
        metas = [s["meta"] for s in samples]
        return {
            "image": images,
            "mask": masks,
            "fill_mask": fill_masks,
            "meta": {
                "sequence": True,
                "sequence_length": self.sequence_length,
                "start_index": start,
                "samples_meta": metas,
                "variables": self.variables,
            },
        }

    def _extract_fields(self, filepath, selection, ds, *, time_index):
        raw_data = {}
        raw_fill_masks = {}
        for var_name in self.variables:
            if var_name not in ds.data_vars:
                logger.error("Variable %s not found in %s", var_name, filepath)
                return self._get_dummy_sample()
            arr = selection[var_name].values.astype(np.float32)
            fill_mask = np.zeros(arr.shape, dtype=bool)
            if "_FillValue" in selection[var_name].attrs:
                fv = selection[var_name].attrs["_FillValue"]
                fill_mask = (arr == fv)
                arr = np.where(fill_mask, np.nan, arr)
            fill_mask = fill_mask | np.isnan(arr)
            raw_data[var_name] = arr
            raw_fill_masks[var_name] = fill_mask

        if not raw_data:
            return self._get_dummy_sample()

        first_shape = next(iter(raw_data.values())).shape
        if len(first_shape) >= 2:
            H, W = first_shape[-2], first_shape[-1]
        else:
            H, W = 1, 1

        depth_arr = None
        if "depth" in selection.coords:
            depth_arr = selection["depth"].values

        if self._preprocessing:
            raw_data = self._preprocessing(raw_data, depth=depth_arr)

        mask_arr = None
        if self.mask_variable and self.mask_variable in ds.data_vars:
            try:
                m = selection[self.mask_variable].values.astype(np.float32)
                while m.ndim < 2:
                    m = m[np.newaxis]
                if m.ndim > 2:
                    m = m.reshape(-1, *m.shape[-2:])[-1]
                mask_arr = m
            except Exception as e:
                logger.warning("Failed to load mask from %s: %s", filepath, e)

        channels = []
        fill_masks_list = []
        for var_name in self.variables:
            arr = raw_data[var_name]
            fm = raw_fill_masks[var_name]

            if var_name in self.scaling:
                vmin, vmax = self.scaling[var_name]
                if self._preprocessing:
                    vmin, vmax = self._preprocessing.transform_scaling(var_name, vmin, vmax)
                ch = self.scale_channel(arr, vmin, vmax)
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
        fill_ratio = float(fill_mask_tensor.mean())

        return {
            "image": image,
            "mask": mask_tensor,
            "fill_mask": fill_mask_tensor,
            "meta": {
                "filepath": filepath,
                "time_index": time_index,
                "slice_mode": self.slice_mode,
                "coverage": coverage,
                "coverage_ok": coverage >= self.coverage_threshold,
                "fill_ratio": fill_ratio,
                "variables": self.variables,
            },
        }

    @staticmethod
    def scale_channel(x: np.ndarray, vmin: float, vmax: float) -> torch.Tensor:
        t = torch.from_numpy(x).float()
        t = torch.nan_to_num(t, nan=vmin)
        t = (t - vmin) / (vmax - vmin)
        return torch.clamp(t, 0.0, 1.0)

    def _get_dummy_sample(self) -> Dict[str, Union[torch.Tensor, Dict]]:
        C = len(self.variables)
        H, W = 64, 64
        return {
            "image": torch.zeros(C, H, W, dtype=torch.float32),
            "mask": torch.zeros(1, H, W, dtype=torch.float32),
            "fill_mask": torch.zeros(C, H, W, dtype=torch.float32),
            "meta": {"error": True, "variables": self.variables},
        }

    @property
    def global_time_values(self):
        """Return sorted unique global time values."""
        return [t for _, t, _ in self._global_sorted]
