import os
from typing import List, Optional

import numpy as np
try:
    from netCDF4 import Dataset as NetCDF4Dataset
except Exception:  # pragma: no cover - optional dependency
    NetCDF4Dataset = None


class NetCDFVerticalSliceDataset:
    """Simple dataset that reads netCDF files and returns vertical slices.

    Each file is expected to contain a variable (default 'image') with shape
    (C, H, W) or (H, W, C). Optionally a 'mask' variable may be present.

    Returns dicts with keys: 'image', optionally 'mask', 'file', 'coverage_ok'.
    """

    def __init__(self, files: Optional[List[str]] = None, variable: str = "image",
                 mask_variable: Optional[str] = "mask", coverage_threshold: float = 0.0):
        files = files or []
        self.files = list(files)
        self.variable = variable
        self.mask_variable = mask_variable
        self.coverage_threshold = float(coverage_threshold)

    def __len__(self):
        return len(self.files)

    def _open(self, path):
        if NetCDF4Dataset is None:
            raise RuntimeError("netCDF4 is required to read netCDF files. Install with `pip install netCDF4`.")
        return NetCDF4Dataset(path, mode="r")

    def __getitem__(self, idx):
        path = self.files[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ds = self._open(path)
        try:
            var = ds.variables[self.variable][:]
        except Exception as e:
            raise RuntimeError(f"Failed to read variable '{self.variable}' from {path}: {e}")

        arr = np.array(var)
        # normalize shape to (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            # could be HWC or CHW; guess by channel dim
            if arr.shape[0] <= 4:  # probably CHW
                pass
            elif arr.shape[2] <= 4:  # HWC -> CHW
                arr = np.transpose(arr, (2, 0, 1))

        sample = {"image": arr.astype(np.float32), "file": path}

        coverage_ok = True
        if self.mask_variable and self.mask_variable in ds.variables:
            mask = np.array(ds.variables[self.mask_variable][:])
            if mask.ndim == 3:
                # reduce to single channel
                mask = mask[0] if mask.shape[0] <= 4 else mask[:, :, 0]
            mask = (mask > 0).astype(np.uint8)
            coverage = float(mask.mean())
            coverage_ok = coverage >= self.coverage_threshold
            sample["mask"] = mask

        sample["coverage_ok"] = coverage_ok
        ds.close()
        return sample
