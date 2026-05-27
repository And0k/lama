"""Loader for NetCDF files with oceanographic data."""

import numpy as np
import xarray as xr
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_netcdf(
    filepath: str,
    variables: Optional[List[str]] = None,
    time_indices: Optional[Union[int, List[int], slice]] = None,
    depth_indices: Optional[Union[int, List[int], slice]] = None,
    replace_fill_value: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load NetCDF file and extract selected variables as raw float32 arrays.

    No normalization or scaling is performed — that responsibility belongs to
    the dataset class, which has access to globally configured physical ranges.

    Args:
        filepath: Path to NetCDF file.
        variables: List of variable names to load. If None, load all data variables.
        time_indices: Time indices to select (int, list, or slice).
        depth_indices: Depth indices to select (int, list, or slice).
        replace_fill_value: Replace _FillValue with NaN.

    Returns:
        Dictionary mapping variable names to raw numpy float32 arrays.
    """
    ds = xr.open_dataset(filepath)

    if variables is None:
        variables = list(ds.data_vars.keys())
    else:
        missing = set(variables) - set(ds.data_vars.keys())
        if missing:
            raise ValueError(f"Variables not found in dataset: {missing}")

    if time_indices is not None:
        ds = ds.isel(time=time_indices)
    if depth_indices is not None:
        ds = ds.isel(depth=depth_indices)

    result = {}
    for var_name in variables:
        var = ds[var_name]
        data = var.values.astype(np.float32)
        if replace_fill_value and '_FillValue' in var.attrs:
            fill_value = var.attrs['_FillValue']
            data = np.where(data == fill_value, np.nan, data)
        result[var_name] = data

    ds.close()
    return result


def get_available_variables(filepath: str) -> List[str]:
    """
    Get list of data variable names in NetCDF file.

    Args:
        filepath: Path to NetCDF file.

    Returns:
        List of variable names.
    """
    ds = xr.open_dataset(filepath)
    variables = list(ds.data_vars.keys())
    ds.close()
    return variables


def get_variable_info(filepath: str, variable: str) -> Dict:
    """
    Get metadata for a specific variable.

    Args:
        filepath: Path to NetCDF file.
        variable: Variable name.

    Returns:
        Dictionary with variable metadata.
    """
    ds = xr.open_dataset(filepath)
    if variable not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable '{variable}' not found in dataset")
    var = ds[variable]
    info = {
        'dtype': str(var.dtype),
        'shape': var.shape,
        'dims': var.dims,
        'attrs': dict(var.attrs),
    }
    ds.close()
    return info