import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Sequence
import xarray as xr


def slice_nc(arr4d: np.ndarray, axes: Tuple[str, str], idx: Tuple[int, int]) -> np.ndarray:
    """
    Universal slicing for 4D arrays with named axes.

    Args:
        arr4d: np.ndarray with shape (T, D, Y, X) in order (time, depth, lat, lon).
        axes: Tuple of two axis names to keep in output, e.g. ("lat", "lon").
        idx: Tuple of two indices for the remaining axes, e.g. (t, d).

    Returns:
        2D array with axes ordered as specified.
    """
    dims = ("time", "depth", "lat", "lon")
    sl = [slice(None)] * 4
    idx_axes = [d for d in dims if d not in axes]
    sl[dims.index(idx_axes[0])] = idx[0]
    sl[dims.index(idx_axes[1])] = idx[1]
    out = arr4d[tuple(sl)]

    # After slicing, axes are in order of their original positions (preserving order)
    # Reorder to match the requested axes order
    # e.g. if axes=('lon', 'lat'), we want output (lon, lat) not (lat, lon)
    out_dims = [d for d in dims if d in axes]
    if tuple(out_dims) != axes:
        # Need to transpose: find where each output axis currently is
        current_pos = [out_dims.index(a) for a in axes]
        out = np.moveaxis(out, current_pos, [0, 1])

    return out


def _resolve_axis_count(
    shape: tuple[int, ...],
    dims: Sequence[str],
    axis_name: str,
    indices: int | slice | Sequence[int] | None,
) -> int:
    """Resolve effective sample count for a single dimension.

    Handles integer, slice, and sequence indexing uniformly while
    falling back to the full axis size when no indices are specified.
    """
    if axis_name not in dims:
        return 1

    axis_size = shape[dims.index(axis_name)]

    match indices:
        case None:
            return axis_size
        case int():
            return 1
        case slice():
            start, stop, step = indices.indices(axis_size)
            return len(range(start, stop, step))
        case _:
            return len(indices)


def compute_sample_count(
    shape: tuple[int, ...],
    dims: Sequence[str],
    slice_mode: Optional[Sequence[str]],
    time_indices: Optional[Union[int, List[int], slice]] = None,
    depth_indices: Optional[Union[int, List[int], slice]] = None,
    lat_indices: Optional[Union[int, List[int], slice]] = None,
    lon_indices: Optional[Union[int, List[int], slice]] = None,
) -> int:
    """Compute expected number of samples by reducing over slice_mode axes.

    slice_mode should be defined as a sequence of dimension names,
    e.g. ["time", "latitude"] instead of "time_lat". This eliminates
    branching entirely—each axis contributes multiplicatively via a
    single uniform resolution path.
    """
    if slice_mode is None:
        slice_mode = ["time", "depth"]

    index_map: dict[str, int | slice | Sequence[int] | None] = {
        "time": time_indices,
        "depth": depth_indices,
        "latitude": lat_indices,
        "longitude": lon_indices,
    }

    counts: tuple[int, ...] = tuple(
        _resolve_axis_count(shape, list(dims), axis, index_map.get(axis))
        for axis in slice_mode
    )

    from functools import reduce
    from operator import mul

    return reduce(mul, counts, 1)


def subset_xarray(
    ds: xr.Dataset,
    slice_mode: Optional[Sequence[str]],
    time_indices: Optional[Union[int, List[int], slice]] = None,
    depth_indices: Optional[Union[int, List[int], slice]] = None,
    lat_indices: Optional[Union[int, List[int], slice]] = None,
    lon_indices: Optional[Union[int, List[int], slice]] = None,
) -> xr.Dataset:
    """Subset an xarray Dataset using isel for lazy loading.

    Args:
        ds: Input xarray Dataset.
        slice_mode: Sequence of dimension names to subset over, e.g. ["time", "depth"].
                   If None, returns the full dataset (no subsetting).
        time_indices, depth_indices, lat_indices, lon_indices:
            Indices to select for each dimension (int, list, or slice).

    Returns:
        Subsetted xarray Dataset.
    """
    if slice_mode is None:
        return ds

    # Build selection dictionary for isel
    sel_kw = {}
    if "time" in slice_mode and time_indices is not None:
        sel_kw["time"] = time_indices
    if "depth" in slice_mode and depth_indices is not None:
        sel_kw["depth"] = depth_indices
    if "latitude" in slice_mode and lat_indices is not None:
        sel_kw["latitude"] = lat_indices
    if "longitude" in slice_mode and lon_indices is not None:
        sel_kw["longitude"] = lon_indices

    if sel_kw:
        return ds.isel(sel_kw)
    return ds