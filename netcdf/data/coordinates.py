"""Advanced coordinate system support for NetCDF data.

Provides utilities for:
- Rotated pole coordinate transformations
- Horizontal regridding (interpolation between grids)
- Coordinate system detection and conversion

Usage::

    from netcdf.data.coordinates import regrid_dataset, detect coordinate_system

    # Detect coordinate system
    cs = detect_coordinate_system("file.nc")

    # Regrid to a regular lat/lon grid
    regrid_dataset("input.nc", "output.nc",
                   target_lat=np.linspace(-90, 90, 180),
                   target_lon=np.linspace(0, 360, 360))
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def detect_coordinate_system(filepath: str) -> Dict:
    """Detect the coordinate system of a NetCDF file.

    Args:
        filepath: Path to NetCDF file.

    Returns:
        Dict with keys:
        - type: "regular", "rotated_pole", "curvilinear", "unknown"
        - lat_name: name of latitude coordinate
        - lon_name: name of longitude coordinate
        - is_rotated: whether rotation parameters are present
        - rotation_pole: (phi, lambda) if rotated
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    lat_name = None
    lon_name = None
    for name in ds.coords:
        lower = name.lower()
        if lat_name is None and ("lat" in lower or lower == "y"):
            lat_name = name
        if lon_name is None and ("lon" in lower or lower == "x"):
            lon_name = name

    if lat_name is None:
        lat_name = "latitude" if "latitude" in ds.coords else None
    if lon_name is None:
        lon_name = "longitude" if "longitude" in ds.coords else None

    result = {
        "type": "unknown",
        "lat_name": lat_name,
        "lon_name": lon_name,
        "is_rotated": False,
        "rotation_pole": None,
    }

    if lat_name is None or lon_name is None:
        ds.close()
        return result

    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        lat_sorted = np.all(np.diff(lat_vals) > 0) or np.all(np.diff(lat_vals) < 0)
        lon_sorted = np.all(np.diff(lon_vals) > 0) or np.all(np.diff(lon_vals) < 0)
        if lat_sorted and lon_sorted:
            result["type"] = "regular"
        else:
            result["type"] = "curvilinear"
    elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
        result["type"] = "curvilinear"

    for attr_name in ("grid_north_pole_latitude", "grid_north_pole_longitude",
                       "north_pole_latitude", "north_pole_longitude"):
        if attr_name in ds.attrs:
            result["is_rotated"] = True
            result["type"] = "rotated_pole"
            np_lat = ds.attrs.get("grid_north_pole_latitude", ds.attrs.get("north_pole_latitude", 90.0))
            np_lon = ds.attrs.get("grid_north_pole_longitude", ds.attrs.get("north_pole_longitude", 0.0))
            result["rotation_pole"] = (float(np_lat), float(np_lon))
            break

    ds.close()
    return result


def rotated_to_regular(
    lat_rot: np.ndarray,
    lon_rot: np.ndarray,
    pole_lat: float = 90.0,
    pole_lon: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert rotated pole coordinates to regular lat/lon.

    Uses the standard CMIP/CORE rotated pole transformation.
    The pole position (pole_lat, pole_lon) specifies where the north pole
    of the rotated grid sits in the regular coordinate system.

    Args:
        lat_rot: Rotated latitude values (degrees).
        lon_rot: Rotated longitude values (degrees).
        pole_lat: Regular latitude of the rotated north pole (degrees).
        pole_lon: Regular longitude of the rotated north pole (degrees).

    Returns:
        (lat_reg, lon_reg) arrays in regular lat/lon (degrees).
    """
    lat_r = np.radians(np.asarray(lat_rot, dtype=np.float64))
    lon_r = np.radians(np.asarray(lon_rot, dtype=np.float64))
    phi = np.radians(pole_lat)
    lam = np.radians(pole_lon)

    x1 = np.cos(lat_r) * np.cos(lon_r - lam)
    y1 = np.cos(lat_r) * np.sin(lon_r - lam)
    z1 = np.sin(lat_r)

    angle = -(np.pi / 2 - phi)
    x2 = x1 * np.cos(angle) + z1 * np.sin(angle)
    y2 = y1
    z2 = -x1 * np.sin(angle) + z1 * np.cos(angle)

    x3 = x2 * np.cos(lam) - y2 * np.sin(lam)
    y3 = x2 * np.sin(lam) + y2 * np.cos(lam)
    z3 = z2

    lat_reg = np.arcsin(np.clip(z3, -1, 1))
    lon_reg = np.arctan2(y3, x3)

    return np.degrees(lat_reg).astype(np.float64), np.degrees(lon_reg).astype(np.float64)


def build_regridder(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "bilinear",
) -> callable:
    """Build a regridding function that interpolates from source to target grid.

    Args:
        src_lat: Source latitude 1-D array.
        src_lon: Source longitude 1-D array.
        target_lat: Target latitude 1-D array.
        target_lon: Target longitude 1-D array.
        method: Interpolation method: "bilinear" or "nearest".

    Returns:
        Function that takes a 2-D array (src_lat x src_lon) and returns
        a 2-D array (target_lat x target_lon).
    """
    from scipy.interpolate import RegularGridInterpolator

    src_lat_sorted = np.sort(src_lat)
    src_lon_sorted = np.sort(src_lon)

    needs_flip_lat = not np.array_equal(src_lat, src_lat_sorted)
    needs_flip_lon = not np.array_equal(src_lon, src_lon_sorted)

    def _sort_array(arr, lat_flip, lon_flip):
        if lat_flip:
            arr = arr[::-1, :]
        if lon_flip:
            arr = arr[:, ::-1]
        return arr

    target_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    points = np.stack([target_grid[0].ravel(), target_grid[1].ravel()], axis=-1)

    def regrid(arr: np.ndarray) -> np.ndarray:
        arr = _sort_array(arr, needs_flip_lat, needs_flip_lon)

        if method == "bilinear":
            try:
                interp = RegularGridInterpolator(
                    (src_lat_sorted, src_lon_sorted),
                    arr,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                return interp(points).reshape(len(target_lat), len(target_lon)).astype(arr.dtype)
            except Exception:
                logger.warning("Bilinear interpolation failed, falling back to nearest")

        interp = RegularGridInterpolator(
            (src_lat_sorted, src_lon_sorted),
            arr,
            method="nearest",
            bounds_error=False,
            fill_value=None,
        )
        return interp(points).reshape(len(target_lat), len(target_lon)).astype(arr.dtype)

    return regrid


def regrid_dataset(
    input_path: str,
    output_path: str,
    target_lat: Optional[np.ndarray] = None,
    target_lon: Optional[np.ndarray] = None,
    target_lat_size: int = 180,
    target_lon_size: int = 360,
    variables: Optional[List[str]] = None,
    method: str = "bilinear",
    lat_name: Optional[str] = None,
    lon_name: Optional[str] = None,
) -> None:
    """Regrid a NetCDF file to a regular lat/lon grid.

    Args:
        input_path: Input NetCDF file path.
        output_path: Output NetCDF file path.
        target_lat: Target latitude array. If None, generated from target_lat_size.
        target_lon: Target longitude array. If None, generated from target_lon_size.
        target_lat_size: Number of target latitude points.
        target_lon_size: Number of target longitude points.
        variables: Variables to regrid. If None, regrid all data variables.
        method: Interpolation method.
        lat_name: Name of latitude coordinate in input file.
        lon_name: Name of longitude coordinate in input file.
    """
    import xarray as xr

    ds = xr.open_dataset(input_path)

    if lat_name is None or lon_name is None:
        cs = detect_coordinate_system(input_path)
        lat_name = cs["lat_name"] or "latitude"
        lon_name = cs["lon_name"] or "longitude"

    src_lat = ds[lat_name].values
    src_lon = ds[lon_name].values

    if src_lat.ndim > 1:
        src_lat_1d = src_lat[:, 0] if src_lat.shape[1] > 0 else src_lat[0, :]
        src_lon_1d = src_lon[0, :] if src_lon.shape[0] > 0 else src_lon[:, 0]
    else:
        src_lat_1d = src_lat
        src_lon_1d = src_lon

    if target_lat is None:
        target_lat = np.linspace(float(np.min(src_lat_1d)), float(np.max(src_lat_1d)), target_lat_size)
    if target_lon is None:
        target_lon = np.linspace(float(np.min(src_lon_1d)), float(np.max(src_lon_1d)), target_lon_size)

    regridder = build_regridder(src_lat_1d, src_lon_1d, target_lat, target_lon, method=method)

    if variables is None:
        variables = list(ds.data_vars.keys())

    result_ds = xr.Dataset(
        coords={
            "latitude": target_lat,
            "longitude": target_lon,
        }
    )

    if "time" in ds.dims:
        result_ds = result_ds.assign_coords(time=ds["time"].values)

    for var_name in variables:
        if var_name not in ds.data_vars:
            logger.warning("Variable %s not found in %s", var_name, input_path)
            continue

        var = ds[var_name]
        data = var.values.astype(np.float32)

        if data.ndim == 2:
            regridded = regridder(data)
            result_ds[var_name] = (("latitude", "longitude"), regridded)
        elif data.ndim == 3:
            time_dim = "time" if "time" in var.dims else var.dims[0]
            # spatial_dims = [d for d in var.dims if d not in (time_dim,)]
            n_time = data.shape[var.dims.index(time_dim)]
            regridded = np.zeros((n_time, len(target_lat), len(target_lon)), dtype=np.float32)
            for t in range(n_time):
                slab = data[t] if time_dim == var.dims[0] else data[:, :, t] if var.dims[-1] == time_dim else data[t]
                regridded[t] = regridder(slab)
            result_ds[var_name] = ((time_dim, "latitude", "longitude"), regridded)
        elif data.ndim == 4:
            time_dim = "time" if "time" in var.dims else var.dims[0]
            depth_dim = "depth" if "depth" in var.dims else var.dims[1]
            n_time = data.shape[var.dims.index(time_dim)]
            n_depth = data.shape[var.dims.index(depth_dim)]
            regridded = np.zeros((n_time, n_depth, len(target_lat), len(target_lon)), dtype=np.float32)
            for t in range(n_time):
                for d in range(n_depth):
                    slab = data[t, d] if var.dims[:2] == (time_dim, depth_dim) else data[t, d]
                    regridded[t, d] = regridder(slab)
            result_ds[var_name] = ((time_dim, depth_dim, "latitude", "longitude"), regridded)
        else:
            logger.warning("Skipping variable %s with unsupported ndim=%d", var_name, data.ndim)
            continue

        for attr in var.attrs:
            result_ds[var_name].attrs[attr] = var.attrs[attr]

    result_ds.to_netcdf(output_path)
    ds.close()
    logger.info("Regridded %s -> %s (%d variables)", input_path, output_path, len(variables))
