"""Shared fixtures for the NetCDF inpainting pipeline tests."""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Dimensions shared across all fixture NetCDF files
# ---------------------------------------------------------------------------
T, D, H, W = 3, 5, 16, 20


@pytest.fixture(scope="session")
def dummy_nc(tmp_path_factory):
    """Create a synthetic 4-D NetCDF file with two 4D variables (thetao, so).

    Shape: (time=3, depth=5, lat=16, lon=20)
    Both variables have _FillValue=-999.0.
    Returns path to the created file.
    """
    import xarray as xr

    path = tmp_path_factory.mktemp("data") / "dummy.nc"
    rng = np.random.RandomState(42)

    ds = xr.Dataset(
        {},
        coords={
            "time": np.arange(T),
            "depth": np.arange(D),
            "latitude": np.arange(H),
            "longitude": np.arange(W),
        },
    )

    for name in ("thetao", "so"):
        arr = rng.randn(T, D, H, W).astype(np.float32) * 10 + 15
        ds[name] = (("time", "depth", "latitude", "longitude"), arr)
        ds[name].attrs["_FillValue"] = -999.0

    ds.to_netcdf(path, engine="netcdf4")
    return path


@pytest.fixture(scope="session")
def dummy_nc_3d(tmp_path_factory):
    """Create a 3-D NetCDF file (no depth) with variable 'bottomT'.

    Shape: (time=2, lat=16, lon=20)
    Returns path to the created file.
    """
    import xarray as xr

    path = tmp_path_factory.mktemp("data") / "dummy_3d.nc"
    rng = np.random.RandomState(99)

    ds = xr.Dataset(
        {},
        coords={
            "time": np.arange(2),
            "latitude": np.arange(H),
            "longitude": np.arange(W),
        },
    )
    arr = rng.randn(2, H, W).astype(np.float32) * 5 + 15
    ds["bottomT"] = (("time", "latitude", "longitude"), arr)
    ds["bottomT"].attrs["_FillValue"] = -999.0

    ds.to_netcdf(path, engine="netcdf4")
    return path


@pytest.fixture(scope="session")
def mask_params():
    """Return mask generator parameters (irregular masks only)."""
    return {
        "_target_": "saicinpainting.training.data.masks.MixedMaskGenerator",
        "irregular_proba": 1.0,
        "box_proba": 0.0,
        "segm_proba": 0.0,
        "squares_proba": 0.0,
        "superres_proba": 0.0,
        "outpainting_proba": 0.0,
    }


@pytest.fixture(scope="session")
def scaling():
    """Return scaling dict for the two 4-D variables."""
    return {"thetao": (-3.0, 40.0), "so": (0.0, 40.0)}
