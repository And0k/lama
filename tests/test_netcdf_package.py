"""Tests for the netCDF subpackage."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from netcdf.data.loader import load_netcdf, get_available_variables, get_variable_info
from netcdf.data.slicer import slice_nc
from netcdf.mask_generator import generate_mask, generate_masks_for_batch
from netcdf.data.dataset import NetCDFDataset, _resolve_axis_count
from netcdf.visualization import plot_slice, plot_mask, plot_comparison

sys.path.insert(0, Path(__file__).parent.parent.as_posix())

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore

try:
    import netCDF4
except ImportError:
    netCDF4 = None  # type: ignore


def _make_dummy_nc(filepath, vars_info, dims=("time", "depth", "latitude", "longitude"), shape=(2, 4, 8, 10)):
    if xr is None:
        pytest.skip("xarray and netCDF4 required for dummy file creation")

    ds = xr.Dataset({}, coords={
        "time": np.arange(shape[0]),
        "depth": np.arange(shape[1]),
        "latitude": np.arange(shape[2]),
        "longitude": np.arange(shape[3]),
    })

    for var_name, var_dtype in vars_info:
        data = np.random.randn(*shape).astype(var_dtype)
        ds[var_name] = (dims, data)
        ds[var_name].attrs["_FillValue"] = -999.0

    ds.to_netcdf(filepath, engine="netcdf4")


class TestLoadNetCDF:

    def test_load_netcdf_basic(self, tmp_path):
        nc_path = tmp_path / "dummy.nc"
        _make_dummy_nc(nc_path, [("temperature", np.float32)])
        data = load_netcdf(str(nc_path), variables=["temperature"])
        assert "temperature" in data
        assert data["temperature"].shape == (2, 4, 8, 10)

    def test_load_netcdf_returns_raw_values(self, tmp_path):
        nc_path = tmp_path / "dummy_raw.nc"
        _make_dummy_nc(nc_path, [("temperature", np.float32)])
        data = load_netcdf(str(nc_path), variables=["temperature"], replace_fill_value=True)
        assert data["temperature"].dtype == np.float32
        assert not np.all((data["temperature"] >= 0) & (data["temperature"] <= 1))

    def test_load_netcdf_replace_fillvalue(self, tmp_path):
        nc_path = tmp_path / "dummy2.nc"
        if xr is not None:
            _make_dummy_nc(nc_path, [("salinity", np.float32)])
            data = load_netcdf(str(nc_path), variables=["salinity"], replace_fill_value=True)
            assert "salinity" in data

    def test_get_available_variables(self, tmp_path):
        nc_path = tmp_path / "dummy3.nc"
        _make_dummy_nc(nc_path, [("temperature", np.float32), ("salinity", np.float32)])
        vars_list = get_available_variables(str(nc_path))
        assert "temperature" in vars_list
        assert "salinity" in vars_list

    def test_get_variable_info(self, tmp_path):
        nc_path = tmp_path / "dummy4.nc"
        _make_dummy_nc(nc_path, [("u", np.float32)])
        info = get_variable_info(str(nc_path), "u")
        assert "shape" in info
        assert "attrs" in info


class TestGenerateMasks:

    def test_generate_mask_shape(self):
        mask = generate_mask((64, 64), mask_params={
            '_target_': 'saicinpainting.training.data.masks.MixedMaskGenerator',
            'irregular_proba': 1.0,
            'box_proba': 0.0,
            'segm_proba': 0.0,
            'squares_proba': 0.0,
            'superres_proba': 0.0,
            'outpainting_proba': 0.0,
        })
        assert mask.shape == (1, 64, 64)
        assert mask.dtype == np.uint8

    def test_generate_masks_for_batch(self):
        mask_params = {
            '_target_': 'saicinpainting.training.data.masks.MixedMaskGenerator',
            'irregular_proba': 1.0,
            'box_proba': 0.0,
            'segm_proba': 0.0,
            'squares_proba': 0.0,
            'superres_proba': 0.0,
            'outpainting_proba': 0.0,
        }
        masks = generate_masks_for_batch((4, 3, 64, 64), mask_params=mask_params)
        assert masks.shape == (4, 1, 64, 64)


class TestNetCDFDataset:

    def test_dataset_len(self, tmp_path):
        nc_path = tmp_path / "dummy7.nc"
        _make_dummy_nc(nc_path, [("temperature", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["temperature"],
            scaling={},
        )
        assert len(ds) == 2  # 2 time steps

    def test_dataset_getitem_returns_2d_field(self, tmp_path):
        nc_path = tmp_path / "dummy8.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={"thetao": [-3.0, 40.0]},
        )
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert "meta" in sample
        assert sample["image"].ndim == 3
        C, H, W = sample["image"].shape
        assert H == 8  # latitude
        assert W == 10  # longitude

    def test_dataset_scaling_range(self, tmp_path):
        nc_path = tmp_path / "dummy9.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={"thetao": [-3.0, 40.0]},
        )
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_dataset_multi_variable(self, tmp_path):
        nc_path = tmp_path / "dummy_multi.nc"
        _make_dummy_nc(
            nc_path,
            [("thetao", np.float32), ("so", np.float32), ("uo", np.float32)],
            shape=(2, 4, 8, 10),
        )
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao", "so", "uo"],
            scaling={"thetao": [-3.0, 40.0], "so": [0.0, 40.0], "uo": [-2.0, 2.0]},
        )
        sample = ds[0]
        assert sample["image"].shape[0] == 3  # 3 channels

    def test_dataset_get_shape(self, tmp_path):
        nc_path = tmp_path / "dummy_shape.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={},
        )
        C, H, W = ds.get_shape()
        assert C == 1
        assert H == 8
        assert W == 10

    def test_dataset_3d_variable(self, tmp_path):
        nc_path = tmp_path / "dummy_3d_ds.nc"
        if xr is not None:
            ds_xr = xr.Dataset({}, coords={
                "time": np.arange(2),
                "latitude": np.arange(8),
                "longitude": np.arange(10),
            })
            data = np.random.randn(2, 8, 10).astype(np.float32) * 5 + 15
            ds_xr["bottomT"] = (("time", "latitude", "longitude"), data)
            ds_xr["bottomT"].attrs["_FillValue"] = -999.0
            ds_xr.to_netcdf(nc_path, engine="netcdf4")

            ds = NetCDFDataset(
                filepaths=[str(nc_path)],
                variables=["bottomT"],
                scaling={"bottomT": [-3.0, 40.0]},
            )
            sample = ds[0]
            assert sample["image"].ndim == 3
            assert sample["image"].shape == (1, 8, 10)



    def test_dataset_slice_mode_time_depth(self, tmp_path):
        nc_path = tmp_path / "dummy_td.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={},
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 4),
        )
        assert len(ds) == 8  # 2 time * 4 depth

    def test_dataset_slice_mode_time_lat(self, tmp_path):
        nc_path = tmp_path / "dummy_tl.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={},
            slice_mode=["time", "latitude"],
            time_indices=slice(0, 2),
            lat_indices=slice(0, 8),
        )
        assert len(ds) == 16  # 2 time * 8 lat
        sample = ds[0]
        assert sample["image"].shape[1] == 4  # depth
        assert sample["image"].shape[2] == 10  # longitude

    def test_dataset_slice_mode_time_lon(self, tmp_path):
        nc_path = tmp_path / "dummy_tlon.nc"
        _make_dummy_nc(nc_path, [("thetao", np.float32)], shape=(2, 4, 8, 10))
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao"],
            scaling={},
            slice_mode=["time", "longitude"],
            time_indices=slice(0, 2),
            lon_indices=slice(0, 10),
        )
        assert len(ds) == 20  # 2 time * 10 lon
        sample = ds[0]
        assert sample["image"].shape[1] == 4  # depth
        assert sample["image"].shape[2] == 8  # latitude

    def test_resolve_axis_count(self):
        from netcdf.data.dataset import _resolve_axis_count
        shape = (2, 4, 8, 10)
        dims = ["time", "depth", "latitude", "longitude"]
        assert _resolve_axis_count(shape, dims, "time", None) == 2
        assert _resolve_axis_count(shape, dims, "time", 0) == 1
        assert _resolve_axis_count(shape, dims, "time", slice(0, 1)) == 1
        assert _resolve_axis_count(shape, dims, "latitude", [0, 1, 2]) == 3


class TestSliceNC:

    def test_slice_nc_lat_lon(self):
        arr = np.arange(2 * 4 * 8 * 10).reshape(2, 4, 8, 10).astype(float)
        out = slice_nc(arr, ('lat', 'lon'), (0, 0))
        assert out.shape == (8, 10)

    def test_slice_nc_depth_lon(self):
        arr = np.arange(2 * 4 * 8 * 10).reshape(2, 4, 8, 10).astype(float)
        out = slice_nc(arr, ('depth', 'lon'), (0, 0))
        assert out.shape == (4, 10)

    def test_slice_nc_depth_lat(self):
        arr = np.arange(2 * 4 * 8 * 10).reshape(2, 4, 8, 10).astype(float)
        out = slice_nc(arr, ('depth', 'lat'), (0, 0))
        assert out.shape == (4, 8)

    def test_slice_nc_reorder(self):
        arr = np.arange(2 * 4 * 8 * 10).reshape(2, 4, 8, 10).astype(float)
        out1 = slice_nc(arr, ('lat', 'lon'), (0, 0))
        out2 = slice_nc(arr, ('lon', 'lat'), (0, 0))
        assert out1.shape == (8, 10)
        assert out2.shape == (10, 8)


class TestVisualization:

    def test_plot_slice(self):
        data = np.random.rand(16, 16)
        ax = plot_slice(data)
        assert ax is not None

    def test_plot_mask(self):
        mask = np.random.randint(0, 2, (16, 16), dtype=np.uint8)
        ax = plot_mask(mask)
        assert ax is not None

    def test_plot_comparison(self):
        orig = np.random.rand(16, 16)
        masked = orig.copy()
        result = np.random.rand(16, 16)
        fig = plot_comparison(orig, masked, result)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
