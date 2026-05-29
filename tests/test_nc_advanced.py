"""Tests for new NetCDF features: preprocessing, multi-file aggregation, coordinates."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, Path(__file__).parent.parent.as_posix())

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import netCDF4
except ImportError:
    netCDF4 = None


def _make_nc(filepath, variables, shape=(3, 5, 8, 10), time_offset=0, attrs=None):
    if xr is None:
        pytest.skip("xarray required")
    dims = ("time", "depth", "latitude", "longitude")
    coords = {
        "time": np.arange(shape[0]) + time_offset,
        "depth": np.arange(shape[1]),
        "latitude": np.arange(shape[2]),
        "longitude": np.arange(shape[3]),
    }
    ds = xr.Dataset({}, coords=coords)
    rng = np.random.RandomState(42 + time_offset)
    for name in variables:
        data = rng.randn(*shape).astype(np.float32) * 10 + 15
        ds[name] = (dims, data)
        ds[name].attrs["_FillValue"] = -999.0
        if attrs and name in attrs:
            for k, v in attrs[name].items():
                ds[name].attrs[k] = v
    ds.to_netcdf(filepath, engine="netcdf4")
    return filepath


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_log_transform(self):
        from netcdf.data.preprocessing import LogTransform
        t = LogTransform(base="natural", offset=1.0)
        arr = np.array([0.0, 3.0, 39.0], dtype=np.float32)
        result = t(arr)
        expected = np.log(arr + 1.0).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_log10_transform(self):
        from netcdf.data.preprocessing import LogTransform
        t = LogTransform(base="10", offset=1.0)
        arr = np.array([1.0, 9.0], dtype=np.float32)
        result = t(arr)
        expected = np.log10(arr + 1.0).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_log_transform_scaling(self):
        from netcdf.data.preprocessing import LogTransform
        t = LogTransform(offset=1.0)
        vmin, vmax = t.transform_scaling(0.0, 40.0)
        np.testing.assert_allclose(vmin, np.log(1.0))
        np.testing.assert_allclose(vmax, np.log(41.0))

    def test_sqrt_transform(self):
        from netcdf.data.preprocessing import SqrtTransform
        t = SqrtTransform()
        arr = np.array([-1.0, 0.0, 4.0, 9.0], dtype=np.float32)
        result = t(arr)
        expected = np.sqrt(np.maximum(arr, 0)).astype(np.float32)
        np.testing.assert_allclose(result, expected)

    def test_sqrt_transform_scaling(self):
        from netcdf.data.preprocessing import SqrtTransform
        t = SqrtTransform()
        vmin, vmax = t.transform_scaling(0.0, 40.0)
        np.testing.assert_allclose(vmin, 0.0)
        np.testing.assert_allclose(vmax, np.sqrt(40.0))

    def test_power_transform(self):
        from netcdf.data.preprocessing import PowerTransform
        t = PowerTransform(exponent=0.5)
        arr = np.array([-4.0, 0.0, 9.0], dtype=np.float32)
        result = t(arr)
        np.testing.assert_allclose(result[0], -2.0, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0)
        np.testing.assert_allclose(result[2], 3.0, atol=1e-6)

    def test_depth_dependent_scaling(self):
        from netcdf.data.preprocessing import DepthDependentScaling
        t = DepthDependentScaling(surface_factor=1.2, deep_factor=0.8, depth_threshold=500.0)
        arr = np.ones((5, 8, 10), dtype=np.float32)
        depth = np.array([0, 250, 500, 750, 1000], dtype=np.float32)
        result = t(arr, depth=depth)
        assert result.shape == arr.shape
        np.testing.assert_allclose(result[0, 0, 0], 1.2, atol=1e-5)
        np.testing.assert_allclose(result[2, 0, 0], 0.8, atol=1e-5)
        np.testing.assert_allclose(result[4, 0, 0], 0.8, atol=1e-5)

    def test_depth_dependent_scaling_no_depth(self):
        from netcdf.data.preprocessing import DepthDependentScaling
        t = DepthDependentScaling(surface_factor=1.5, deep_factor=0.5)
        arr = np.ones((3, 4), dtype=np.float32)
        result = t(arr, depth=None)
        np.testing.assert_allclose(result, 1.5)

    def test_build_transform_from_string(self):
        from netcdf.data.preprocessing import build_transform
        t = build_transform("log")
        assert t is not None

    def test_build_transform_from_dict(self):
        from netcdf.data.preprocessing import build_transform
        t = build_transform({"type": "power", "exponent": 0.3})
        assert t is not None

    def test_build_transform_none(self):
        from netcdf.data.preprocessing import build_transform
        assert build_transform(None) is None

    def test_pipeline_from_config(self):
        from netcdf.data.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_config({
            "so": {"type": "log", "offset": 1.0},
            "thetao": "sqrt",
        })
        assert "so" in pipeline.transforms
        assert "thetao" in pipeline.transforms

    def test_pipeline_call(self):
        from netcdf.data.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_config({
            "so": {"type": "log", "offset": 1.0},
        })
        raw = {
            "so": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "thetao": np.array([5.0, 10.0, 15.0], dtype=np.float32),
        }
        result = pipeline(raw)
        np.testing.assert_allclose(result["so"], np.log(raw["so"] + 1.0), atol=1e-5)
        np.testing.assert_array_equal(result["thetao"], raw["thetao"])

    def test_pipeline_transform_scaling(self):
        from netcdf.data.preprocessing import PreprocessingPipeline
        pipeline = PreprocessingPipeline.from_config({
            "so": {"type": "log", "offset": 1.0},
        })
        vmin, vmax = pipeline.transform_scaling("so", 0.0, 40.0)
        np.testing.assert_allclose(vmin, np.log(1.0))
        np.testing.assert_allclose(vmax, np.log(41.0))
        vmin2, vmax2 = pipeline.transform_scaling("thetao", 0.0, 40.0)
        assert vmin2 == 0.0
        assert vmax2 == 40.0

    def test_dataset_with_variable_transforms(self, tmp_path):
        nc_path = tmp_path / "preproc_test.nc"
        _make_nc(nc_path, ["thetao", "so"], shape=(2, 4, 8, 10))
        from netcdf.data.dataset import NetCDFDataset
        ds = NetCDFDataset(
            filepaths=[str(nc_path)],
            variables=["thetao", "so"],
            scaling={"thetao": [-3.0, 40.0], "so": [0.0, 40.0]},
            variable_transforms={"so": {"type": "log", "offset": 1.0}},
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 4),
        )
        sample = ds[0]
        assert sample["image"].shape[0] == 2
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0


# ---------------------------------------------------------------------------
# Multi-file time-series aggregation tests
# ---------------------------------------------------------------------------

class TestMultiFile:

    def test_scan_time_coordinates(self, tmp_path):
        from netcdf.data.multi_file import scan_time_coordinates
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=3)
        info = scan_time_coordinates([str(f1), str(f2)])
        assert len(info) == 2
        assert info[0]["n_times"] == 3
        assert info[1]["n_times"] == 3

    def test_align_time_coordinates_no_overlap(self, tmp_path):
        from netcdf.data.multi_file import scan_time_coordinates, align_time_coordinates
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=3)
        info = scan_time_coordinates([str(f1), str(f2)])
        aligned = align_time_coordinates(info, deduplicate=True)
        assert len(aligned) == 2
        total_times = sum(a["n_times"] for a in aligned)
        assert total_times == 6

    def test_align_time_coordinates_with_overlap(self, tmp_path):
        from netcdf.data.multi_file import scan_time_coordinates, align_time_coordinates
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=2)
        info = scan_time_coordinates([str(f1), str(f2)])
        aligned = align_time_coordinates(info, deduplicate=True)
        # f1: times [0,1,2], f2: times [2,3,4]
        # Dedup: unique = [0,1,2,3,4] = 5 unique times
        total_times = sum(a["n_times"] for a in aligned)
        assert total_times == 5
        assert aligned[0]["n_times"] == 3
        assert aligned[1]["n_times"] == 2

    def test_multi_file_dataset_length(self, tmp_path):
        from netcdf.data.multi_file import MultiFileNetCDFDataset
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=3)
        ds = MultiFileNetCDFDataset(
            filepaths=[str(f1), str(f2)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
            deduplicate=True,
        )
        assert len(ds) == 6

    def test_multi_file_dataset_getitem(self, tmp_path):
        from netcdf.data.multi_file import MultiFileNetCDFDataset
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=3)
        ds = MultiFileNetCDFDataset(
            filepaths=[str(f1), str(f2)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
        )
        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].ndim == 3

    def test_multi_file_deduplicate(self, tmp_path):
        from netcdf.data.multi_file import MultiFileNetCDFDataset
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=2)
        ds_dedup = MultiFileNetCDFDataset(
            filepaths=[str(f1), str(f2)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
            deduplicate=True,
        )
        ds_all = MultiFileNetCDFDataset(
            filepaths=[str(f1), str(f2)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
            deduplicate=False,
        )
        assert len(ds_dedup) < len(ds_all)

    def test_multi_file_sequence(self, tmp_path):
        from netcdf.data.multi_file import MultiFileNetCDFDataset
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(5, 5, 8, 10), time_offset=0)
        ds = MultiFileNetCDFDataset(
            filepaths=[str(f1)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
            sequence_length=3,
        )
        assert len(ds) == 3
        sample = ds[0]
        assert sample["meta"]["sequence"] is True
        assert sample["meta"]["sequence_length"] == 3
        assert sample["image"].ndim == 4

    def test_multi_file_global_time_values(self, tmp_path):
        from netcdf.data.multi_file import MultiFileNetCDFDataset
        f1 = _make_nc(tmp_path / "f1.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=0)
        f2 = _make_nc(tmp_path / "f2.nc", ["thetao"], shape=(3, 5, 8, 10), time_offset=3)
        ds = MultiFileNetCDFDataset(
            filepaths=[str(f1), str(f2)],
            variables=["thetao"],
            scaling={"thetao": (-3.0, 40.0)},
        )
        times = ds.global_time_values
        assert len(times) == 6
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


# ---------------------------------------------------------------------------
# Coordinate system tests
# ---------------------------------------------------------------------------

class TestCoordinates:

    def test_detect_regular(self, tmp_path):
        from netcdf.data.coordinates import detect_coordinate_system
        nc_path = tmp_path / "regular.nc"
        _make_nc(nc_path, ["thetao"], shape=(2, 3, 8, 10))
        cs = detect_coordinate_system(str(nc_path))
        assert cs["type"] == "regular"
        assert cs["lat_name"] is not None
        assert cs["lon_name"] is not None

    def test_rotated_to_regular(self):
        from netcdf.data.coordinates import rotated_to_regular
        lat_rot = np.array([0.0, 10.0, -10.0])
        lon_rot = np.array([0.0, 15.0, -15.0])
        lat_reg, lon_reg = rotated_to_regular(lat_rot, lon_rot, pole_lat=90.0, pole_lon=0.0)
        np.testing.assert_allclose(lat_reg, lat_rot, atol=1e-10)
        np.testing.assert_allclose(lon_reg, lon_rot, atol=1e-10)

    def test_rotated_to_regular_rotated_pole(self):
        from netcdf.data.coordinates import rotated_to_regular
        pole_lat = 39.25
        pole_lon = -162.0
        lat_rot = np.array([0.0])
        lon_rot = np.array([0.0])
        lat_reg, lon_reg = rotated_to_regular(lat_rot, lon_rot, pole_lat=pole_lat, pole_lon=pole_lon)
        assert lat_reg.shape == (1,)
        assert lon_reg.shape == (1,)

    def test_build_regridder(self, tmp_path):
        from netcdf.data.coordinates import build_regridder
        src_lat = np.linspace(-90, 90, 18)
        src_lon = np.linspace(0, 360, 36)
        target_lat = np.linspace(-90, 90, 9)
        target_lon = np.linspace(0, 360, 18)
        regridder = build_regridder(src_lat, src_lon, target_lat, target_lon, method="nearest")
        data = np.random.rand(18, 36).astype(np.float32)
        result = regridder(data)
        assert result.shape == (9, 18)

    def test_regrid_dataset(self, tmp_path):
        from netcdf.data.coordinates import regrid_dataset
        input_path = str(tmp_path / "input.nc")
        output_path = str(tmp_path / "output.nc")
        _make_nc(input_path, ["thetao"], shape=(2, 3, 8, 10))
        regrid_dataset(
            input_path, output_path,
            target_lat_size=4, target_lon_size=5,
            variables=["thetao"],
        )
        ds_out = xr.open_dataset(output_path)
        assert "latitude" in ds_out.dims
        assert "longitude" in ds_out.dims
        assert ds_out["thetao"].shape == (2, 3, 4, 5)
        ds_out.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
