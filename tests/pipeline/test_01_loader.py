"""Step 1 — NetCDF Loader: read raw arrays from a .nc file."""

import numpy as np
import pytest

from netcdf.data.loader import load_netcdf, get_available_variables, get_variable_info


class TestLoadNetCDF:
    """Validate load_netcdf, get_available_variables, get_variable_info."""

    def test_load_variables_present(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao", "so"])
        assert set(data.keys()) == {"thetao", "so"}

    def test_load_shape(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao"])
        assert data["thetao"].shape == (3, 5, 16, 20)

    def test_load_dtype_is_float32(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao"])
        assert data["thetao"].dtype == np.float32

    def test_fill_value_replaced_with_nan(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao"], replace_fill_value=True)
        assert np.issubdtype(data["thetao"].dtype, np.floating)

    def test_load_with_time_slice(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao"], time_indices=slice(0, 2))
        assert data["thetao"].shape[0] == 2

    def test_load_with_depth_index(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=["thetao"], depth_indices=0)
        # scalar index squeezes the depth dim: shape becomes (T, H, W)
        assert data["thetao"].ndim == 3

    def test_load_all_variables_when_none(self, dummy_nc):
        data = load_netcdf(str(dummy_nc), variables=None)
        assert "thetao" in data and "so" in data

    def test_missing_variable_raises(self, dummy_nc):
        with pytest.raises(ValueError, match="not found"):
            load_netcdf(str(dummy_nc), variables=["nonexistent"])

    def test_get_available_variables(self, dummy_nc):
        vars_list = get_available_variables(str(dummy_nc))
        assert "thetao" in vars_list and "so" in vars_list

    def test_get_variable_info(self, dummy_nc):
        info = get_variable_info(str(dummy_nc), "thetao")
        assert "shape" in info
        assert "dims" in info
        assert info["shape"] == (3, 5, 16, 20)
