"""Step 2 — 4-D Slicer: extract 2-D slices from 4-D arrays."""

import numpy as np

from hydro_lama_nc.data.slicer import slice_nc, compute_sample_count


T, D, H, W = 3, 5, 16, 20


def _make_4d():
    return np.arange(T * D * H * W, dtype=float).reshape(T, D, H, W)


class TestSliceNC:

    def test_lat_lon_shape(self):
        out = slice_nc(_make_4d(), ("lat", "lon"), (0, 0))
        assert out.shape == (H, W)

    def test_depth_lon_shape(self):
        out = slice_nc(_make_4d(), ("depth", "lon"), (0, 0))
        assert out.shape == (D, W)

    def test_depth_lat_shape(self):
        out = slice_nc(_make_4d(), ("depth", "lat"), (0, 0))
        assert out.shape == (D, H)

    def test_ordering_lon_lat(self):
        out = slice_nc(_make_4d(), ("lon", "lat"), (0, 0))
        assert out.shape == (W, H)

    def test_values_correct_t0_d0(self):
        arr = _make_4d()
        out = slice_nc(arr, ("lat", "lon"), (0, 0))
        np.testing.assert_array_equal(out, arr[0, 0])

    def test_values_correct_t1_d2(self):
        arr = _make_4d()
        out = slice_nc(arr, ("lat", "lon"), (1, 2))
        np.testing.assert_array_equal(out, arr[1, 2])


class TestComputeSampleCount:

    def test_none_slice_mode(self):
        # slice_mode=None defaults to ["time", "depth"] → T * D samples
        assert compute_sample_count((T, D, H, W), ["time", "depth", "latitude", "longitude"], None) == T * D

    def test_time_depth(self):
        n = compute_sample_count(
            (T, D, H, W),
            ["time", "depth", "latitude", "longitude"],
            ["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 4),
        )
        assert n == 2 * 4

    def test_time_latitude(self):
        n = compute_sample_count(
            (T, D, H, W),
            ["time", "depth", "latitude", "longitude"],
            ["time", "latitude"],
            time_indices=slice(0, 2),
            lat_indices=slice(0, H),
        )
        assert n == 2 * H

    def test_time_longitude(self):
        n = compute_sample_count(
            (T, D, H, W),
            ["time", "depth", "latitude", "longitude"],
            ["time", "longitude"],
            time_indices=1,
            lon_indices=slice(0, W),
        )
        assert n == 1 * W
