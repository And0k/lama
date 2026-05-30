"""TDD tests for Hydro T+S visualization and metric functions."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import os
import tempfile

from netcdf.evaluation import compute_metrics, optimal_interpolation


# ── compute_metrics ──────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_identical_fields_zero_rmse(self):
        f = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        below = np.zeros((8, 8), dtype=bool)
        m = compute_metrics(f, f, below)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-7)
        assert m["mse"] == pytest.approx(0.0, abs=1e-7)

    def test_all_below_returns_nan(self):
        f = np.ones((4, 4), dtype=np.float32)
        below = np.ones((4, 4), dtype=bool)
        m = compute_metrics(f, f * 2, below)
        assert np.isnan(m["rmse"])

    def test_psnr_high_for_small_error(self):
        rng = np.random.default_rng(1)
        t = rng.random((16, 16)).astype(np.float32)
        p = t + rng.normal(0, 0.001, t.shape).astype(np.float32)
        below = np.zeros_like(t, dtype=bool)
        m = compute_metrics(p, t, below)
        assert m["psnr"] > 40.0


# ── optimal_interpolation ────────────────────────────────────────────────


class TestOptimalInterpolation:
    def test_returns_correct_shape(self):
        inp = np.random.default_rng(0).random((2, 8, 8)).astype(np.float32)
        mask = np.zeros((1, 8, 8), dtype=np.float32)
        mask[0, :4, :4] = 1.0
        T_oi, S_oi = optimal_interpolation(inp, mask, 8, 8)
        assert T_oi.shape == (8, 8)
        assert S_oi.shape == (8, 8)

    def test_few_points_returns_zeros(self):
        inp = np.random.default_rng(0).random((2, 4, 4)).astype(np.float32)
        mask = np.zeros((1, 4, 4), dtype=np.float32)
        mask[0, 0, 0] = 1.0
        mask[0, 1, 1] = 1.0
        T_oi, S_oi = optimal_interpolation(inp, mask, 4, 4)
        assert T_oi.sum() == 0.0

    def test_perfect_when_all_observed(self):
        inp = np.random.default_rng(2).random((2, 6, 6)).astype(np.float32)
        mask = np.ones((1, 6, 6), dtype=np.float32)
        T_oi, S_oi = optimal_interpolation(inp, mask, 6, 6)
        assert np.allclose(T_oi, inp[0], atol=0.1)
        assert np.allclose(S_oi, inp[1], atol=0.1)


# ── plot_input_channels ─────────────────────────────────────────────────


class TestPlotInputChannels:
    def test_creates_png(self):
        from netcdf.visualization import plot_input_channels

        rng = np.random.default_rng(42)
        u = rng.random((16, 16)).astype(np.float32)
        T = rng.random((16, 16)).astype(np.float32)
        v = rng.random((16, 16)).astype(np.float32)
        S = rng.random((16, 16)).astype(np.float32)
        bathy = rng.random(16).astype(np.float32)
        mask_ctd = np.zeros((16, 16), dtype=np.float32)
        mask_ctd[:, 3] = 1.0
        mask_cmems = np.zeros((16, 16), dtype=np.float32)
        mask_cmems[::4, ::4] = 1.0

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            plot_input_channels(
                T, S, u, v,
                bathy=bathy, mask_ctd=mask_ctd, mask_cmems=mask_cmems,
                save_path=path,
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ── plot_two_field_rows ──────────────────────────────────────────────────


class TestPlotTwoFieldRows:
    def test_creates_png(self):
        from netcdf.visualization import plot_two_field_rows

        rng = np.random.default_rng(42)
        top = rng.random((16, 16)).astype(np.float32)
        bottom = rng.random((16, 16)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            plot_two_field_rows(
                top, bottom,
                title_top="T field", title_bottom="S field",
                xlabel="x", ylabel="z",
                save_path=path,
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_with_explicit_limits(self):
        from netcdf.visualization import plot_two_field_rows

        rng = np.random.default_rng(42)
        top = rng.random((16, 16)).astype(np.float32)
        bottom = rng.random((16, 16)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            plot_two_field_rows(
                top, bottom,
                title_top="T", title_bottom="S",
                vmin_top=0.0, vmax_top=1.0,
                vmin_bottom=0.0, vmax_bottom=1.0,
                cmap="plasma",
                save_path=path,
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ── plot_fields_result (wrapper 2.2) ────────────────────────────────────


class TestPlotFieldsResult:
    def test_oi_and_lama_calls(self):
        from netcdf.visualization import plot_fields_result

        rng = np.random.default_rng(42)
        T_true = rng.random((16, 16)).astype(np.float32)
        S_true = rng.random((16, 16)).astype(np.float32)
        T_pred = T_true + rng.normal(0, 0.05, (16, 16)).astype(np.float32)
        S_pred = S_true + rng.normal(0, 0.05, (16, 16)).astype(np.float32)
        below = np.zeros((16, 16), dtype=bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_fields_result(T_pred, S_pred, method="lama",
                               below=below, output_dir=tmpdir, suffix="s1")
            assert os.path.exists(os.path.join(tmpdir, "lama_T_S_s1.png"))

            plot_fields_result(T_true + 0.01, S_true + 0.01, method="oi",
                               below=below, output_dir=tmpdir, suffix="s1")
            assert os.path.exists(os.path.join(tmpdir, "oi_T_S_s1.png"))


# ── plot_fields_error (wrapper 2.1) ─────────────────────────────────────


class TestPlotFieldsError:
    def test_creates_png(self):
        from netcdf.visualization import plot_fields_error

        rng = np.random.default_rng(42)
        T_err = rng.normal(0, 0.1, (16, 16)).astype(np.float32)
        S_err = rng.normal(0, 0.1, (16, 16)).astype(np.float32)
        below = np.zeros((16, 16), dtype=bool)

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_fields_error(T_err, S_err, method="lama",
                              below=below, output_dir=tmpdir, suffix="s1")
            assert os.path.exists(os.path.join(tmpdir, "lama_error_s1.png"))

            plot_fields_error(T_err * 0.5, S_err * 0.5, method="oi",
                              below=below, output_dir=tmpdir, suffix="s1")
            assert os.path.exists(os.path.join(tmpdir, "oi_error_s1.png"))


# ── plot_netcdf_inference (updated) ─────────────────────────────────────


class TestPlotNetcdfInferenceUpdated:
    def test_has_error_column(self):
        from netcdf.visualization import plot_netcdf_inference

        rng = np.random.default_rng(42)
        H, W = 16, 16
        channels = [rng.random((H, W)).astype(np.float32) for _ in range(2)]
        inpainted = [c + rng.normal(0, 0.05, (H, W)).astype(np.float32)
                     for c in channels]
        fill_masks = [np.zeros((H, W), dtype=bool) for _ in range(2)]
        generated_mask = np.zeros((1, H, W), dtype=np.float32)
        generated_mask[0, 4:8, 4:8] = 1.0

        fig = plot_netcdf_inference(
            channels, inpainted, fill_masks, generated_mask,
            var_names=["T", "S"],
        )
        # Should have 2 rows x 3 cols (Original, Masked, Error)
        assert fig.axes is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
