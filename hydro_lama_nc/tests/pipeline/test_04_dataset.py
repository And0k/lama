"""Step 4 — NetCDFDataset: PyTorch Dataset and DataLoader integration."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from hydro_lama_nc.data.dataset import NetCDFDataset


class TestNetCDFDatasetBasic:

    def test_length_no_slice_mode(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
        )
        assert len(ds) == 3  # one sample per time step

    def test_getitem_shapes(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
        )
        sample = ds[0]
        assert sample["image"].shape == (1, 16, 20)
        assert sample["mask"].shape == (1, 16, 20)

    def test_scaling_range(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
        )
        sample = ds[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_multi_variable_channels(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao", "so"],
            scaling=scaling,
        )
        sample = ds[0]
        assert sample["image"].shape[0] == 2

    def test_get_shape(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
        )
        C, H, W = ds.get_shape()
        assert C == 1
        assert H == 16
        assert W == 20

    def test_meta_has_required_keys(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
        )
        meta = ds[0]["meta"]
        for key in ("filepath", "time_index", "depth_index", "variables"):
            assert key in meta


class TestNetCDFDatasetSliceModes:

    def test_time_depth(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 4),
        )
        assert len(ds) == 2 * 4
        sample = ds[0]
        assert sample["image"].shape == (1, 16, 20)

    def test_time_latitude(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "latitude"],
            time_indices=slice(0, 2),
            lat_indices=slice(0, 16),
        )
        assert len(ds) == 2 * 16
        sample = ds[0]
        C, H, W = sample["image"].shape
        assert C == 1
        assert H == 5   # depth
        assert W == 20  # lon

    def test_time_longitude(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "longitude"],
            time_indices=slice(0, 2),
            lon_indices=slice(0, 20),
        )
        assert len(ds) == 2 * 20
        sample = ds[0]
        C, H, W = sample["image"].shape
        assert C == 1
        assert H == 5   # depth
        assert W == 16  # lat


class TestNetCDFDataset3DVariable:

    def test_3d_variable(self, dummy_nc_3d):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc_3d)],
            variables=["bottomT"],
            scaling={"bottomT": (-3.0, 40.0)},
        )
        sample = ds[0]
        assert sample["image"].ndim == 3
        assert sample["image"].shape == (1, 16, 20)


class TestNetCDFDatasetDataLoader:

    def test_dataloader_batch(self, dummy_nc, scaling):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao", "so"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 3),
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].ndim == 4
        assert batch["image"].shape[1] == 2  # 2 channels
        assert batch["image"].shape[2] == 16  # H
        assert batch["image"].shape[3] == 20  # W
        assert batch["mask"].ndim == 4
        assert batch["mask"].shape[1] == 1

    def test_dataloader_with_mask_generator(self, dummy_nc, scaling, mask_params):
        ds = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 3),
            mask_generator=mask_params,
        )
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape[0] == 2
        assert batch["mask"].sum() > 0, "mask should have masked pixels"
