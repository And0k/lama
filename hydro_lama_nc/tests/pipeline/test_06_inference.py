"""Step 6 — End-to-end inference: NetCDF → Dataset → DataLoader → Model → output."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hydro_lama_nc.data.dataset import NetCDFDataset
from models.factory import build_model
from hydro_lama_nc.visualization import plot_comparison


class TestEndToEndInference:

    def test_single_variable_pipeline(self, dummy_nc, scaling):
        """Full pipeline: load NC → Dataset → DataLoader → Model → verify output."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 3),
        )
        assert len(dataset) == 6  # 2 time × 3 depth

        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        images = batch["image"]
        masks = batch["mask"]
        assert images.shape == (2, 1, 16, 20)
        assert masks.shape == (2, 1, 16, 20)

        model = build_model(in_channels=1, num_classes=1)
        model.eval()
        with torch.no_grad():
            inpainted = model(images)
        assert inpainted.shape == images.shape

    def test_multi_variable_pipeline(self, dummy_nc, scaling):
        """Pipeline with 3-channel input (thetao + so + third channel)."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao", "so"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 2),
            depth_indices=slice(0, 3),
        )
        loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        images = batch["image"]
        C = images.shape[1]
        model = build_model(in_channels=C, num_classes=C)
        model.eval()
        with torch.no_grad():
            inpainted = model(images)
        assert inpainted.shape == images.shape

    def test_inference_output_range(self, dummy_nc, scaling):
        """Inference output should be finite (may not be in [0,1] without clamp)."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=0,
            depth_indices=0,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        model = build_model(in_channels=1, num_classes=1)
        model.eval()
        with torch.no_grad():
            inpainted = model(batch["image"])
        assert torch.isfinite(inpainted).all()

    def test_mask_applied(self, dummy_nc, scaling):
        """Masked regions in input should differ from original; model produces output."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=0,
            depth_indices=slice(0, 3),
            mask_generator={
                "_target_": "saicinpainting.training.data.masks.MixedMaskGenerator",
                "irregular_proba": 1.0,
                "box_proba": 0.0,
                "segm_proba": 0.0,
                "squares_proba": 0.0,
                "superres_proba": 0.0,
                "outpainting_proba": 0.0,
            },
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        images = batch["image"]
        masks = batch["mask"]

        # At least some pixels should be masked
        assert masks.sum() > 0

        model = build_model(in_channels=1, num_classes=1)
        model.eval()
        with torch.no_grad():
            inpainted = model(images)
        assert inpainted.shape == images.shape

    def test_visualization_works(self, dummy_nc, scaling):
        """Inference result can be passed to plot_comparison without error."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=0,
            depth_indices=0,
        )
        sample = dataset[0]
        image = sample["image"][0].numpy()  # (H, W)
        mask = sample["mask"][0].numpy()    # (1, H, W) → (H, W)
        if mask.ndim == 3:
            mask = mask[0]

        masked = image.copy()
        masked[mask > 0] = 0.0

        fake_result = np.clip(image + np.random.randn(*image.shape) * 0.1, 0, 1)

        fig = plot_comparison(
            original=image,
            masked=masked,
            result=fake_result,
            titles=["Original", "Masked", "Inpainted"],
        )
        assert fig is not None

    def test_full_pipeline_all_samples(self, dummy_nc, scaling):
        """Iterate through ALL samples from the dataset through the model."""
        dataset = NetCDFDataset(
            filepaths=[str(dummy_nc)],
            variables=["thetao"],
            scaling=scaling,
            slice_mode=["time", "depth"],
            time_indices=slice(0, 3),
            depth_indices=slice(0, 5),
        )
        total = len(dataset)
        assert total == 15  # 3 time × 5 depth

        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        model = build_model(in_channels=1, num_classes=1)
        model.eval()

        all_outputs = []
        with torch.no_grad():
            for batch in loader:
                out = model(batch["image"])
                all_outputs.append(out)

        combined = torch.cat(all_outputs, dim=0)
        assert combined.shape[0] == total
        assert torch.isfinite(combined).all()
