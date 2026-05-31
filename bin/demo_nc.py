#!/usr/bin/env python3
"""
NetCDF → LaMa inpainting demonstration (Hydra-driven).

Uses ``@hydra.main`` with ``configs/nc/demo/default.yaml`` (via
``config_path="../configs"``, ``config_name="nc/demo/default"``) to
configure model paths, scaling, variables, projections, and mask generation.

Model choice rationale
----------------------
``big-lama`` is the most general-purpose pretrained checkpoint trained on
Places365 with FFC-ResNet.  See the original ``demo_nc.py`` docstring for
details.

Limitations
-----------
- The model was trained on natural images (RGB), not oceanographic data.
- Spatial dimensions must be divisible by ``2**n_downsampling`` (8 for
  big-lama).  The script pads reflectively and crops back.
- Only irregular masks are generated (no segmentation-aware masks).
- ``num_workers > 0`` may fail with netCDF3 files.

Usage
-----
    python bin/demo_nc.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

from demo_utils import _load_lama_model, _resolve_model, _resolve_nc_file, demo_loading_and_slicing

from netcdf.data.dataset import NetCDFDataset
from netcdf.data.loader import read_units, resolve_output_channels
from netcdf.evaluation import compute_ssim, resolve_vel_scaling
from netcdf.utils import pad_to_modulo
from netcdf.visualization import plot_netcdf_inference

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# Demo — Inference with all projections + SSIM evaluation
# ===================================================================

def demo_inference(nc_file: str, cfg: DictConfig) -> None:
    """Run pretrained LaMa on every projection and evaluate with SSIM."""
    log.info("=== DEMO 2: Inference + SSIM ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    label, model_dir = _resolve_model(cfg)
    log.info("Model '%s' from %s", label, model_dir)
    model = _load_lama_model(model_dir, device)
    log.info("Model loaded.")

    scaling_dict = OmegaConf.to_object(cfg.scaling)
    projections = OmegaConf.to_object(cfg.projections)
    computed_vars = OmegaConf.to_object(cfg.computed_variables)
    horizontal_mask_cfg = OmegaConf.to_object(cfg.mask_generator)
    vertical_mask_cfg = OmegaConf.to_object(cfg.vertical_mask)
    clim = OmegaConf.to_object(cfg.colorbar) if "colorbar" in cfg else None

    output_channels = resolve_output_channels(nc_file, cfg.output_channels)
    log.info("Output channels: %s", [ch["name"] for ch in output_channels])

    var_names = [ch["name"] for ch in output_channels]
    units = read_units(nc_file, var_names)

    vel_scaling = resolve_vel_scaling(OmegaConf.to_object(cfg.scaling))
    scaling_dict["|V|"] = vel_scaling

    out_dir = Path(tempfile.mkdtemp(prefix="netcdf_inference_"))
    log.info("Output directory: %s", out_dir)

    for proj_name, proj_cfg in projections.items():
        slice_mode = proj_cfg["slice_mode"]
        dim_y, dim_x = proj_cfg["dim_labels"]

        mask_type = proj_cfg.get("mask_type", "horizontal")
        if mask_type == "vertical":
            mask_gen_cfg = {"type": "vertical", **vertical_mask_cfg}
        else:
            mask_gen_cfg = horizontal_mask_cfg

        dataset = NetCDFDataset(
            filepaths=[nc_file],
            variables=var_names,
            computed_variables=computed_vars,
            scaling=scaling_dict,
            slice_mode=slice_mode,
            time_indices=OmegaConf.to_object(cfg.time_indices),
            depth_indices=OmegaConf.to_object(cfg.depth_indices),
            mask_generator=mask_gen_cfg,
            **proj_cfg["extra_kwargs"],
        )
        n_samples = len(dataset)
        C, H, W = dataset.get_shape()
        log.info(
            "Projection '%s': slice_mode=%s, %d samples, "
            "output shape (C=%d, %s=%d, %s=%d)",
            proj_name, slice_mode, n_samples, C, dim_y, H, dim_x, W,
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        ssim_values = []

        for i, batch in enumerate(dataloader):
            image = batch["image"]
            mask = batch["mask"]
            fmask = batch["fill_mask"]

            img_pad, (orig_H, orig_W) = pad_to_modulo(image, 8)
            msk_pad, _ = pad_to_modulo((mask > 0).float(), 8)

            with torch.no_grad():
                result = model({"image": img_pad.to(device), "mask": msk_pad.to(device)})

            out = result["inpainted"][:, :, :orig_H, :orig_W].cpu()

            channels_phys = []
            inpainted_phys = []
            fill_masks_np = []

            for ch_idx, ch_info in enumerate(output_channels):
                vmin, vmax = scaling_dict[ch_info["scaling_key"]]

                orig_phys = image[0, ch_idx].numpy() * (vmax - vmin) + vmin
                out_phys = out[0, ch_idx].numpy() * (vmax - vmin) + vmin
                fill_mask_np = fmask[0, ch_idx].bool().numpy()

                channels_phys.append(orig_phys)
                inpainted_phys.append(out_phys)
                fill_masks_np.append(fill_mask_np)

            ssim_val = compute_ssim(
                np.nan_to_num(channels_phys[0].copy(), nan=0.0),
                np.nan_to_num(inpainted_phys[0].copy(), nan=0.0),
            )
            ssim_values.append(ssim_val)

            fig = plot_netcdf_inference(
                channels=channels_phys,
                inpainted=inpainted_phys,
                fill_masks=fill_masks_np,
                generated_mask=mask[0].numpy(),
                var_names=[ch["name"] for ch in output_channels],
                dim_x=dim_x,
                dim_y=dim_y,
                orig_shape=(orig_H, orig_W),
                ssim_val=ssim_val,
                units=units,
                suptitle=f"projection={proj_name}  sample={i}",
                clim=clim,
            )

            fig_path = out_dir / f"{proj_name}_3ch_t0_{dim_y}{orig_H}_{dim_x}{orig_W}_sample{i}.png"
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            log.info("  sample %d: SSIM=%.4f saved %s", i, ssim_val, fig_path.name)

        if ssim_values:
            log.info(
                "Projection '%s': mean SSIM=%.4f over %d samples",
                proj_name, np.mean(ssim_values), len(ssim_values),
            )

    log.info("All projections complete.  Results in %s", out_dir)

# ===================================================================
# Main
# ===================================================================

@hydra.main(config_path="../configs", config_name="demo_nc", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg._group_
    log.info("NetCDF Integration Demo for LaMa Inpainting (Hydra-driven)")

    nc_file, is_synthetic = _resolve_nc_file(cfg)
    if is_synthetic:
        log.info("No CMEMS file found — using synthetic data: %s", nc_file)
    else:
        log.info("Using CMEMS file: %s", nc_file)

    try:
        demo_loading_and_slicing(nc_file)
        demo_inference(nc_file, cfg)
        log.info("All demos completed successfully.")
    except Exception:
        log.exception("Demo failed")
        sys.exit(1)


if __name__ == "__main__":
    main()