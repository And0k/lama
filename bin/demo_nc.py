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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from netcdf.data.dataset import NetCDFDataset
from netcdf.visualization import plot_comparison
from saicinpainting.evaluation.losses.ssim import SSIM
from saicinpainting.training.trainers import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ===================================================================
# Helpers
# ===================================================================

def _create_demo_nc(path: str) -> str:
    """Write a synthetic 4-D NetCDF file used when no real data is found."""
    T, D, H, W = 4, 6, 32, 40
    rng = np.random.RandomState(0)
    ds = xr.Dataset(
        {},
        coords={
            "time": np.arange(T),
            "depth": np.arange(D),
            "latitude": np.arange(H),
            "longitude": np.arange(W),
        },
    )
    thetao = rng.uniform(-2, 35, (T, D, H, W)).astype(np.float32)
    ds["thetao"] = (("time", "depth", "latitude", "longitude"), thetao)
    ds["thetao"].attrs["_FillValue"] = -999.0
    ds["thetao"].attrs["units"] = "degC"

    so = rng.uniform(2, 40, (T, D, H, W)).astype(np.float32)
    ds["so"] = (("time", "depth", "latitude", "longitude"), so)
    ds["so"].attrs["_FillValue"] = -999.0
    ds["so"].attrs["units"] = "PSU"

    uo = rng.uniform(-1.5, 1.5, (T, D, H, W)).astype(np.float32)
    ds["uo"] = (("time", "depth", "latitude", "longitude"), uo)
    ds["uo"].attrs["_FillValue"] = -999.0
    ds["uo"].attrs["units"] = "m s-1"

    vo = rng.uniform(-1.5, 1.5, (T, D, H, W)).astype(np.float32)
    ds["vo"] = (("time", "depth", "latitude", "longitude"), vo)
    ds["vo"].attrs["_FillValue"] = -999.0
    ds["vo"].attrs["units"] = "m s-1"

    ds.to_netcdf(path, engine="netcdf4")
    return path


def _resolve_nc_file(cfg: DictConfig) -> tuple[str, bool]:
    """Return ``(path, is_synthetic)``."""
    for candidate in cfg.cems_candidates:
        if os.path.isfile(candidate):
            return candidate, False
    nc_dir = tempfile.mkdtemp(prefix="netcdf_demo_")
    nc_file = os.path.join(nc_dir, "demo.nc")
    _create_demo_nc(nc_file)
    return nc_file, True


def _resolve_model(cfg: DictConfig) -> tuple[str, str]:
    """Return ``(label, model_dir)`` for the first available model."""
    for entry in cfg.model_candidates:
        label, model_dir = entry.label, entry.path
        if os.path.isfile(os.path.join(model_dir, "config.yaml")) and os.path.isfile(
            os.path.join(model_dir, "models", "best.ckpt")
        ):
            return label, model_dir
    searched = "\n".join(f"  {e.path}" for e in cfg.model_candidates)
    raise FileNotFoundError(f"No pretrained model found.  Searched:\n{searched}")


def _load_lama_model(model_dir: str, device: torch.device) -> torch.nn.Module:
    """Load and freeze a pretrained LaMa model from *model_dir*."""
    config_path = os.path.join(model_dir, "config.yaml")
    ckpt_path = os.path.join(model_dir, "models", "best.ckpt")
    with open(config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"
    log.info("Training config:\n%s", OmegaConf.to_yaml(train_config))
    model = load_checkpoint(train_config, ckpt_path, strict=False, map_location=device)
    model.freeze()
    model.to(device)
    model.eval()
    return model


def _next_modulo(x: int, mod: int) -> int:
    return ((x + mod - 1) // mod) * mod


def _pad_to_modulo(tensor: torch.Tensor, mod: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims to multiples of *mod*; return ``(padded, (orig_H, orig_W))``."""
    _, _, H, W = tensor.shape
    pad_H = _next_modulo(H, mod) - H
    pad_W = _next_modulo(W, mod) - W
    if pad_H == 0 and pad_W == 0:
        return tensor, (H, W)
    return torch.nn.functional.pad(tensor, (0, pad_W, 0, pad_H), mode="reflect"), (H, W)


def _compute_ssim(orig: np.ndarray, inpainted: np.ndarray) -> float:
    """Compute SSIM between two single-channel ``(H, W)`` numpy arrays in [0, 1]."""
    ssim_fn = SSIM(window_size=11, size_average=True).eval()
    t1 = torch.from_numpy(orig).float().unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(inpainted).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return ssim_fn(t1, t2).item()


def _resolve_vel_scaling(cfg: DictConfig) -> tuple[float, float]:
    """Compute |V| scaling from uo/vo ranges if not explicitly configured."""
    scaling = cfg.scaling
    vel_cfg = scaling.get("|V|")
    if vel_cfg is None or vel_cfg[0] is None:
        uo_min, uo_max = scaling.uo
        vo_min, vo_max = scaling.vo
        vmax = np.sqrt(max(abs(uo_min), uo_max)**2 + max(abs(vo_min), vo_max)**2)
        return (0.0, float(vmax))
    return tuple(vel_cfg)


def _resolve_output_channels(nc_file: str, cfg: DictConfig) -> list[dict]:
    """Return output channel descriptors, falling back if required vars missing."""
    with xr.open_dataset(nc_file) as ds:
        available = set(ds.data_vars.keys())

    required = {"uo", "vo", "thetao", "so"}
    if required.issubset(available):
        return [{"name": name, "scaling_key": name} for name in cfg.output_channels]

    log.warning("Not all required variables (%s) found in %s.  Available: %s",
                required, nc_file, available)
    fallback = [v for v in cfg.output_channels if v in available and v != "|V|"]
    if not fallback:
        fallback = list(available)[:3]
    return [{"name": name, "scaling_key": name} for name in fallback[:3]]


def _read_units(nc_file: str, var_names: list[str]) -> dict[str, str]:
    """Read units for each variable from the NetCDF file."""
    units = {}
    with xr.open_dataset(nc_file) as ds:
        for v in var_names:
            if v == "|V|":
                inp = ds["uo"].attrs.get("units", "m s-1") if "uo" in ds.data_vars else "m s-1"
                units[v] = inp
            elif v in ds.data_vars:
                units[v] = ds[v].attrs.get("units", "")
            else:
                units[v] = ""
    return units


# ===================================================================
# Demo 1 — Loading and slicing
# ===================================================================

def demo_loading_and_slicing(nc_file: str) -> None:
    """Inspect a NetCDF file and extract 2-D slices."""
    log.info("=== DEMO 1: Loading and Slicing ===")

    from netcdf.data.loader import get_available_variables, load_netcdf
    from netcdf.data.slicer import slice_nc

    variables = get_available_variables(nc_file)
    log.info("Available variables: %s", variables)

    data = load_netcdf(
        nc_file,
        variables=["thetao", "so", "uo", "vo"],
        time_indices=slice(0, 2),
        depth_indices=slice(0, 4),
        replace_fill_value=True,
    )
    log.info("thetao shape: %s", data["thetao"].shape)
    log.info("so shape: %s", data["so"].shape)
    log.info("uo shape: %s", data["uo"].shape)

    arr = data["thetao"]
    vert = slice_nc(arr, ("depth", "lat"), (0, 0))
    log.info("slice_nc(axes=('depth','lat'), idx=(0,0)) -> shape %s", vert.shape)
    horiz = slice_nc(arr, ("lat", "lon"), (0, 2))
    log.info("slice_nc(axes=('lat','lon'), idx=(0,2))   -> shape %s", horiz.shape)

    # Demonstrate velocity magnitude computation
    u = data["uo"]
    v = data["vo"]
    vel_mag = np.sqrt(u**2 + v**2)
    log.info("|V| = sqrt(uo^2 + vo^2) -> shape %s", vel_mag.shape)

    # Demonstrate slicing on computed velocity
    vel_slice = slice_nc(vel_mag, ("lat", "lon"), (0, 0))
    log.info("|V| slice_nc(('lat','lon'), idx=(0,0)) -> shape %s", vel_slice.shape)


# ===================================================================
# Demo 2 — Inference with all projections + SSIM evaluation
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

    import yaml

    scaling_dict = OmegaConf.to_object(cfg.scaling)
    mask_gen_cfg = OmegaConf.to_object(cfg.mask_generator)
    projections = OmegaConf.to_object(cfg.projections)
    computed_vars = OmegaConf.to_object(cfg.computed_variables)

    output_channels = _resolve_output_channels(nc_file, cfg)
    log.info("Output channels: %s", [ch["name"] for ch in output_channels])

    var_names = [ch["name"] for ch in output_channels]
    units = _read_units(nc_file, var_names)

    vel_scaling = _resolve_vel_scaling(cfg)
    scaling_dict["|V|"] = vel_scaling

    out_dir = Path(tempfile.mkdtemp(prefix="netcdf_inference_"))
    log.info("Output directory: %s", out_dir)

    for proj_name, proj_cfg in projections.items():
        slice_mode = proj_cfg["slice_mode"]
        dim_y, dim_x = proj_cfg["dim_labels"]

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

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        ssim_values = []

        for i, batch in enumerate(dataloader):
            image = batch["image"]
            mask = batch["mask"]
            fmask = batch["fill_mask"]

            img_pad, (orig_H, orig_W) = _pad_to_modulo(image, 8)
            msk_pad, _ = _pad_to_modulo((mask > 0).float(), 8)

            with torch.no_grad():
                result = model({"image": img_pad.to(device), "mask": msk_pad.to(device)})

            out = result["inpainted"][:, :, :orig_H, :orig_W].cpu()

            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            ssim_row = []

            for ch_idx, ch_info in enumerate(output_channels):
                vmin, vmax = scaling_dict[ch_info["scaling_key"]]
                var_name = ch_info["name"]

                orig_phys = image[0, ch_idx].numpy() * (vmax - vmin) + vmin
                out_phys = out[0, ch_idx].numpy() * (vmax - vmin) + vmin

                fill_mask_np = fmask[0, ch_idx].bool().numpy()
                orig_phys[fill_mask_np] = np.nan
                out_phys[fill_mask_np] = np.nan

                mask_np = mask[0, 0].numpy()
                masked_phys = orig_phys.copy()
                masked_phys[mask_np > 0] = np.nan

                if ch_idx == 0:
                    ssim_val = _compute_ssim(
                        np.nan_to_num(orig_phys.copy(), nan=0.0),
                        np.nan_to_num(out_phys.copy(), nan=0.0),
                    )
                    ssim_values.append(ssim_val)
                    ssim_row.append(ssim_val)

                cmap = plt.cm.viridis.copy()
                cmap.set_bad(color='white')

                for col, (data_2d, title) in enumerate([
                    (orig_phys, "Original"),
                    (masked_phys, "Masked"),
                    (out_phys, "Inpainted"),
                ]):
                    masked_data = np.ma.masked_invalid(data_2d)
                    im = axes[ch_idx, col].imshow(masked_data, cmap=cmap, aspect='auto')
                    axes[ch_idx, col].set_xlabel(dim_x)
                    axes[ch_idx, col].set_ylabel(dim_y)
                    col_title = f"{var_name} — {title}"
                    if ch_idx == 0 and col == 2:
                        col_title += f"  (SSIM={ssim_val:.4f})"
                    axes[ch_idx, col].set_title(col_title)

                unit_label = units.get(var_name, "")
                unit_suffix = f" [{unit_label}]" if unit_label else ""
                fig.colorbar(im, ax=axes[ch_idx, :].tolist(),
                             label=f"{var_name}{unit_suffix}", shrink=0.85)

            fig.suptitle(
                f"projection={proj_name}  sample={i}  "
                f"{dim_y}×{dim_x}=({orig_H},{orig_W})",
                fontsize=10,
            )
            plt.tight_layout()
            fig_path = out_dir / f"{proj_name}_3ch_t0_{dim_y}{orig_H}_{dim_x}{orig_W}_sample{i}.png"
            fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
            plt.close(fig)

            ssim_str = f" SSIM={ssim_val:.4f}" if ssim_row else ""
            log.info("  sample %d:%s saved %s", i, ssim_str, fig_path.name)

        if ssim_values:
            log.info(
                "Projection '%s': mean SSIM=%.4f over %d samples",
                proj_name, np.mean(ssim_values), len(ssim_values),
            )

    log.info("All projections complete.  Results in %s", out_dir)


# ===================================================================
# Main
# ===================================================================

@hydra.main(config_path="../configs", config_name="nc/demo/default", version_base=None)
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