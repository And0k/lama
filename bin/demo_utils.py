# ===================================================================
# Helpers
# ===================================================================

import logging
import os
import tempfile

import hydra
import numpy as np
import torch
import xarray as xr
import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

def _create_demo_nc(path: str) -> str:
    """Write a synthetic 4-D NetCDF file used when no real data is found.

    Generates realistic shallow sea oceanographic data:
    - Temperature decreases with depth (thermocline ~20m)
    - Salinity stratified
    - Velocity fields with spatial patterns
    """
    T, D, H, W = 4, 24, 96, 128
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

    # Temperature: surface warm (~20°C), decreases to ~4°C at depth (shallow sea thermocline)
    depth_temps = np.linspace(20, 4, D).astype(np.float32)
    base_temp = depth_temps[:, None, None] * np.ones((D, H, W), dtype=np.float32)
    noise_temp = rng.uniform(-2, 2, (T, D, H, W)).astype(np.float32)
    ds["thetao"] = (("time", "depth", "latitude", "longitude"), base_temp + noise_temp)
    ds["thetao"].attrs["_FillValue"] = -999.0
    ds["thetao"].attrs["units"] = "degC"

    # Salinity: surface lower (~30 PSU), increases with depth
    depth_so = np.linspace(30, 35, D).astype(np.float32)
    base_so = depth_so[:, None, None] * np.ones((D, H, W), dtype=np.float32)
    noise_so = rng.uniform(-1, 1, (T, D, H, W)).astype(np.float32)
    ds["so"] = (("time", "depth", "latitude", "longitude"), base_so + noise_so)
    ds["so"].attrs["_FillValue"] = -999.0
    ds["so"].attrs["units"] = "PSU"

    # Velocity: horizontal with weak vertical shear
    uo = rng.uniform(-0.5, 0.5, (T, D, H, W)).astype(np.float32)
    ds["uo"] = (("time", "depth", "latitude", "longitude"), uo)
    ds["uo"].attrs["_FillValue"] = -999.0
    ds["uo"].attrs["units"] = "m s-1"

    vo = rng.uniform(-0.5, 0.5, (T, D, H, W)).astype(np.float32)
    ds["vo"] = (("time", "depth", "latitude", "longitude"), vo)
    ds["vo"].attrs["_FillValue"] = -999.0
    ds["vo"].attrs["units"] = "m s-1"

    ds.to_netcdf(path, engine="netcdf4")
    return path


def _resolve_nc_file(cfg: DictConfig) -> tuple[str, bool]:
    """Return ``(path, is_synthetic)``.

    Checks for NetCDF file in this order:
    1. cfg.cems_candidates — list of CMEMS file paths
    2. cfg.dataset.filepaths — filepaths from data config
    Falls back to synthetic data if no valid file found.
    """
    # Check cems_candidates first
    cems_candidates = cfg.get("cems_candidates", None)
    if cems_candidates:
        for candidate in cems_candidates:
            if os.path.isfile(candidate):
                return candidate, False
            log.warning("cems_candidate not found: %s", candidate)

    # Check dataset.filepaths from data config
    ds_cfg = cfg.get("dataset", None)
    if ds_cfg is not None:
        filepaths = ds_cfg.get("filepaths", None)
        if filepaths:
            for fp in filepaths:
                if os.path.isfile(fp):
                    return fp, False
                log.warning("dataset filepath not found: %s", fp)

    log.info("No valid NetCDF file found — using synthetic data")
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

    from saicinpainting.training.trainers import load_checkpoint

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


# ===================================================================
# Demo — Loading and slicing
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
# Main
# ===================================================================

@hydra.main(config_path="../configs", config_name="nc/demo/default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = cfg._group_
    log.info("NetCDF Loading and slicing Demo (Hydra-driven)")

    nc_file, is_synthetic = _resolve_nc_file(cfg)
    if is_synthetic:
        log.info("No CMEMS file found — using synthetic data: %s", nc_file)
    else:
        log.info("Using CMEMS file: %s", nc_file)
    demo_loading_and_slicing(nc_file)

if __name__ == "__main__":
    main()