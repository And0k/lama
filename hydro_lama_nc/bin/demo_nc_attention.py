#!/usr/bin/env python3
"""
Hydro T+S inference demo — loads best checkpoint from date-based output dir.

Usage:
    python bin/demo_nc_attention.py
    python bin/demo_nc_attention.py outdir=outputs/hydro_2026-05-30

Saves per-sample PNGs using the same naming convention as training:
    {method}_T_S_{suffix}.png
    {method}_error_{suffix}.png
    input_{suffix}.png
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydro_lama_nc.resolvers  # noqa: F401

log = logging.getLogger(__name__)


def _find_best_checkpoint(ckpt_dir: Path):
    """Return path to best checkpoint or None."""
    candidates = sorted(ckpt_dir.glob("hydro-*val_loss=*.ckpt"))
    if not candidates:
        return None
    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[1])
        except (IndexError, ValueError):
            return float("inf")
    return min(candidates, key=_loss)


def _load_hydro_model(ckpt_path: Path, device: torch.device):
    """Load HydroLightningModule from checkpoint."""
    from notebooks.train_hydro_colab import HydroLightningModule
    model = HydroLightningModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(device)
    return model


def _generate_sample(cfg: DictConfig, device: torch.device):
    """Generate one synthetic sample and run inference."""
    from hydro_lama_nc.data.synthetic_dataset import HydroSyntheticDataset
    from hydro_lama_nc.evaluation import compute_metrics, optimal_interpolation
    from hydro_lama_nc.visualization import (
        plot_input_channels, plot_fields_result, plot_fields_error,
    )

    ds = HydroSyntheticDataset(
        n=1, nx=cfg.nx, nz=cfg.nz, seed=cfg.seed,
    )
    sample = ds[0]
    batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
             for k, v in sample.items() if isinstance(v, torch.Tensor)}

    return batch


@hydra.main(config_path="../configs", config_name="nc/training/hydro_train",
            version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    cfg_dict = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                 if not k.startswith("_")}
    outdir = cfg_dict.get("outdir", "")
    clim = cfg_dict.get("colorbar")
    if not outdir:
        outdir = str(Path(PROJECT_ROOT) / "outputs" / f"hydro_{__import__('datetime').date.today()}")
    outdir = Path(outdir)
    if not outdir.exists():
        log.error("Output directory not found: %s", outdir)
        sys.exit(1)

    ckpt_dir = outdir / "checkpoints"
    viz_dir = outdir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # ── Load best checkpoint ──────────────────────────────────────────────
    ckpt_path = _find_best_checkpoint(ckpt_dir)
    if ckpt_path is None:
        log.error("No checkpoint found in %s", ckpt_dir)
        sys.exit(1)
    log.info("Loading checkpoint: %s", ckpt_path)

    model = _load_hydro_model(ckpt_path, device)
    log.info("HydroGenerator loaded (%.1fK params)",
             sum(p.numel() for p in model.generator.parameters()) / 1e3)

    # ── Generate samples and visualize ────────────────────────────────────
    from hydro_lama_nc.data.synthetic_dataset import HydroSyntheticDataset
    from hydro_lama_nc.evaluation import compute_metrics, optimal_interpolation
    from hydro_lama_nc.visualization import (
        plot_input_channels, plot_fields_result, plot_fields_error,
    )

    n_demo = cfg_dict.get("n_demo", 4)
    ds = HydroSyntheticDataset(n=n_demo, nx=cfg_dict.get("nz", 64),
                                nz=cfg_dict.get("nz", 64), seed=cfg_dict.get("seed", 42))

    for i in range(n_demo):
        sample = ds[i]
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                 for k, v in sample.items() if isinstance(v, torch.Tensor)}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            batch = model(batch)
            pred = batch["predicted_image"].cpu().numpy()

        tgt_np = batch.get("target", batch["image"])[:1].cpu().numpy()
        inp_np = batch["image"][:1].cpu().numpy()
        obs_mask_np = batch.get("obs_mask")
        below_np = batch["mask"][0, 0].cpu().numpy().astype(bool)

        if obs_mask_np is None:
            obs_mask_np = batch["mask"][:1].cpu().numpy()
        obs_mask_np = obs_mask_np[:1].cpu().numpy()

        _, _, H, W = inp_np.shape
        oi_T, oi_S = optimal_interpolation(inp_np[0, :2], obs_mask_np[0], H, W)

        suffix = f"sample{i:03d}"

        mask_ctd=inp_np[0, 2]
        mask_cmems=inp_np[0, 3]
        bathy_arr=inp_np[0, 6, 0, :]
        T_obs = inp_np[0, 0]
        S_obs = inp_np[0, 1]

        # Input channels
        plot_input_channels(
            T_obs=T_obs,
            S_obs=S_obs,
            u=inp_np[0, 4],
            v=inp_np[0, 5],
            bathy=bathy_arr,
            mask_ctd=inp_np[0, 2],
            mask_cmems=inp_np[0, 3],
            below=below_np,
            suptitle=f"Sample {i} — inputs",
            save_path=str(viz_dir / f"input_{suffix}.png"),
            clim=clim,
        )
        # OI and LaMa fields and error heatmaps
        for method, TS in (("OI", (oi_T, oi_S)), ("LaMa", (pred[0, 0], pred[0, 1]))):
            for b_error_plot, fun in ((False, plot_fields_result), (True, plot_fields_error)):
                fun(
                    *TS,
                    T_obs=T_obs,
                    S_obs=S_obs,
                    T_bg=tgt_np[0, 0] if b_error_plot else None,
                    S_bg=tgt_np[0, 1] if b_error_plot else None,
                    method=method,
                    below=below_np,
                    mask_ctd=mask_ctd,
                    mask_cmems=mask_cmems,
                    bathy=bathy_arr,
                    output_dir=str(viz_dir),
                    suffix=suffix,
                    suptitle=f"Sample {i} — {method}{' error' if b_error_plot else ''}",
                    clim=clim,
                )



        # Metrics
        for name, p, t in [("T", pred[0, 0], tgt_np[0, 0]),
                            ("S", pred[0, 1], tgt_np[0, 1])]:
            mL = compute_metrics(p, t, below_np)
            mO = compute_metrics(
                oi_T if name == "T" else oi_S, t, below_np,
            )
            log.info("  sample %d %s: LaMa RMSE=%.4f OI RMSE=%.4f",
                     i, name, mL["rmse"], mO["rmse"])

    log.info("Demo complete.  Results in %s", viz_dir)


if __name__ == "__main__":
    main()
