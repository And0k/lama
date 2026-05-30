#!/usr/bin/env python3
"""
Hydro T+S Training Script — Colab + Local

Usage (Colab):
    from notebooks.train_hydro_colab import run_training
    run_training(epochs=40, batch_size=4)

Usage (CLI):
    python notebooks/train_hydro_colab.py epochs=40 batch_size=4 viz_every=5
"""

import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, LightningModule, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import netcdf.resolvers  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hydro_train")


# ── Visualization callback ──────────────────────────────────────────────


class HydroVisualizationCallback(Callback):
    """Generates per-epoch T/S comparison plots with OI baseline.

    Saves to viz_dir:
        input_ep{N}.png         — input channels (u,T / v,S)
        oi_T_S_ep{N}.png        — OI T+S fields
        lama_T_S_ep{N}.png      — LaMa T+S fields
        oi_error_ep{N}.png      — OI error heatmaps
        lama_error_ep{N}.png    — LaMa error heatmaps
    """

    def __init__(self, viz_dir: Path, viz_every: int = 5):
        self.viz_dir = viz_dir
        self.viz_every = viz_every

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.viz_every != 0 and epoch != 0:
            return

        from netcdf.evaluation import compute_metrics, optimal_interpolation
        from netcdf.visualization import (
            plot_input_channels, plot_fields_result, plot_fields_error,
        )

        device = next(pl_module.parameters()).device
        pl_module.eval()

        val_dl = trainer.val_dataloaders
        if val_dl is None:
            return
        batch = next(iter(val_dl))
        if isinstance(batch, (list, tuple)):
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b
                     for b in batch]
            batch = batch[0] if isinstance(batch, (list, tuple)) else batch

        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        with torch.no_grad():
            batch_dev = pl_module(batch_dev)
            pred = batch_dev["predicted_image"].cpu().numpy()

        tgt_np = batch_dev.get("target", batch_dev["image"])[:1].cpu().numpy()
        inp_np = batch_dev["image"][:1].cpu().numpy()
        obs_mask_np = batch_dev.get("obs_mask")
        below_np = batch_dev["mask"][0, 0].cpu().numpy().astype(bool)

        if obs_mask_np is None:
            obs_mask_np = batch_dev["mask"][:1].cpu().numpy()
        obs_mask_np = obs_mask_np[:1].cpu().numpy()

        _, _, H, W = inp_np.shape

        # OI baseline
        oi_T, oi_S = optimal_interpolation(inp_np[0, :2], obs_mask_np[0], H, W)
        oi_np = np.stack([oi_T, oi_S], axis=0)

        # Metrics
        suffix = f"ep{epoch:04d}"
        for name, p, t in [("T", pred[0, 0], tgt_np[0, 0]),
                            ("S", pred[0, 1], tgt_np[0, 1])]:
            mL = compute_metrics(p, t, below_np)
            mO = compute_metrics(
                oi_np[0 if name == "T" else 1], t, below_np,
            )
            logger.info("  [viz ep=%d] %s LaMa RMSE=%.4f OI RMSE=%.4f",
                        epoch, name, mL["rmse"], mO["rmse"])

        # ── Save input channels ──────────────────────────────────────────
        u_lr = inp_np[0, 4]
        v_lr = inp_np[0, 5]
        T_obs = inp_np[0, 0]
        S_obs = inp_np[0, 1]
        bathy_arr = inp_np[0, 6, 0, :]
        mask_ctd = inp_np[0, 2]
        mask_cmems = inp_np[0, 3]

        plot_input_channels(
            u_lr, T_obs, v_lr, S_obs,
            bathy=bathy_arr, mask_ctd=mask_ctd, mask_cmems=mask_cmems,
            below=below_np,
            suptitle=f"Epoch {epoch} — inputs",
            save_path=str(self.viz_dir / f"input_{suffix}.png"),
        )

        # ── Save OI and LaMa fields ─────────────────────────────────────
        plot_fields_result(
            oi_T, oi_S, method="oi", below=below_np,
            output_dir=str(self.viz_dir), suffix=suffix,
            suptitle=f"Epoch {epoch} — OI",
        )
        plot_fields_result(
            pred[0, 0], pred[0, 1], method="lama", below=below_np,
            output_dir=str(self.viz_dir), suffix=suffix,
            suptitle=f"Epoch {epoch} — LaMa",
        )

        # ── Save error heatmaps ──────────────────────────────────────────
        plot_fields_error(
            oi_T - tgt_np[0, 0], oi_S - tgt_np[0, 1],
            method="oi", below=below_np,
            output_dir=str(self.viz_dir), suffix=suffix,
            suptitle=f"Epoch {epoch} — OI error",
        )
        plot_fields_error(
            pred[0, 0] - tgt_np[0, 0], pred[0, 1] - tgt_np[0, 1],
            method="lama", below=below_np,
            output_dir=str(self.viz_dir), suffix=suffix,
            suptitle=f"Epoch {epoch} — LaMa error",
        )

        # ── TensorBoard ──────────────────────────────────────────────────
        if trainer.logger is not None:
            fig_path = self.viz_dir / f"lama_error_{suffix}.png"
            if fig_path.exists():
                img = plt.imread(str(fig_path))
                trainer.logger.experiment.add_image(
                    "hydro/lama_error", img,
                    global_step=trainer.global_step, dataformats="HWC",
                )

        pl_module.train()


# ── Training module ─────────────────────────────────────────────────────


class HydroLightningModule(LightningModule):
    """Minimal PL module for Hydro T+S training with manual optimization."""

    def __init__(self, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        from saicinpainting.training.modules.hydro import HydroGenerator
        self.generator = HydroGenerator(
            input_nc=8, output_nc=2, base_ch=32, n_blocks=6,
        )
        self.lr = lr
        self._val_outputs = []
        n_params = sum(p.numel() for p in self.generator.parameters())
        logger.info("HydroGenerator: %.1fK params", n_params / 1e3)

    def forward(self, batch):
        x = batch["image"]
        origh, origw = x.shape[2], x.shape[3]
        pad_h = (8 - origh % 8) % 8
        pad_w = (8 - origw % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        pred = self.generator(x)
        if pad_h > 0 or pad_w > 0:
            pred = pred[:, :, :origh, :origw]
        batch["predicted_image"] = pred
        target = batch.get("target", batch["image"])
        batch["inpainted"] = (
            batch["mask"] * pred + (1 - batch["mask"]) * target
        )
        return batch

    def training_step(self, batch, batch_idx):
        from saicinpainting.training.losses.physical import (
            inverse_variance_mse, hydrostatic_stability_loss,
            bottom_boundary_layer_loss, geostrophic_balance_loss,
            smoothness_loss,
        )

        opt = self.optimizers()
        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch.get("target", batch["image"])
        mask = batch["mask"]

        loss = 10.0 * F.l1_loss(pred * mask, tgt * mask)
        loss += 1.0 * F.l1_loss(pred * (1 - mask), tgt * (1 - mask))
        loss += 1.0 * inverse_variance_mse(
            pred, tgt, batch.get("sigma_obs", mask),
            batch.get("obs_mask", mask), below_mask=mask,
        )
        loss += 0.3 * hydrostatic_stability_loss(
            pred, channel_index=0, salinity_index=1, below_mask=mask,
        )
        loss += 0.2 * bottom_boundary_layer_loss(
            pred, bathy_indices=batch.get("bathy_indices"),
            channel_index=0, salinity_index=1,
        )
        loss += 0.15 * geostrophic_balance_loss(
            pred, u_input=batch.get("u_input"), below_mask=mask,
        )
        loss += 1.0 * smoothness_loss(pred, mask=mask)

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        from saicinpainting.training.losses.physical import (
            inverse_variance_mse, hydrostatic_stability_loss,
            geostrophic_balance_loss, smoothness_loss,
        )

        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch.get("target", batch["image"])
        mask = batch["mask"]

        loss = 10.0 * F.l1_loss(pred * mask, tgt * mask)
        loss += 1.0 * F.l1_loss(pred * (1 - mask), tgt * (1 - mask))
        loss += 1.0 * inverse_variance_mse(
            pred, tgt, batch.get("sigma_obs", mask),
            batch.get("obs_mask", mask), below_mask=mask,
        )
        loss += 0.3 * hydrostatic_stability_loss(
            pred, channel_index=0, salinity_index=1, below_mask=mask,
        )
        loss += 0.15 * geostrophic_balance_loss(
            pred, u_input=batch.get("u_input"), below_mask=mask,
        )
        loss += 1.0 * smoothness_loss(pred, mask=mask)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self._val_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        self._val_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.0, 0.999),
        )


# ── Training entry point ────────────────────────────────────────────────


def _find_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return path to best checkpoint in *ckpt_dir*, or None."""
    candidates = sorted(ckpt_dir.glob("hydro-*val_loss=*.ckpt"))
    if not candidates:
        return None
    # Parse val_loss from filename: hydro-NN-val_loss=X.XXXX.ckpt
    def _loss(p: Path) -> float:
        try:
            return float(p.stem.split("val_loss=")[1])
        except (IndexError, ValueError):
            return float("inf")
    return min(candidates, key=_loss)


def run_training(
    epochs: int = 40,
    batch_size: int = 4,
    val_batch_size: int = 2,
    lr: float = 2e-4,
    viz_every: int = 5,
    n_train: int = 400,
    n_val: int = 80,
    seed: int = 42,
    outdir: str = "",
    **_kwargs,
):
    """Run Hydro T+S training with synthetic Baltic data.

    Output directory is date-based (``hydro_YYYY-MM-DD``).  If a
    directory with today's date already exists, training resumes from
    the best checkpoint found there.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        val_batch_size: Validation batch size.
        lr: Learning rate.
        viz_every: Generate comparison plots every N epochs.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        seed: Random seed.
        outdir: Output directory path.  Auto-generated from date if empty.
    """
    import pytorch_lightning as pl
    from netcdf.data.synthetic_dataset import HydroSyntheticDataset
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Output directory (date-based, same-day resume) ────────────────────
    if not outdir:
        outdir = str(Path(PROJECT_ROOT) / "outputs" / f"hydro_{date.today()}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tb_dir = outdir / "tb_logs"
    viz_dir = outdir / "viz"
    ckpt_dir = outdir / "checkpoints"
    tb_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    logger.info("Device: %s", device)
    logger.info("Output directory: %s", outdir)

    # ── Auto-resume from best checkpoint ──────────────────────────────────
    resume_ckpt = _find_best_checkpoint(ckpt_dir)
    if resume_ckpt is not None:
        logger.info("Resuming from best checkpoint: %s", resume_ckpt)
    else:
        logger.info("No previous checkpoint found — starting fresh")

    # ── Datasets ──────────────────────────────────────────────────────────
    full_train = HydroSyntheticDataset(
        n=n_train, nx=64, nz=64, seed=seed,
    )
    val_ds = HydroSyntheticDataset(
        n=n_val, nx=64, nz=64, seed=seed + 100,
    )
    logger.info("Train: %d  Val: %d", len(full_train), len(val_ds))

    train_dl = DataLoader(
        full_train, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroLightningModule(lr=lr)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="hydro-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=15, mode="min", verbose=True,
    )
    viz_cb = HydroVisualizationCallback(viz_dir=viz_dir, viz_every=viz_every)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir=str(tb_dir), name="hydro_T_S")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb, early_stop_cb, viz_cb],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(resume_ckpt) if resume_ckpt else None)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.exception("Training failed: %s", e)

    logger.info("Training complete.  Outputs: %s", outdir)
    logger.info("  Checkpoints:  %s", ckpt_dir)
    logger.info("  TensorBoard:  %s", tb_dir)
    logger.info("  Visualizations: %s", viz_dir)

    return outdir


@hydra.main(config_path="../configs/nc/training", config_name="hydro_train",
            version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = {k: v for k, v in OmegaConf.to_container(cfg, resolve=True).items()
                if not k.startswith("_")}
    outdir = run_training(**cfg_dict)
    print(f"\nOutput directory: {outdir}")
    print(f"TensorBoard: tensorboard --logdir {outdir}/tb_logs")
    print(f"Checkpoints: {outdir}/checkpoints")
    print(f"Visualizations: {outdir}/viz")


if __name__ == "__main__":
    main()
