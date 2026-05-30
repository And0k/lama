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
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*batch_size.*ambiguous.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
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
from netcdf.utils import find_best_checkpoint, cleanup_checkpoints

logger = logging.getLogger("hydro_train")


# ── Visualization callback ──────────────────────────────────────────────


class CurriculumCallback(Callback):
    """Calls dataset.set_epoch() at each epoch start for curriculum learning."""

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_dataset.set_epoch(trainer.current_epoch, trainer.max_epochs)


class HydroVisualizationCallback(Callback):
    """Generates per-epoch T/S comparison plots with OI baseline.

    Saves to viz_dir:
        input_ep{N}.png         — input channels (u,T / v,S), fixed val sample
        oi_T_S_ep{N}.png        — OI T+S fields (fixed val sample)
        lama_T_S_ep{N}.png      — LaMa T+S fields (fixed val sample)
        oi_error_ep{N}.png      — OI error heatmaps (fixed val sample)
        lama_error_ep{N}.png    — LaMa error heatmaps (fixed val sample)
        train_input_ep{N}.png   — training sample (varies per epoch, lazy=True)

    Args:
        viz_dir: Output directory for plots.
        viz_every: Generate plots every N epochs.
        clim: Colorbar limits dict.
        scaling: Scaling ranges for physical units.
        train_dataset: Training HydroSyntheticDataset for diversity viz.
    """

    def __init__(self, viz_dir: Path, viz_every: int = 5, clim: dict = None,
                 scaling: dict = None, train_dataset=None):
        self.viz_dir = viz_dir
        self.viz_every = viz_every
        self.clim = clim
        self.scaling = scaling
        self.train_dataset = train_dataset

    def _visualize_sample(self, batch, pl_module, epoch, suffix_prefix="", trainer=None):
        """Run model on a single batch and save all viz plots.

        Args:
            batch: One sample dict from HydroSyntheticDataset.
            pl_module: HydroLightningModule.
            epoch: Current epoch number.
            suffix_prefix: Optional prefix for file suffix (e.g. "train_").
            trainer: PL Trainer for TensorBoard logging (optional).
        """
        from netcdf.evaluation import compute_metrics, optimal_interpolation
        from netcdf.visualization import (
            plot_input_channels, plot_fields_result, plot_fields_error,
        )

        device = next(pl_module.parameters()).device
        pl_module.eval()

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

        oi_T, oi_S = optimal_interpolation(inp_np[0, :2], obs_mask_np[0], H, W)
        oi_np = np.stack([oi_T, oi_S], axis=0)

        suffix = f"{suffix_prefix}ep{epoch:04d}" if suffix_prefix else f"ep{epoch:04d}"

        for name, p, t in [("T", pred[0, 0], tgt_np[0, 0]),
                            ("S", pred[0, 1], tgt_np[0, 1])]:
            mL = compute_metrics(p, t, below_np)
            mO = compute_metrics(
                oi_np[0 if name == "T" else 1], t, below_np,
            )
            logger.info("  [viz ep=%d %s] %s LaMa RMSE=%.4f OI RMSE=%.4f",
                        epoch, suffix_prefix or "val", name, mL["rmse"], mO["rmse"])

        u_lr = inp_np[0, 4]
        v_lr = inp_np[0, 5]
        T_obs = inp_np[0, 0]
        S_obs = inp_np[0, 1]
        bathy_arr = inp_np[0, 6, 0, :]
        mask_ctd = inp_np[0, 2]
        mask_cmems = inp_np[0, 3]

        # Input channels
        plot_input_channels(
            T_obs, S_obs, u_lr, v_lr,
            T_bg=tgt_np[0, 0], S_bg=tgt_np[0, 1],
            bathy=bathy_arr, mask_ctd=mask_ctd, mask_cmems=mask_cmems,
            below=below_np,
            suptitle=f"Epoch {epoch} — inputs",
            save_path=str(self.viz_dir / f"input_{suffix}.png"),
            clim=self.clim,
            scaling=self.scaling,
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
                    output_dir=str(self.viz_dir),
                    suffix=suffix,
                    suptitle=f"Epoch {epoch} — {method}{' error' if b_error_plot else ''}",
                    clim=self.clim,
                )

        if trainer.logger is not None:
            fig_path = self.viz_dir / f"lama_error_{suffix}.png"
            if fig_path.exists():
                img = plt.imread(str(fig_path))
                trainer.logger.experiment.add_image(
                    f"hydro/lama_error_{suffix_prefix}val" if suffix_prefix else "hydro/lama_error",
                    img,
                    global_step=trainer.global_step, dataformats="HWC",
                )

        pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.viz_every != 0 and epoch != 0:
            return

        # ── Viz 1: fixed val sample (lazy=False, same input every epoch) ──
        # Good for tracking model progress on a consistent baseline.
        val_dl = trainer.val_dataloaders
        if val_dl is None:
            return
        if isinstance(val_dl, (list, tuple)):
            val_dl = val_dl[0]
        val_batch = next(iter(val_dl))
        self._visualize_sample(val_batch, pl_module, epoch, suffix_prefix="", trainer=trainer)

        # ── Viz 2: training sample (lazy=True, varies per epoch) ────────
        # Shows scene diversity: bathymetry, T/S fields, etc. change.
        if self.train_dataset is not None:
            self.train_dataset._epoch = epoch
            train_batch = self.train_dataset[0]
            train_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                           for k, v in train_batch.items()}
            self._visualize_sample(train_batch, pl_module, epoch, suffix_prefix="train_", trainer=trainer)


# ── Training module ─────────────────────────────────────────────────────


class HydroLightningModule(LightningModule):
    """PL module for Hydro T+S training with uncertainty weighting and OneCycleLR."""

    def __init__(self, lr=2e-4, base_ch=32, n_blocks=6, ratio_g=0.5):
        super().__init__()
        self.save_hyperparameters()
        from saicinpainting.training.modules.hydro import HydroGenerator
        self.generator = HydroGenerator(
            in_ch=8, base_ch=base_ch, n_blocks=n_blocks, ratio_g=ratio_g,
        )
        self.lr = lr
        self._val_outputs = []
        # Learnable uncertainty weights (Kendall & Gal 2017): one per loss term
        # Terms: [obs_l1, domain_l1, inv_var, stability, bbl, geostrophic]
        self.log_vars = nn.Parameter(torch.zeros(6))
        n_params = sum(p.numel() for p in self.generator.parameters())
        logger.info("HydroGenerator: %.1fK params (base_ch=%d, n_blocks=%d, ratio_g=%.2f)",
                    n_params / 1e3, base_ch, n_blocks, ratio_g)

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

        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch.get("target", batch["image"])
        mask = batch["mask"]

        # Individual loss terms (unweighted)
        losses = [
            F.l1_loss(pred * mask, tgt * mask),           # obs_l1
            F.l1_loss(pred * (1 - mask), tgt * (1 - mask)),  # domain_l1
            inverse_variance_mse(
                pred, tgt, batch.get("sigma_obs", mask),
                batch.get("obs_mask", mask), below_mask=mask,
            ),
            hydrostatic_stability_loss(
                pred, channel_index=0, salinity_index=1, below_mask=mask,
            ),
            bottom_boundary_layer_loss(
                pred, bathy_indices=batch.get("bathy_indices"),
                channel_index=0, salinity_index=1,
            ),
            geostrophic_balance_loss(
                pred, u_input=batch.get("u_input"), below_mask=mask,
            ),
        ]
        smooth = smoothness_loss(pred, mask=mask)

        # Uncertainty weighting (Kendall & Gal 2017)
        total = sum(
            torch.exp(-self.log_vars[i]) * l + 0.5 * self.log_vars[i]
            for i, l in enumerate(losses)
        ) + smooth  # smoothness stays outside uncertainty weighting

        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        for i, l in enumerate(losses):
            self.log(f"loss/{i}", l.detach(), on_step=True, on_epoch=False)
        # Log learned sigma = exp(0.5 * log_var) for monitoring
        for i in range(6):
            self.log(f"sigma/{i}", torch.exp(0.5 * self.log_vars[i]).detach(),
                     on_step=True, on_epoch=False)
        return total

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        from saicinpainting.training.losses.physical import (
            inverse_variance_mse, hydrostatic_stability_loss,
            geostrophic_balance_loss, smoothness_loss,
        )

        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch.get("target", batch["image"])
        mask = batch["mask"]

        losses = [
            F.l1_loss(pred * mask, tgt * mask),
            F.l1_loss(pred * (1 - mask), tgt * (1 - mask)),
            inverse_variance_mse(
                pred, tgt, batch.get("sigma_obs", mask),
                batch.get("obs_mask", mask), below_mask=mask,
            ),
            hydrostatic_stability_loss(
                pred, channel_index=0, salinity_index=1, below_mask=mask,
            ),
            geostrophic_balance_loss(
                pred, u_input=batch.get("u_input"), below_mask=mask,
            ),
        ]
        smooth = smoothness_loss(pred, mask=mask)

        total = sum(
            torch.exp(-self.log_vars[i]) * l + 0.5 * self.log_vars[i]
            for i, l in enumerate(losses)
        ) + smooth

        loss_name = "val_loss" if dataloader_idx == 0 else "ood_val_loss"
        prog_bar = dataloader_idx == 0
        self.log(loss_name, total, prog_bar=prog_bar, on_epoch=True, sync_dist=True)
        self._val_outputs.append(total)
        return total

    def on_validation_epoch_end(self):
        self._val_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.0, 0.999),
        )
        total_steps = self.trainer.estimated_stepping_batches
        already_done = self.trainer.global_step
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            last_epoch=already_done - 1,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_train_start(self):
        """Patch OneCycleLR total_steps after PL restores scheduler state.

        When resuming from a checkpoint, PL restores the scheduler state
        from the checkpoint - including the old total_steps.  We need to
        update it to match the new max_epochs.
        """
        sched = self.lr_schedulers()
        if sched is not None and hasattr(sched, 'total_steps'):
            new_total = self.trainer.estimated_stepping_batches
            if sched.total_steps != new_total:
                logger.info("Patching scheduler total_steps: %d → %d",
                            sched.total_steps, new_total)
                sched.total_steps = new_total


# ── Training entry point ────────────────────────────────────────────────


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
    base_ch: int = 32,
    n_blocks: int = 6,
    ratio_g: float = 0.5,
    augment: bool = True,
    lazy: bool = True,
    colorbar: dict = None,
    data_scaling: dict = None,
    # Dataset generation params (also affect scene diversity)
    n_ctd: int = 5,
    n_cmems_x: int = 8,
    n_cmems_z: int = 10,
    noise_ctd: float = 0.05,
    noise_cmems: float = 0.10,
    mode_mix: dict = None,
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
        n_train: Number of training samples (dataset length, not unique scenes).
        n_val: Number of validation samples.
        seed: Random seed.
        outdir: Output directory path.  Auto-generated from date if empty.
        base_ch: Base channel count for HydroGenerator.
        n_blocks: Number of residual blocks.
        ratio_g: Fraction of channels for global (Fourier) branch.
        augment: Enable physical augmentation in training dataset.
        lazy: Generate scenes on-the-fly (infinite diversity per epoch).
            When True, n_train is the dataset length but each access
            creates a fresh scene — the model never sees the same T/S
            field twice.  Val datasets stay pre-generated for stability.
        n_ctd: Base CTD profiles per sample (also max for curriculum).
        n_cmems_x: CMEMS columns per sample.
        n_cmems_z: CMEMS depth levels per sample.
        noise_ctd: CTD measurement noise sigma.
        noise_cmems: CMEMS measurement noise sigma.
        mode_mix: Dict of mode->probability, e.g.
            {"realistic": 0.7, "extended": 0.2, "stress": 0.1}.  None
            defaults to 70/20/10.
    """
    if not logging.getLogger(__name__).handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

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
    resume_ckpt = find_best_checkpoint(ckpt_dir)
    if resume_ckpt is not None:
        try:
            ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
            model_state = ckpt.get("state_dict", {})
            # Verify checkpoint matches current model config by checking
            # the first conv weight shape: [base_ch, input_nc, 7, 7]
            enc0_shape = model_state.get("generator.enc.0.weight")
            if enc0_shape is not None and enc0_shape.shape[0] != base_ch:
                logger.info("Checkpoint base_ch=%d != config base_ch=%d — starting fresh",
                            enc0_shape.shape[0], base_ch)
                resume_ckpt = None
            elif "log_vars" not in model_state:
                logger.info("Checkpoint missing log_vars (pre-uncertainty-weighting) — starting fresh")
                resume_ckpt = None
            else:
                logger.info("Resuming from best checkpoint: %s", resume_ckpt)
                n_removed = cleanup_checkpoints(ckpt_dir, max_keep=1)
                if n_removed:
                    logger.info("Cleaned up %d worse checkpoint(s)", n_removed)
        except Exception:
            logger.info("Checkpoint incompatible — starting fresh")
            resume_ckpt = None
    else:
        logger.info("No previous checkpoint found — starting fresh")

    # ── Datasets ──────────────────────────────────────────────────────────
    logger.info("Generating train dataset: %d samples, lazy=%s, augment=%s, mode_mix=%s",
                n_train, lazy, augment, mode_mix)
    full_train = HydroSyntheticDataset(
        n=n_train, nx=64, nz=64, seed=seed, augment=augment, lazy=lazy,
        n_ctd=n_ctd, n_cmems_x=n_cmems_x, n_cmems_z=n_cmems_z,
        noise_ctd=noise_ctd, noise_cmems=noise_cmems,
        mode_mix=mode_mix,
    )
    logger.info("Generating val dataset: %d samples, fixed scenes", n_val)
    val_ds = HydroSyntheticDataset(
        n=n_val, nx=64, nz=64, seed=seed + 100, augment=False,
        n_variants=1, lazy=False,
    )
    logger.info("Generating OOD val dataset: %d samples, stress mode, n_ctd=2", n_val // 2)
    ood_val_ds = HydroSyntheticDataset(
        n=n_val // 2, nx=64, nz=64, seed=seed + 200, augment=False,
        n_ctd=2, noise_ctd=0.08, noise_cmems=0.15,
        mode_mix={"realistic": 0.0, "extended": 0.3, "stress": 0.7},
        n_variants=1, lazy=False,
    )
    logger.info("Train: %d  Val: %d  OOD-Val: %d",
                len(full_train), len(val_ds), len(ood_val_ds))

    train_dl = DataLoader(
        full_train, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0,
    )
    ood_val_dl = DataLoader(
        ood_val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroLightningModule(lr=lr, base_ch=base_ch, n_blocks=n_blocks,
                                  ratio_g=ratio_g)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="hydro-{epoch:02d}-val_loss={val_loss/dataloader_idx_0:.4f}",
        monitor="val_loss/dataloader_idx_0", mode="min", save_top_k=10,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss/dataloader_idx_0", patience=15, mode="min", verbose=True,
    )
    viz_cb = HydroVisualizationCallback(
        viz_dir=viz_dir, viz_every=viz_every, clim=colorbar, train_dataset=full_train
    )
    curriculum_cb = CurriculumCallback(full_train)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(save_dir=str(tb_dir), name="hydro_T_S")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb, early_stop_cb, viz_cb, curriculum_cb],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    try:
        trainer.fit(model, train_dl, [val_dl, ood_val_dl],
                    ckpt_path=str(resume_ckpt) if resume_ckpt else None)
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
