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

import hydro_lama_nc.resolvers  # noqa: F401
from hydro_lama_nc.utils import find_best_checkpoint, cleanup_checkpoints

logger = logging.getLogger("hydro_train")


def _setup_logging() -> None:
    """Replace the root console handler with colorlog if available."""
    try:
        from colorlog import ColoredFormatter
        fmt = ColoredFormatter(
            "%(cyan)s%(asctime)s%(reset)s|%(blue)s%(name)s%(reset)s"
            "|%(log_color)s%(levelname)s%(reset)s|%(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "purple", "INFO": "green",
                "WARNING": "yellow", "ERROR": "red", "CRITICAL": "red",
            },
        )
    except ImportError:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    root = logging.getLogger()
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setFormatter(fmt)
            return
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


# ── Visualization callback ──────────────────────────────────────────────


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
        viz_n_ctd: Number of CTD profiles in the fixed visualization sample.
    """

    def __init__(self, viz_dir: Path, viz_every: int = 5, clim: dict = None,
                 scaling: dict = None, train_dataset=None, viz_n_ctd: int = 4):
        self.viz_dir = viz_dir
        self.viz_every = viz_every
        self.clim = clim
        self.scaling = scaling
        self.train_dataset = train_dataset
        self.viz_n_ctd = viz_n_ctd
        self._viz_sample = None
        self._val_input_saved = False

    def _make_viz_sample(self, nx: int, nz: int, n_ctd: int = None,
                         seed: int = 999) -> dict:
        """Generate a fixed visualization sample with guaranteed CTD profiles.

        Uses private scene-generation helpers from ctd_dropout to produce a
        single deterministic sample that always has the requested n_ctd.

        Args:
            nx: Horizontal grid size.
            nz: Vertical grid size.
            n_ctd: Number of CTD profiles (defaults to self.viz_n_ctd).
            seed: RNG seed for reproducibility.

        Returns:
            Dict with keys matching the dataset output format.
        """
        from hydro_lama_nc.data.ctd_dropout import (
            _generate_scene, _make_cmems_background, _sample_ctd_profiles,
            _sample_cmems_observations, _assemble_7ch, _make_cmems_raw_7ch,
        )

        if n_ctd is None:
            n_ctd = self.viz_n_ctd

        rng = np.random.default_rng(seed)
        sc = _generate_scene(seed, "realistic", "summer", nx, nz)
        T, S = sc["T"], sc["S"]
        below, bathy = sc["below"], sc["bathy"]
        u_lr, v_lr = sc["u_lr"], sc["v_lr"]

        T_bg, S_bg = _make_cmems_background(T, S, below, nx, nz, 2.0, 6, rng)
        T_ctd, S_ctd, mask_ctd = _sample_ctd_profiles(
            T, S, below, bathy, nx, nz, n_ctd, 0.05, rng,
        )
        _, _, mask_cmems = _sample_cmems_observations(
            T_bg, S_bg, below, mask_ctd, nx, nz, 8, 10, 0.10, rng,
        )

        inp = _assemble_7ch(
            T_bg, S_bg, T_ctd, S_ctd, mask_ctd, mask_cmems,
            u_lr, v_lr, bathy, nz,
        )
        tgt = np.stack([T, S], axis=0).astype(np.float32)
        cmems_raw = _make_cmems_raw_7ch(T_bg, S_bg, mask_cmems)

        return {
            "image": torch.from_numpy(inp).unsqueeze(0),
            "target": torch.from_numpy(tgt).unsqueeze(0),
            "mask": torch.from_numpy(below.astype(np.float32)[None]).unsqueeze(0),
            "mask_ctd": torch.from_numpy(mask_ctd[None]).unsqueeze(0),
            "mask_cmems": torch.from_numpy(mask_cmems[None]).unsqueeze(0),
            "cmems_raw": torch.from_numpy(cmems_raw).unsqueeze(0),
            "u_input": torch.from_numpy(u_lr[None].astype(np.float32)).unsqueeze(0),
            "v_input": torch.from_numpy(v_lr[None].astype(np.float32)).unsqueeze(0),
        }

    def _visualize_sample(self, batch, pl_module, epoch, suffix_prefix="", trainer=None):
        """Run model on a single batch and save all viz plots."""
        from hydro_lama_nc.evaluation import compute_metrics, optimal_interpolation
        from hydro_lama_nc.visualization import (
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

        tgt_np = batch_dev.get("target", batch["image"])[:1].cpu().numpy()
        inp_np = batch["image"][:1].cpu().numpy()
        mask_ctd_np = batch.get("mask_ctd", torch.zeros_like(batch["image"][:1, :1]))[:1].cpu().numpy()
        mask_cmems_np = batch.get("mask_cmems", torch.zeros_like(batch["image"][:1, :1]))[:1].cpu().numpy()
        below_np = batch["mask"][0, 0].cpu().numpy().astype(bool)

        obs_mask_np = np.clip(mask_ctd_np + mask_cmems_np, 0, 1)

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
        mask_ctd = mask_ctd_np[0, 0]
        mask_cmems = mask_cmems_np[0, 0]

        # Input channels — skip for fixed val sample after first save
        is_val_viz = not suffix_prefix
        if not (is_val_viz and self._val_input_saved):
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
            if is_val_viz:
                self._val_input_saved = True
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

        # Build the fixed viz sample once (needs nx/nz from the val loader)
        if self._viz_sample is None:
            val_dl = trainer.val_dataloaders
            if val_dl is None:
                return
            if isinstance(val_dl, (list, tuple)):
                val_dl = val_dl[0]
            first_batch = next(iter(val_dl))
            # Infer nx, nz from the input image shape: (B, 7, nz, nx)
            img_shape = first_batch["image"].shape
            nz_inferred, nx_inferred = img_shape[-2], img_shape[-1]
            logger.info("Creating fixed viz sample: n_ctd=%d, nx=%d, nz=%d",
                        self.viz_n_ctd, nx_inferred, nz_inferred)
            self._viz_sample = self._make_viz_sample(nx_inferred, nz_inferred)

        self._visualize_sample(
            self._viz_sample, pl_module, epoch, suffix_prefix="", trainer=trainer,
        )

        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)
            train_batch = self.train_dataset[0]
            train_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                           for k, v in train_batch.items()}
            self._visualize_sample(train_batch, pl_module, epoch, suffix_prefix="train_", trainer=trainer)


# ── Training module ─────────────────────────────────────────────────────


class HydroLightningModule(LightningModule):
    """PL module for Hydro T+S training with 3-loss Kendall & Gal weighting."""

    def __init__(self, lr=2e-4, base_ch=32, n_blocks=6, ratio_g=0.5):
        super().__init__()
        self.save_hyperparameters()
        from hydro_lama_nc.model import HydroGenerator
        self.generator = HydroGenerator(
            in_ch=7, base_ch=base_ch, n_blocks=n_blocks, ratio_g=ratio_g,
        )
        self.lr = lr
        self._val_outputs = []
        # Learnable uncertainty weights (Kendall & Gal 2017): one per loss term
        # Terms: [observation_loss, physics_loss, geostrophic_loss]
        self.log_vars = nn.Parameter(torch.zeros(3))
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
        return batch

    def training_step(self, batch, batch_idx):
        from hydro_lama_nc.losses import observation_loss, physics_loss, geostrophic_loss

        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch["target"]
        mask_below = batch["mask"]
        mask_ctd = batch["mask_ctd"]
        mask_cmems = batch["mask_cmems"]
        cmems_raw = batch["cmems_raw"]
        u_input = batch["u_input"]
        v_input = batch.get("v_input", u_input)

        losses = [
            observation_loss(pred, tgt, mask_ctd, mask_cmems, cmems_raw),
            physics_loss(pred, mask_below=mask_below),
            geostrophic_loss(pred[:, :1], pred[:, 1:], u_input, v_input),
        ]

        total = sum(
            torch.exp(-self.log_vars[i].clamp(-2.0, 2.0)) * l
            + 0.5 * self.log_vars[i].clamp(-2.0, 2.0)
            for i, l in enumerate(losses)
        )

        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)

        keys = ["obs", "physics", "geo"]
        for i, l in enumerate(losses):
            lv = self.log_vars[i]
            weight = torch.exp(-lv.clamp(-2.0, 2.0))
            raw = l.detach()
            k = keys[i]
            self.log(f"log_vars/log_var_{k}", lv.detach(), on_step=False, on_epoch=True)
            self.log(f"weights/weight_{k}", weight.detach(), on_step=False, on_epoch=True)
            self.log(f"loss_raw/{k}_raw", raw, on_step=False, on_epoch=True)
            self.log(f"loss_weighted/{k}_weighted", (weight * raw).detach(), on_step=False, on_epoch=True)

        return total

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        from hydro_lama_nc.losses import observation_loss, physics_loss, geostrophic_loss

        batch = self(batch)
        pred = batch["predicted_image"]
        tgt = batch["target"]
        mask_below = batch["mask"]
        mask_ctd = batch["mask_ctd"]
        mask_cmems = batch["mask_cmems"]
        cmems_raw = batch["cmems_raw"]
        u_input = batch["u_input"]
        v_input = batch.get("v_input", u_input)

        losses = [
            observation_loss(pred, tgt, mask_ctd, mask_cmems, cmems_raw),
            physics_loss(pred, mask_below=mask_below),
            geostrophic_loss(pred[:, :1], pred[:, 1:], u_input, v_input),
        ]

        total = sum(
            torch.exp(-self.log_vars[i].clamp(-2.0, 2.0)) * l
            + 0.5 * self.log_vars[i].clamp(-2.0, 2.0)
            for i, l in enumerate(losses)
        )

        loss_name = "val_loss" if dataloader_idx == 0 else "ood_val_loss"
        prog_bar = dataloader_idx == 0
        self.log(loss_name, total, prog_bar=prog_bar, on_epoch=True, sync_dist=True,
                 add_dataloader_idx=False)

        prefix = "" if dataloader_idx == 0 else "ood_"
        keys = ["obs", "physics", "geo"]
        for i, l in enumerate(losses):
            lv = self.log_vars[i]
            weight = torch.exp(-lv.clamp(-2.0, 2.0))
            raw = l.detach()
            k = keys[i]
            self.log(f"{prefix}log_vars/log_var_{k}", lv.detach(), on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)
            self.log(f"{prefix}weights/weight_{k}", weight.detach(), on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)
            self.log(f"{prefix}loss_raw/{k}_raw", raw, on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)
            self.log(f"{prefix}loss_weighted/{k}_weighted", (weight * raw).detach(),
                     on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        self._val_outputs.append(total)
        return total

    def on_validation_epoch_end(self):
        self._val_outputs = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            list(self.generator.parameters()) + [self.log_vars],
            lr=self.lr, betas=(0.0, 0.999),
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
        """Patch OneCycleLR total_steps after PL restores scheduler state."""
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
    nx: int = 64,
    nz: int = 64,
    colorbar: dict = None,
    data_scaling: dict = None,
    viz_n_ctd: int = 4,
    # CTD dropout params
    ctd_max: int = 7,
    ctd_min: int = 0,
    n_cmems_x: int = 8,
    n_cmems_z: int = 10,
    noise_ctd: float = 0.05,
    noise_cmems: float = 0.10,
    cmems_smoothing_sigma: float = 2.0,
    cmems_shift_max_px: int = 6,
    mode_mix: dict = None,
    num_workers: int = 4,
    # Val dataset
    val_seed_offset: int = 100,
    val_n_ctd_list: list = None,
    val_mode_mix: dict = None,
    # OOD val dataset
    ood_n_ratio: float = 0.5,
    ood_seed_offset: int = 200,
    ood_n_ctd: int = 2,
    ood_noise_ctd: float = 0.08,
    ood_noise_cmems: float = 0.15,
    ood_mode_mix: dict = None,
    **_kwargs,
):
    """Run Hydro T+S training with CTD dropout and 3-loss pipeline.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        val_batch_size: Validation batch size.
        lr: Learning rate.
        viz_every: Generate comparison plots every N epochs.
        n_train: Number of training samples (dataset length).
        n_val: Number of validation samples.
        seed: Random seed.
        outdir: Output directory path. Auto-generated from date if empty.
        base_ch: Base channel count for HydroGenerator.
        n_blocks: Number of residual blocks.
        ratio_g: Fraction of channels for global (Fourier) branch.
        augment: Enable physical augmentation in training dataset.
        lazy: Generate scenes on-the-fly.
        nx: Horizontal grid points.
        nz: Vertical grid points.
        viz_n_ctd: Number of CTD profiles in the fixed visualization sample.
        ctd_max: Maximum CTD profiles per sample.
        ctd_min: Minimum CTD profiles per sample.
        n_cmems_x: CMEMS columns per sample.
        n_cmems_z: CMEMS depth levels per column.
        noise_ctd: CTD measurement noise sigma.
        noise_cmems: CMEMS measurement noise sigma.
        cmems_smoothing_sigma: CMEMS background Gaussian filter sigma.
        cmems_shift_max_px: CMEMS background max vertical shift.
        mode_mix: Dict of mode→probability.
        num_workers: DataLoader workers.
        val_seed_offset: Val seed offset.
        val_n_ctd_list: List of n_ctd values for degradation curve.
        val_mode_mix: Val scene mode mixture (default: [0.7, 0.2, 0.1]).
        ood_n_ratio: OOD n_val ratio.
        ood_seed_offset: OOD seed offset.
        ood_n_ctd: CTD profiles for OOD dataset.
        ood_noise_ctd: OOD CTD noise.
        ood_noise_cmems: OOD CMEMS noise.
        ood_mode_mix: OOD mode mixture.
    """
    _setup_logging()

    import pytorch_lightning as pl
    from hydro_lama_nc.data.ctd_dropout import (
        HybridCTDDataset, StratifiedCTDSampler, FixedCTDValDataset,
        stratified_collate_fn, DegradationCurveCallback,
    )
    from torch.utils.data import DataLoader

    if val_n_ctd_list is None:
        val_n_ctd_list = [0, 1, 2, 3, 5, 7]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Output directory (date-based, same-day resume) ────────────────────
    if not outdir:
        outdir = str(Path(PROJECT_ROOT) / "outputs" / f"{date.today()}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tb_dir = outdir / "tb_logs"
    viz_dir = outdir / "viz"
    ckpt_dir = outdir / "checkpoints"
    tb_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    logger.info("Device: %s. Output directory: %s", device, outdir)

    # ── Auto-resume from best checkpoint ──────────────────────────────────
    resume_ckpt = find_best_checkpoint(ckpt_dir)
    if resume_ckpt is not None:
        try:
            ckpt = torch.load(str(resume_ckpt), map_location="cpu", weights_only=False)
            model_state = ckpt.get("state_dict", {})
            enc0_shape = model_state.get("generator.enc.0.weight")
            if enc0_shape is not None and enc0_shape.shape[1] != 7:
                logger.info("Checkpoint in_ch=%d != config in_ch=7 — starting fresh",
                            enc0_shape.shape[1])
                resume_ckpt = None
            elif "log_vars" in model_state and model_state["log_vars"].shape[0] != 3:
                logger.info("Checkpoint log_vars shape=%s != 3 — starting fresh",
                            str(model_state["log_vars"].shape))
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
    logger.info("Generating train dataset: %d samples, nx=%d, nz=%d, lazy=%s, "
                "ctd=[%d..%d], mode_mix=%s",
                n_train, nx, nz, lazy, ctd_min, ctd_max, mode_mix)
    full_train = HybridCTDDataset(
        n=n_train, nx=nx, nz=nz,
        min_ctd=ctd_min, max_ctd=ctd_max,
        n_cmems_x=n_cmems_x, n_cmems_z=n_cmems_z,
        noise_ctd=noise_ctd, noise_cmems=noise_cmems,
        smoothing_sigma=cmems_smoothing_sigma,
        shift_max_px=cmems_shift_max_px,
        mode_mix=mode_mix,
        seed=seed, augment=augment, lazy=lazy,
    )

    logger.info("Generating val dataset: %d samples, fixed scenes", n_val)
    val_ds = FixedCTDValDataset(
        n_scenes=n_val,
        val_n_ctd_list=val_n_ctd_list,
        nx=nx, nz=nz,
        seed=seed + val_seed_offset,
        n_cmems_x=n_cmems_x, n_cmems_z=n_cmems_z,
        noise_ctd=noise_ctd, noise_cmems=noise_cmems,
        smoothing_sigma=cmems_smoothing_sigma,
        shift_max_px=cmems_shift_max_px,
        mode_mix=val_mode_mix,
    )

    ood_n = max(1, int(n_val * ood_n_ratio))
    logger.info("Generating OOD val dataset: %d samples, n_ctd=%d", ood_n, ood_n_ctd)
    ood_val_ds = HybridCTDDataset(
        n=ood_n, nx=nx, nz=nz,
        min_ctd=ood_n_ctd, max_ctd=ood_n_ctd,
        n_cmems_x=n_cmems_x, n_cmems_z=n_cmems_z,
        noise_ctd=ood_noise_ctd, noise_cmems=ood_noise_cmems,
        smoothing_sigma=cmems_smoothing_sigma,
        shift_max_px=cmems_shift_max_px,
        mode_mix=ood_mode_mix,
        seed=seed + ood_seed_offset, augment=False, lazy=False,
    )

    logger.info("Train: %d  Val: %d (%d n_ctd values)  OOD-Val: %d",
                len(full_train), len(val_ds), len(val_n_ctd_list), len(ood_val_ds))

    # Stratified sampler for balanced n_ctd representation
    train_sampler = StratifiedCTDSampler(full_train, batch_size=batch_size)
    train_dl = DataLoader(
        full_train, batch_sampler=train_sampler, num_workers=num_workers,
        collate_fn=stratified_collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=stratified_collate_fn,
    )
    ood_val_dl = DataLoader(
        ood_val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=stratified_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = HydroLightningModule(lr=lr, base_ch=base_ch, n_blocks=n_blocks,
                                  ratio_g=ratio_g)

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="hydro-{epoch:02d}-val_loss={val_loss:.4f}",
        monitor="val_loss", mode="min", save_top_k=10,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=15, mode="min", verbose=True,
        check_on_train_epoch_end=False,
    )
    viz_cb = HydroVisualizationCallback(
        viz_dir=viz_dir, viz_every=viz_every, clim=colorbar,
        scaling=data_scaling,
        train_dataset=full_train,
        viz_n_ctd=viz_n_ctd,
    )
    degradation_cb = DegradationCurveCallback(
        val_dataset=val_ds, batch_size=val_batch_size,
    )

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

    # Run degradation curve logging manually after each validation
    class _DegradationHook(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            degradation_cb.log_degradation_curve(trainer, pl_module)

    trainer.callbacks.append(_DegradationHook())

    try:
        trainer.fit(model, train_dl, [val_dl, ood_val_dl],
                    ckpt_path=str(resume_ckpt) if resume_ckpt else None)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception:
        logger.exception("Training failed")

    logger.info("Training complete.  Outputs: %s", outdir)
    logger.info("  Checkpoints:  %s", ckpt_dir)
    logger.info("  TensorBoard:  %s", tb_dir)
    logger.info("  Visualizations: %s", viz_dir)

    return outdir


@hydra.main(config_path="../configs", config_name="nc/training/hydro_train",
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
