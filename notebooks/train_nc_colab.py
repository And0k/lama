#!/usr/bin/env python3
"""LaMa NetCDF Training — single-cell Colab notebook
=====================================================
Copy-paste into one Jupyter cell.  Repo + deps assumed installed.
See README_nc.md for the full workflow.
"""

# ════════════════════════════════════════════════════════════════
#  1. USER SETTINGS — edit these before running
# ════════════════════════════════════════════════════════════════

REPO_DIR = "/content/lama"

NC_FILE_PATH = (
    "/content/drive/MyDrive/"
    "cmems_mod_bal_phy_anfc_P1D-m_multi-vars_10.51E-21.32E_54.02N-58.97N_"
    "0.50-91.31m_2023-10-01-2023-12-01.nc"
)

OUTPUT_DIR = "/content/drive/MyDrive/lama_nc_output"

PRETRAINED_CKPT = None  # e.g. "/content/drive/MyDrive/big-lama/models/best.ckpt"
RESUME_FROM_CHECKPOINT = None  # None → auto-detect last.ckpt

# Reproducibility seed.  Set to an integer to get deterministic data order,
# mask generation, and weight initialisation across runs.  Set to None to
# disable (faster but non-reproducible).
SEED = 42

MAX_EPOCHS = 40
BATCH_SIZE = 2
ADVERSARIAL_WEIGHT = 0  # 0 = Phase 1 (reconstruction-only)
FEATURE_MATCHING_WEIGHT = 0

ACCELERATOR = "auto"  # "gpu" / "cpu" — PL detects automatically
DEVICES = "auto"
LIMIT_VAL_BATCHES = 200

TB_VERSION = "run0"  # fixed → all runs on one continuous graph

# ════════════════════════════════════════════════════════════════
#  2. ENVIRONMENT
# ════════════════════════════════════════════════════════════════

import logging
import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_nc")

# Suppress noisy PL/torch warnings that are harmless in our setup:
#  - LeafSpec deprecation: PyTorch 2.12 internal, not our code
#  - num_workers bottleneck: resolved, now using 4 workers
#  - eval-mode modules: FID/SSIM evaluators are frozen by design
#  - dataloader not resumable: mid-epoch resume restarts the epoch, which is fine
import warnings

warnings.filterwarnings("ignore", message=".*LeafSpec.*")
warnings.filterwarnings("ignore", message=".*module.*in eval mode.*")
warnings.filterwarnings("ignore", message=".*dataloader is not resumable.*")
warnings.filterwarnings("ignore", message=".*must be in.*range.*")

# ── Colab Drive mount (silent skip when not on Colab) ──────────
_is_colab = False
try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    _is_colab = True
except Exception:
    pass

# ── Repo detection ─────────────────────────────────────────────
if os.path.isdir(REPO_DIR) and os.path.isfile(os.path.join(REPO_DIR, "saicinpainting", "__init__.py")):
    pass
elif os.path.isfile(os.path.join(os.getcwd(), "saicinpainting", "__init__.py")):
    REPO_DIR = os.getcwd()
else:
    raise FileNotFoundError("Cannot find the lama repo. Set REPO_DIR or run from repo root.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ── NetCDF file resolution (real file or synthetic fallback) ───
_nc_file = NC_FILE_PATH
_is_synthetic = False

if not os.path.isfile(_nc_file):
    _is_synthetic = True
    log.info("NC file not found — will use in-memory SyntheticOceanDataset")

# Redirect Drive-based output to /tmp when running locally
if not _is_colab and OUTPUT_DIR.startswith("/content/drive"):
    OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "lama_nc_output")

# ════════════════════════════════════════════════════════════════
#  3. CONFIG — load from YAML, apply overrides
# ════════════════════════════════════════════════════════════════

from netcdf.config import load_nc_training_config, detect_precision

config = load_nc_training_config(os.path.join(REPO_DIR, "configs"))
_precision = detect_precision()

# ── Reproducibility ─────────────────────────────────────────────
if SEED is not None:
    import pytorch_lightning as pl

    pl.seed_everything(SEED, workers=True)
    log.info("Seed:           %d (deterministic mode)", SEED)
else:
    log.info("Seed:           None (non-deterministic)")

# ── Output directories ─────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
tb_dir = os.path.join(OUTPUT_DIR, "tb_logs")
checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
samples_dir = os.path.join(OUTPUT_DIR, "samples")
for d in (tb_dir, checkpoints_dir, samples_dir):
    os.makedirs(d, exist_ok=True)

# ── Dataset sizing (synthetic vs real) ─────────────────────────
if _is_synthetic:
    _n_samples_train = 400
    _n_samples_val = 80
    _run_title = "synthetic_demo"
    _synth_cfg = dict(
        nx=64,
        nz=64,
        n_ctd=5,
        n_cmems_x=8,
        noise_ctd=0.05,
        noise_cmems=0.10,
    )
else:
    _lat_indices = "${slice:48,126}"
    _lon_indices = "${slice:141,365}"
    _n_samples_train = 4836  # T=62 × lat=78
    _n_samples_val = 4836
    _run_title = "cmems_vertical_nc"

_steps_per_epoch = _n_samples_train // BATCH_SIZE

# ── Apply overrides ────────────────────────────────────────────
config.run_title = _run_title
config.training_model.visualize_each_iters = 999999
config.training_model.store_discr_outputs_for_vis = False
config.losses.adversarial.weight = ADVERSARIAL_WEIGHT
config.losses.feature_matching.weight = FEATURE_MATCHING_WEIGHT
config.losses.resnet_pl.weight = 0

if not _is_synthetic:
    config.data.nc.data.batch_size = BATCH_SIZE
    config.data.nc.data.num_workers = 4
    config.data.nc.data.dataset.filepaths = [_nc_file]
    config.data.nc.data.dataset.lat_indices = _lat_indices
    config.data.nc.data.dataset.lon_indices = _lon_indices

# Replace trainer kwargs entirely — the YAML has PL 1.x / DDP keys
# (gpus, accelerator=ddp, gradient_clip_val, …) that crash PL 2.6.
config.trainer.kwargs = OmegaConf.create({
    "accelerator": ACCELERATOR,
    "devices": DEVICES,
    "max_epochs": MAX_EPOCHS,
    "limit_train_batches": _steps_per_epoch,
    "val_check_interval": _steps_per_epoch,
    "limit_val_batches": LIMIT_VAL_BATCHES,
    "log_every_n_steps": 1,
    "precision": _precision,
    "num_sanity_val_steps": 0,
})
config.trainer.checkpoint_kwargs.pop("period", None)  # PL 2.6: use every_n_epochs

config.location.tb_dir = tb_dir
config.location.out_root_dir = OUTPUT_DIR
config.location.data_root_dir = os.path.dirname(_nc_file) if _nc_file else OUTPUT_DIR

config.visualizer.outdir = samples_dir
config.visualizer.key_order = ["image", "predicted_image", "inpainted"]
config.visualizer.rescale_keys = []

# NOTE: ${slice:...} returns Python slice objects that OmegaConf cannot store
# as node values.  Do NOT call OmegaConf.resolve(config) — resolution happens
# lazily when values are accessed.

log.info("─" * 50)
log.info("Mode:           %s", "SYNTHETIC (in-memory)" if _is_synthetic else "REAL DATA")
log.info("NC file:        %s", _nc_file if _nc_file else "N/A (in-memory)")
log.info("Output dir:     %s", OUTPUT_DIR)
log.info("Steps/epoch:    %s", _steps_per_epoch)
log.info("Max epochs:     %s", MAX_EPOCHS)
log.info("Total steps:    %s", MAX_EPOCHS * _steps_per_epoch)
log.info("Adversarial:    %s", "ON" if ADVERSARIAL_WEIGHT > 0 else "OFF (Phase 1)")

# ════════════════════════════════════════════════════════════════
#  4. MODEL
# ════════════════════════════════════════════════════════════════

from saicinpainting.training.trainers import make_training_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info("Device:         %s", device)
log.info("Precision:      %s (auto-detected for %s)", _precision, device)

model = make_training_model(config)

n_gen = sum(p.numel() for p in model.generator.parameters())
n_discr = sum(p.numel() for p in model.discriminator.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info("Generator:      %.1fM params", n_gen / 1e6)
log.info("Discriminator:  %.1fM params", n_discr / 1e6)
log.info("Trainable:      %.1fM params", n_train / 1e6)

# ════════════════════════════════════════════════════════════════
#  5. WEIGHTS — resume checkpoint > pretrained > scratch
# ════════════════════════════════════════════════════════════════

_resume_ckpt = RESUME_FROM_CHECKPOINT
if _resume_ckpt is None:
    _auto = os.path.join(checkpoints_dir, "last.ckpt")
    if os.path.isfile(_auto):
        _resume_ckpt = _auto

if _resume_ckpt is not None:
    log.info("Resume:         %s", _resume_ckpt)
elif PRETRAINED_CKPT and os.path.isfile(PRETRAINED_CKPT):
    log.info("Pretrained:     loading generator from %s", PRETRAINED_CKPT)
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
    gen_state = {
        k.removeprefix("generator."): v for k, v in ckpt["state_dict"].items() if k.startswith("generator.")
    }
    missing, unexpected = model.generator.load_state_dict(gen_state, strict=False)
    log.info("  Loaded %d keys, missing=%d, unexpected=%d", len(gen_state), len(missing), len(unexpected))
elif PRETRAINED_CKPT:
    log.warning("Pretrained checkpoint not found: %s", PRETRAINED_CKPT)
else:
    log.info("Weights:        training from scratch")

# ════════════════════════════════════════════════════════════════
#  6. TRAIN
# ════════════════════════════════════════════════════════════════

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Fixed version → all reruns write to the same TensorBoard event dir.
# PL restores global_step from checkpoint → x-axis is continuous.
metrics_logger = TensorBoardLogger(tb_dir, name="nc_training", version=TB_VERSION)

trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)

trainer = pl.Trainer(
    callbacks=[
        ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
    ],
    logger=metrics_logger,
    default_root_dir=OUTPUT_DIR,
    enable_model_summary=False,  # our log lines above replace the PL table
    **trainer_kwargs,
)

# ── Synthetic dataloaders (in-memory, no NetCDF) ──────────────
_train_dl = None
_val_dl = None

if _is_synthetic:
    from torch.utils.data import DataLoader
    from netcdf.data.synthetic_dataset import SyntheticOceanDataset

    _synth_seed = SEED if SEED is not None else 42
    train_ds = SyntheticOceanDataset(n=_n_samples_train, seed=_synth_seed, **_synth_cfg)
    val_ds = SyntheticOceanDataset(n=_n_samples_val, seed=_synth_seed + 100, **_synth_cfg)
    _train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    _val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    log.info("Synthetic train: %d samples, val: %d samples", len(train_ds), len(val_ds))

log.info("─" * 50)
log.info(
    "Starting: %d epochs × %d steps/epoch = %d total",
    MAX_EPOCHS,
    _steps_per_epoch,
    MAX_EPOCHS * _steps_per_epoch,
)
log.info("Checkpoints → %s", checkpoints_dir)
log.info("TensorBoard → %s", os.path.join(tb_dir, "nc_training", TB_VERSION))

trainer.fit(model, train_dataloaders=_train_dl, val_dataloaders=_val_dl, ckpt_path=_resume_ckpt)

log.info("─" * 50)
log.info("Training complete!")
log.info("Best checkpoint: %s", checkpoints_dir)
log.info("Launch TensorBoard:  %%tensorboard --logdir %s", os.path.join(tb_dir, "nc_training"))
