#!/usr/bin/env python3
"""LaMa NetCDF Training — Google Colab or local
================================================
Copy this entire file into a single Jupyter notebook cell.

Prerequisites
-------------
1. Clone the repo and install deps in a Colab cell::

       !git clone <repo-url> /content/lama
       !pip install -r /content/lama/requirements.txt

2. Upload the CMEMS NetCDF file to Google Drive.

3. (Optional) Upload a pretrained big-lama checkpoint (``best.ckpt``).

Auto-resume
-----------
If ``{OUTPUT_DIR}/checkpoints/last.ckpt`` exists, training resumes
automatically from that checkpoint (global_step, optimizer state,
LR scheduler — everything).  Pretrained weights are only loaded on
the very first run when no checkpoint exists yet.

TensorBoard
-----------
All runs log to the **same** ``version`` directory so that metrics
appear on one continuous graph across restarts::

    %load_ext tensorboard
    %tensorboard --logdir {OUTPUT_DIR}/tb_logs/nc_training
"""

import logging
import os
import sys
import tempfile

# ╔══════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION — edit these before running                ║
# ╚══════════════════════════════════════════════════════════════╝

# Repo root: where the lama checkout lives.  On Colab this is /content/lama;
# when running locally it is auto-detected from the current working directory.
REPO_DIR = "/content/lama"

# Path to the CMEMS NetCDF file on Google Drive.
# If the file does not exist (Drive not mounted, running locally, etc.) the
# script falls back to a synthetic demo dataset (same as bin/demo_nc.py).
NC_FILE_PATH = (
    "/content/drive/MyDrive/"
    "cmems_mod_bal_phy_anfc_P1D-m_multi-vars_10.51E-21.32E_54.02N-58.97N_0.50-91.31m_2023-10-01-2023-12-01.nc"
)

# Set to the big-lama ``best.ckpt`` path to fine-tune from pretrained weights.
# Only used on the **first** run (when no previous checkpoint exists).
PRETRAINED_CKPT = None  # e.g. "/content/drive/MyDrive/big-lama/models/best.ckpt"

# Override: set explicitly to force a specific checkpoint path.
# When None, the script auto-detects ``{OUTPUT_DIR}/checkpoints/last.ckpt``.
RESUME_FROM_CHECKPOINT = None

OUTPUT_DIR = "/content/drive/MyDrive/lama_nc_output"

# Training schedule (overridden automatically when using synthetic fallback)
MAX_EPOCHS = 40
BATCH_SIZE = 2
STEPS_PER_EPOCH = 4836 // BATCH_SIZE  # 2418 for batch_size=2

# Phase 1: reconstruction-only (stable convergence, no GAN).
# After 5 epochs, restart with ADVERSARIAL_WEIGHT=10, FEATURE_MATCHING_WEIGHT=100.
ADVERSARIAL_WEIGHT = 0
FEATURE_MATCHING_WEIGHT = 0

# Trainer
PRECISION = "32"      # "16-mixed" for ~2× GPU speedup
ACCELERATOR = "auto"  # "gpu" if Colab runtime has GPU, else "cpu"
DEVICES = "auto"
LIMIT_VAL_BATCHES = 200  # validation samples per check (~4 min on GPU)

# Fixed TensorBoard version — all runs append to the same event directory
# so that metrics form one continuous graph across restarts.
TB_VERSION = "run0"

# ╔══════════════════════════════════════════════════════════════╗
# ║  2. ENVIRONMENT — Drive mount, sys.path, file resolution     ║
# ╚══════════════════════════════════════════════════════════════╝

_is_colab = False
try:
    from google.colab import drive  # type: ignore[import-untyped]

    drive.mount("/content/drive")
    _is_colab = True
except Exception:
    pass  # not on Colab, or mount failed — skip

# Detect repo root: prefer REPO_DIR if it exists, else cwd.
if os.path.isdir(REPO_DIR) and os.path.isfile(os.path.join(REPO_DIR, "saicinpainting", "__init__.py")):
    pass
elif os.path.isfile(os.path.join(os.getcwd(), "saicinpainting", "__init__.py")):
    REPO_DIR = os.getcwd()
else:
    raise FileNotFoundError(
        f"Cannot find the lama repo.  Set REPO_DIR at the top of this "
        f"script or run from the repo root.  Looked in: {REPO_DIR!r}, {os.getcwd()!r}"
    )

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# Resolve the NetCDF file: use NC_FILE_PATH if it exists, otherwise fall
# back to the same synthetic demo dataset that bin/demo_nc.py uses.
_nc_file = NC_FILE_PATH
_is_synthetic = False

if not os.path.isfile(_nc_file):
    from bin.demo_utils import _create_demo_nc

    _demo_dir = tempfile.mkdtemp(prefix="netcdf_train_")
    _nc_file = os.path.join(_demo_dir, "demo.nc")
    _create_demo_nc(_nc_file)
    _is_synthetic = True

# When running locally without the real file, write output to /tmp instead
# of a Google Drive path that does not exist.
if not _is_colab and OUTPUT_DIR.startswith("/content/drive"):
    OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "lama_nc_output")

# ╔══════════════════════════════════════════════════════════════╗
# ║  3. LOGGING + OMEGACONF RESOLVERS                            ║
# ╚══════════════════════════════════════════════════════════════╝

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_nc")

from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "slice",
    lambda *a: slice(*[None if x == "None" else int(x) for x in a]),
    replace=True,
)
OmegaConf.register_new_resolver("indices", lambda *a: [int(x) for x in a], replace=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  4. BUILD TRAINING CONFIG                                    ║
# ╚══════════════════════════════════════════════════════════════╝

# Auto-adjust dataset parameters for synthetic fallback.
# Synthetic file: T=4, D=24, H=96, W=128  →  slice_mode=["time","latitude"]
#   lat_indices=slice(10,50) → 40 values, lon_indices=None → 128
#   samples = 4 × 40 = 160,  sample shape = 24×128 (depth×lon)
if _is_synthetic:
    _lat_indices = "${slice:10,50}"
    _lon_indices = None
    _n_samples = 160  # T=4 × lat=40
    _run_title = "synthetic_demo"
else:
    _lat_indices = "${slice:48,126}"
    _lon_indices = "${slice:141,365}"
    _n_samples = 4836  # T=62 × lat=78
    _run_title = "cmems_vertical_nc"

_steps_per_epoch = _n_samples // BATCH_SIZE

os.makedirs(OUTPUT_DIR, exist_ok=True)
tb_dir = os.path.join(OUTPUT_DIR, "tb_logs")
checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
samples_dir = os.path.join(OUTPUT_DIR, "samples")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# ── Auto-detect previous checkpoint ────────────────────────────
_resume_ckpt = RESUME_FROM_CHECKPOINT
if _resume_ckpt is None:
    _auto = os.path.join(checkpoints_dir, "last.ckpt")
    if os.path.isfile(_auto):
        _resume_ckpt = _auto

config = OmegaConf.create(
    {
        "run_title": _run_title,
        # ── training model wrapper ──────────────────────────────
        "training_model": {
            "kind": "default",
            "visualize_each_iters": 500,
            "concat_mask": True,
            "store_discr_outputs_for_vis": False,
        },
        # ── losses ──────────────────────────────────────────────
        "losses": {
            "l1": {"weight_missing": 1.0, "weight_known": 10.0},
            "l2": {"weight_missing": 0.5, "weight_known": 5.0},
            "perceptual": {"weight": 0},
            "adversarial": {
                "kind": "r1",
                "weight": ADVERSARIAL_WEIGHT,
                "gp_coef": 0.001,
                "mask_as_fake_target": True,
                "allow_scale_mask": True,
            },
            "feature_matching": {"weight": FEATURE_MATCHING_WEIGHT},
            "resnet_pl": {"weight": 0},
            "domain": {
                "smoothness": {"weight": 1.0},
                "bounds": {
                    "weight": 0.5,
                    "bounds": [[0, 1], [0, 1], [0, 1]],
                },
                "gradient": {"weight": 0.1},
            },
        },
        # ── data ────────────────────────────────────────────────
        "data": {
            "nc": {
                "data": {
                    "batch_size": BATCH_SIZE,
                    "val_batch_size": 1,
                    "num_workers": 0,
                    "dataset": {
                        "_target_": "netcdf.data.dataset.NetCDFDataset",
                        "filepaths": [_nc_file],
                        "variables": ["|V|", "thetao", "so"],
                        "computed_variables": {
                            "|V|": {
                                "inputs": ["uo", "vo"],
                                "operation": "velocity_magnitude",
                            },
                        },
                        "scaling": {
                            "|V|": [0.0, 2.5],
                            "thetao": [-2.0, 22.0],
                            "so": [29.0, 36.0],
                            "uo": [-2.0, 2.0],
                            "vo": [-2.0, 2.0],
                        },
                        "time_indices": None,
                        "depth_indices": None,
                        "lat_indices": _lat_indices,
                        "lon_indices": _lon_indices,
                        "slice_mode": ["time", "latitude"],
                        "mask_variable": None,
                        "mask_generator": {
                            "type": "vertical",
                            "n_lines": 15,
                            "x_num": 20,
                            "y_num": 24,
                            "y_law": "log",
                            "seed": 42,
                        },
                        "coverage_threshold": 0.0,
                        "fill_ratio_threshold": 1.0,
                        "transform": {
                            "_target_": "netcdf.data.augment.get_netcdf_transforms",
                            "transform_variant": "light",
                            "flip_lat": True,
                            "flip_lon": False,
                            "noise_sigma": 0.01,
                            "depth_dropout": 0.1,
                            "crop_size": None,
                            "crop_p": 0.5,
                        },
                    },
                },
            },
        },
        # ── generator (FFC-ResNet, 75% global ratio) ────────────
        "generator": {
            "kind": "ffc_resnet",
            "input_nc": 4,
            "output_nc": 3,
            "ngf": 64,
            "n_downsampling": 3,
            "n_blocks": 9,
            "add_out_act": "sigmoid",
            "init_conv_kwargs": {
                "ratio_gin": 0,
                "ratio_gout": 0,
                "enable_lfu": False,
            },
            "downsample_conv_kwargs": {
                "ratio_gin": 0,
                "ratio_gout": 0,
                "enable_lfu": False,
            },
            "resnet_conv_kwargs": {
                "ratio_gin": 0.75,
                "ratio_gout": 0.75,
                "enable_lfu": False,
            },
        },
        # ── discriminator (PatchGAN) ────────────────────────────
        "discriminator": {
            "kind": "pix2pixhd_nlayer",
            "input_nc": 3,
            "ndf": 64,
            "n_layers": 4,
        },
        # ── optimizers ──────────────────────────────────────────
        "optimizers": {
            "generator": {
                "lr": 1e-4,
                "betas": [0.0, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
            },
            "discriminator": {
                "lr": 1e-4,
                "betas": [0.0, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
            },
        },
        # ── visualizer ──────────────────────────────────────────
        "visualizer": {
            "kind": "directory",
            "outdir": samples_dir,
            "key_order": ["image", "predicted_image", "inpainted"],
            "max_items_in_batch": 4,
            "last_without_mask": True,
            "rescale_keys": [],
        },
        # ── evaluator ───────────────────────────────────────────
        "evaluator": {
            "kind": "default",
            "ssim": True,
            "lpips": False,
            "fid": True,
            "inpainted_key": "inpainted",
            "integral_kind": "ssim_fid100_f1",
        },
        # ── PL Trainer ──────────────────────────────────────────
        "trainer": {
            "kwargs": {
                "accelerator": ACCELERATOR,
                "devices": DEVICES,
                "max_epochs": MAX_EPOCHS,
                "limit_train_batches": _steps_per_epoch,
                "val_check_interval": _steps_per_epoch,
                "limit_val_batches": LIMIT_VAL_BATCHES,
                "log_every_n_steps": 50,
                "precision": PRECISION,
                "check_val_every_n_epoch": 1,
                "num_sanity_val_steps": 0,
                # NOTE: gradient_clip_val is NOT supported with manual
                # optimization.  DefaultInpaintingTrainingModule handles
                # its own backward / optimizer steps.
            },
            "checkpoint_kwargs": {
                "verbose": True,
                "save_top_k": 5,
                "save_last": True,
                "monitor": "val_ssim_fid100_f1_total_mean",
                "mode": "max",
            },
        },
        # ── location ────────────────────────────────────────────
        "location": {
            "tb_dir": tb_dir,
            "out_root_dir": OUTPUT_DIR,
            "data_root_dir": os.path.dirname(_nc_file),
        },
    }
)

# NOTE: Do NOT call OmegaConf.resolve(config) — the ${slice:...} resolver
# returns Python slice objects that OmegaConf cannot store as node values.
# Resolution happens lazily when values are accessed (same as bin/train.py).

log.info("Mode:           %s", "SYNTHETIC DEMO" if _is_synthetic else "REAL DATA")
log.info("NC file:        %s", _nc_file)
log.info("Output dir:     %s", OUTPUT_DIR)
log.info("Steps/epoch:    %d", _steps_per_epoch)
log.info("Max epochs:     %d", MAX_EPOCHS)
log.info("Total steps:    %d", MAX_EPOCHS * _steps_per_epoch)
log.info("Adversarial:    %s", "ON" if ADVERSARIAL_WEIGHT > 0 else "OFF (Phase 1)")
log.info("Precision:      %s", PRECISION)

# ╔══════════════════════════════════════════════════════════════╗
# ║  5. CREATE MODEL                                             ║
# ╚══════════════════════════════════════════════════════════════╝

import torch
from saicinpainting.training.trainers import make_training_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", device)

model = make_training_model(config)

n_gen = sum(p.numel() for p in model.generator.parameters())
n_discr = sum(p.numel() for p in model.discriminator.parameters())
log.info("Generator:     %.1fM params", n_gen / 1e6)
log.info("Discriminator: %.1fM params", n_discr / 1e6)

# ╔══════════════════════════════════════════════════════════════╗
# ║  6. LOAD WEIGHTS                                             ║
# ║     Priority: resume checkpoint > pretrained > scratch       ║
# ╚══════════════════════════════════════════════════════════════╝

if _resume_ckpt is not None:
    log.info("Will resume from checkpoint: %s", _resume_ckpt)
    log.info("  (global_step, optimizer state, LR — everything restored)")
elif PRETRAINED_CKPT and os.path.isfile(PRETRAINED_CKPT):
    log.info("No previous checkpoint — loading pretrained generator from %s", PRETRAINED_CKPT)
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu", weights_only=False)
    gen_state = {
        k.replace("generator.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("generator.")
    }
    missing, unexpected = model.generator.load_state_dict(gen_state, strict=False)
    log.info(
        "Loaded %d generator keys. Missing: %d, Unexpected: %d",
        len(gen_state), len(missing), len(unexpected),
    )
    if missing:
        log.info("  Missing (first 5): %s", missing[:5])
    log.info("  FFC-ResNet blocks transferred; first/last conv will retrain.")
elif PRETRAINED_CKPT:
    log.warning("Pretrained checkpoint not found: %s — training from scratch", PRETRAINED_CKPT)
else:
    log.info("No checkpoint, no pretrained weights — training from scratch")

# ╔══════════════════════════════════════════════════════════════╗
# ║  7. TRAIN                                                    ║
# ╚══════════════════════════════════════════════════════════════╝

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Use a FIXED version so all reruns append to the same TensorBoard
# event directory.  PL restores global_step from the checkpoint,
# so the x-axis is continuous across restarts.
metrics_logger = TensorBoardLogger(tb_dir, name="nc_training", version=TB_VERSION)

# config.trainer.kwargs has no ${slice:...} interpolations so resolve=True is safe.
trainer_kwargs = OmegaConf.to_container(config.trainer.kwargs, resolve=True)
log.info("Trainer kwargs: %s", trainer_kwargs)

trainer = pl.Trainer(
    callbacks=[
        ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
    ],
    logger=metrics_logger,
    default_root_dir=OUTPUT_DIR,
    **trainer_kwargs,
)

log.info(
    "Starting training: %d epochs x %d steps/epoch = %d total steps",
    MAX_EPOCHS, _steps_per_epoch, MAX_EPOCHS * _steps_per_epoch,
)
log.info("Checkpoints -> %s", checkpoints_dir)
log.info("TensorBoard -> %s", os.path.join(tb_dir, "nc_training", TB_VERSION))

trainer.fit(model, ckpt_path=_resume_ckpt)

log.info("Training complete!")
log.info("Best checkpoint: %s", checkpoints_dir)
log.info("Launch TensorBoard:  %%tensorboard --logdir %s", os.path.join(tb_dir, "nc_training"))
