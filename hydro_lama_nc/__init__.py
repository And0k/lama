"""Hydro LAMA NC — standalone Hydro T+S ocean field inpainting.

Self-contained package extracted from the LaMa repository.
No dependency on saicinpainting/ — all models, losses, and metrics included.

Modules:
    model       — HydroGenerator (LaMa-lite FFC + T↔S cross-attention)
    losses      — Physics-motivated loss functions
    metrics     — PairwiseScore, SSIM, evaluation metrics
    data        — Synthetic datasets, NetCDF loading, preprocessing
    evaluation  — compute_metrics, optimal_interpolation, SSIM
    visualization — Plot functions for ocean fields
    utils       — Tensor padding, checkpoint management
    config      — Config composition from YAML
    resolvers   — OmegaConf custom resolvers
"""

from .model import HydroGenerator, LamaGenerator
from .evaluation import compute_metrics, optimal_interpolation, compute_ssim
from .utils import pad_to_modulo, next_modulo, find_best_checkpoint, cleanup_checkpoints
