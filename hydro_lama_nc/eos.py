"""Equation of State for Baltic Sea — single source of truth.

All density-related physics lives here.  Other modules import from
this file instead of hardcoding coefficients or formulas.

Linearized EOS:  ρ = 1 − α·T + β·S   with T, S ∈ [0, 1].

Coefficients calibrated for normalised fields so that β/α = 3.75
(haline-dominated, matching Baltic Sea dynamics).
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Canonical coefficients ────────────────────────────────────────────
ALPHA: float = 0.04
BETA: float = 0.15


# ── Forward: T, S → ρ ────────────────────────────────────────────────

def linearized_density(
    T: Union[np.ndarray, torch.Tensor],
    S: Union[np.ndarray, torch.Tensor],
    alpha: float = ALPHA,
    beta: float = BETA,
) -> Union[np.ndarray, torch.Tensor]:
    """ρ = 1 − α·T + β·S.

    Works with both NumPy arrays and PyTorch tensors.

    Args:
        T: Temperature, any shape.
        S: Salinity, same shape as T.
        alpha: Thermal expansion coefficient.
        beta: Haline contraction coefficient.

    Returns:
        Density, same shape and type as inputs.
    """
    return 1.0 - alpha * T + beta * S


# ── Gradient helpers (Torch) ──────────────────────────────────────────

def density_gradient(
    T: torch.Tensor,
    S: torch.Tensor,
    dim: int,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> torch.Tensor:
    """∂ρ/∂x_dim = −α·∂T/∂x_dim + β·∂S/∂x_dim.

    Args:
        T: (B, 1, H, W) temperature tensor.
        S: (B, 1, H, W) salinity tensor.
        dim: spatial dimension to differentiate (2=z, 3=x).
        alpha: Thermal expansion coefficient.
        beta: Haline contraction coefficient.

    Returns:
        Density gradient tensor.
    """
    return -alpha * torch.gradient(T, dim=dim)[0] \
           + beta * torch.gradient(S, dim=dim)[0]


# ── Inverse: ρ, T → S ────────────────────────────────────────────────

def salinity_from_density(
    rho: Union[np.ndarray, float],
    T: Union[np.ndarray, float],
    alpha: float = ALPHA,
    beta: float = BETA,
) -> Union[np.ndarray, float]:
    """S = (ρ − 1 + α·T) / β.  Inverse of linearized_density w.r.t. S."""
    return (rho - 1.0 + alpha * T) / beta


# ── Interface stability constraint ────────────────────────────────────

def min_salinity_amplitude(
    A_T: Union[np.ndarray, float],
    min_drho: float = 0.005,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> Union[np.ndarray, float]:
    """Minimum S amplitude for a stable density interface.

    For a tanh interface with T-amplitude A_T, the net density jump is
    −α·A_T + β·A_S.  For stable stratification (jump > 0):

        A_S ≥ (α·A_T + min_drho) / β

    Args:
        A_T: Temperature interface amplitude(s).
        min_drho: Minimum density jump (positive).
        alpha: Thermal expansion coefficient.
        beta: Haline contraction coefficient.

    Returns:
        Minimum A_S ensuring stability.
    """
    return (alpha * A_T + min_drho) / beta
