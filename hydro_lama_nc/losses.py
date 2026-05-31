"""Three-loss pipeline for oceanographic field inpainting.

Replaces the previous 7+1 loss design with three physically-motivated losses,
all under Kendall & Gal uncertainty weighting with clamps:

    [0] observation_loss  — value following + structural gradient matching
    [1] physics_loss      — density inversions (stratification constraint)
    [2] geostrophic_loss  — thermal wind balance

All loss functions return raw (unweighted) scalar tensors.  External
weights (Kendall & Gal log_vars) are applied at the call site.
"""

import logging

import torch
import torch.nn.functional as F

from .eos import linearized_density, density_gradient, ALPHA, BETA

logger = logging.getLogger(__name__)


def observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw,
                     sigma_ctd=0.05, sigma_cmems=0.10, sigma_bg=0.25):
    """Combined observation loss: value + structure.

    Value loss: weighted MSE at observation points (CTD + CMEMS).
    Structure loss: gradient matching at CMEMS points — CMEMS values are biased
    (smoothed+shifted) but gradients partially preserve real structure
    (thermocline position, front orientation).

    Args:
        pred:       (B, 2, nz, nx) predicted [T, S]
        target:     (B, 2, nz, nx) ground truth [T, S]
        mask_ctd:   (B, 1, nz, nx) CTD mask (sparse columns)
        mask_cmems: (B, 1, nz, nx) CMEMS observation mask (sparse grid)
        cmems_raw:  (B, 2, nz, nx) CMEMS background field (smoothed+shifted)
        sigma_ctd:  float, CTD measurement noise σ
        sigma_cmems: float, CMEMS measurement noise σ
        sigma_bg:   float, background (no obs) uncertainty σ

    Returns:
        Scalar loss.
    """
    obs_mask = (mask_ctd + mask_cmems).clamp(0, 1)

    sigma_map = (sigma_ctd * mask_ctd + sigma_cmems * mask_cmems
                 + sigma_bg * (1 - obs_mask).clamp(0, 1))

    sigma2 = sigma_map.pow(2).clamp(min=1e-4)

    mse = (pred - target).pow(2) / sigma2
    obs_bool = obs_mask.expand_as(pred).bool()
    value_loss = mse[obs_bool].mean()

    grad_pred = torch.gradient(pred, dim=2)[0]
    grad_cmems = torch.gradient(cmems_raw, dim=2)[0]
    structure_loss = F.l1_loss(grad_pred * mask_cmems, grad_cmems * mask_cmems)

    return value_loss + 0.3 * structure_loss


def physics_loss(pred, mask_below=None):
    """Consolidated physics: density must increase with depth.

    Penalizes density inversions in the model output, normalized by the
    typical density gradient scale so the loss magnitude is comparable to
    the other losses.

    Applied everywhere (including CTD columns).  observation_loss
    gradient at CTD is ~100× stronger, so physics cannot over-smooth
    CTD data — it only prevents the model from amplifying noise.

    Args:
        pred:        (B, 2, nz, nx) predicted [T, S] in [0,1]
        mask_below:  (B, 1, nz, nx) 1.0 below bottom

    Returns:
        Scalar loss.
    """
    T, S = pred[:, :1], pred[:, 1:]

    rho = linearized_density(T, S)

    drho_dz = rho[:, :, 2:, :] - rho[:, :, :-2, :]

    domain = torch.ones_like(drho_dz)
    if mask_below is not None:
        mask = (1 - mask_below[:, :, 2:, :]) * (1 - mask_below[:, :, :-2, :])
        domain = domain * mask

    scale = drho_dz[domain.bool()].std().detach().clamp(min=1e-6)
    violations = F.relu(-drho_dz) * domain

    return (violations / scale).pow(2).mean()


def geostrophic_loss(pred_T, pred_S, u, v):
    """Thermal wind balance: ∂v/∂z ∝ +∂ρ/∂x.

    In a 2D (x, z) cross-section, only the v-equation is computable:
        ∂v/∂z = (g / fρ₀) · ∂ρ/∂x
    The u-equation requires ∂ρ/∂y which doesn't exist in 2D.

    Both sides normalised to unit std (pattern matching, not magnitude).
    Loss = MSE of (dv/dz_norm − dρ/dx_norm) — wants them equal.

    Args:
        pred_T: (B, 1, nz, nx) predicted temperature in [0,1]
        pred_S: (B, 1, nz, nx) predicted salinity in [0,1]
        u:      (B, 1, nz, nx) zonal velocity (unused in 2D)
        v:      (B, 1, nz, nx) meridional velocity

    Returns:
        Scalar loss.
    """
    drho_dx = density_gradient(pred_T, pred_S, dim=3)
    dv_dz = torch.gradient(v, dim=2)[0]

    h = min(drho_dx.shape[2], dv_dz.shape[2])
    w = min(drho_dx.shape[3], dv_dz.shape[3])

    drho = drho_dx[:, :, :h, :w]
    dv = dv_dz[:, :, :h, :w]

    drho_n = drho / (drho.std() + 1e-6)
    dv_n = dv / (dv.std() + 1e-6)

    return ((drho_n - dv_n).pow(2)).mean()
