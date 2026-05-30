"""Domain-specific losses for oceanographic field inpainting.

Provides spatial regularization and physical constraint losses that
encourage physically plausible reconstructions of ocean fields.

Physics-motivated losses (ported from bin/lama_hydro_simple_end2end/hydro_attention.py):
    hydrostatic_stability_loss: penalizes density inversions (∂ρ/∂z < 0)
    bottom_boundary_layer_loss: penalizes non-zero vertical gradient at bottom
    observation_weighted_mse: MSE weighted by observation mask + domain mask
    geostrophic_balance_loss: thermal wind balance ∂u/∂z ~ -∂ρ/∂x
    inverse_variance_mse: inverse-variance (sigma) weighted MSE at observation points

All loss functions return raw (unweighted) scalar tensors.  External
weights are applied at the call site in the loss composition, not inside
the loss function.  This keeps the loss definitions pure physics and
makes weight tuning visible in one place (config or training script).
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def linearized_density(T, S, alpha=0.20, beta=0.08):
    """Linearized equation of state for Baltic Sea.

    ρ ≈ 1 - α·T + β·S  with T, S in [0, 1].

    Args:
        T: Temperature tensor, any shape.
        S: Salinity tensor, same shape as T.
        alpha: Thermal expansion coefficient.
        beta: Haline contraction coefficient.

    Returns:
        Density tensor, same shape as T.
    """
    return 1.0 - alpha * T + beta * S


def smoothness_loss(pred, mask=None):
    """Spatial smoothness loss using total variation.

    Args:
        pred: (B, C, H, W) predicted field
        mask: (B, 1, H, W) binary mask, 1.0 below bottom (ignored in loss)

    Returns:
        Scalar loss
    """
    if mask is not None:
        effective = pred * (1 - mask)
    else:
        effective = pred

    grad_h = effective[:, :, 1:, :] - effective[:, :, :-1, :]
    grad_w = effective[:, :, :, 1:] - effective[:, :, :, :-1]
    return (grad_h.abs().mean() + grad_w.abs().mean()) * 0.5


def physical_bounds_loss(pred, bounds, mask=None):
    """Penalize predictions outside physically valid ranges.

    Args:
        pred: (B, C, H, W) predicted field in [0,1] scaled space
        bounds: list of (vmin, vmax) tuples, one per channel
        mask: (B, 1, H, W) binary mask, 1.0 below bottom (ignored in loss)

    Returns:
        Scalar loss
    """
    total_loss = 0.0
    n_channels = pred.shape[1]

    for c in range(min(n_channels, len(bounds))):
        vmin, vmax = bounds[c]
        ch = pred[:, c:c+1, :, :]

        violation_lo = F.relu(vmin - ch)
        violation_hi = F.relu(ch - vmax)
        violation = violation_lo + violation_hi

        if mask is not None:
            violation = violation * (1 - mask)

        total_loss = total_loss + violation.pow(2).mean()

    return total_loss


def gradient_consistency_loss(pred, target=None, mask=None):
    """Laplacian-based gradient consistency loss.

    Args:
        pred: (B, C, H, W) predicted field
        target: (B, C, H, W) optional target field
        mask: (B, 1, H, W) binary mask, 1.0 below bottom (ignored in loss)

    Returns:
        Scalar loss
    """
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=pred.dtype, device=pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    channels = pred.shape[1]
    kernel = kernel.expand(channels, 1, 3, 3).contiguous()

    lap_pred = F.conv2d(pred, kernel, padding=0, groups=channels)

    if target is not None:
        lap_target = F.conv2d(target, kernel, padding=0, groups=channels)
        diff = (lap_pred - lap_target).pow(2)
    else:
        diff = lap_pred.pow(2)

    if mask is not None:
        mask_cropped = mask[:, :, 1:-1, 1:-1]
        diff = diff * (1 - mask_cropped)

    return diff.mean()


# ── Physics-motivated losses from lama_hydro_simple_end2end ────────────


def hydrostatic_stability_loss(pred, channel_index=1, below_mask=None,
                                alpha=0.20, beta=0.08,
                                salinity_index=None):
    """Penalize density inversions (hydrostatic instability).

    In a stably stratified ocean, density must increase with depth:
        ∂ρ/∂z ≥ 0.
    Using a linearized equation of state  ρ ≈ 1 - α·T + β·S, instability
    occurs when  α·∂T/∂z - β·∂S/∂z > 0.  Only penalizes the violating
    part via ReLU.

    When salinity_index is None, falls back to single-channel T-only EOS
    (backward compatible with T-only pipeline).

    Args:
        pred:          (B, C, H, W) predicted field in [0, 1].
        channel_index: int, index of the temperature channel in dim 1.
        below_mask:    (B, 1, H, W) float mask, 1.0 below bottom.
        alpha:         float, thermal expansion coefficient.
        beta:          float, haline contraction coefficient.
        salinity_index: int or None. If provided, use full T+S EOS.
                        If None, use T-only (ρ ≈ 1 - α·T).

    Returns:
        Scalar loss.
    """
    T = pred[:, channel_index:channel_index + 1, :, :]  # (B, 1, H, W)

    if salinity_index is not None:
        S = pred[:, salinity_index:salinity_index + 1, :, :]
        rho = linearized_density(T, S, alpha=alpha, beta=beta)
    else:
        rho = 1.0 - alpha * T

    drho_dz = rho[:, :, 2:, :] - rho[:, :, :-2, :]  # (B, 1, H-2, W)

    if below_mask is not None:
        domain = 1.0 - below_mask
        mask = domain[:, :, 2:, :] * domain[:, :, :-2, :]
    else:
        mask = 1.0

    unstable = F.relu(-drho_dz)
    return (unstable * mask).mean()


def bottom_boundary_layer_loss(pred, bathy_indices, channel_index=0,
                                salinity_index=None):
    """Penalize non-zero vertical gradient of T (and optionally S) near the seabed.

    Physically: turbulent mixing in the BBL homogenizes T and S →
    ∂T/∂z → 0, ∂S/∂z → 0 at the bottom.

    Uses torch.gather for differentiable indexing — no Python loops,
    no .item() scalar extraction.

    Args:
        pred:          (B, C, H, W) predicted field, requires_grad=True.
        bathy_indices: (B, W) int tensor of bottom depth indices.
        channel_index: int, index of the temperature channel.
        salinity_index: int or None. If provided, also penalize ∂S/∂z.

    Returns:
        Scalar loss (differentiable).
    """
    B, C, H, W = pred.shape

    zi = (bathy_indices.long() - 2).clamp(1, H - 2)
    valid = (zi > 1) & (zi < H - 1)

    zi_exp = zi.unsqueeze(1).unsqueeze(2).expand(B, C, 1, W)
    zi1_exp = (zi - 1).clamp(0).unsqueeze(1).unsqueeze(2).expand(B, C, 1, W)

    val_bot = pred.gather(2, zi_exp).squeeze(2)
    val_above = pred.gather(2, zi1_exp).squeeze(2)

    vf = valid.float()
    dT = val_bot[:, channel_index, :] - val_above[:, channel_index, :]
    loss = (dT ** 2 * vf).sum()
    count = vf.sum().clamp(min=1)

    if salinity_index is not None:
        dS = val_bot[:, salinity_index, :] - val_above[:, salinity_index, :]
        loss = loss + (dS ** 2 * vf).sum()
        count = count * 2

    return loss / count


def observation_weighted_mse(pred, target, obs_mask, below_mask=None,
                              weight_known=1.0, weight_domain=0.1):
    """MSE weighted by observation and domain masks.

    weight_known and weight_domain are formula-internal: they control the
    relative contribution of observed vs domain points within the loss.
    External weighting is applied by the caller.

    Args:
        pred:         (B, C, H, W) predicted field.
        target:       (B, C, H, W) target field.
        obs_mask:     (B, 1, H, W) float mask, 1.0 at observation locations.
        below_mask:   (B, 1, H, W) float mask, 1.0 below bottom.
        weight_known: multiplier for the observation-term MSE (formula-internal).
        weight_domain: multiplier for the domain-term MSE (formula-internal).

    Returns:
        Scalar loss.
    """
    if below_mask is not None:
        domain = 1.0 - below_mask
    else:
        domain = torch.ones_like(obs_mask)

    l_obs = F.mse_loss(pred * obs_mask, target * obs_mask)
    l_dom = F.mse_loss(pred * domain, target * domain)

    return weight_known * l_obs + weight_domain * l_dom


def inverse_variance_mse(pred, target, sigma, obs_mask, below_mask=None,
                          weight_domain=0.1):
    """Inverse-variance weighted MSE at observation points.

    Points with lower sigma (e.g. CTD, sigma=0.05) get higher weight
    than points with higher sigma (e.g. CMEMS, sigma=0.10).
    weight = 1 / (sigma + eps)^2  — maximum likelihood under Gaussian noise.

    L = mean(weight * (pred - target)^2) at observed points
      + weight_domain * MSE(pred, target) in domain

    weight_domain is formula-internal (controls domain contribution).
    External weighting is applied by the caller.

    Args:
        pred:       (B, C, H, W) predicted field.
        target:     (B, C, H, W) target field.
        sigma:      (B, 1, H, W) per-point uncertainty map.
        obs_mask:   (B, 1, H, W) binary mask, 1.0 at observations.
        below_mask: (B, 1, H, W) binary mask, 1.0 below bottom.
        weight_domain: multiplier for the domain MSE term (formula-internal).

    Returns:
        Scalar loss.
    """
    if below_mask is not None:
        domain = 1.0 - below_mask
    else:
        domain = torch.ones_like(obs_mask)

    inv_var = obs_mask / (sigma.clamp(min=0.01).pow(2))
    l_obs = (inv_var * (pred - target).pow(2)).mean()
    l_dom = F.mse_loss(pred * domain, target * domain)

    return l_obs + weight_domain * l_dom


def geostrophic_balance_loss(pred, u_input, below_mask=None):
    """Thermal wind balance on 2D (x, z) slice.

    Geostrophic relation:  ∂u/∂z = -(g / f·ρ₀) · ∂ρ/∂x
    Penalize mismatch between predicted ∂ρ/∂x and observed ∂u/∂z.
    Both terms are normalized to unit standard deviation before comparing
    so that magnitude differences don't dominate.

    Args:
        pred:       (B, 2, H, W) predicted [T, S] in [0, 1].
        u_input:    (B, 1, H, W) zonal velocity input (LR upsampled).
        below_mask: (B, 1, H, W) binary mask, 1.0 below bottom.

    Returns:
        Scalar loss.
    """
    T, S = pred[:, :1], pred[:, 1:]
    rho = linearized_density(T, S)

    if below_mask is not None:
        domain = 1.0 - below_mask
    else:
        domain = torch.ones_like(T)

    drho_dx = rho[:, :, :, 2:] - rho[:, :, :, :-2]   # (B, 1, H, W-2)
    du_dz = u_input[:, :, 2:, :] - u_input[:, :, :-2, :]  # (B, 1, H-2, W)

    h_min = min(drho_dx.shape[2], du_dz.shape[2])
    w_min = min(drho_dx.shape[3], du_dz.shape[3])
    d = drho_dx[:, :, :h_min, :w_min]
    u_shear = du_dz[:, :, :h_min, :w_min]
    m = domain[:, :, :h_min, :w_min]

    d_n = d / (d.std() + 1e-6)
    u_n = u_shear / (u_shear.std() + 1e-6)

    return (m * (d_n + u_n).pow(2)).mean()
