"""Domain-specific losses for oceanographic field inpainting.

Provides spatial regularization and physical constraint losses that
encourage physically plausible reconstructions of ocean fields.

Physics-motivated losses (ported from bin/lama_hydro_simple_end2end/train.py):
    hydrostatic_stability_loss: penalizes density inversions (∂ρ/∂z < 0)
    bottom_boundary_layer_loss: penalizes non-zero vertical gradient at bottom
    observation_weighted_mse: MSE weighted by observation mask + domain mask
"""

import torch
import torch.nn.functional as F


def smoothness_loss(pred, mask=None, weight=1.0):
    """Spatial smoothness loss using total variation.

    Penalizes large spatial gradients in the predicted field.
    Useful for oceanographic fields where physical quantities vary smoothly.

    Args:
        pred: (B, C, H, W) predicted field
        mask: (B, 1, H, W) optional binary mask (1 = masked region to ignore)
        weight: scalar multiplier

    Returns:
        Scalar loss
    """
    if mask is not None:
        effective = pred * (1 - mask)
    else:
        effective = pred

    grad_h = effective[:, :, 1:, :] - effective[:, :, :-1, :]
    grad_w = effective[:, :, :, 1:] - effective[:, :, :, :-1]
    return weight * (grad_h.abs().mean() + grad_w.abs().mean())


def physical_bounds_loss(pred, bounds, mask=None, weight=1.0):
    """Penalize predictions outside physically valid ranges.

    Each channel is checked against its (vmin, vmax) bounds.
    Loss is MSE of the violation (how far out of range).

    Args:
        pred: (B, C, H, W) predicted field in [0,1] scaled space
        bounds: list of (vmin, vmax) tuples, one per channel. Values are in
                the same [0,1] scaled space (i.e. already normalized).
                If bounds are in physical space, convert to [0,1] first.
        mask: (B, 1, H, W) optional binary mask (1 = masked region to ignore)
        weight: scalar multiplier

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

    return weight * total_loss


def gradient_consistency_loss(pred, target=None, mask=None, weight=1.0):
    """Laplacian-based gradient consistency loss.

    If target is provided, penalizes difference in Laplacian between pred and target.
    If target is None, penalizes the Laplacian magnitude (encourages smooth fields).

    Uses a 3x3 Laplacian kernel applied without padding (valid convolution)
    to avoid border artifacts:

        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]]

    Args:
        pred: (B, C, H, W) predicted field
        target: (B, C, H, W) optional target field
        mask: (B, 1, H, W) optional binary mask (1 = masked region to ignore)
        weight: scalar multiplier

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

    return weight * diff.mean()


# ── Physics-motivated losses from lama_hydro_simple_end2end ────────────


def hydrostatic_stability_loss(pred, channel_index=1, below_mask=None,
                                alpha=0.25, weight=1.0):
    """Penalize density inversions (hydrostatic instability).

    In a stably stratified ocean, density must increase with depth:
        ∂ρ/∂z ≥ 0.
    Using a linearized equation of state  ρ ≈ 1 - α·T_norm  (with T_norm
    in [0, 1]), this is equivalent to  ∂T/∂z ≤ 0.  Any positive vertical
    temperature gradient (warmer water below cooler water) indicates an
    unstable density inversion.

    Central finite difference is used for the vertical derivative (axis 2).

    Args:
        pred:         (B, C, H, W) predicted field in [0, 1].
        channel_index: int, index of the temperature channel in dim 1.
        below_mask:   (B, 1, H, W) float mask, 1.0 below bottom (outside
                      domain), 0.0 above.  If None, no masking is applied.
        alpha:        float, thermal expansion coefficient for linearized EOS.
                      Only affects the sign; the loss penalizes dT/dz > 0
                      regardless of alpha value.
        weight:       scalar multiplier.

    Returns:
        Scalar loss.
    """
    T = pred[:, channel_index:channel_index + 1, :, :]  # (B, 1, H, W)

    # Central difference: dT/dz at interior points
    dT_dz = T[:, :, 2:, :] - T[:, :, :-2, :]  # (B, 1, H-2, W)

    if below_mask is not None:
        domain = 1.0 - below_mask
        mask = domain[:, :, 2:, :] * domain[:, :, :-2, :]
    else:
        mask = 1.0

    # Only penalize positive dT/dz (warm water below cold = unstable)
    instability = F.relu(dT_dz)
    return weight * (instability * mask).mean()


def bottom_boundary_layer_loss(pred, bathy_indices, channel_index=1,
                                weight=1.0):
    """Penalize non-zero vertical temperature gradient near the bottom.

    In the bottom boundary layer (BBL), turbulent mixing homogenizes
    temperature, so dT/dz → 0 at the seafloor.  This loss penalizes
    the squared difference between the temperature at the last two
    grid cells above the bottom.

    Args:
        pred:          (B, C, H, W) predicted field in [0, 1].
        bathy_indices: (B, W) int tensor of bottom depth indices (in grid
                       cells).  Values ≤ 1 are skipped (too shallow).
        channel_index: int, index of the temperature channel in dim 1.
        weight:        scalar multiplier.

    Returns:
        Scalar loss.
    """
    T = pred[:, channel_index]  # (B, H, W)
    B, H, W = T.shape

    total = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    count = 0

    for bi in range(B):
        for xi in range(W):
            zi = int(bathy_indices[bi, xi].item()) - 2
            if 1 < zi < H - 1:
                dT = T[bi, zi, xi] - T[bi, zi - 1, xi]
                total = total + dT ** 2
                count += 1

    return weight * total / max(count, 1)


def observation_weighted_mse(pred, target, obs_mask, below_mask=None,
                              weight_known=1.0, weight_domain=0.1,
                              weight=1.0):
    """MSE weighted by observation and domain masks.

    Two-term loss:
        L = weight_known  · MSE(pred, target) in observed pixels
          + weight_domain · MSE(pred, target) in domain (above bottom)

    Args:
        pred:         (B, C, H, W) predicted field.
        target:       (B, C, H, W) target field.
        obs_mask:     (B, 1, H, W) float mask, 1.0 at observation locations
                      (CTD + CMEMS), 0.0 elsewhere.
        below_mask:   (B, 1, H, W) float mask, 1.0 below bottom (outside
                      domain), 0.0 above.  If None, domain = 1 everywhere.
        weight_known: multiplier for the observation-term MSE.
        weight_domain: multiplier for the domain-term MSE.
        weight:       scalar multiplier for the total.

    Returns:
        Scalar loss.
    """
    if below_mask is not None:
        domain = 1.0 - below_mask
    else:
        domain = torch.ones_like(obs_mask)

    l_obs = F.mse_loss(pred * obs_mask, target * obs_mask)
    l_dom = F.mse_loss(pred * domain, target * domain)

    return weight * (weight_known * l_obs + weight_domain * l_dom)
