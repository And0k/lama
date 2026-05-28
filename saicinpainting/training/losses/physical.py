"""Domain-specific losses for oceanographic field inpainting.

Provides spatial regularization and physical constraint losses that
encourage physically plausible reconstructions of ocean fields.
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
