"""Evaluation metrics for NetCDF inpainting results.

Includes standard image-quality metrics (SSIM, RMSE, correlation) and
physics-motivated metrics for oceanographic field reconstruction:
    - StabilityViolationScore: fraction of columns with density inversions
    - BBLGradientScore: mean |dT/dz| near the bottom (should → 0)
    - DomainRMSEScore: RMSE computed only in valid (above-bottom) pixels
    - OIBaselineScore: RMSE of OI interpolation for comparison

Utility functions (used by training scripts and visualization):
    - compute_metrics: MSE/MAE/RMSE/PSNR in ocean domain
    - optimal_interpolation: cubic OI from sparse observations
"""

import logging

import numpy as np
import torch
from scipy.interpolate import griddata

from hydro_lama_nc.metrics.base import PairwiseScore

logger = logging.getLogger(__name__)


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    below: np.ndarray) -> dict:
    """MSE, MAE, RMSE, PSNR in ocean domain only.

    Args:
        pred:   (H, W) predicted field.
        target: (H, W) target field.
        below:  (H, W) bool, True = below bottom (excluded).

    Returns:
        Dict with keys: mse, mae, rmse, psnr.
    """
    mask = ~below
    if mask.sum() == 0:
        return dict(mse=np.nan, mae=np.nan, rmse=np.nan, psnr=np.nan)
    p, t = pred[mask], target[mask]
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))
    psnr = float(20 * np.log10(1.0 / (rmse + 1e-8)))
    return dict(mse=mse, mae=mae, rmse=rmse, psnr=psnr)


def optimal_interpolation(inp_np: np.ndarray, obs_mask_np: np.ndarray,
                          nz: int, nx: int) -> tuple:
    """Cubic interpolation from observed points onto full grid.

    Args:
        inp_np:      (2, nz, nx) [T_obs, S_obs] observed values.
        obs_mask_np: (1, nz, nx) observation mask (1.0 = observed).
        nz:          Number of depth levels.
        nx:          Number of horizontal points.

    Returns:
        (T_oi, S_oi) each (nz, nx) float32.
    """
    mask = obs_mask_np[0]
    pts = np.argwhere(mask > 0.5)
    if len(pts) < 4:
        return (np.zeros((nz, nx), np.float32),
                np.zeros((nz, nx), np.float32))
    zi_all, xi_all = np.arange(nz), np.arange(nx)
    ZI, XI = np.meshgrid(zi_all, xi_all, indexing="ij")
    grid_coords = np.stack([ZI.ravel(), XI.ravel()], axis=1)

    T_oi = griddata(pts, inp_np[0][pts[:, 0], pts[:, 1]],
                    grid_coords, method="cubic", fill_value=0.0)
    S_oi = griddata(pts, inp_np[1][pts[:, 0], pts[:, 1]],
                    grid_coords, method="cubic", fill_value=0.0)
    return (T_oi.reshape(nz, nx).astype(np.float32),
            S_oi.reshape(nz, nx).astype(np.float32))


def compute_ssim(
    orig: np.ndarray,
    inpainted: np.ndarray,
    window_size: int = 11,
) -> float:
    """Compute SSIM between two single-channel (H, W) numpy arrays in [0, 1].

    Args:
        orig: Original image array.
        inpainted: Inpainted image array.
        window_size: SSIM window size.

    Returns:
        SSIM value in [0, 1].
    """
    from hydro_lama_nc.metrics.ssim import SSIM

    ssim_fn = SSIM(window_size=window_size, size_average=True).eval()
    t1 = torch.from_numpy(orig).float().unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(inpainted).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return ssim_fn(t1, t2).item()


def resolve_vel_scaling(
    scaling: dict,
    uo_key: str = "uo",
    vo_key: str = "vo",
    vel_key: str = "|V|",
) -> tuple[float, float]:
    """Compute |V| scaling from uo/vo ranges if not explicitly configured.

    Args:
        scaling: Dict mapping variable names to (min, max) tuples.
        uo_key: Key for eastward velocity.
        vo_key: Key for northward velocity.
        vel_key: Key for velocity magnitude.

    Returns:
        (vmin, vmax) for velocity magnitude.
    """
    vel_cfg = scaling.get(vel_key)
    if vel_cfg is None or vel_cfg[0] is None:
        uo_min, uo_max = scaling[uo_key]
        vo_min, vo_max = scaling[vo_key]
        vmax = np.sqrt(max(abs(uo_min), uo_max) ** 2 + max(abs(vo_min), vo_max) ** 2)
        return (0.0, float(vmax))
    return tuple(vel_cfg)


class RMSEScore(PairwiseScore):
    """Per-sample RMSE between prediction and target."""

    def __init__(self):
        super().__init__()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        pred = pred_batch.detach().cpu()
        target = target_batch.detach().cpu()
        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()
        batch_values = torch.sqrt((pred_flat - target_flat).pow(2).mean(dim=-1)).numpy()
        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values


class CorrelationScore(PairwiseScore):
    """Per-sample Pearson correlation between prediction and target."""

    def __init__(self):
        super().__init__()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        pred = pred_batch.detach().cpu().numpy()
        target = target_batch.detach().cpu().numpy()
        B = pred.shape[0]
        batch_values = np.empty(B)
        for i in range(B):
            p = pred[i].flatten()
            t = target[i].flatten()
            batch_values[i] = np.corrcoef(p, t)[0, 1]
        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values


# ── Physics-motivated metrics ──────────────────────────────────────────


class StabilityViolationScore(PairwiseScore):
    """Fraction of water columns with density inversions.

    A density inversion occurs when warmer water underlies cooler water
    (∂T/∂z > 0 in normalised [0,1] space), indicating hydrostatic
    instability.  The score is the fraction of columns with at least one
    inversion — lower is better (0 = fully stable).

    Operates on a single channel (default: channel 1 = thetao).

    Args:
        channel_index: which channel of the (B, C, H, W) tensor to check.
    """

    def __init__(self, channel_index: int = 1):
        super().__init__()
        self.channel_index = channel_index
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        # pred_batch: (B, C, H, W)
        T = pred_batch[:, self.channel_index]  # (B, H, W)
        B, H, W = T.shape

        # Central difference dT/dz (axis=1 is vertical)
        dT_dz = T[:, 2:, :] - T[:, :-2, :]  # (B, H-2, W)

        # Count columns with any inversion (dT/dz > 0)
        inversions_per_col = (dT_dz > 0).any(dim=1)  # (B, W) — True if column has inversion
        violation_fraction = inversions_per_col.float().mean(dim=1)  # (B,)

        batch_values = violation_fraction.numpy()
        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values


class BBLGradientScore(PairwiseScore):
    """Mean |dT/dz| in the bottom two cells — measures BBL homogenisation.

    In a well-mixed bottom boundary layer, dT/dz → 0 near the seafloor.
    This metric measures the mean absolute vertical gradient in the
    bottom 10% of the water column — lower is better.

    Operates on a single channel (default: channel 1 = thetao).

    Args:
        channel_index: which channel to measure.
        bottom_fraction: fraction of H to treat as BBL (default 0.1).
    """

    def __init__(self, channel_index: int = 1, bottom_fraction: float = 0.1):
        super().__init__()
        self.channel_index = channel_index
        self.bottom_fraction = bottom_fraction
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        T = pred_batch[:, self.channel_index]  # (B, H, W)
        B, H, W = T.shape

        bbl_depth = max(1, int(H * self.bottom_fraction))
        T_bbl = T[:, -bbl_depth:, :]  # (B, bbl_depth, W)

        if bbl_depth < 2:
            batch_values = np.zeros(B)
        else:
            dT = T_bbl[:, 1:, :] - T_bbl[:, :-1, :]  # (B, bbl_depth-1, W)
            mean_grad = dT.abs().mean(dim=(1, 2))  # (B,)
            batch_values = mean_grad.numpy()

        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values


class DomainRMSEScore(PairwiseScore):
    """RMSE computed only in valid (above-bottom) pixels.

    If a fill_mask is provided (1.0 = below bottom / invalid), only
    pixels where fill_mask == 0 are included.  Otherwise falls back
    to global RMSE.

    Args:
        channel_index: which channel to evaluate (default 0 = all).
    """

    def __init__(self, channel_index: int = -1):
        super().__init__()
        self.channel_index = channel_index
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        pred = pred_batch.detach().cpu()
        target = target_batch.detach().cpu()

        if self.channel_index >= 0:
            pred = pred[:, self.channel_index]
            target = target[:, self.channel_index]

        B = pred.shape[0]
        pred_flat = pred.reshape(B, -1).float()
        target_flat = target.reshape(B, -1).float()

        if mask is not None:
            # mask: (B, 1, H, W) or (B, C, H, W) — 1.0 = invalid
            m = mask.detach().cpu().reshape(B, -1).float()
            valid = (m < 0.5).float()  # (B, N) — 1.0 where valid
            sq_err = (pred_flat - target_flat).pow(2) * valid
            count = valid.sum(dim=1).clamp(min=1)
            batch_values = torch.sqrt(sq_err.sum(dim=1) / count).numpy()
        else:
            batch_values = torch.sqrt((pred_flat - target_flat).pow(2).mean(dim=1)).numpy()

        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values


class OIBaselineScore(PairwiseScore):
    """RMSE of optimal interpolation baseline for comparison.

    Runs scipy.griddata (cubic) on observed points and computes RMSE
    against the target.  This is a reference score, not a model metric.

    Args:
        obs_mask_key: key in batch to find observation mask.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def forward(self, pred_batch, target_batch, mask=None):
        from scipy.interpolate import griddata

        target = target_batch.detach().cpu().numpy()
        B = target.shape[0]
        batch_values = np.empty(B)

        for i in range(B):
            tgt = target[i, 0] if target.ndim == 4 else target[i]  # (H, W)
            H, W = tgt.shape

            if mask is not None:
                m = mask[i, 0].detach().cpu().numpy() if mask.ndim == 4 else mask[i].detach().cpu().numpy()
                obs_pts = np.argwhere(m > 0.5)
            else:
                # If no mask, OI = target (perfect)
                batch_values[i] = 0.0
                continue

            if len(obs_pts) < 4:
                batch_values[i] = float('nan')
                continue

            vals = tgt[obs_pts[:, 0], obs_pts[:, 1]]
            zi_all = np.arange(H)
            xi_all = np.arange(W)
            ZI, XI = np.meshgrid(zi_all, xi_all, indexing='ij')
            grid = np.stack([ZI.ravel(), XI.ravel()], axis=1)

            interp = griddata(obs_pts, vals, grid, method='cubic', fill_value=0.0)
            oi = interp.reshape(H, W).astype(np.float32)

            rmse = np.sqrt(np.mean((oi - tgt) ** 2))
            batch_values[i] = rmse

        self.individual_values = np.hstack([self.individual_values, batch_values])
        return batch_values
