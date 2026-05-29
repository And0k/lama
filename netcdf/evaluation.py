"""Evaluation metrics for NetCDF inpainting results."""

import logging

import numpy as np
import torch

from saicinpainting.evaluation.losses.base_loss import PairwiseScore

logger = logging.getLogger(__name__)


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
    from saicinpainting.evaluation.losses.ssim import SSIM

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
