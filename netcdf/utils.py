"""General utilities for NetCDF tensor operations."""

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def next_modulo(x: int, mod: int) -> int:
    """Round x up to the nearest multiple of mod."""
    return ((x + mod - 1) // mod) * mod


def pad_to_modulo(
    tensor: torch.Tensor, mod: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad spatial dims to multiples of mod; return (padded, (orig_H, orig_W)).

    Args:
        tensor: Input tensor of shape (B, C, H, W).
        mod: Modulus to pad to.

    Returns:
        Tuple of (padded tensor, (original_height, original_width)).
    """
    _, _, H, W = tensor.shape
    pad_H = next_modulo(H, mod) - H
    pad_W = next_modulo(W, mod) - W
    if pad_H == 0 and pad_W == 0:
        return tensor, (H, W)
    mode = "reflect" if pad_H < H and pad_W < W else "constant"
    return (
        torch.nn.functional.pad(tensor, (0, pad_W, 0, pad_H), mode=mode),
        (H, W),
    )
