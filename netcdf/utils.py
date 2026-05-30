"""General utilities for NetCDF tensor operations and training."""

import logging
from pathlib import Path
from typing import Optional, Tuple

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


# ── Checkpoint utilities ────────────────────────────────────────────────────


def _parse_val_loss(path: Path) -> float:
    """Extract val_loss from a flat checkpoint filename.

    Expected format: ``hydro-{epoch:02d}-val_loss={value:.4f}.ckpt``
    Also handles legacy format: ``hydro-epoch=00-val_loss=2.7003.ckpt``
    """
    try:
        s = path.stem
        return float(s.split("val_loss=")[-1])
    except (IndexError, ValueError):
        return float("inf")


def find_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return checkpoint with lowest val_loss, or None.

    Searches ``ckpt_dir`` for files matching ``hydro-*.ckpt`` (flat layout).
    Parses val_loss from the filename.

    Args:
        ckpt_dir: Directory containing checkpoint files.

    Returns:
        Path to checkpoint with lowest val_loss, or None.
    """
    candidates = sorted(ckpt_dir.glob("hydro-*.ckpt"))
    if not candidates:
        return None
    return min(candidates, key=_parse_val_loss)


def cleanup_checkpoints(ckpt_dir: Path, max_keep: int = 10) -> int:
    """Delete worst checkpoints, keeping only the best *max_keep*.

    Args:
        ckpt_dir: Directory containing checkpoint files.
        max_keep: Number of best checkpoints to keep.

    Returns:
        Number of checkpoint files deleted.
    """
    candidates = sorted(ckpt_dir.glob("hydro-*.ckpt"), key=_parse_val_loss)
    if len(candidates) <= max_keep:
        return 0
    to_remove = candidates[max_keep:]
    for p in to_remove:
        try:
            p.unlink()
            logger.info("Removed checkpoint: %s", p.name)
        except OSError:
            logger.debug("Could not remove %s", p)
    return len(to_remove)
