"""Mask generator for NetCDF data.

Supports vertical masks natively.  LaMa MixedMaskGenerator is optional
(only needed for legacy image-based masks).
"""

import json
import logging
from typing import Optional

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

try:
    from saicinpainting.training.data.masks import MixedMaskGenerator, SegmentationMask
except ImportError:  # pragma: no cover
    MixedMaskGenerator = None
    SegmentationMask = None

_generator_cache: dict = {}


def _get_generator(mask_params: dict):
    """Cache and return an instantiated mask generator from Hydra config params."""
    cache_key = json.dumps(mask_params, sort_keys=True)
    if cache_key not in _generator_cache:
        _generator_cache[cache_key] = instantiate(OmegaConf.create(mask_params))
    return _generator_cache[cache_key]


def generate_mask(
    shape: tuple[int, int],
    mask_params: dict,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a mask for inpainting.

    Args:
        shape: Shape of the mask as (height, width).
        mask_params: Configuration dict. Supports:
            - `_target_`: LaMa mask generator class name
            - `type: "vertical"`: Use vertical mask generator for vertical slices
            - `n_lines`, `x_num`, `y_num`, `y_law`: Parameters for vertical masks
        seed: Random seed for reproducibility.

    Returns:
        A binary mask of shape (1, height, width) with values 0 or 1.
    """
    mask_type = mask_params.get("type")
    if mask_type == "vertical":
        from .data.vertical_masks import generate_vertical_mask
        n_lines = mask_params.get("n_lines", 20)
        x_num = mask_params.get("x_num", 20)
        y_num = mask_params.get("y_num", 30)
        y_law = mask_params.get("y_law", "log")
        return generate_vertical_mask(
            shape=shape,
            n_lines=n_lines,
            x_num=x_num,
            y_num=y_num,
            y_law=y_law,
            seed=seed,
        )

    if MixedMaskGenerator is None or SegmentationMask is None:
        raise RuntimeError("saicinpainting is required for mask generation. Install with `pip install saicinpainting`.")

    generator = _get_generator(mask_params)

    if seed is not None:
        np.random.seed(seed)

    dummy_img = np.zeros((1,) + shape, dtype=np.float32)
    mask = generator(dummy_img)
    mask = (mask > 0).astype(np.uint8)
    mask = np.squeeze(mask)
    mask = mask[None, ...]
    return mask


def generate_masks_for_batch(
    batch_shape: tuple[int, int, int, int],
    mask_params: dict,
    seeds: Optional[list[int]] = None,
) -> np.ndarray:
    """
    Generate masks for a batch of images.

    Args:
        batch_shape: Shape of the batch as (batch_size, channels, height, width).
        mask_params: Configuration dict with `_target_` key pointing to the mask generator class.
        seeds: List of seeds for each item in the batch. If None, use random seeds.

    Returns:
        A binary mask of shape (batch_size, 1, height, width).
    """
    batch_size, _, height, width = batch_shape
    masks = []
    for i in range(batch_size):
        seed = seeds[i] if seeds is not None and i < len(seeds) else None
        mask = generate_mask((height, width), mask_params=mask_params, seed=seed)
        masks.append(mask)
    return np.stack(masks, axis=0)