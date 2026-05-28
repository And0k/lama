"""Visualization utilities for NetCDF oceanographic data."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

FILL_COLOR = "0.85"
MASK_COLOR = "white"


def _prepare_2d_display(img: np.ndarray) -> np.ndarray:
    """Collapse non-spatial dimensions for 2D rendering."""
    if img.ndim != 3:
        return img

    channels, height, width = img.shape
    match channels:
        case 1:
            return img[0]
        case 3 | 4:
            return np.transpose(img, (1, 2, 0))
        case _:
            return img[0]


def _attach_external_colorbar(ax: Axes, mappable: plt.cm.ScalarMappable, **kwargs) -> None:
    """Attach a colorbar outside the plotting area without shrinking the axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, **kwargs)


def plot_slice(
    data: np.ndarray,
    ax: Axes,
    *,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Render a 2D slice onto an existing axes with external colorbar."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    _attach_external_colorbar(ax, im)


def plot_mask(mask: np.ndarray, ax: Axes, *, title: str = "Mask") -> None:
    """Render a binary mask with discrete external colorbar."""
    display_mask = mask[0] if mask.ndim == 3 else mask
    im = ax.imshow(display_mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    _attach_external_colorbar(ax, im, ticks=[0, 1])


def plot_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    result: np.ndarray,
    *,
    titles: Optional[List[str]] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (15, 5),
) -> Figure:
    """Display original, masked, and inpainted results in a seamless row.

    Args:
        original: Original image of shape (H, W) or (C, H, W).
        masked: Image with mask applied (missing values) of same shape as original.
        result: Inpainted result of same shape as original.
    """
    resolved_titles = titles or ["Original", "Masked", "Result"]
    datasets = tuple(map(_prepare_2d_display, (original, masked, result)))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex="col", sharey="row")
    fig.subplots_adjust(wspace=0.05)

    for ax, data, title in zip(axes, datasets, resolved_titles):
        plot_slice(data, ax=ax, title=title, cmap=cmap)

    # Structurally removes internal labels/ticks; keeps only outer boundaries
    for ax in axes.flat:
        ax.label_outer()

    return fig


def plot_netcdf_inference(
    channels: List[np.ndarray],
    inpainted: List[np.ndarray],
    fill_masks: List[np.ndarray],
    generated_mask: np.ndarray,
    *,
    var_names: List[str],
    dim_x: str = "longitude",
    dim_y: str = "latitude",
    orig_shape: Optional[Tuple[int, int]] = None,
    ssim_val: Optional[float] = None,
    units: Optional[Dict[str, str]] = None,
    suptitle: str = "",
    figsize: Tuple[int, int] = (18, 15),
) -> Figure:
    """Plot multi-channel inference results with distinct fill-value and mask overlays.

    Fill values (original ``_FillValue`` pixels) are rendered in light gray.
    Generated inpainting masks are rendered in white.

    Parameters
    ----------
    channels : list of (H, W) arrays
        Original data in physical units (one per variable).
    inpainted : list of (H, W) arrays
        Inpainted data in physical units (one per variable).
    fill_masks : list of (H, W) boolean arrays
        True where the original NetCDF had a fill value.
    generated_mask : (1, H, W) or (H, W) array
        Binary mask used for inpainting.
    var_names : list of str
        Variable names for row labels.
    dim_x, dim_y : str
        Axis labels.
    orig_shape : tuple, optional
        (H, W) of the original (pre-pad) field.
    ssim_val : float, optional
        SSIM to annotate on the first channel's inpainted column.
    units : dict, optional
        Per-variable unit strings for the colorbar label.
    suptitle : str
        Overall figure title.
    """
    n_ch = len(channels)
    units = units or {}

    mask_2d = generated_mask[0] if generated_mask.ndim == 3 else generated_mask

    cmap_data = plt.cm.viridis.copy()
    cmap_data.set_bad(color=MASK_COLOR)

    cmap_fill = plt.cm.Greys_r.copy()

    fig, axes = plt.subplots(
        n_ch,
        3,
        figsize=figsize,
        sharex="col",
        sharey="row",
        constrained_layout=True,  # layout engine handles colorbars too
    )
    if n_ch == 1:
        axes = axes.reshape(1, -1)

    for ch_idx in range(n_ch):
        orig_phys = channels[ch_idx]
        inp_phys = inpainted[ch_idx]
        fill_mask = fill_masks[ch_idx]

        masked_phys = orig_phys.copy()
        masked_phys[mask_2d > 0] = np.nan

        unit_label = units.get(var_names[ch_idx], "")
        unit_suffix = f" [{unit_label}]" if unit_label else ""

        for col, (data_2d, title) in enumerate([
            (orig_phys, "Original"),
            (masked_phys, "Masked"),
            (inp_phys, "Inpainted"),
        ]):
            ax = axes[ch_idx, col]

            display = data_2d.copy()
            display[fill_mask] = np.nan

            im = ax.imshow(
                np.ma.masked_invalid(display), cmap=cmap_data, aspect="auto",
            )

            if fill_mask.any():
                fill_overlay = np.ma.masked_where(
                    ~fill_mask, np.full_like(data_2d, 0.5),
                )
                ax.imshow(
                    fill_overlay, cmap=cmap_fill, vmin=0, vmax=1,
                    alpha=0.45, aspect="auto",
                )

            ax.set_xlabel(dim_x)
            ax.set_ylabel(dim_y)

            col_title = f"{var_names[ch_idx]} \u2014 {title}"
            if ch_idx == 0 and col == 2 and ssim_val is not None:
                col_title += f"  (SSIM={ssim_val:.4f})"
            ax.set_title(col_title)

        _attach_external_colorbar(axes[ch_idx, -1], im, label=f"{var_names[ch_idx]}{unit_suffix}")

    for ax in axes.flat:
        ax.label_outer()

    dim_y_label = orig_shape[0] if orig_shape else "?"
    dim_x_label = orig_shape[1] if orig_shape else "?"
    fig.suptitle(
        f"{suptitle}  {dim_y}\u00d7{dim_x}=({dim_y_label},{dim_x_label})",
        fontsize=10,
    )
    # fig.subplots_adjust(wspace=0.15, hspace=0.4)
    return fig


def visualize_sample(
    sample: Dict[str, Union[np.ndarray, torch.Tensor, Dict]],
    *,
    idx: int = 0,
    channel: int = 0,
    save_path: Optional[str] = None,
) -> Figure:
    """Visualize dataset sample with automatic internal label suppression."""
    image = sample["image"].numpy() if isinstance(sample["image"], torch.Tensor) else sample["image"]
    mask = sample["mask"].numpy() if isinstance(sample["mask"], torch.Tensor) else sample["mask"]
    meta = sample["meta"]

    # Handle batch dimension
    if image.ndim == 4:
        image = image[idx]
    if mask.ndim == 4:
        mask = mask[idx]

    image_chan = image[channel] if image.ndim == 3 else image
    mask_chan = mask[0] if mask.ndim == 3 else mask

    # Fill masked regions with median for robust visualization
    masked_image = image_chan.copy()
    masked_image[mask_chan > 0] = np.nanmedian(image_chan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex="col", sharey="row")
    fig.subplots_adjust(wspace=0.05)

    plot_slice(image_chan, ax=axes[0], title=f"Original (Ch {channel})")
    plot_mask(mask_chan, ax=axes[1])
    plot_slice(masked_image, ax=axes[2], title="Masked Image")

    # Enforce clean grid: internal labels/ticks removed automatically
    for ax in axes.flat:
        ax.label_outer()

    time_idx = meta.get("time_index", "N/A")
    file_path = meta.get("filepath", "Unknown")
    fig.suptitle(f"Time index: {time_idx}\nFile: {file_path}", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved visualization to %s", save_path)

    return fig
