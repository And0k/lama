"""Visualization utilities for NetCDF oceanographic data.

Functions:
    - plot_slice, plot_mask, plot_comparison: basic 2D field rendering
    - plot_netcdf_inference: multi-channel inference with error heatmap
    - plot_input_channels: HydroGenerator 8ch input visualization
    - plot_two_field_rows: universal 2-row stacked field plot
    - plot_fields_result: wrapper for T+S field visualization
    - plot_fields_error: wrapper for T+S error heatmap visualization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
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


def _attach_external_colorbar(ax: Axes, mappable: plt.cm.ScalarMappable,
                              **kwargs) -> None:
    """Attach a colorbar outside the plotting area without shrinking the axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mappable, cax=cax, **kwargs)


def _mask_below(arr: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Set below-bottom pixels to NaN for masked rendering."""
    v = arr.copy()
    v[below] = np.nan
    return v


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
    """Display original, masked, and inpainted results in a seamless row."""
    resolved_titles = titles or ["Original", "Masked", "Result"]
    datasets = tuple(map(_prepare_2d_display, (original, masked, result)))

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex="col", sharey="row")
    fig.subplots_adjust(wspace=0.05)

    for ax, data, title in zip(axes, datasets, resolved_titles):
        plot_slice(data, ax=ax, title=title, cmap=cmap)

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
    """Plot multi-channel inference with error heatmap instead of inpainted.

    Layout per variable row: Original | Masked | Error (|inpainted − original|).
    Fill values rendered in light gray.  White observation mask drawn on
    Original column for first variable only (other variables share the mask).

    Parameters
    ----------
    channels : list of (H, W) arrays — original data in physical units.
    inpainted : list of (H, W) arrays — inpainted data.
    fill_masks : list of (H, W) boolean arrays — True at _FillValue pixels.
    generated_mask : (1, H, W) or (H, W) — binary inpainting mask.
    var_names : list of str — variable names for row labels.
    dim_x, dim_y : str — axis labels.
    orig_shape : tuple, optional — (H, W) pre-pad field.
    ssim_val : float, optional — SSIM annotation.
    units : dict, optional — per-variable unit strings.
    suptitle : str — overall figure title.
    """
    n_ch = len(channels)
    units = units or {}

    mask_2d = generated_mask[0] if generated_mask.ndim == 3 else generated_mask

    cmap_data = plt.cm.viridis.copy()
    cmap_data.set_bad(color=MASK_COLOR)

    cmap_fill = plt.cm.Greys_r.copy()
    cmap_err = plt.cm.RdBu_r.copy()

    fig, axes = plt.subplots(
        n_ch, 3, figsize=figsize,
        sharex="col", sharey="row", constrained_layout=True,
    )
    if n_ch == 1:
        axes = axes.reshape(1, -1)

    for ch_idx in range(n_ch):
        orig_phys = channels[ch_idx]
        inp_phys = inpainted[ch_idx]
        fill_mask = fill_masks[ch_idx]

        masked_phys = orig_phys.copy()
        masked_phys[mask_2d > 0] = np.nan

        err = inp_phys - orig_phys
        err_water = err.copy()
        err_water[fill_mask] = np.nan

        unit_label = units.get(var_names[ch_idx], "")
        unit_suffix = f" [{unit_label}]" if unit_label else ""

        emax = float(np.nanmax(np.abs(err_water))) or 1.0
        err_norm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)

        col_data = [
            (orig_phys, "Original", cmap_data, None),
            (masked_phys, "Masked", cmap_data, None),
            (err_water, "Error", cmap_err, err_norm),
        ]

        for col, (data_2d, col_title, cmap, norm) in enumerate(col_data):
            ax = axes[ch_idx, col]
            display = data_2d.copy()
            if col < 2:
                display[fill_mask] = np.nan

            im = ax.imshow(
                np.ma.masked_invalid(display), cmap=cmap,
                norm=norm, aspect="auto",
            )

            if fill_mask.any() and col < 2:
                fill_overlay = np.ma.masked_where(
                    ~fill_mask, np.full_like(data_2d, 0.5),
                )
                ax.imshow(
                    fill_overlay, cmap=cmap_fill, vmin=0, vmax=1,
                    alpha=0.45, aspect="auto",
                )

            # White obs mask on Original column, first variable only
            if col == 0 and ch_idx == 0:
                obs_vis = np.ma.masked_where(
                    mask_2d < 0.5, np.ones_like(mask_2d),
                )
                ax.imshow(obs_vis, cmap="Greys", vmin=0, vmax=1,
                          alpha=0.6, aspect="auto")

            ax.set_xlabel(dim_x)
            ax.set_ylabel(dim_y)

            title = f"{var_names[ch_idx]} \u2014 {col_title}"
            if ch_idx == 0 and col == 0 and ssim_val is not None:
                title += f"  (SSIM={ssim_val:.4f})"
            ax.set_title(title)

        _attach_external_colorbar(
            axes[ch_idx, 0], im,
            label=f"{var_names[ch_idx]}{unit_suffix}",
        )
        _attach_external_colorbar(
            axes[ch_idx, 2],
            plt.cm.ScalarMappable(norm=err_norm, cmap=cmap_err),
            label=f"\u0394{var_names[ch_idx]}{unit_suffix}",
        )

    for ax in axes.flat:
        ax.label_outer()

    dim_y_label = orig_shape[0] if orig_shape else "?"
    dim_x_label = orig_shape[1] if orig_shape else "?"
    fig.suptitle(
        f"{suptitle}  {dim_y}\u00d7{dim_x}=({dim_y_label},{dim_x_label})",
        fontsize=10,
    )
    return fig


# ── Hydro T+S visualization ─────────────────────────────────────────────


def plot_input_channels(
    u: np.ndarray,
    T: np.ndarray,
    v: np.ndarray,
    S: np.ndarray,
    *,
    bathy: Optional[np.ndarray] = None,
    mask_ctd: Optional[np.ndarray] = None,
    mask_cmems: Optional[np.ndarray] = None,
    below: Optional[np.ndarray] = None,
    suptitle: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """Visualize all 4 physical input channels: u,T / v,S.

    Row 0: u (with bathy gray, CTD/CMEMS white overlays), T
    Row 1: v, S

    Only u gets mask overlays — other panels are clean fields.

    Args:
        u, T, v, S: (H, W) fields in physical or normalised space.
        bathy:      (W,) normalised bottom depth in [0, 1].  Bottom filled gray on u.
        mask_ctd:   (H, W) CTD observation mask (1.0 = observed).  White on u.
        mask_cmems: (H, W) CMEMS observation mask (1.0 = observed).  White on u.
        below:      (H, W) bool, True = below bottom.
        suptitle:   Figure title.
        save_path:  If set, save PNG to this path.
        figsize:    Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex="col", sharey="row")
    fig.subplots_adjust(wspace=0.25, hspace=0.30)

    panels = [
        (axes[0, 0], u, "u (velocity)", "RdBu_r"),
        (axes[0, 1], T, "T (temperature)", "plasma"),
        (axes[1, 0], v, "v (velocity)", "RdBu_r"),
        (axes[1, 1], S, "S (salinity)", "viridis"),
    ]

    for ax, field, title, cmap in panels:
        display = field.copy()
        if below is not None:
            display[below] = np.nan
        im = ax.imshow(np.ma.masked_invalid(display), cmap=cmap,
                        aspect="auto", origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (distance)")
        ax.set_ylabel("z (depth)")
        _attach_external_colorbar(ax, im)

    # Overlays on u panel only
    ax_u = axes[0, 0]
    if below is not None and bathy is not None:
        H, W = u.shape
        z_norm = np.linspace(0, 1, H)
        for xi in range(W):
            zi = int(bathy[xi] * H)
            if zi < H:
                ax_u.axhspan(zi, H, color="0.5", alpha=0.3)

    if mask_ctd is not None:
        ctd_vis = np.ma.masked_where(mask_ctd < 0.5, np.ones_like(mask_ctd))
        ax_u.imshow(ctd_vis, cmap="Greys", vmin=0, vmax=1,
                    alpha=0.7, aspect="auto")

    if mask_cmems is not None:
        cmems_vis = np.ma.masked_where(mask_cmems < 0.5,
                                        np.ones_like(mask_cmems))
        ax_u.imshow(cmems_vis, cmap="Greys", vmin=0, vmax=1,
                    alpha=0.4, aspect="auto")

    fig.suptitle(suptitle, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved input channels to %s", save_path)
    plt.close(fig)
    return fig


def plot_two_field_rows(
    field_top: np.ndarray,
    field_bottom: np.ndarray,
    *,
    title_top: str = "T",
    title_bottom: str = "S",
    xlabel: str = "x (distance)",
    ylabel: str = "z (depth)",
    vmin_top: Optional[float] = None,
    vmax_top: Optional[float] = None,
    vmin_bottom: Optional[float] = None,
    vmax_bottom: Optional[float] = None,
    cmap: str = "plasma",
    below: Optional[np.ndarray] = None,
    mask_overlay: Optional[np.ndarray] = None,
    suptitle: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 8),
    label_top: str = "",
    label_bottom: str = "",
) -> Figure:
    """Universal 2-row stacked field plot with independent colorbars.

    Args:
        field_top, field_bottom: (H, W) arrays to display.
        title_top, title_bottom: subplot titles.
        xlabel, ylabel: axis labels.
        vmin_top/vmax_top, vmin_bottom/vmax_bottom: color limits (auto if None).
        cmap: colormap name.
        below: (H, W) bool — below-bottom mask (NaN).
        mask_overlay: (H, W) float — white overlay mask (e.g. obs locations).
        suptitle: figure super-title.
        save_path: if set, save PNG.
        figsize: figure size.
        label_top, label_bottom: colorbar labels.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    sharex="col")
    fig.subplots_adjust(hspace=0.35)

    for ax, field, title, vm, vM, cbar_label in [
        (ax1, field_top, title_top, vmin_top, vmax_top, label_top),
        (ax2, field_bottom, title_bottom, vmin_bottom, vmax_bottom, label_bottom),
    ]:
        display = field.copy()
        if below is not None:
            display[below] = np.nan

        im = ax.imshow(np.ma.masked_invalid(display), cmap=cmap,
                        vmin=vm, vmax=vM, aspect="auto", origin="upper")

        if mask_overlay is not None:
            ov = np.ma.masked_where(mask_overlay < 0.5,
                                     np.ones_like(mask_overlay))
            ax.imshow(ov, cmap="Greys", vmin=0, vmax=1,
                      alpha=0.6, aspect="auto")

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        _attach_external_colorbar(ax, im, label=cbar_label)

    ax2.set_xlabel(xlabel)
    ax1.tick_params(labelbottom=False)

    fig.suptitle(suptitle, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved to %s", save_path)
    plt.close(fig)
    return fig


def plot_fields_result(
    T_field: np.ndarray,
    S_field: np.ndarray,
    *,
    method: str = "lama",
    below: Optional[np.ndarray] = None,
    output_dir: str = ".",
    suffix: str = "sample",
    vmin_T: Optional[float] = None,
    vmax_T: Optional[float] = None,
    vmin_S: Optional[float] = None,
    vmax_S: Optional[float] = None,
    suptitle: str = "",
) -> Figure:
    """Wrapper: save T+S result fields to ``{output_dir}/{method}_T_S_{suffix}.png``."""
    save_path = str(Path(output_dir) / f"{method}_T_S_{suffix}.png")
    return plot_two_field_rows(
        T_field, S_field,
        title_top=f"{method.upper()} T",
        title_bottom=f"{method.upper()} S",
        cmap="plasma",
        below=below,
        vmin_top=vmin_T, vmax_top=vmax_T,
        vmin_bottom=vmin_S, vmax_bottom=vmax_S,
        label_top="T", label_bottom="S",
        suptitle=suptitle or f"{method.upper()} reconstruction",
        save_path=save_path,
    )


def plot_fields_error(
    T_err: np.ndarray,
    S_err: np.ndarray,
    *,
    method: str = "lama",
    below: Optional[np.ndarray] = None,
    output_dir: str = ".",
    suffix: str = "sample",
    suptitle: str = "",
) -> Figure:
    """Wrapper: save T+S error heatmaps to ``{output_dir}/{method}_error_{suffix}.png``."""
    save_path = str(Path(output_dir) / f"{method}_error_{suffix}.png")

    T_abs = np.abs(T_err[~below]) if below is not None else np.abs(T_err)
    S_abs = np.abs(S_err[~below]) if below is not None else np.abs(S_err)
    emax_T = float(T_abs.max()) if T_abs.size > 0 else 1.0
    emax_S = float(S_abs.max()) if S_abs.size > 0 else 1.0

    return plot_two_field_rows(
        T_err, S_err,
        title_top=f"{method.upper()} T error",
        title_bottom=f"{method.upper()} S error",
        cmap="RdBu_r",
        below=below,
        vmin_top=-emax_T, vmax_top=emax_T,
        vmin_bottom=-emax_S, vmax_bottom=emax_S,
        label_top="\u0394T", label_bottom="\u0394S",
        suptitle=suptitle or f"{method.upper()} error",
        save_path=save_path,
    )


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

    if image.ndim == 4:
        image = image[idx]
    if mask.ndim == 4:
        mask = mask[idx]

    image_chan = image[channel] if image.ndim == 3 else image
    mask_chan = mask[0] if mask.ndim == 3 else mask

    masked_image = image_chan.copy()
    masked_image[mask_chan > 0] = np.nanmedian(image_chan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex="col", sharey="row")
    fig.subplots_adjust(wspace=0.05)

    plot_slice(image_chan, ax=axes[0], title=f"Original (Ch {channel})")
    plot_mask(mask_chan, ax=axes[1])
    plot_slice(masked_image, ax=axes[2], title="Masked Image")

    for ax in axes.flat:
        ax.label_outer()

    time_idx = meta.get("time_index", "N/A")
    file_path = meta.get("filepath", "Unknown")
    fig.suptitle(f"Time index: {time_idx}\nFile: {file_path}", fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved visualization to %s", save_path)

    return fig
