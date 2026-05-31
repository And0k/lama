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
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .eos import linearized_density

logger = logging.getLogger(__name__)

FILL_COLOR = "0.85"
MASK_COLOR = "white"

# ── Layout constants — single source of truth for all hydro plots ───────
# All per-axes dimensions are derived from CELL_W × CELL_H so that every
# plot type (2×2 inputs, 2×1 fields, n×3 inference) produces identically
# sized imshow areas regardless of grid layout.
CELL_W = 5.0  # single axes width  [inches]
CELL_H = 4.5  # single axes height [inches]
CBAR_W = 0.15  # colorbar width [inches]
CBAR_PAD = 0.0  # gap between axes edge and colorbar [inches]


def _figsize(
    ncols: int, nrows: int, *, extra_w: float = 1.5, extra_h: float = 1.0, cbar_cols: int = 0
) -> Tuple[float, float]:
    """Compute figure size from grid dimensions.

    Args:
        ncols: Number of data columns in the grid.
        nrows: Number of rows in the grid.
        extra_w: Width reserved for y-axis labels and margins [inches].
        extra_h: Height reserved for titles and x-axis labels [inches].
        cbar_cols: Number of colorbar columns already included in the
            GridSpec (their width is ``cbar_cols * CBAR_W``).

    Returns:
        (width, height) in inches for ``plt.figure(figsize=...)``.
    """
    w = ncols * CELL_W + cbar_cols * CBAR_W + extra_w
    h = nrows * CELL_H + extra_h
    return (w, h)


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


def _attach_external_colorbar(
    ax: Axes, mappable: plt.cm.ScalarMappable, size: float = CBAR_W, pad: float = CBAR_PAD, **kwargs
) -> None:
    """Attach a colorbar flush against the plot.

    Args:
        ax: Parent axes.
        mappable: Mappable for the colorbar.
        size: Colorbar width as fraction of axes width.
        pad: Gap between axes and colorbar.
        **kwargs: Forwarded to ``plt.colorbar``.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    plt.colorbar(mappable, cax=cax, **kwargs)


def _mask_below(arr: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Set below-bottom pixels to NaN for masked rendering."""
    v = arr.copy()
    v[below] = np.nan
    return v


def _apply_scaling(
    data: np.ndarray,
    scaling: Optional[Dict[str, Tuple[float, float]]],
    var_key: Optional[str],
) -> np.ndarray:
    """Denormalize from [0, 1] to physical units using *scaling[var_key]*.

    If *scaling* is ``None`` or does not contain *var_key*, the data is
    returned unchanged (already in physical units).
    """
    if scaling is None or var_key is None:
        return data
    rng = scaling.get(var_key)
    if rng is None:
        return data
    vmin, vmax = rng
    return data * (vmax - vmin) + vmin


def plot_slice(
    data: np.ndarray,
    ax: Axes,
    *,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clim: Optional[Dict[str, Any]] = None,
    var_key: Optional[str] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Render a 2D slice onto an existing axes with external colorbar.

    If *scaling* is provided and contains *var_key*, the data is
    denormalized from [0, 1] to physical units before rendering.
    If *clim* is provided and contains *var_key*, the ``vmin``/``vmax``
    from config take precedence over the function arguments.
    """
    data = _apply_scaling(data, scaling, var_key)
    if clim and var_key and var_key in clim:
        limits = clim[var_key]
        if limits is not None:
            vmin = limits.get("vmin", vmin) if isinstance(limits, dict) else vmin
            vmax = limits.get("vmax", vmax) if isinstance(limits, dict) else vmax
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
    figsize: Optional[Tuple[int, int]] = None,
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Figure:
    """Display original, masked, and inpainted results in a seamless row."""
    resolved_titles = titles or ["Original", "Masked", "Result"]
    datasets = tuple(map(_prepare_2d_display, (original, masked, result)))

    fig, axes = plt.subplots(
        1, 3, figsize=figsize if figsize is not None else _figsize(3, 1), sharex="col", sharey="row"
    )
    fig.subplots_adjust(wspace=0.05)

    for ax, data, title in zip(axes, datasets, resolved_titles):
        plot_slice(data, ax=ax, title=title, cmap=cmap, clim=clim, var_key="T", scaling=scaling)

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
    figsize: Optional[Tuple[int, int]] = None,
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
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
        n_ch,
        3,
        figsize=figsize if figsize is not None else _figsize(3, n_ch),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    if n_ch == 1:
        axes = axes.reshape(1, -1)

    for ch_idx in range(n_ch):
        orig_phys = _apply_scaling(channels[ch_idx], scaling, var_names[ch_idx])
        inp_phys = _apply_scaling(inpainted[ch_idx], scaling, var_names[ch_idx])
        fill_mask = fill_masks[ch_idx]

        masked_phys = orig_phys.copy()
        masked_phys[mask_2d > 0] = np.nan

        err_water = inp_phys - orig_phys
        err_water[fill_mask] = np.nan

        unit_label = units.get(var_names[ch_idx], "")
        unit_suffix = f" [{unit_label}]" if unit_label else ""

        var_cfg = (clim or {}).get(var_names[ch_idx]) if clim else None
        data_vmin = var_cfg.get("vmin") if isinstance(var_cfg, dict) else None
        data_vmax = var_cfg.get("vmax") if isinstance(var_cfg, dict) else None

        err_cfg = (clim or {}).get("error") if clim else None
        cfg_emax = err_cfg.get("emax") if isinstance(err_cfg, dict) else None
        emax = cfg_emax if cfg_emax is not None else (float(np.nanmax(np.abs(err_water))) or 1.0)
        err_norm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)

        col_data = [
            (orig_phys, "Original", cmap_data, None, data_vmin, data_vmax),
            (masked_phys, "Masked", cmap_data, None, data_vmin, data_vmax),
            (err_water, "Error", cmap_err, err_norm, None, None),
        ]

        for col, (data_2d, col_title, cmap, norm, v_min, v_max) in enumerate(col_data):
            ax = axes[ch_idx, col]
            display = data_2d.copy()
            if col < 2:
                display[fill_mask] = np.nan

            im = ax.imshow(
                np.ma.masked_invalid(display),
                cmap=cmap,
                norm=norm,
                vmin=v_min,
                vmax=v_max,
                aspect="auto",
                interpolation="none",
            )

            if fill_mask.any() and col < 2:
                fill_overlay = np.ma.masked_where(
                    ~fill_mask,
                    np.full_like(data_2d, 0.5),
                )
                ax.imshow(
                    fill_overlay,
                    cmap=cmap_fill,
                    vmin=0,
                    vmax=1,
                    alpha=0.45,
                    aspect="auto",
                    interpolation="none",
                )

            # White obs mask on Original column, first variable only
            if col == 0 and ch_idx == 0:
                obs_vis = np.ma.masked_where(
                    mask_2d < 0.5,
                    np.ones_like(mask_2d),
                )
                ax.imshow(
                    obs_vis, cmap="Greys", vmin=0, vmax=1, alpha=0.6, aspect="auto", interpolation="none"
                )

            ax.set_xlabel(dim_x)
            ax.set_ylabel(dim_y)

            title = f"{var_names[ch_idx]} \u2014 {col_title}"
            if ch_idx == 0 and col == 0 and ssim_val is not None:
                title += f"  (SSIM={ssim_val:.4f})"
            ax.set_title(title)

        _attach_external_colorbar(
            axes[ch_idx, 0],
            im,
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
    T_obs: np.ndarray,
    S_obs: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    T_bg: Optional[np.ndarray] = None,
    S_bg: Optional[np.ndarray] = None,
    bathy: Optional[np.ndarray] = None,
    mask_ctd: Optional[np.ndarray] = None,
    mask_cmems: Optional[np.ndarray] = None,
    below: Optional[np.ndarray] = None,
    suptitle: str = "",
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Figure:
    """Visualize all 4 physical input channels: u,T / v,S.

    Row 0: u (bathymetry fill, location markers), T (background + obs overlay)
    Row 1: v, S (background + obs overlay)

    Layout: 2 data columns + 2 narrow colorbar columns per row via GridSpec.

    Args:
        T_obs, S_obs, u, v:   (H, W) — T/S sparse observation values, u/v velocity.
        T_bg, S_bg:   (H, W) full target fields for background (optional).
        bathy:        (W,) normalised bottom depth in [0, 1].
        mask_ctd:     (H, W) CTD observation mask (1.0 = observed).
        mask_cmems:   (H, W) CMEMS observation mask (1.0 = observed).
        below:        (H, W) bool, True = below bottom.
        suptitle:     Figure title.
        save_path:    If set, save PNG to this path.
        figsize:      Figure size (default: computed from CELL_W × CELL_H).
    """
    if figsize is None:
        figsize = _figsize(2, 2, cbar_cols=2)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 0.05, 1, 0.05])
    axes = np.array([[fig.add_subplot(gs[r, c * 2]) for c in range(2)] for r in range(2)])
    cbar_axes = np.array([[fig.add_subplot(gs[r, c * 2 + 1]) for c in range(2)] for r in range(2)])

    panels = [
        (axes[0, 0], u, "u (velocity)", "RdBu_r", "u"),
        (axes[0, 1], T_bg if T_bg is not None else T_obs, "T (temperature)", "plasma", "T"),
        (axes[1, 0], v, "v (velocity)", "RdBu_r", "v"),
        (axes[1, 1], S_bg if S_bg is not None else S_obs, "S (salinity)", "viridis", "S"),
    ]

    for i, (ax, field, title, cmap, var_key) in enumerate(panels):
        display = _apply_scaling(field, scaling, var_key)
        if below is not None:
            display[below] = np.nan
        var_cfg = (clim or {}).get(var_key) if clim else None
        v_min = var_cfg.get("vmin") if isinstance(var_cfg, dict) else None
        v_max = var_cfg.get("vmax") if isinstance(var_cfg, dict) else None
        im = ax.imshow(
            np.ma.masked_invalid(display),
            cmap=cmap,
            aspect="auto",
            origin="upper",
            interpolation="none",
            vmin=v_min,
            vmax=v_max,
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, cax=cbar_axes[i // 2, i % 2])

    # ── Overlays ───────────────────────────────────────────────────────────
    H, W = u.shape

    # u panel: bathymetry fill + location markers
    ax_u = axes[0, 0]
    if below is not None and bathy is not None:
        x_vals = np.arange(W)
        y_bottom = bathy * H
        ax_u.fill_between(x_vals, y_bottom, H, color="0.3", alpha=0.3)

    if mask_ctd is not None:
        ctd_xs = np.where(mask_ctd.any(axis=0))[0]
        for xi in ctd_xs:
            ax_u.axvline(xi, color="white", linewidth=0.6, alpha=0.5)

    if mask_cmems is not None:
        cmems_ys, cmems_xs = np.where(mask_cmems > 0.5)
        ax_u.scatter(
            cmems_xs, cmems_ys, s=3.5, c=None, alpha=1, marker="o", edgecolors="green", linewidths=0.5
        )

    # T and S panels: observation values at mask positions over background
    obs_panels = [
        (axes[0, 1], _apply_scaling(T_obs, scaling, "T"), "plasma"),
        (axes[1, 1], _apply_scaling(S_obs, scaling, "S"), "viridis"),
    ]
    for ax, obs_field, cmap_name in obs_panels:
        if mask_ctd is not None:
            obs = np.full((H, W), np.nan)
            obs[mask_ctd > 0.5] = obs_field[mask_ctd > 0.5]
            ax.imshow(
                np.ma.masked_invalid(obs),
                cmap=cmap_name,
                aspect="auto",
                origin="upper",
                interpolation="none",
                alpha=0.9,
            )

        if mask_cmems is not None:
            obs = np.full((H, W), np.nan)
            obs[mask_cmems > 0.5] = obs_field[mask_cmems > 0.5]
            ax.imshow(
                np.ma.masked_invalid(obs),
                cmap=cmap_name,
                aspect="auto",
                origin="upper",
                interpolation="none",
                alpha=0.7,
            )

    # All panels: gray vertical lines in below-bottom region at CTD x-positions
    if below is not None and bathy is not None and mask_ctd is not None:
        ctd_xs = np.where(mask_ctd.any(axis=0))[0]
        for ax in axes.flat:
            for xi in ctd_xs:
                y_bathy = int(round(bathy[xi] * H))
                if y_bathy < H:
                    ax.plot([xi, xi], [y_bathy + 1, H - 1], color="0.5", linewidth=1.0, alpha=0.6)

    for ax in axes[-1]:
        ax.set_xlabel("x (distance)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z (depth)")
    for ax in axes.flat:
        ax.label_outer()

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
    mask_ctd: Optional[np.ndarray] = None,
    bathy: Optional[np.ndarray] = None,
    density_field: Optional[np.ndarray] = None,
    suptitle: str = "",
    save_path: Optional[str] = None,
    label_top: str = "",
    label_bottom: str = "",
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
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
        mask_ctd: (H, W) CTD mask — for below-bottom gray lines.
        bathy: (W,) normalised bottom depth — for below-bottom gray lines.
        density_field: (H, W) optional density field for isoline overlay.
        suptitle: figure super-title.
        save_path: if set, save PNG.
        label_top, label_bottom: colorbar labels.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=_figsize(1, 2),
        sharex="col",
        constrained_layout=True,
    )

    for ax, field, title, vm, vM, cbar_label, var_key in [
        (ax1, field_top, title_top, vmin_top, vmax_top, label_top, "T"),
        (ax2, field_bottom, title_bottom, vmin_bottom, vmax_bottom, label_bottom, "S"),
    ]:
        if clim and var_key in clim and clim[var_key] is not None:
            cfg = clim[var_key]
            if vm is None:
                vm = cfg.get("vmin")
            if vM is None:
                vM = cfg.get("vmax")
        display = _apply_scaling(field, scaling, var_key)
        if below is not None:
            display[below] = np.nan

        im = ax.imshow(
            np.ma.masked_invalid(display),
            cmap=cmap,
            vmin=vm,
            vmax=vM,
            aspect="auto",
            origin="upper",
            interpolation="none",
        )

        if mask_overlay is not None:
            ov = np.ma.masked_where(mask_overlay < 0.5, np.ones_like(mask_overlay))
            ax.imshow(ov, cmap="Greys", vmin=0, vmax=1, alpha=0.6, aspect="auto", interpolation="none")

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        _attach_external_colorbar(ax, im, label=cbar_label)

    ax2.set_xlabel(xlabel)
    for ax in (ax1, ax2):
        ax.label_outer()

    # Gray vertical lines in below-bottom region at CTD x-positions
    if below is not None and bathy is not None and mask_ctd is not None:
        H = field_top.shape[0]
        ctd_xs = np.where(mask_ctd.any(axis=0))[0]
        for ax in (ax1, ax2):
            for xi in ctd_xs:
                y_bathy = int(round(bathy[xi] * H))
                if y_bathy < H:
                    ax.plot([xi, xi], [y_bathy + 1, H - 1], color="0.5", linewidth=1.0, alpha=0.6)

    # Density isolines overlay (contour lines only)
    if density_field is not None:
        rho = np.ma.masked_invalid(density_field)
        levels = np.linspace(np.nanmin(density_field), np.nanmax(density_field), 8)
        for ax in (ax1, ax2):
            ax.contour(
                rho,
                levels=levels,
                colors="black",
                linewidths=0.5,
                alpha=0.4,
                linestyles="solid",
            )

    fig.suptitle(suptitle, fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.debug("Saved to %s", save_path)
    plt.close(fig)
    return fig


def plot_fields_result(
    T_field: np.ndarray,
    S_field: np.ndarray,
    *,
    method: str = "LaMa",
    below: Optional[np.ndarray] = None,
    mask_ctd: Optional[np.ndarray] = None,
    mask_cmems: Optional[np.ndarray] = None,
    bathy: Optional[np.ndarray] = None,
    output_dir: str = ".",
    suffix: str = "sample",
    vmin_T: Optional[float] = None,
    vmax_T: Optional[float] = None,
    vmin_S: Optional[float] = None,
    vmax_S: Optional[float] = None,
    suptitle: str = "",
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs,  # not used
) -> Figure:
    """Wrapper: save T+S result fields to ``{output_dir}/{method}_T_S_{suffix}.png``."""
    save_path = str(Path(output_dir) / f"{method.lower()}_T_S_{suffix}.png")
    return plot_two_field_rows(
        T_field,
        S_field,
        title_top=f"{method} T",
        title_bottom=f"{method} S",
        cmap="plasma",
        below=below,
        mask_ctd=mask_ctd,
        bathy=bathy,
        vmin_top=vmin_T,
        vmax_top=vmax_T,
        vmin_bottom=vmin_S,
        vmax_bottom=vmax_S,
        label_top="T",
        label_bottom="S",
        suptitle=suptitle or f"{method} reconstruction",
        save_path=save_path,
        clim=clim,
        scaling=scaling,
    )


def plot_fields_error(
    T_field: np.ndarray,
    S_field: np.ndarray,
    T_bg: np.ndarray,
    S_bg: np.ndarray,
    T_obs: Optional[np.ndarray] = None,
    S_obs: Optional[np.ndarray] = None,
    *,
    method: str = "LaMa",
    below: Optional[np.ndarray] = None,
    mask_ctd: Optional[np.ndarray] = None,
    mask_cmems: Optional[np.ndarray] = None,
    bathy: Optional[np.ndarray] = None,
    output_dir: str = ".",
    suffix: str = "sample",
    suptitle: str = "",
    clim: Optional[Dict[str, Any]] = None,
    scaling: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Figure:
    """
    Wrapper: save T+S error heatmaps with below-bottom gray lines.
    and obs-point error overlay.

    If *T_obs*/*S_obs* and masks are provided, |field − obs| error is drawn
    at observation positions using the ``hot`` colormap.
    """
    save_path = str(Path(output_dir) / f"{method.lower()}_error_{suffix}.png")

    err_T = _apply_scaling(T_field - T_bg, scaling, "T")
    err_S = _apply_scaling(S_field - S_bg, scaling, "S")

    # Overlay |pred − obs| error at observation positions
    if mask_ctd is not None:
        b_ctd = mask_ctd > 0.5
        if T_obs is not None:
            err_T[b_ctd] = _apply_scaling(T_field - T_obs, scaling, "T")[b_ctd]
        if S_obs is not None:
            err_S[b_ctd] = _apply_scaling(S_field - S_obs, scaling, "S")[b_ctd]

    below_phys = below
    T_abs = np.abs(err_T[~below_phys]) if below_phys is not None else np.abs(err_T)
    S_abs = np.abs(err_S[~below_phys]) if below_phys is not None else np.abs(err_S)

    err_cfg = (clim or {}).get("error") if clim else None
    cfg_emax = err_cfg.get("emax") if isinstance(err_cfg, dict) else None

    if cfg_emax is not None:
        emax_T = cfg_emax
        emax_S = cfg_emax
    else:
        emax_T = float(T_abs.max()) if T_abs.size > 0 else 1.0
        emax_S = float(S_abs.max()) if S_abs.size > 0 else 1.0

    return plot_two_field_rows(
        err_T,
        err_S,
        title_top=f"{method} T error",
        title_bottom=f"{method} S error",
        cmap="RdBu_r",
        below=below,
        mask_ctd=mask_ctd,
        bathy=bathy,
        density_field=linearized_density(T_bg, S_bg),
        vmin_top=-emax_T,
        vmax_top=emax_T,
        vmin_bottom=-emax_S,
        vmax_bottom=emax_S,
        label_top="\u0394T",
        label_bottom="\u0394S",
        suptitle=suptitle or f"{method} error",
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

    fig, axes = plt.subplots(1, 3, figsize=_figsize(3, 1), sharex="col", sharey="row")
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
