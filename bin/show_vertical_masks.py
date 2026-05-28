import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def show_vertical_masks_composition(nc_file: str, cfg: DictConfig) -> None:
    """Demonstrate vertical slice mask generation and visualization."""
    log.info("=== DEMO 3: Vertical Slice Masks ===")

    from netcdf.data.loader import load_netcdf
    from netcdf.data.slicer import slice_nc
    from netcdf.data.vertical_masks import (
        add_model_points,
        generate_vertical_mask,
        vertical_random_lines_mask,
    )

    data = load_netcdf(
        nc_file,
        variables=["thetao"],
        replace_fill_value=True,
    )
    arr = data["thetao"]

    vert_slice = slice_nc(arr, ("depth", "lat"), (0, 0))
    h, w = vert_slice.shape
    log.info("Vertical slice (depth, lat) shape: (%d, %d)", h, w)

    vert_mask_cfg = OmegaConf.to_object(cfg.vertical_mask) if hasattr(cfg, "vertical_mask") else {}

    # Random lines mask
    mask_lines = vertical_random_lines_mask(
        width=w, height=h,
        n_lines=vert_mask_cfg.get("n_lines", 15),
        seed=vert_mask_cfg.get("seed", None),
    )

    # Model points grid (without random lines)
    grid_mask = np.ones((h, w), dtype=np.uint8) * 255
    grid_mask = add_model_points(
        grid_mask,
        x_num=vert_mask_cfg.get("x_num", 20),
        y_num=vert_mask_cfg.get("y_num", 24),
        y_law=vert_mask_cfg.get("y_law", "log"),
    )

    # Combined mask: 1=masked, 0=valid
    combined_mask = generate_vertical_mask(
        shape=(h, w),
        n_lines=vert_mask_cfg.get("n_lines", 15),
        x_num=vert_mask_cfg.get("x_num", 20),
        y_num=vert_mask_cfg.get("y_num", 24),
        y_law=vert_mask_cfg.get("y_law", "log"),
        seed=vert_mask_cfg.get("seed", 42),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    im1 = axes[0, 0].imshow(vert_slice, cmap="viridis", aspect="auto")
    axes[0, 0].set_title("Original vertical slice")
    fig.colorbar(im1, ax=axes[0, 0], label="thetao value")

    axes[0, 1].imshow(mask_lines, cmap="gray", aspect="auto")
    axes[0, 1].set_title("Random lines mask")

    axes[1, 0].imshow(grid_mask, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Model points grid")

    axes[1, 1].imshow(combined_mask[0], cmap="gray", aspect="auto")
    axes[1, 1].set_title("Combined mask (1=masked, 0=valid)")

    plt.tight_layout()
    out_path = Path(tempfile.mkdtemp(prefix="netcdf_masks_")) / "vertical_masks_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Vertical mask visualization saved to %s", out_path)


def show_apply_vertical_mask(nc_file: str, cfg: DictConfig) -> None:
    """Demonstrate vertical mask generation on vertical slices (without model inference)."""
    log.info("=== DEMO 4: Vertical Slice Inference ===")

    from netcdf.data.loader import load_netcdf
    from netcdf.data.slicer import slice_nc
    from netcdf.data.vertical_masks import generate_vertical_mask

    vert_mask_cfg = OmegaConf.to_object(cfg.vertical_mask) if hasattr(cfg, "vertical_mask") else {}
    scaling_dict = OmegaConf.to_object(cfg.scaling)

    data = load_netcdf(
        nc_file,
        variables=["thetao"],
        replace_fill_value=True,
    )
    arr = data["thetao"]

    vert_slice = slice_nc(arr, ("depth", "lon"), (0, 0))
    h, w = vert_slice.shape
    log.info("Vertical slice (depth, lon) shape: (%d, %d)", h, w)

    for seed in [42, 123, 456]:
        mask = generate_vertical_mask(
            shape=(h, w),
            n_lines=vert_mask_cfg.get("n_lines", 10),
            x_num=vert_mask_cfg.get("x_num", 10),
            y_num=min(vert_mask_cfg.get("y_num", 15), h),
            seed=seed,
        )
        coverage = mask[0].mean() * 100
        log.info(" seed %d: mask shape %s, coverage %.1f%%", seed, mask.shape, coverage)

    out_dir = Path(tempfile.mkdtemp(prefix="netcdf_vertical_inf_"))
    log.info("Output directory: %s", out_dir)

    # Show original and masked side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin, vmax = scaling_dict.get("thetao", (4.0, 20.0))
    scaled = vert_slice * (vmax - vmin) + vmin

    mask_demo = generate_vertical_mask(shape=(h, w), n_lines=15, x_num=20, y_num=24, seed=42)
    masked = scaled.copy()
    masked[mask_demo[0] == 1] = np.nan

    im1 = axes[0].imshow(scaled, cmap="viridis", aspect="auto")
    axes[0].set_title("Original vertical slice")
    axes[0].set_xlabel("longitude")
    axes[0].set_ylabel("depth")
    fig.colorbar(im1, ax=axes[0], label="thetao (°C)")

    im2 = axes[1].imshow(np.ma.masked_invalid(masked), cmap="viridis", aspect="auto")
    axes[1].set_title("Masked for inpainting")
    axes[1].set_xlabel("longitude")
    axes[1].set_ylabel("depth")
    fig.colorbar(im2, ax=axes[1], label="thetao (°C)")

    plt.tight_layout()
    fig_path = out_dir / "vertical_inference_demo.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Vertical inference demo saved to %s", fig_path)


# ===================================================================
# Main
# ===================================================================


@hydra.main(config_path="../configs", config_name="nc/demo/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """ Demo Vertical slice masks """
    cfg = cfg._group_
    log.info("NetCDF Integration Demo for LaMa Inpainting (Hydra-driven)")

    nc_file, is_synthetic = _resolve_nc_file(cfg)
    if is_synthetic:
        log.info("No CMEMS file found — using synthetic data: %s", nc_file)
    else:
        log.info("Using CMEMS file: %s", nc_file)


    show_vertical_masks_composition(nc_file, cfg)
    show_apply_vertical_mask(nc_file, cfg)
    log.info("All demos completed successfully.")



if __name__ == "__main__":
    main()
