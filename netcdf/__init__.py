"""NetCDF oceanographic data support for LaMa inpainting.

This package provides tools for loading, slicing, and processing NetCDF data
for vertical profile reconstruction with LaMa inpainting.

Key components:
- load_netcdf: Load NetCDF files with oceanographic variables
- extract_slices: Slice 4D data (time, depth, lat, lon) into 2D images
- generate_mask: Generate masks for inpainting training
- NetCDFDataset: PyTorch Dataset for NetCDF data
"""

from .data.dataset import NetCDFDataset
from .data.loader import load_netcdf, get_available_variables, read_units, resolve_output_channels
from .data.slicer import slice_nc
from .data.vertical_masks import (
    vertical_random_lines_mask,
    add_model_points,
    generate_vertical_mask,
)
from .evaluation import compute_ssim, resolve_vel_scaling
from .mask_generator import generate_mask, generate_masks_for_batch
from .utils import next_modulo, pad_to_modulo
from .visualization import plot_slice, plot_mask, plot_comparison, plot_netcdf_inference, visualize_sample

__all__ = [
    "load_netcdf",
    "get_available_variables",
    "read_units",
    "resolve_output_channels",
    "generate_mask",
    "generate_masks_for_batch",
    "NetCDFDataset",
    "slice_nc",
    "vertical_random_lines_mask",
    "add_model_points",
    "generate_vertical_mask",
    "compute_ssim",
    "resolve_vel_scaling",
    "next_modulo",
    "pad_to_modulo",
    "plot_slice",
    "plot_mask",
    "plot_comparison",
    "plot_netcdf_inference",
    "visualize_sample",
]
