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
from .data.loader import load_netcdf, get_available_variables, get_variable_info
from .data.slicer import slice_nc
from .mask_generator import generate_mask, generate_masks_for_batch
from .visualization import plot_slice, plot_mask, plot_comparison, visualize_sample

__all__ = [
    "load_netcdf",
    "get_available_variables",
    "get_variable_info",
    "generate_mask",
    "generate_masks_for_batch",
    "NetCDFDataset",
    "slice_nc",
    "plot_slice",
    "plot_mask",
    "plot_comparison",
    "visualize_sample",
]
