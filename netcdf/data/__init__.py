"""
NetCDF data utilities for oceanographic data loading and processing.
"""

from .loader import load_netcdf, get_available_variables
from .dataset import NetCDFDataset
from .vertical_masks import (
    vertical_random_lines_mask,
    add_model_points,
    generate_vertical_mask,
)
from ..mask_generator import generate_mask
from ..visualization import visualize_sample, plot_comparison

__all__ = [
    'load_netcdf',
    'get_available_variables',
    'NetCDFDataset',
    'generate_mask',
    'vertical_random_lines_mask',
    'add_model_points',
    'generate_vertical_mask',
    'visualize_sample',
    'plot_comparison',
]