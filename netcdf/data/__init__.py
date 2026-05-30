"""
NetCDF data utilities for oceanographic data loading and processing.
"""

from .loader import load_netcdf, get_available_variables, read_units, resolve_output_channels
from .dataset import NetCDFDataset
from .vertical_masks import (
    vertical_random_lines_mask,
    add_model_points,
    generate_vertical_mask,
)
from .preprocessing import (
    VariableTransform,
    LogTransform,
    SqrtTransform,
    PowerTransform,
    DepthDependentScaling,
    PreprocessingPipeline,
    build_transform,
)
from .multi_file import MultiFileNetCDFDataset, scan_time_coordinates, align_time_coordinates
from .coordinates import (
    detect_coordinate_system,
    rotated_to_regular,
    build_regridder,
    regrid_dataset,
)
from ..mask_generator import generate_mask
from ..visualization import (
    visualize_sample,
    plot_comparison,
    plot_netcdf_inference,
    plot_input_channels,
    plot_two_field_rows,
    plot_fields_result,
    plot_fields_error,
)
from .synthetic import (
    synthetic_TS_field,
    synthetic_V_field,
    sloping_bathymetry,
    make_synthetic_sample,
    make_hydro_synthetic_sample,
    sample_observations,
)
from .synthetic_dataset import SyntheticOceanDataset, HydroSyntheticDataset

__all__ = [
    'load_netcdf',
    'get_available_variables',
    'read_units',
    'resolve_output_channels',
    'NetCDFDataset',
    'MultiFileNetCDFDataset',
    'scan_time_coordinates',
    'align_time_coordinates',
    'VariableTransform',
    'LogTransform',
    'SqrtTransform',
    'PowerTransform',
    'DepthDependentScaling',
    'PreprocessingPipeline',
    'build_transform',
    'detect_coordinate_system',
    'rotated_to_regular',
    'build_regridder',
    'regrid_dataset',
    'generate_mask',
    'vertical_random_lines_mask',
    'add_model_points',
    'generate_vertical_mask',
    'visualize_sample',
    'plot_comparison',
    'plot_netcdf_inference',
    'plot_input_channels',
    'plot_two_field_rows',
    'plot_fields_result',
    'plot_fields_error',
    'synthetic_TS_field',
    'synthetic_V_field',
    'sloping_bathymetry',
    'make_synthetic_sample',
    'make_hydro_synthetic_sample',
    'sample_observations',
    'SyntheticOceanDataset',
    'HydroSyntheticDataset',
]