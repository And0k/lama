"""Variable-specific preprocessing transforms for NetCDF data.

Provides per-variable transforms applied before [0,1] scaling, including
log transforms for salinity and depth-dependent scaling factors.

Usage in dataset config::

    variable_transforms:
      so:
        type: log
        offset: 1.0
      thetao:
        type: depth_dependent
        surface_factor: 1.2
        deep_factor: 0.8
        depth_threshold: 500.0
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class VariableTransform(ABC):
    """Base class for per-variable preprocessing transforms."""

    @abstractmethod
    def __call__(self, arr: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply transform to array.

        Args:
            arr: Raw float32 array (may contain NaN).
            depth: Optional 1-D depth coordinate array (meters).
                   Required for depth-dependent transforms.

        Returns:
            Transformed array with same shape.
        """

    @abstractmethod
    def transform_scaling(self, vmin: float, vmax: float) -> Tuple[float, float]:
        """Transform physical scaling range to match the transform.

        Args:
            vmin: Original minimum of physical range.
            vmax: Original maximum of physical range.

        Returns:
            (new_vmin, new_vmax) for the transformed data.
        """


class LogTransform(VariableTransform):
    """Logarithmic transform: log(x + offset).

    Useful for salinity or other variables where log-space
    better represents the dynamic range.
    """

    def __init__(self, base: str = "natural", offset: float = 1.0):
        """
        Args:
            base: Logarithm base: "natural" (ln), "10" (log10), "2" (log2).
            offset: Added to x before log to avoid log(0).
        """
        self.offset = offset
        if base == "natural":
            self._log_fn = np.log
        elif base == "10":
            self._log_fn = np.log10
        elif base == "2":
            self._log_fn = np.log2
        else:
            raise ValueError(f"Unsupported log base: {base}")
        self.base = base

    def __call__(self, arr: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        return self._log_fn(np.maximum(arr, 0) + self.offset).astype(np.float32)

    def transform_scaling(self, vmin: float, vmax: float) -> Tuple[float, float]:
        new_vmin = self._log_fn(max(vmin, 0) + self.offset)
        new_vmax = self._log_fn(max(vmax, 0) + self.offset)
        return float(new_vmin), float(new_vmax)


class SqrtTransform(VariableTransform):
    """Square-root transform: sqrt(max(x, 0))."""

    def __call__(self, arr: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        return np.sqrt(np.maximum(arr, 0)).astype(np.float32)

    def transform_scaling(self, vmin: float, vmax: float) -> Tuple[float, float]:
        return float(np.sqrt(max(vmin, 0))), float(np.sqrt(max(vmax, 0)))


class PowerTransform(VariableTransform):
    """General power transform: sign(x) * |x|^exponent."""

    def __init__(self, exponent: float = 0.5):
        self.exponent = exponent

    def __call__(self, arr: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        return (np.sign(arr) * np.power(np.abs(arr), self.exponent)).astype(np.float32)

    def transform_scaling(self, vmin: float, vmax: float) -> Tuple[float, float]:
        sign_lo = np.sign(vmin)
        sign_hi = np.sign(vmax)
        new_vmin = sign_lo * np.power(abs(vmin), self.exponent)
        new_vmax = sign_hi * np.power(abs(vmax), self.exponent)
        return float(new_vmin), float(new_vmax)


class DepthDependentScaling(VariableTransform):
    """Apply depth-dependent scaling factors.

    Scales data by a factor that varies with depth:
    - surface_factor applied at depth=0
    - deep_factor applied at depth >= depth_threshold
    - Linear interpolation between them

    This is useful when surface and deep waters have different
    dynamic ranges that benefit from separate normalization.
    """

    def __init__(
        self,
        surface_factor: float = 1.2,
        deep_factor: float = 0.8,
        depth_threshold: float = 500.0,
    ):
        """
        Args:
            surface_factor: Multiplicative factor at depth=0.
            deep_factor: Multiplicative factor at depth >= depth_threshold.
            depth_threshold: Depth (meters) at which deep_factor is fully applied.
        """
        self.surface_factor = surface_factor
        self.deep_factor = deep_factor
        self.depth_threshold = depth_threshold

    def __call__(self, arr: np.ndarray, depth: Optional[np.ndarray] = None) -> np.ndarray:
        if depth is None or depth.size == 0:
            return arr * self.surface_factor

        factor = self._compute_factor(depth)
        broadcast_shape = [1] * arr.ndim
        if arr.ndim >= 1:
            broadcast_shape[0] = -1
        return (arr * factor.reshape(broadcast_shape)).astype(np.float32)

    def _compute_factor(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float64)
        t = np.clip(depth / max(self.depth_threshold, 1e-8), 0.0, 1.0)
        return self.surface_factor + t * (self.deep_factor - self.surface_factor)

    def transform_scaling(self, vmin: float, vmax: float) -> Tuple[float, float]:
        max_factor = max(self.surface_factor, self.deep_factor)
        min_factor = min(self.surface_factor, self.deep_factor)
        return vmin * min_factor, vmax * max_factor


_TRANSFORM_REGISTRY = {
    "log": LogTransform,
    "sqrt": SqrtTransform,
    "power": PowerTransform,
    "depth_dependent": DepthDependentScaling,
}


def build_transform(config: Union[Dict, str, None]) -> Optional[VariableTransform]:
    """Build a VariableTransform from a config dict or string.

    Args:
        config: Transform configuration. Can be:
            - None: no transform
            - str: transform type name (uses defaults)
            - dict with 'type' key and optional parameters

    Returns:
        VariableTransform instance or None.
    """
    if config is None:
        return None
    if isinstance(config, str):
        config = {"type": config}
    if not isinstance(config, dict) or "type" not in config:
        raise ValueError(f"Invalid transform config: {config}")

    cfg = dict(config)
    ttype = cfg.pop("type")
    if ttype not in _TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform type: {ttype}. Available: {list(_TRANSFORM_REGISTRY.keys())}")
    return _TRANSFORM_REGISTRY[ttype](**cfg)


class PreprocessingPipeline:
    """Chain of variable-specific transforms applied before scaling.

    Attributes:
        transforms: Mapping from variable name to transform.
    """

    def __init__(self, transforms: Optional[Dict[str, VariableTransform]] = None):
        self.transforms = transforms or {}

    def __call__(
        self,
        raw_data: Dict[str, np.ndarray],
        depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Apply transforms to raw data dict.

        Args:
            raw_data: Mapping of variable name to raw float32 array.
            depth: Optional depth coordinate array for depth-dependent transforms.

        Returns:
            Dict with transformed arrays.
        """
        result = {}
        for var_name, arr in raw_data.items():
            if var_name in self.transforms:
                result[var_name] = self.transforms[var_name](arr, depth)
            else:
                result[var_name] = arr
        return result

    def transform_scaling(
        self, var_name: str, vmin: float, vmax: float
    ) -> Tuple[float, float]:
        """Transform scaling range for a specific variable.

        Args:
            var_name: Variable name.
            vmin: Original minimum.
            vmax: Original maximum.

        Returns:
            (new_vmin, new_vmax) if transform exists, else (vmin, vmax).
        """
        if var_name in self.transforms:
            return self.transforms[var_name].transform_scaling(vmin, vmax)
        return vmin, vmax

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Union[str, Dict]]]) -> "PreprocessingPipeline":
        """Build pipeline from YAML-style config.

        Args:
            config: Mapping of variable name to transform config.

        Returns:
            PreprocessingPipeline instance.
        """
        if not config:
            return cls()
        transforms = {}
        for var_name, cfg in config.items():
            t = build_transform(cfg)
            if t is not None:
                transforms[var_name] = t
                logger.info("Built transform for variable '%s': %s", var_name, cfg)
        return cls(transforms)
