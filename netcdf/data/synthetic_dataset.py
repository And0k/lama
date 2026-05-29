"""In-memory synthetic ocean dataset for training without NetCDF files.

Uses generators from ``netcdf.data.synthetic`` to create physically-
motivated 2D cross-sections of |V|, temperature, and salinity.

All samples are generated at construction time with a fixed seed,
ensuring strict reproducibility without disk I/O.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import make_synthetic_sample

logger = logging.getLogger(__name__)


class SyntheticOceanDataset(Dataset):
    """In-memory dataset of synthetic oceanographic cross-sections.

    Each sample matches the ``NetCDFDataset.__getitem__`` output format:
        - image:     (3, H, W) float32 in [0, 1]  — [|V|, thetao, so]
        - mask:      (1, H, W) float32             — observation mask
        - fill_mask: (3, H, W) float32             — 1.0 below bottom
        - meta:      dict with metadata

    Args:
        n: number of samples to generate.
        nx: horizontal grid points (distance).
        nz: vertical grid points (depth).
        n_ctd: number of CTD profiles per sample.
        n_cmems_x: number of CMEMS columns per sample.
        noise_ctd: standard deviation of CTD measurement noise.
        noise_cmems: standard deviation of CMEMS measurement noise.
        seed: RNG seed for reproducibility.
        scaling: optional dict of (vmin, vmax) per variable name for
            metadata.  Does NOT affect the data (already in [0, 1]).
    """

    def __init__(
        self,
        n: int = 400,
        nx: int = 64,
        nz: int = 64,
        n_ctd: int = 5,
        n_cmems_x: int = 8,
        noise_ctd: float = 0.05,
        noise_cmems: float = 0.10,
        seed: int = 42,
        scaling: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.n = n
        self.nx = nx
        self.nz = nz
        self.scaling = scaling or {}
        self.variables = ["|V|", "thetao", "so"]

        rng = np.random.default_rng(seed)

        logger.info(
            "Generating %d synthetic samples (nx=%d, nz=%d, seed=%d)...",
            n, nx, nz, seed,
        )

        self.samples = []
        for i in range(n):
            sample = make_synthetic_sample(
                nx=nx,
                nz=nz,
                n_ctd=n_ctd,
                n_cmems_x=n_cmems_x,
                noise_ctd=noise_ctd,
                noise_cmems=noise_cmems,
                rng=rng,
            )
            self.samples.append(sample)

        logger.info("Synthetic dataset ready: %d samples", n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        s = self.samples[idx]
        return {
            "image": torch.from_numpy(s["image"]),
            "mask": torch.from_numpy(s["mask"]),
            "fill_mask": torch.from_numpy(s["fill_mask"]),
            "meta": {
                "filepath": "synthetic",
                "time_index": idx,
                "depth_index": 0,
                "lat_index": 0,
                "lon_index": 0,
                "slice_mode": ["time", "latitude"],
                "coverage": float(s["mask"].mean()),
                "coverage_ok": True,
                "fill_ratio": float(s["fill_mask"].mean()),
                "variables": self.variables,
                "bathy": s["bathy"],
                "below": s["below"],
            },
        }

    def get_shape(self) -> Tuple[int, int, int]:
        return 3, self.nz, self.nx
