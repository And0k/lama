"""In-memory synthetic ocean datasets for training without NetCDF files.

Two dataset classes:
    - ``SyntheticOceanDataset``: 3-channel (|V|, thetao, so) using simple
      sigmoid+eddy generators from ``netcdf.data.synthetic``.
    - ``HydroSyntheticDataset``: 8-channel hydro input + 2-channel T/S target
      using the full Baltic SceneGenerator for physically realistic data.

All samples are generated at construction time with a fixed seed,
ensuring strict reproducibility without disk I/O.
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import make_synthetic_sample, sample_observations

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


class HydroSyntheticDataset(Dataset):
    """8-channel hydro inpainting dataset using Baltic SceneGenerator.

    Produces batches compatible with ``HydroGenerator`` and the
    ``DefaultInpaintingTrainingModule`` loss pipeline.

    Ground truth scenes (T, S, u, v, bathymetry) are pre-generated once
    at construction time.  Observation masks (CTD positions + CMEMS
    grid) are sampled **on-the-fly** in ``__getitem__``, so each access
    produces a different random observation pattern for the same scene.

    Each sample:
        - image:       (8, nz, nx) float32 — hydro input channels
        - target:      (2, nz, nx) float32 — ground truth [T, S]
        - mask:        (1, nz, nx) float32 — below-bottom mask
        - fill_mask:   (3, nz, nx) float32 — below-bottom (compatibility)
        - sigma_obs:   (1, nz, nx) float32 — per-point uncertainty
        - obs_mask:    (1, nz, nx) float32 — observation mask (CTD ∪ CMEMS)
        - u_input:     (1, nz, nx) float32 — zonal velocity (for geostrophic loss)
        - bathy_indices: (nx,)      int64  — bottom depth indices (for BBL loss)
        - meta:        dict

    Args:
        n: Number of samples to generate.
        nx: Horizontal grid points.
        nz: Vertical grid points.
        n_ctd: CTD profiles per sample.
        n_cmems_x: CMEMS columns per sample.
        n_cmems_z: CMEMS depth levels per column.
        noise_ctd: CTD measurement noise σ.
        noise_cmems: CMEMS measurement noise σ.
        mode_mix: Dict of mode→probability. Default 70/20/10 realistic/extended/stress.
        seed: RNG seed for reproducibility of scene generation.
    """

    def __init__(
        self,
        n: int = 400,
        nx: int = 64,
        nz: int = 64,
        n_ctd: int = 5,
        n_cmems_x: int = 8,
        n_cmems_z: int = 10,
        noise_ctd: float = 0.05,
        noise_cmems: float = 0.10,
        mode_mix: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        self.n = n
        self.nx = nx
        self.nz = nz
        self.n_ctd = n_ctd
        self.n_cmems_x = n_cmems_x
        self.n_cmems_z = n_cmems_z
        self.noise_ctd = noise_ctd
        self.noise_cmems = noise_cmems
        self.variables = ["T", "S"]

        rng = np.random.default_rng(seed)

        modes = list((mode_mix or {"realistic": 0.7, "extended": 0.2, "stress": 0.1}).keys())
        weights = np.array(
            [(mode_mix or {"realistic": 0.7, "extended": 0.2, "stress": 0.1})[m]
             for m in modes],
            dtype=float,
        )
        weights /= weights.sum()

        seasons = ["winter", "spring", "summer", "autumn"]

        logger.info(
            "Generating %d hydro scenes (nx=%d, nz=%d, modes=%s, seed=%d)...",
            n, nx, nz, modes, seed,
        )

        from .synthetic import _normalize_field
        from .synthetic_baltic import SceneGenerator, Domain

        self.scenes = []
        for i in range(n):
            mode = rng.choice(modes, p=weights)
            season = rng.choice(seasons)

            seed_val = int(rng.integers(0, 2**31))
            gen = SceneGenerator(seed=seed_val)
            domain = Domain(nx=nx, nz=nz)
            scene = gen.generate(domain, mode=mode, season=season)

            T_raw = scene["T"]
            S_raw = scene["S"]
            u_raw = scene["u"]
            v_raw = scene["v"]
            bottom_m = scene["bottom"]
            mask_water = scene["mask"]

            bathy = (bottom_m / domain.zmax).astype(np.float32)
            below = ~mask_water

            T = _normalize_field(T_raw, mask_water)
            S = _normalize_field(S_raw, mask_water)

            u_arr = np.nan_to_num(u_raw.astype(np.float64), nan=0.0)
            v_arr = np.nan_to_num(v_raw.astype(np.float64), nan=0.0)
            valid_u = u_arr[mask_water]
            valid_v = v_arr[mask_water]
            u_max = max(float(np.abs(valid_u).max()) if valid_u.size > 0 else 1.0, 1e-8)
            v_max = max(float(np.abs(valid_v).max()) if valid_v.size > 0 else 1.0, 1e-8)
            u_arr = np.clip(u_arr / u_max, -1.0, 1.0)
            v_arr = np.clip(v_arr / v_max, -1.0, 1.0)

            factor = max(1, min(nx, nz) // 16)
            if factor > 1:
                u_ds = u_arr[::factor, ::factor]
                v_ds = v_arr[::factor, ::factor]
                u_lr = np.kron(u_ds, np.ones((factor, factor)))[:nz, :nx].astype(np.float32)
                v_lr = np.kron(v_ds, np.ones((factor, factor)))[:nz, :nx].astype(np.float32)
            else:
                u_lr = u_arr.astype(np.float32)
                v_lr = v_arr.astype(np.float32)

            u_lr = np.clip((u_lr + 1.0) / 2.0, 0.0, 1.0)
            v_lr = np.clip((v_lr + 1.0) / 2.0, 0.0, 1.0)
            u_lr[below] = 0.0
            v_lr[below] = 0.0

            T[below] = 0.0
            S[below] = 0.0

            self.scenes.append({
                "T": T, "S": S,
                "u_lr": u_lr, "v_lr": v_lr,
                "bathy": bathy, "below": below,
            })

        logger.info("Hydro synthetic dataset ready: %d scenes", n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sc = self.scenes[idx]
        T, S = sc["T"], sc["S"]
        below, bathy = sc["below"], sc["bathy"]
        u_lr, v_lr = sc["u_lr"], sc["v_lr"]
        nz, nx = self.nz, self.nx

        rng = np.random.default_rng()

        T_obs, S_obs, mask_ctd, mask_cmems, sigma_obs = sample_observations(
            T, S, below, bathy, nx, nz,
            self.n_ctd, self.n_cmems_x, self.n_cmems_z,
            self.noise_ctd, self.noise_cmems, rng,
        )

        bathy_ch = np.tile(bathy[None, :], (nz, 1)).astype(np.float32)
        inp = np.stack(
            [T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathy_ch, sigma_obs],
            axis=0,
        ).astype(np.float32)

        tgt = np.stack([T, S], axis=0).astype(np.float32)
        obs_mask = np.clip(mask_ctd + mask_cmems, 0, 1).astype(np.float32)
        bathy_indices = np.clip((bathy * nz).astype(int), 1, nz - 1)
        fill_mask = np.stack([below.astype(np.float32)] * 3, axis=0)

        return {
            "image": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
            "mask": torch.from_numpy(below.astype(np.float32)[None]),
            "fill_mask": torch.from_numpy(fill_mask),
            "sigma_obs": torch.from_numpy(sigma_obs[None]),
            "obs_mask": torch.from_numpy(obs_mask[None]),
            "u_input": torch.from_numpy(u_lr[None]),
            "bathy_indices": torch.from_numpy(bathy_indices).long(),
            "meta": {
                "filepath": "synthetic_baltic",
                "time_index": idx,
                "depth_index": 0,
                "lat_index": 0,
                "lon_index": 0,
                "slice_mode": ["time", "latitude"],
                "coverage": float(obs_mask.mean()),
                "coverage_ok": True,
                "fill_ratio": float(below.mean()),
                "variables": self.variables,
                "bathy": bathy,
                "below": below,
            },
        }

    def get_shape(self) -> Tuple[int, int, int]:
        return 8, self.nz, self.nx
