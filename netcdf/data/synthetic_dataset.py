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


def augment_hydro_sample(
    inp: np.ndarray,
    tgt: np.ndarray,
    bathy_indices: np.ndarray,
    u_input: np.ndarray,
    below: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply physics-constrained augmentations to a hydro sample.

    Augmentations (applied lazily in __getitem__):
        1. Horizontal reflection (x → -x): physically valid mirror.
           Also reverses bathy_indices and negates u_input.
        2. Amplitude scaling ±20%: simulates seasonal variation.
           Only applied to T,S observation channels (0,1) and target.

    Note: CTD dropout is now handled by stochastic n_ctd thinning in
    __getitem__ before sample_observations, not in augmentation.

    Args:
        inp:          (8, nz, nx) input tensor.
        tgt:          (2, nz, nx) target tensor.
        bathy_indices: (nx,) bottom depth indices.
        u_input:      (1, nz, nx) zonal velocity.
        below:        (nz, nx) bool below-bottom mask.
        rng:          Seeded NumPy generator.

    Returns:
        Tuple of (inp, tgt, bathy_indices, u_input, below) after augmentation.
    """
    nz, nx = inp.shape[1], inp.shape[2]

    # 1. Horizontal reflection
    if rng.random() < 0.5:
        inp = inp[:, :, ::-1].copy()
        tgt = tgt[:, :, ::-1].copy()
        bathy_indices = bathy_indices[::-1].copy()
        u_input = u_input[:, :, ::-1].copy()
        u_input = 1.0 - u_input
        below = below[:, ::-1].copy()

    # 2. Amplitude scaling (seasonal variation) ±20%
    alpha = rng.uniform(0.8, 1.2)
    inp[0] *= alpha
    inp[1] *= alpha
    tgt[0] *= alpha
    tgt[1] *= alpha

    return inp, tgt, bathy_indices, u_input, below


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

        logger.debug(
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

        logger.debug("Synthetic dataset ready: %d samples", n)

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

    Curriculum learning: pre-generates samples with n_ctd ∈ {2..7}.
    Call set_epoch(epoch, total_epochs) at each epoch start to gradually
    reduce the minimum n_ctd (more instruments early, fewer late).

    Args:
        n: Number of samples to generate.
        nx: Horizontal grid points.
        nz: Vertical grid points.
        n_ctd: Base CTD profiles per sample (also max for curriculum).
        n_cmems_x: CMEMS columns per sample.
        n_cmems_z: CMEMS depth levels per column.
        noise_ctd: CTD measurement noise σ.
        noise_cmems: CMEMS measurement noise σ.
        mode_mix: Dict of mode→probability. Default 70/20/10 realistic/extended/stress.
        seed: RNG seed for reproducibility of scene generation.
        augment: Enable physical augmentation in __getitem__.
        n_variants: Number of scene variants per sample (pre-generated mode only).
        lazy: If True, generate scenes on-the-fly in __getitem__ instead of
            pre-generating.  Each access produces a fresh scene with a
            deterministic seed derived from (idx, epoch).  This gives
            infinite effective diversity — the model never sees the same
            T/S field twice across epochs.  Cost: ~1ms per access.
            When lazy=True, n_variants is ignored.
    """

    # Curriculum: n_ctd values to draw from (2 = hardest, 7 = easiest)
    _CURRICULUM_N_CTD = [2, 3, 4, 5, 6, 7]

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
        augment: bool = False,
        n_variants: int = 4,
        lazy: bool = False,
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
        self.augment = augment
        self._epoch = 0
        self._total_epochs = 1
        self._lazy = lazy
        self._seed = seed
        self._n_variants = max(n_variants, 1)
        # Curriculum: start with max n_ctd, decrease over training
        self._min_n_ctd = max(self._CURRICULUM_N_CTD)

        # mode_mix and seasons — stored for lazy generation
        self._mode_mix = mode_mix or {"realistic": 0.7, "extended": 0.2, "stress": 0.1}
        self._modes = list(self._mode_mix.keys())
        self._weights = np.array([self._mode_mix[m] for m in self._modes], dtype=float)
        self._weights /= self._weights.sum()
        self._all_seasons = ["winter", "spring", "summer", "autumn"]

        if lazy:
            logger.info(
                "Hydro synthetic dataset: lazy mode, n=%d, infinite diversity per epoch",
                n,
            )
            self._scene_variants = None
            self._scene_n_ctd = None
            return

        # --- Pre-generate mode (original behavior) ---
        rng = np.random.default_rng(seed)

        total_scenes = n * self._n_variants
        logger.info(
            "Generating %d hydro scenes (%d samples × %d variants, nx=%d, nz=%d, seed=%d, augment=%s)...",
            total_scenes, n, self._n_variants, nx, nz, seed, augment,
        )

        from .synthetic import _normalize_field

        self._scene_variants = []
        self._scene_n_ctd = []

        for i in range(n):
            sample_n_ctd = int(rng.choice(self._CURRICULUM_N_CTD))
            base_seed = int(rng.integers(0, 2**31))

            variants = []
            for v in range(self._n_variants):
                mode = rng.choice(self._modes, p=self._weights)
                season = rng.choice(self._all_seasons)

                seed_val = base_seed + v
                scene = self._generate_scene(seed_val, mode, season, nx, nz, _normalize_field)
                variants.append(scene)

            self._scene_variants.append(variants)
            self._scene_n_ctd.append(sample_n_ctd)

        logger.info("Hydro synthetic dataset ready: %d samples × %d variants", n, self._n_variants)

    @staticmethod
    def _generate_scene(seed_val, mode, season, nx, nz, normalize_fn):
        """Generate and process a single scene variant.

        Returns dict with T, S, u_lr, v_lr, bathy, below — all float32/bool.
        """
        from .synthetic_baltic import SceneGenerator, Domain

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

        T = normalize_fn(T_raw, mask_water)
        S = normalize_fn(S_raw, mask_water)

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

        return {"T": T, "S": S, "u_lr": u_lr, "v_lr": v_lr, "bathy": bathy, "below": below}

    def set_epoch(self, epoch: int, total_epochs: int):
        """Update curriculum state for the current epoch.

        Gradually reduces minimum n_ctd from max (easiest) to min (hardest).

        Args:
            epoch: Current epoch number (0-indexed).
            total_epochs: Total number of training epochs.
        """
        self._epoch = epoch
        self._total_epochs = max(total_epochs, 1)
        frac = epoch / self._total_epochs
        # Linear schedule: start at max n_ctd, end at min n_ctd
        max_ctd = max(self._CURRICULUM_N_CTD)
        min_ctd = min(self._CURRICULUM_N_CTD)
        self._min_n_ctd = max(min_ctd, int(max_ctd - frac * (max_ctd - min_ctd)))
        logger.debug("Curriculum epoch=%d: min_n_ctd=%d", epoch, self._min_n_ctd)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nz, nx = self.nz, self.nx
        from .synthetic import _normalize_field

        if self._lazy:
            # On-the-fly: fresh scene every call, deterministic seed
            scene_seed = self._seed + idx + self._epoch * 1_000_003
            scene_rng = np.random.default_rng(scene_seed)
            mode = scene_rng.choice(self._modes, p=self._weights)
            season = scene_rng.choice(self._all_seasons)
            sc = self._generate_scene(scene_seed, mode, season, nx, nz, _normalize_field)
            scene_n_ctd = int(scene_rng.choice(self._CURRICULUM_N_CTD))
        else:
            # Pre-generated: select random variant
            variants = self._scene_variants[idx]
            variant_idx = np.random.default_rng().integers(0, len(variants))
            sc = variants[variant_idx]
            scene_n_ctd = self._scene_n_ctd[idx]

        T, S = sc["T"], sc["S"]
        below, bathy = sc["below"], sc["bathy"]
        u_lr, v_lr = sc["u_lr"], sc["v_lr"]

        # Curriculum: use scene's assigned n_ctd, clamped to current minimum
        effective_n_ctd = max(scene_n_ctd, self._min_n_ctd)

        # Stochastic thinning when augment is enabled:
        # randomly reduce n_ctd by up to 30% to simulate sparse coverage.
        # Done here (before sample_observations) rather than as post-hoc
        # augmentation — cleaner, avoids generating observations that are
        # immediately discarded.
        if self.augment:
            thin_rng = np.random.default_rng(seed=idx + self._epoch * 10000 + 999)
            thin_factor = thin_rng.uniform(0.7, 1.0)
            effective_n_ctd = max(2, int(effective_n_ctd * thin_factor))

        rng = np.random.default_rng()

        T_obs, S_obs, mask_ctd, mask_cmems, sigma_obs = sample_observations(
            T, S, below, bathy, nx, nz,
            effective_n_ctd, self.n_cmems_x, self.n_cmems_z,
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
        u_input = u_lr[None].astype(np.float32)

        # Apply augmentation lazily with deterministic per-sample RNG
        if self.augment:
            aug_rng = np.random.default_rng(seed=idx + self._epoch * 10000)
            inp, tgt, bathy_indices, u_input, below = augment_hydro_sample(
                inp, tgt, bathy_indices, u_input, below, aug_rng,
            )
            obs_mask = np.clip(inp[2] + inp[3], 0, 1).astype(np.float32)
            fill_mask = np.stack([below.astype(np.float32)] * 3, axis=0)

        return {
            "image": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
            "mask": torch.from_numpy(below.astype(np.float32)[None]),
            "fill_mask": torch.from_numpy(fill_mask),
            "sigma_obs": torch.from_numpy(sigma_obs[None]),
            "obs_mask": torch.from_numpy(obs_mask[None]),
            "u_input": torch.from_numpy(u_input),
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
