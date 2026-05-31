"""CTD dropout training datasets, sampler, and callbacks.

Provides:
    HybridCTDDataset     — 7-channel dataset with random CTD dropout
    StratifiedCTDSampler — balanced n_ctd representation per batch
    FixedCTDValDataset   — fixed scenes × multiple n_ctd for degradation curves
    stratified_collate_fn — standard collate for 7-channel format
    DegradationCurveCallback — logs per-n_ctd validation RMSE

Input channel mapping (7 channels):
    Ch 0: T_obs      — CMEMS background (smoothed+shifted), CTD overlaid
    Ch 1: S_obs      — same
    Ch 2: mask_ctd   — 1.0 at CTD profile columns
    Ch 3: mask_cmems — 1.0 at CMEMS observation grid points
    Ch 4: u_lr       — zonal velocity (LR upsampled)
    Ch 5: v_lr       — meridional velocity
    Ch 6: bathy_ch   — tiled bathymetry

sigma_obs removed — computed from masks in loss function.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader

logger = logging.getLogger(__name__)


def _normalize_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], NaN/masked regions → 0."""
    valid = field[mask]
    if valid.size == 0:
        return np.zeros_like(field, dtype=np.float32)
    vmin, vmax = float(valid.min()), float(valid.max())
    out = np.zeros_like(field, dtype=np.float32)
    out[mask] = (field[mask] - vmin) / (vmax - vmin + 1e-8)
    return out


def _enforce_stratification(
    T: np.ndarray,
    S: np.ndarray,
    below: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce stable density stratification: ∂ρ/∂z ≥ 0 everywhere.

    After normalization, adjusts S per-column so that density
    ρ = 1 − α·T + β·S increases monotonically with depth.
    T is preserved; S is adjusted minimally.
    Where S would exceed [0, 1], T absorbs the excess.

    Args:
        T: (nz, nx) temperature in [0, 1].
        S: (nz, nx) salinity in [0, 1].
        below: (nz, nx) bool, True below bottom.

    Returns:
        (T_adj, S_adj) — corrected fields in [0, 1].
    """
    from ..eos import linearized_density, salinity_from_density, ALPHA, BETA

    mask = ~below
    nz, nx = T.shape
    S_adj = S.copy()
    T_adj = T.copy()

    for x in range(nx):
        water_z = np.where(mask[:, x])[0]
        if len(water_z) < 2:
            continue
        z0, zN = water_z[0], water_z[-1]

        T_col = T_adj[z0:zN + 1, x].copy()
        S_col = S_adj[z0:zN + 1, x].copy()
        rho = linearized_density(T_col, S_col)

        for i in range(1, len(rho)):
            if rho[i] < rho[i - 1] + 1e-4:
                rho[i] = rho[i - 1] + 1e-4

        S_new = salinity_from_density(rho, T_col)

        out_of_range = (S_new > 1.0) | (S_new < 0.0)
        if out_of_range.any():
            S_new = np.clip(S_new, 0.0, 1.0)
            T_new = (1.0 + BETA * S_new - rho) / ALPHA
            T_adj[z0:zN + 1, x] = np.clip(T_new, 0.0, 1.0)

        S_adj[z0:zN + 1, x] = S_new

    return T_adj, S_adj


def _generate_scene(seed_val: int, mode: str, season: str,
                    nx: int, nz: int) -> dict:
    """Generate and normalize a single Baltic scene.

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

    return {"T": T, "S": S, "u_lr": u_lr, "v_lr": v_lr, "bathy": bathy, "below": below}


def _make_cmems_background(T: np.ndarray, S: np.ndarray,
                           below: np.ndarray, nx: int, nz: int,
                           smoothing_sigma: float = 2.0,
                           shift_max_px: int = 6,
                           rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create biased CMEMS background from truth via smoothing + vertical shift.

    CMEMS reanalysis is a smoothed, slightly shifted version of truth.
    The shift simulates thermocline depth bias in operational products.

    Args:
        T, S: (nz, nx) ground truth in [0, 1].
        below: (nz, nx) bool below-bottom mask.
        nx, nz: grid dimensions.
        smoothing_sigma: Gaussian filter σ for spatial smoothing.
        shift_max_px: max vertical shift in pixels.
        rng: NumPy RNG.

    Returns:
        Tuple of (T_bg, S_bg), each (nz, nx) float32.
    """
    from scipy.ndimage import gaussian_filter

    if rng is None:
        rng = np.random.default_rng(42)

    T_bg = gaussian_filter(T.astype(np.float64), sigma=smoothing_sigma).astype(np.float32)
    S_bg = gaussian_filter(S.astype(np.float64), sigma=smoothing_sigma).astype(np.float32)

    shift = int(rng.integers(-shift_max_px, shift_max_px + 1))
    if shift != 0:
        T_bg = np.roll(T_bg, shift, axis=0)
        S_bg = np.roll(S_bg, shift, axis=0)
        if shift > 0:
            T_bg[:shift, :] = 0.0
            S_bg[:shift, :] = 0.0
        else:
            T_bg[shift:, :] = 0.0
            S_bg[shift:, :] = 0.0

    T_bg[below] = 0.0
    S_bg[below] = 0.0

    return T_bg, S_bg


def _sample_ctd_profiles(T: np.ndarray, S: np.ndarray, below: np.ndarray,
                         bathy: np.ndarray, nx: int, nz: int,
                         n_ctd: int, noise_ctd: float,
                         rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample CTD profiles at random x positions from truth.

    CTD profiles are unbiased ground truth with small measurement noise.

    Returns:
        Tuple of (T_ctd, S_ctd, mask_ctd), each (nz, nx) float32.
    """
    T_ctd = np.zeros((nz, nx), np.float32)
    S_ctd = np.zeros((nz, nx), np.float32)
    mask_ctd = np.zeros((nz, nx), np.float32)

    if n_ctd <= 0:
        return T_ctd, S_ctd, mask_ctd

    ctd_xs = np.sort(rng.choice(nx, min(n_ctd, nx), replace=False))
    for xi in ctd_xs:
        d = int(bathy[xi] * nz)
        if d < 2:
            continue
        mask_ctd[:d, xi] = 1.0
        noise_T = rng.normal(0, noise_ctd, d).astype(np.float32)
        noise_S = rng.normal(0, noise_ctd, d).astype(np.float32)
        T_ctd[:d, xi] = np.clip(T[:d, xi] + noise_T, 0, 1)
        S_ctd[:d, xi] = np.clip(S[:d, xi] + noise_S, 0, 1)

    return T_ctd, S_ctd, mask_ctd


def _sample_cmems_observations(T_bg: np.ndarray, S_bg: np.ndarray,
                               below: np.ndarray, mask_ctd: np.ndarray,
                               nx: int, nz: int,
                               n_cmems_x: int, n_cmems_z: int,
                               noise_cmems: float,
                               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample sparse CMEMS observations from the biased background.

    CMEMS observations are sampled FROM the background (biased), not from truth.

    Returns:
        Tuple of (T_cmems, S_cmems, mask_cmems), each (nz, nx) float32.
    """
    T_cmems = np.zeros((nz, nx), np.float32)
    S_cmems = np.zeros((nz, nx), np.float32)
    mask_cmems = np.zeros((nz, nx), np.float32)

    cmems_z_raw = np.logspace(0, np.log10(max(nz - 1, 1)), n_cmems_z)
    cmems_zs = np.unique(np.clip(cmems_z_raw.astype(int), 0, nz - 1))
    cmems_xs = np.linspace(0, nx - 1, n_cmems_x, dtype=int)

    # Jitter CMEMS grid positions
    if n_cmems_x > 1:
        dx = max(1, (nx - 1) / (n_cmems_x - 1) * 0.4)
        cmems_xs = np.clip(
            cmems_xs + rng.integers(-int(dx), int(dx) + 1, size=n_cmems_x),
            0, nx - 1,
        )
    if len(cmems_zs) > 1:
        dz = max(1, int(np.median(np.diff(cmems_zs)) * 0.4))
        cmems_zs = np.unique(np.clip(
            cmems_zs + rng.integers(-dz, dz + 1, size=len(cmems_zs)),
            0, nz - 1,
        ))

    for zi in cmems_zs:
        for xi in cmems_xs:
            if not below[zi, xi] and mask_ctd[zi, xi] < 0.5:
                mask_cmems[zi, xi] = 1.0
                noise_T = np.float32(rng.normal(0, noise_cmems))
                noise_S = np.float32(rng.normal(0, noise_cmems))
                T_cmems[zi, xi] = np.clip(T_bg[zi, xi] + noise_T, 0, 1)
                S_cmems[zi, xi] = np.clip(S_bg[zi, xi] + noise_S, 0, 1)

    return T_cmems, S_cmems, mask_cmems


def _assemble_7ch(T_bg: np.ndarray, S_bg: np.ndarray,
                  T_ctd: np.ndarray, S_ctd: np.ndarray,
                  mask_ctd: np.ndarray, mask_cmems: np.ndarray,
                  u_lr: np.ndarray, v_lr: np.ndarray,
                  bathy: np.ndarray, nz: int) -> np.ndarray:
    """Assemble 7-channel input tensor.

    Ch 0-1: CMEMS background with CTD overlay at CTD positions.
    At mask_ctd=1: T_obs = T_ctd (truth + noise, overwrites bg)
    At mask_cmems=1: T_obs = T_bg + noise (already in T_cmems, but we use bg directly)
    Elsewhere: T_obs = T_bg

    Returns:
        (7, nz, nx) float32.
    """
    T_obs = T_bg.copy()
    S_obs = S_bg.copy()

    ctd_cols = mask_ctd > 0.5
    T_obs[ctd_cols] = T_ctd[ctd_cols]
    S_obs[ctd_cols] = S_ctd[ctd_cols]

    bathy_ch = np.tile(bathy[None, :], (nz, 1)).astype(np.float32)

    return np.stack(
        [T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathy_ch],
        axis=0,
    ).astype(np.float32)


def _make_cmems_raw_7ch(T_bg: np.ndarray, S_bg: np.ndarray,
                        mask_cmems: np.ndarray) -> np.ndarray:
    """Create cmems_raw for structural loss: (2, nz, nx) with CMEMS bg values at obs points."""
    return np.stack([T_bg, S_bg], axis=0).astype(np.float32)


class HybridCTDDataset(Dataset):
    """7-channel hydro dataset with random CTD dropout.

    Each sample:
        - image:       (7, nz, nx) float32 — hydro input channels
        - target:      (2, nz, nx) float32 — ground truth [T, S]
        - mask:        (1, nz, nx) float32 — below-bottom mask
        - mask_ctd:    (1, nz, nx) float32 — CTD profile mask
        - mask_cmems:  (1, nz, nx) float32 — CMEMS observation mask
        - cmems_raw:   (2, nz, nx) float32 — CMEMS background (for structure loss)
        - u_input:     (1, nz, nx) float32 — zonal velocity (for geostrophic loss)
        - v_input:     (1, nz, nx) float32 — meridional velocity (for geostrophic loss)
        - meta:        dict

    Args:
        n: Number of samples.
        nx: Horizontal grid points.
        nz: Vertical grid points.
        min_ctd: Minimum CTD profiles per sample.
        max_ctd: Maximum CTD profiles per sample.
        n_cmems_x: CMEMS columns per sample.
        n_cmems_z: CMEMS depth levels per column.
        noise_ctd: CTD measurement noise σ.
        noise_cmems: CMEMS measurement noise σ.
        smoothing_sigma: CMEMS background Gaussian filter σ.
        shift_max_px: CMEMS background max vertical shift.
        mode_mix: Scene mode mixture.
        seed: RNG seed.
        augment: Enable augmentation.
        lazy: Generate scenes on-the-fly.
    """

    def __init__(
        self,
        n: int = 400,
        nx: int = 64,
        nz: int = 64,
        min_ctd: int = 0,
        max_ctd: int = 7,
        n_cmems_x: int = 8,
        n_cmems_z: int = 10,
        noise_ctd: float = 0.05,
        noise_cmems: float = 0.10,
        smoothing_sigma: float = 2.0,
        shift_max_px: int = 6,
        mode_mix: Optional[Dict[str, float]] = None,
        seed: int = 42,
        augment: bool = False,
        lazy: bool = True,
    ):
        self.n = n
        self.nx = nx
        self.nz = nz
        self.min_ctd = min_ctd
        self.max_ctd = max_ctd
        self.n_cmems_x = n_cmems_x
        self.n_cmems_z = n_cmems_z
        self.noise_ctd = noise_ctd
        self.noise_cmems = noise_cmems
        self.smoothing_sigma = smoothing_sigma
        self.shift_max_px = shift_max_px
        self.augment = augment
        self._lazy = lazy
        self._seed = seed
        self._epoch = 0

        self._mode_mix = mode_mix or {"realistic": 0.7, "extended": 0.2, "stress": 0.1}
        self._modes = list(self._mode_mix.keys())
        self._weights = np.array([self._mode_mix[m] for m in self._modes], dtype=float)
        self._weights /= self._weights.sum()
        self._all_seasons = ["winter", "spring", "summer", "autumn"]

        if lazy:
            logger.info(
                "HybridCTDDataset: lazy mode, n=%d, n_ctd=[%d..%d], infinite diversity",
                n, min_ctd, max_ctd,
            )

    def set_epoch(self, epoch: int):
        """Update epoch counter for lazy seed derivation."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nz, nx = self.nz, self.nx
        rng = np.random.default_rng(self._seed + idx + self._epoch * 1_000_003)

        mode = rng.choice(self._modes, p=self._weights)
        season = rng.choice(self._all_seasons)
        scene_seed = int(rng.integers(0, 2**31))

        sc = _generate_scene(scene_seed, mode, season, nx, nz)
        T_raw, S_raw = sc["T"], sc["S"]
        below, bathy = sc["below"], sc["bathy"]
        u_lr, v_lr = sc["u_lr"], sc["v_lr"]

        n_ctd = int(rng.integers(self.min_ctd, self.max_ctd + 1))

        T_bg, S_bg = _make_cmems_background(
            T_raw, S_raw, below, nx, nz,
            self.smoothing_sigma, self.shift_max_px, rng,
        )

        T_ctd, S_ctd, mask_ctd = _sample_ctd_profiles(
            T_raw, S_raw, below, bathy, nx, nz, n_ctd, self.noise_ctd, rng,
        )

        _, _, mask_cmems = _sample_cmems_observations(
            T_bg, S_bg, below, mask_ctd, nx, nz,
            self.n_cmems_x, self.n_cmems_z, self.noise_cmems, rng,
        )

        ctd_cols = np.where(mask_ctd[0, :] > 0.5)[0]
        non_ctd_cols = np.array([x for x in range(nx) if x not in set(ctd_cols)])
        if non_ctd_cols.size > 0:
            T_tgt, S_tgt = _enforce_stratification(T_raw, S_raw, below)
            T_tgt[:, ctd_cols] = T_raw[:, ctd_cols]
            S_tgt[:, ctd_cols] = S_raw[:, ctd_cols]
        else:
            T_tgt, S_tgt = T_raw, S_raw

        inp = _assemble_7ch(
            T_bg, S_bg, T_ctd, S_ctd, mask_ctd, mask_cmems,
            u_lr, v_lr, bathy, nz,
        )

        tgt = np.stack([T_tgt, S_tgt], axis=0).astype(np.float32)

        cmems_raw = _make_cmems_raw_7ch(T_bg, S_bg, mask_cmems)

        return {
            "image": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
            "mask": torch.from_numpy(below.astype(np.float32)[None]),
            "mask_ctd": torch.from_numpy(mask_ctd[None]),
            "mask_cmems": torch.from_numpy(mask_cmems[None]),
            "cmems_raw": torch.from_numpy(cmems_raw),
            "u_input": torch.from_numpy(u_lr[None].astype(np.float32)),
            "v_input": torch.from_numpy(v_lr[None].astype(np.float32)),
        }


class StratifiedCTDSampler(Sampler):
    """Balanced sampler ensuring each batch has a mix of n_ctd values.

    Groups dataset indices by their n_ctd value. Each batch draws equally
    from each group so every gradient step sees all difficulty levels.

    Args:
        dataset: HybridCTDDataset instance.
        batch_size: Batch size (must be divisible by number of n_ctd groups).
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(self, dataset: HybridCTDDataset, batch_size: int,
                 drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        n_ctd_range = list(range(dataset.min_ctd, dataset.max_ctd + 1))
        self.n_groups = len(n_ctd_range)

        # Pre-assign n_ctd to each index for consistent grouping
        rng = np.random.default_rng(dataset._seed)
        self._index_n_ctd = np.array([
            int(rng.integers(dataset.min_ctd, dataset.max_ctd + 1))
            for _ in range(len(dataset))
        ])

        self._groups = {n: np.where(self._index_n_ctd == n)[0].tolist()
                        for n in n_ctd_range}

        logger.debug("StratifiedCTDSampler: %d groups, batch_size=%d", self.n_groups, batch_size)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng()
        groups = {k: list(v) for k, v in self._groups.items()}
        for g in groups.values():
            rng.shuffle(g)

        # Round-robin interleave: take one from each group per round,
        # shuffle within each round for stochasticity.
        all_indices = []
        keys = sorted(groups.keys())
        max_len = max((len(g) for g in groups.values()), default=0)
        for i in range(max_len):
            round_indices = []
            for k in keys:
                if i < len(groups[k]):
                    round_indices.append(groups[k][i])
            rng.shuffle(round_indices)
            all_indices.extend(round_indices)

        for start in range(0, len(all_indices), self.batch_size):
            batch = all_indices[start:start + self.batch_size]
            if len(batch) == self.batch_size or (not self.drop_last and batch):
                yield batch


class FixedCTDValDataset(Dataset):
    """Validation dataset: same scenes × multiple fixed n_ctd values.

    Each scene is repeated for every n_ctd in val_n_ctd_list, enabling
    degradation curve analysis (RMSE vs number of CTD profiles).

    Args:
        n_scenes: Number of unique validation scenes.
        val_n_ctd_list: List of n_ctd values to evaluate.
        nx, nz: Grid dimensions.
        seed: RNG seed for scene generation.
        n_cmems_x, n_cmems_z: CMEMS grid parameters.
        noise_ctd, noise_cmems: Noise levels.
        smoothing_sigma, shift_max_px: CMEMS background parameters.
    """

    def __init__(
        self,
        n_scenes: int = 20,
        val_n_ctd_list: Optional[List[int]] = None,
        nx: int = 64,
        nz: int = 64,
        seed: int = 142,
        n_cmems_x: int = 8,
        n_cmems_z: int = 10,
        noise_ctd: float = 0.05,
        noise_cmems: float = 0.10,
        smoothing_sigma: float = 2.0,
        shift_max_px: int = 6,
        mode_mix: Optional[Dict[str, float]] = None,
    ):
        self.n_scenes = n_scenes
        self.val_n_ctd_list = val_n_ctd_list or [0, 1, 2, 3, 5, 7]
        self.nx = nx
        self.nz = nz
        self.seed = seed
        self.n_cmems_x = n_cmems_x
        self.n_cmems_z = n_cmems_z
        self.noise_ctd = noise_ctd
        self.noise_cmems = noise_cmems
        self.smoothing_sigma = smoothing_sigma
        self.shift_max_px = shift_max_px

        self._scenes = []
        rng = np.random.default_rng(seed)
        modes = ["realistic", "extended", "stress"]
        if mode_mix is not None:
            mode_weights = np.array([mode_mix.get(m, 0.0) for m in modes])
        else:
            mode_weights = np.array([0.7, 0.2, 0.1])
        seasons = ["winter", "spring", "summer", "autumn"]

        for i in range(n_scenes):
            mode = rng.choice(modes, p=mode_weights)
            season = rng.choice(seasons)
            scene_seed = int(rng.integers(0, 2**31))
            sc = _generate_scene(scene_seed, mode, season, nx, nz)
            self._scenes.append(sc)

        logger.info(
            "FixedCTDValDataset: %d scenes × %d n_ctd values = %d samples",
            n_scenes, len(self.val_n_ctd_list), len(self),
        )

    def __len__(self) -> int:
        return self.n_scenes * len(self.val_n_ctd_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_idx = idx // len(self.val_n_ctd_list)
        n_ctd_idx = idx % len(self.val_n_ctd_list)
        n_ctd = self.val_n_ctd_list[n_ctd_idx]

        sc = self._scenes[scene_idx]
        T_raw, S_raw = sc["T"], sc["S"]
        below, bathy = sc["below"], sc["bathy"]
        u_lr, v_lr = sc["u_lr"], sc["v_lr"]
        nz, nx = self.nz, self.nx

        rng = np.random.default_rng(self.seed + scene_idx * 1000 + n_ctd)

        T_bg, S_bg = _make_cmems_background(
            T_raw, S_raw, below, nx, nz,
            self.smoothing_sigma, self.shift_max_px, rng,
        )

        T_ctd, S_ctd, mask_ctd = _sample_ctd_profiles(
            T_raw, S_raw, below, bathy, nx, nz, n_ctd, self.noise_ctd, rng,
        )

        _, _, mask_cmems = _sample_cmems_observations(
            T_bg, S_bg, below, mask_ctd, nx, nz,
            self.n_cmems_x, self.n_cmems_z, self.noise_cmems, rng,
        )

        ctd_cols = np.where(mask_ctd[0, :] > 0.5)[0]
        non_ctd_cols = np.array([x for x in range(nx) if x not in set(ctd_cols)])
        if non_ctd_cols.size > 0:
            T_tgt, S_tgt = _enforce_stratification(T_raw, S_raw, below)
            T_tgt[:, ctd_cols] = T_raw[:, ctd_cols]
            S_tgt[:, ctd_cols] = S_raw[:, ctd_cols]
        else:
            T_tgt, S_tgt = T_raw, S_raw

        inp = _assemble_7ch(
            T_bg, S_bg, T_ctd, S_ctd, mask_ctd, mask_cmems,
            u_lr, v_lr, bathy, nz,
        )

        tgt = np.stack([T_tgt, S_tgt], axis=0).astype(np.float32)
        cmems_raw = _make_cmems_raw_7ch(T_bg, S_bg, mask_cmems)

        return {
            "image": torch.from_numpy(inp),
            "target": torch.from_numpy(tgt),
            "mask": torch.from_numpy(below.astype(np.float32)[None]),
            "mask_ctd": torch.from_numpy(mask_ctd[None]),
            "mask_cmems": torch.from_numpy(mask_cmems[None]),
            "cmems_raw": torch.from_numpy(cmems_raw),
            "u_input": torch.from_numpy(u_lr[None].astype(np.float32)),
            "v_input": torch.from_numpy(v_lr[None].astype(np.float32)),
            "n_ctd": torch.tensor(n_ctd),
        }


def stratified_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Standard collate for 7-channel format."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals, dim=0)
        else:
            result[k] = vals
    return result


class DegradationCurveCallback:
    """Logs per-n_ctd validation RMSE to TensorBoard.

    At the end of each validation epoch, runs the model on FixedCTDValDataset
    and logs RMSE for each n_ctd value.

    Args:
        val_dataset: FixedCTDValDataset instance.
        batch_size: Batch size for evaluation.
        log_every: Log every N epochs.
    """

    def __init__(self, val_dataset: FixedCTDValDataset,
                 batch_size: int = 4, log_every: int = 1):
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.log_every = log_every
        self.n_ctd_list = val_dataset.val_n_ctd_list
        self._oi_by_n: dict = {}  # cached OI RMSE (model-independent constants)

    def log_degradation_curve(self, trainer, pl_module):
        """Evaluate and log per-n_ctd RMSE.

        LaMa RMSE is logged every call.  OI baseline RMSE is computed once
        (first call) and logged at that step — it is model-independent and
        constant across epochs, serving as a reference line in TensorBoard.
        """
        from scipy.interpolate import griddata

        epoch = trainer.current_epoch
        if epoch % self.log_every != 0:
            return

        device = next(pl_module.parameters()).device
        pl_module.eval()

        dl = DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4,
            collate_fn=stratified_collate_fn,
        )

        rmse_by_n = {n: [] for n in self.n_ctd_list}
        oi_by_n: dict = {}  # per-n accumulators for OI (filled only on first call)
        compute_oi = not self._oi_by_n  # first call only

        with torch.no_grad():
            for batch in dl:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}
                batch_dev = pl_module(batch_dev)
                pred = batch_dev["predicted_image"]
                tgt = batch_dev["target"]
                mask_below = batch_dev["mask"]
                mask_ctd = batch_dev.get("mask_ctd")
                mask_cmems = batch_dev.get("mask_cmems")

                n_ctd_vals = batch_dev["n_ctd"]
                for i in range(pred.shape[0]):
                    n = int(n_ctd_vals[i])
                    if n not in rmse_by_n:
                        continue
                    domain = 1.0 - mask_below[i]
                    err = (pred[i] - tgt[i]) * domain
                    mse = err.pow(2).mean().item()
                    rmse_by_n[n].append(mse ** 0.5)

                    if compute_oi and mask_ctd is not None and mask_cmems is not None:
                        obs = (mask_ctd[i, 0].cpu().numpy() + mask_cmems[i, 0].cpu().numpy())
                        obs = np.clip(obs, 0, 1)
                        tgt_np = tgt[i].cpu().numpy()
                        below_np = mask_below[i, 0].cpu().numpy().astype(bool)
                        ch_in, nz, nx = tgt_np.shape
                        pts = np.argwhere(obs > 0.5)
                        if len(pts) >= 4:
                            zi_all = np.arange(nz)
                            xi_all = np.arange(nx)
                            ZI, XI = np.meshgrid(zi_all, xi_all, indexing="ij")
                            grid_coords = np.stack([ZI.ravel(), XI.ravel()], axis=1)
                            oi_channels = []
                            for ch in range(ch_in):
                                vals = tgt_np[ch][pts[:, 0], pts[:, 1]]
                                interp = griddata(pts, vals, grid_coords,
                                                  method="cubic", fill_value=0.0)
                                oi_channels.append(interp.reshape(nz, nx))
                            oi = np.stack(oi_channels, axis=0).astype(np.float32)
                            domain_np = (1.0 - below_np.astype(np.float32))
                            oi_err = (oi - tgt_np) * domain_np
                            oi_mse = float(np.mean(oi_err ** 2))
                            oi_by_n.setdefault(n, []).append(oi_mse ** 0.5)

        # Log LaMa RMSE every call
        for n in self.n_ctd_list:
            vals = rmse_by_n[n]
            if vals:
                avg_rmse = sum(vals) / len(vals)
                trainer.logger.log_metrics(
                    {f"val_rmse/n{n}": avg_rmse}, step=trainer.global_step,
                )
                logger.info("  [degradation] n_ctd=%d LaMa RMSE=%.4f (%d samples)",
                            n, avg_rmse, len(vals))

        # Log OI baseline once (model-independent constants)
        if compute_oi and oi_by_n:
            for n, vals in oi_by_n.items():
                self._oi_by_n[n] = sum(vals) / len(vals)
            metrics = {f"val_rmse/oi_n{n}": v for n, v in self._oi_by_n.items()}
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            logger.info("  [degradation] OI baseline (logged once): %s",
                        {n: f"{v:.4f}" for n, v in self._oi_by_n.items()})

        pl_module.train()
