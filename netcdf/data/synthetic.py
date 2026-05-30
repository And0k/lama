"""Realistic synthetic oceanographic data generators.

Ported and enhanced from bin/lama_hydro_simple_end2end/train.py.
Generates physically-motivated 2D cross-sections of temperature,
salinity, and velocity fields for training without real NetCDF data.

Two generation backends:
    - Simple (sigmoid+eddy): ``make_synthetic_sample()`` — fast, minimal physics
    - Baltic (SceneGenerator): ``make_hydro_synthetic_sample()`` — full climatology,
      multiple scene types, seasonal variation, halocline/CIL physics

All generators use a seeded ``np.random.Generator`` for strict
reproducibility.
"""
import logging

import numpy as np
from numpy.typing import NDArray
from typing import Final, Optional, Tuple

logger = logging.getLogger(__name__)

def _sigmoid(z: NDArray[np.float64], z0: float, k: float) -> NDArray[np.float64]:
    """Compute numerically stable sigmoid profile."""
    return 1.0 / (1.0 + np.exp(k * (z - z0)))


def _gaussian_eddy(
    XX: NDArray[np.float64],
    ZZ: NDArray[np.float64],
    cx: float,
    cz: float,
    rx: float,
    rz: float,
) -> NDArray[np.float64]:
    """Generate a normalized 2D Gaussian anomaly field."""
    return np.exp(-((XX - cx) ** 2 / (2 * rx**2) + (ZZ - cz) ** 2 / (2 * rz**2)))


def _normalize_to_unit(field: NDArray[np.float64]) -> NDArray[np.float32]:
    """Min-max normalize to [0, 1] and cast to float32."""
    eps: Final[float] = 1e-8
    f_min, f_max = field.min(), field.max()
    return ((field - f_min) / (f_max - f_min + eps)).astype(np.float32)


def synthetic_TS_field(
    nx: int, nz: int, rng: np.random.Generator
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate correlated T/S cross-sections via shared parameterized components.

    Eliminates redundancy by defining stratification and perturbation specs
    as data, applying identical operations through vectorized iteration.

    Args:
        nx: Horizontal grid points.
        nz: Vertical grid points.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        Tuple of (temperature, salinity) arrays shaped (nz, nx),
        min-max normalized to [0, 1] as float32.
    """
    XX, ZZ = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nz))

    # Shared eddy geometry computed once for thermohaline covariance
    cx, cz = rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.5)
    rx, rz = rng.uniform(0.1, 0.25), rng.uniform(0.05, 0.15)
    eddy = _gaussian_eddy(XX, ZZ, cx, cz, rx, rz)

    return tuple(
        _normalize_to_unit(
            _sigmoid(ZZ, *rng.uniform(*z_r), sign * rng.uniform(*k_r))
            + rng.uniform(*tilt_r) * XX
            + rng.uniform(*amp_r) * eddy
        )
        for z_r, k_r, sign, tilt_r, amp_r in (
            ((0.15, 0.40), (8.0, 20.0), 1.0, (-0.15, 0.15), (-0.25, 0.25)),  # Temperature
            ((0.20, 0.45), (6.0, 16.0), -1.0, (-0.10, 0.10), (-0.15, 0.15)),  # Salinity
        )
    )


def synthetic_V_field(nx: int, nz: int, T_field: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """Generate velocity magnitude from geostrophic balance.

    In geostrophic balance, horizontal velocity gradients are proportional
    to horizontal density (temperature) gradients.  The velocity magnitude
    is modelled as |∂T/∂x| with added vertical shear and noise.

    Args:
        nx: horizontal grid points.
        nz: vertical grid points.
        T_field: (nz, nx) temperature field in [0, 1].
        rng: seeded NumPy generator.

    Returns:
        (nz, nx) float32 array normalised to [0, 1].
    """
    # Geostrophic: |V| ~ |dT/dx|
    dT_dx = np.gradient(T_field, axis=1)
    V = np.abs(dT_dx)

    # Add vertical shear component
    z = np.linspace(0, 1, nz)[:, None]
    shear = rng.uniform(0.5, 1.5) * np.exp(-z * rng.uniform(1.0, 3.0))
    V += shear * rng.uniform(0.05, 0.15)

    # Mesoscale eddy contribution
    x = np.linspace(0, 1, nx)
    XX, ZZ = np.meshgrid(x, z.flat)
    cx, cz = rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.5)
    rx, rz = rng.uniform(0.1, 0.25), rng.uniform(0.05, 0.15)
    amp = rng.uniform(0.05, 0.20)
    V += amp * np.exp(-((XX - cx) ** 2 / (2 * rx ** 2) + (ZZ - cz) ** 2 / (2 * rz ** 2)))

    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V.astype(np.float32)


def sloping_bathymetry(nx: int, rng: np.random.Generator) -> np.ndarray:
    """Generate monotonically sloping bathymetry.

    Returns:
        (nx,) float32 array of normalised bottom depth in [0, 1].
    """
    base = np.linspace(rng.uniform(0.5, 0.7), rng.uniform(0.75, 0.95), nx)
    noise = rng.normal(0, 0.02, nx).cumsum() * 0.1
    bathy = np.clip(base + noise, 0.4, 0.99)
    return bathy.astype(np.float32)


def cmems_z_indices(nz: int, n_cols: int = 8) -> np.ndarray:
    """Quasi-regular vertical grid with log-spacing near the surface.

    Mimics standard CMEMS depth levels (denser near surface).

    Returns:
        Array of integer indices into [0, nz).
    """
    raw = np.logspace(0, np.log10(nz), n_cols, endpoint=False)
    return np.unique(np.clip(raw.astype(int), 0, nz - 1))


def make_synthetic_sample(nx: int, nz: int, n_ctd: int, n_cmems_x: int,
                           noise_ctd: float, noise_cmems: float,
                           rng: np.random.Generator) -> dict:
    """Generate a single synthetic training sample.

    Returns a dict with keys matching the NetCDFDataset output format:
        image:      (3, nz, nx) float32 — [|V|_field, thetao_field, so_field]
        mask:       (1, nz, nx) float32 — observation mask (CTD + CMEMS)
        fill_mask:  (3, nz, nx) float32 — 1.0 below bottom (invalid)
        bathy:      (nx,)       float32 — normalised bottom depth
        below:      (nz, nx)    bool    — True below bottom

    The image channels are in [0, 1] (normalised physical space).
    The mask channel is 1.0 where observations exist, 0.0 elsewhere.
    """
    T, S = synthetic_TS_field(nx, nz, rng)
    V = synthetic_V_field(nx, nz, T, rng)
    bathy = sloping_bathymetry(nx, rng)

    # Mask below bottom
    z_norm = np.linspace(0, 1, nz)[:, None]  # (nz, 1)
    below = z_norm > bathy[None, :]  # (nz, nx) bool

    T[below] = 0.0
    S[below] = 0.0
    V[below] = 0.0

    # --- CTD profiles: random x positions, vertical lines ---
    ctd_xs = np.sort(rng.choice(nx, n_ctd, replace=False))
    mask_ctd = np.zeros((nz, nx), np.float32)
    field_T = np.zeros((nz, nx), np.float32)
    field_S = np.zeros((nz, nx), np.float32)
    field_V = np.zeros((nz, nx), np.float32)

    for xi in ctd_xs:
        depth_col = int(bathy[xi] * nz)
        mask_ctd[:depth_col, xi] = 1.0
        field_T[:depth_col, xi] = T[:depth_col, xi] + rng.normal(
            0, noise_ctd, depth_col
        ).astype(np.float32)
        field_S[:depth_col, xi] = S[:depth_col, xi] + rng.normal(
            0, noise_ctd, depth_col
        ).astype(np.float32)
        field_V[:depth_col, xi] = V[:depth_col, xi] + rng.normal(
            0, noise_ctd, depth_col
        ).astype(np.float32)

    # --- CMEMS points: sparse quasi-regular grid ---
    cmems_zs = cmems_z_indices(nz, n_cmems_x)
    cmems_xs = np.linspace(0, nx - 1, n_cmems_x, dtype=int)
    mask_cmems = np.zeros((nz, nx), np.float32)

    for zi in cmems_zs:
        for xi in cmems_xs:
            if z_norm[zi, 0] < bathy[xi]:
                mask_cmems[zi, xi] = 1.0
                noise_factor = np.float32(rng.normal(0, noise_cmems))
                if mask_ctd[zi, xi] == 0:
                    field_T[zi, xi] = T[zi, xi] + noise_factor
                    field_S[zi, xi] = S[zi, xi] + noise_factor
                    field_V[zi, xi] = V[zi, xi] + abs(noise_factor) * 0.5

    # Combined observation mask
    mask_obs = np.clip(mask_ctd + mask_cmems, 0, 1)

    # Stack channels: [|V|, thetao, so]
    image = np.stack([field_V, field_T, field_S], axis=0)  # (3, nz, nx)

    # Fill mask: 1.0 where invalid (below bottom)
    fill_mask = np.stack([below.astype(np.float32)] * 3, axis=0)  # (3, nz, nx)

    return {
        "image": image,
        "mask": mask_obs[None],  # (1, nz, nx)
        "fill_mask": fill_mask,
        "bathy": bathy,
        "below": below,
    }


# ── Hydro pipeline: Baltic SceneGenerator integration ──────────────────────


def _normalize_field(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1], NaN/masked regions → 0.

    Args:
        field: (nz, nx) raw field with NaN below bottom.
        mask:  (nz, nx) bool, True where water (valid).

    Returns:
        (nz, nx) float32 in [0, 1].
    """
    valid = field[mask]
    if valid.size == 0:
        return np.zeros_like(field, dtype=np.float32)
    vmin, vmax = float(valid.min()), float(valid.max())
    out = np.zeros_like(field, dtype=np.float32)
    out[mask] = (field[mask] - vmin) / (vmax - vmin + 1e-8)
    return out


def sample_observations(
    T: NDArray[np.float32],
    S: NDArray[np.float32],
    below: NDArray[np.bool_],
    bathy: NDArray[np.float32],
    nx: int,
    nz: int,
    n_ctd: int = 5,
    n_cmems_x: int = 8,
    n_cmems_z: int = 10,
    noise_ctd: float = 0.05,
    noise_cmems: float = 0.10,
    rng: np.random.Generator | None = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32],
           NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Sample random observation positions and generate noisy values.

    Generates CTD profiles (vertical lines at random x) and CMEMS
    points (sparse log-spaced grid) with Gaussian measurement noise.

    Args:
        T: (nz, nx) ground truth temperature in [0, 1].
        S: (nz, nx) ground truth salinity in [0, 1].
        below: (nz, nx) bool, True below seabed.
        bathy: (nx,) normalised bottom depth in [0, 1].
        nx: Horizontal grid points.
        nz: Vertical grid points.
        n_ctd: Number of CTD profiles.
        n_cmems_x: Number of CMEMS x-columns.
        n_cmems_z: Number of CMEMS z-levels.
        noise_ctd: CTD measurement noise σ.
        noise_cmems: CMEMS measurement noise σ.
        rng: Seeded NumPy generator.

    Returns:
        Tuple of (T_obs, S_obs, mask_ctd, mask_cmems, sigma_obs),
        each (nz, nx) float32.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T_obs = np.zeros((nz, nx), np.float32)
    S_obs = np.zeros((nz, nx), np.float32)
    mask_ctd = np.zeros((nz, nx), np.float32)
    mask_cmems = np.zeros((nz, nx), np.float32)
    sigma_obs = np.zeros((nz, nx), np.float32)

    ctd_xs = np.sort(rng.choice(nx, min(n_ctd, nx), replace=False))
    for xi in ctd_xs:
        d = int(bathy[xi] * nz)
        if d < 2:
            continue
        mask_ctd[:d, xi] = 1.0
        sigma_obs[:d, xi] = noise_ctd
        T_obs[:d, xi] = np.clip(
            T[:d, xi] + rng.normal(0, noise_ctd, d).astype(np.float32), 0, 1
        )
        S_obs[:d, xi] = np.clip(
            S[:d, xi] + rng.normal(0, noise_ctd, d).astype(np.float32), 0, 1
        )

    cmems_z_raw = np.logspace(0, np.log10(max(nz - 1, 1)), n_cmems_z)
    cmems_zs = np.unique(np.clip(cmems_z_raw.astype(int), 0, nz - 1))
    cmems_xs = np.linspace(0, nx - 1, n_cmems_x, dtype=int)

    for zi in cmems_zs:
        for xi in cmems_xs:
            if not below[zi, xi] and mask_ctd[zi, xi] < 0.5:
                mask_cmems[zi, xi] = 1.0
                sigma_obs[zi, xi] = noise_cmems
                T_obs[zi, xi] = np.clip(
                    T[zi, xi] + np.float32(rng.normal(0, noise_cmems)), 0, 1
                )
                S_obs[zi, xi] = np.clip(
                    S[zi, xi] + np.float32(rng.normal(0, noise_cmems)), 0, 1
                )

    return T_obs, S_obs, mask_ctd, mask_cmems, sigma_obs


def make_hydro_synthetic_sample(
    nx: int = 64,
    nz: int = 64,
    n_ctd: int = 5,
    n_cmems_x: int = 8,
    n_cmems_z: int = 10,
    noise_ctd: float = 0.05,
    noise_cmems: float = 0.10,
    rng: np.random.Generator | None = None,
    mode: str = "realistic",
    season: str = "summer",
) -> dict:
    """Generate a single 8-channel hydro training sample using Baltic physics.

    Uses ``synthetic_baltic.SceneGenerator`` for physically realistic T/S/u/v
    fields, then samples CTD profiles and CMEMS points to build the 8-channel
    input tensor expected by ``HydroGenerator``.

    Input channels (8, nz, nx):
        0: T_obs       — observed T values, 0 outside observations
        1: S_obs       — observed S values, 0 outside observations
        2: mask_ctd    — 1 where CTD profile exists (high-trust vertical line)
        3: mask_cmems  — 1 where CMEMS point exists (low-trust sparse point)
        4: u_lr        — zonal velocity, low-res upsampled
        5: v_lr        — meridional velocity, low-res upsampled
        6: bathymetry  — continuous bottom depth ∈ [0,1], broadcast over z
        7: sigma_obs   — per-point uncertainty: 0.05 CTD, 0.10 CMEMS, 0 elsewhere

    Target channels (2, nz, nx):
        0: T_pred      — ground truth temperature ∈ [0, 1]
        1: S_pred      — ground truth salinity ∈ [0, 1]

    Args:
        nx: Horizontal grid points.
        nz: Vertical grid points.
        n_ctd: Number of CTD profiles per sample.
        n_cmems_x: Number of CMEMS columns per sample.
        n_cmems_z: Number of CMEMS depth levels per column.
        noise_ctd: CTD measurement noise σ.
        noise_cmems: CMEMS measurement noise σ.
        rng: Seeded NumPy generator. Created from seed if None.
        mode: Baltic generation mode ("realistic", "extended", "stress").
        season: Season ("winter", "spring", "summer", "autumn").

    Returns:
        Dict with keys: image, target, mask, fill_mask, bathy, below,
        sigma_obs, obs_mask, u_input, bathy_indices.
    """
    from .synthetic_baltic import SceneGenerator, Domain

    if rng is None:
        rng = np.random.default_rng(42)

    seed_val = int(rng.integers(0, 2**31))
    gen = SceneGenerator(seed=seed_val)
    domain = Domain(nx=nx, nz=nz)
    scene = gen.generate(domain, mode=mode, season=season)

    T_raw = scene["T"]   # (nz, nx), NaN below bottom
    S_raw = scene["S"]
    u_raw = scene["u"]
    v_raw = scene["v"]
    bottom_m = scene["bottom"]  # (nx,) in metres
    mask_water = scene["mask"]  # (nz, nx) bool, True where water

    # Normalised bathymetry ∈ [0, 1]
    bathy = (bottom_m / domain.zmax).astype(np.float32)

    # Below-seabed mask
    z_norm = np.linspace(0, 1, nz)[:, None]
    below = ~mask_water  # (nz, nx) bool

    # Normalise T, S to [0, 1] within water domain
    T = _normalize_field(T_raw, mask_water)
    S = _normalize_field(S_raw, mask_water)

    # Normalise u, v: replace NaN below bottom with 0, then scale to [0, 1]
    u_arr = np.nan_to_num(u_raw.astype(np.float64), nan=0.0)
    v_arr = np.nan_to_num(v_raw.astype(np.float64), nan=0.0)
    valid_u = u_arr[mask_water]
    valid_v = v_arr[mask_water]
    u_max = max(float(np.abs(valid_u).max()) if valid_u.size > 0 else 1.0, 1e-8)
    v_max = max(float(np.abs(valid_v).max()) if valid_v.size > 0 else 1.0, 1e-8)
    u_arr = np.clip(u_arr / u_max, -1.0, 1.0)
    v_arr = np.clip(v_arr / v_max, -1.0, 1.0)

    # LR simulation: downsample velocity then upsample (mimics CMEMS coarse grid)
    factor = max(1, min(nx, nz) // 16)
    if factor > 1:
        u_ds = u_arr[::factor, ::factor]
        v_ds = v_arr[::factor, ::factor]
        u_lr = np.kron(u_ds, np.ones((factor, factor)))[:nz, :nx].astype(np.float32)
        v_lr = np.kron(v_ds, np.ones((factor, factor)))[:nz, :nx].astype(np.float32)
    else:
        u_lr = u_arr.astype(np.float32)
        v_lr = v_arr.astype(np.float32)

    # Shift u_lr, v_lr from [-1, 1] to [0, 1] for network input
    u_lr = np.clip((u_lr + 1.0) / 2.0, 0.0, 1.0)
    v_lr = np.clip((v_lr + 1.0) / 2.0, 0.0, 1.0)

    # Zero out below-bottom in velocity channels
    u_lr[below] = 0.0
    v_lr[below] = 0.0

    # --- Observation channels ---
    T_obs, S_obs, mask_ctd, mask_cmems, sigma_obs = sample_observations(
        T, S, below, bathy, nx, nz,
        n_ctd, n_cmems_x, n_cmems_z, noise_ctd, noise_cmems, rng,
    )

    # Bathymetry channel: continuous depth value broadcast over z
    bathy_ch = np.tile(bathy[None, :], (nz, 1)).astype(np.float32)

    # Mask below seabed in T, S targets
    T[below] = 0.0
    S[below] = 0.0

    # Stack 8 input channels
    inp = np.stack(
        [T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathy_ch, sigma_obs],
        axis=0,
    ).astype(np.float32)  # (8, nz, nx)

    # Stack 2 target channels
    tgt = np.stack([T, S], axis=0).astype(np.float32)  # (2, nz, nx)

    # Combined observation mask (CTD ∪ CMEMS)
    obs_mask = np.clip(mask_ctd + mask_cmems, 0, 1).astype(np.float32)

    # Bathymetry as integer indices for BBL loss
    bathy_indices = np.clip((bathy * nz).astype(int), 1, nz - 1)

    # Below-bottom as float for fill_mask (3ch for compatibility)
    fill_mask = np.stack([below.astype(np.float32)] * 3, axis=0)

    return {
        "image": inp,           # (8, nz, nx) — generator input
        "target": tgt,          # (2, nz, nx) — ground truth T,S
        "mask": below.astype(np.float32)[None],  # (1, nz, nx) — below-bottom mask
        "fill_mask": fill_mask, # (3, nz, nx) — compatibility
        "bathy": bathy,         # (nx,) — normalised bottom depth
        "below": below,         # (nz, nx) bool
        "sigma_obs": sigma_obs[None],  # (1, nz, nx) — per-point uncertainty
        "obs_mask": obs_mask[None],    # (1, nz, nx) — observation locations
        "u_input": u_lr[None],         # (1, nz, nx) — zonal velocity input
        "bathy_indices": bathy_indices, # (nx,) — int indices for BBL loss
    }
