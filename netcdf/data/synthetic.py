"""Realistic synthetic oceanographic data generators.

Ported and enhanced from bin/lama_hydro_simple_end2end/train.py.
Generates physically-motivated 2D cross-sections of temperature,
salinity, and velocity fields for training without real NetCDF data.

All generators use a seeded ``np.random.Generator`` for strict
reproducibility.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Final, Optional, Tuple

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
