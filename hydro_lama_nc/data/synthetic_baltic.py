"""
Baltic Sea Synthetic Hydrophysical Section Generator
=====================================================
Generates 2-D (x×z = 256×256) fields T, S, u, v over a 200 km × 200 m domain
representative of the central and SE Baltic Sea.

Physics model — superposition of primitive kernels:
  Interface  →  tanh transition   (thermocline / halocline / front)
  Lens       →  2-D Gaussian      (CIL / intrusion / anomaly)
  Jet        →  2-D Gaussian      (current)
  Bathymetry →  Σ Gaussians + base

Three generation modes (mix for inpainting training: 70 / 20 / 10 %):
  realistic  — climatological Baltic ranges
  extended   — ×1.5 extended ranges, stronger slopes & waves
  stress     — extreme / rare cases (sharp fronts, large CIL, exotic profiles)

Four seasons: winter | spring | summer | autumn
  Season controls surface T range, CIL presence (Baltic-specific ~2–5 °C
  at 30–80 m), thermocline sharpness, and halocline depth.

Output dict per scene
---------------------
  T, S, u, v  : (nz, nx) float64  — NaN below bottom
  bottom       : (nx,)   float64  — depth of sea floor [m]
  mask         : (nz, nx) bool    — True where water
  x            : (nx,)   float64  — horizontal coordinate [m]
  z            : (nz,)   float64  — vertical coordinate, positive down [m]
  meta         : dict    — {mode, season, scene_type, seed}

References
----------
HELCOM Baltic Sea Environment Fact Sheets; Väli et al. 2013 (Baltic CTD climatology);
spec doc "design-to-code contract" (internal, 2025).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, TypedDict

# ---------------------------------------------------------------------------
# TYPE ALIASES
# ---------------------------------------------------------------------------

Mode    = Literal["realistic", "extended", "stress"]
Season  = Literal["winter", "spring", "summer", "autumn"]
ScType  = Literal["background", "internal_waves", "upwelling",
                  "downwelling", "front", "intrusion", "jets"]

# ---------------------------------------------------------------------------
# CONSTANTS — Baltic climatology (central + SE Baltic)
# ---------------------------------------------------------------------------

# Surface T [°C] per season — realistic mode
SURF_T: dict[Season, tuple[float, float]] = {
    "winter":  (0.0,  4.0),
    "spring":  (2.0, 14.0),
    "summer": (14.0, 22.0),
    "autumn":  (6.0, 16.0),
}

# Bottom T [°C] — weakly seasonal in Baltic
BOTT_T: dict[Season, tuple[float, float]] = {
    "winter": (2.0, 6.0),
    "spring": (2.0, 6.0),
    "summer": (2.0, 8.0),
    "autumn": (2.0, 7.0),
}

# CIL (Cold Intermediate Layer) — Baltic-specific residual winter water
# z range [m], T range [°C], PSU contribution ≈ 0
CIL_Z:    tuple[float, float] = (30.0,  80.0)   # depth band
CIL_T:    tuple[float, float] = (2.0,   5.0)    # temperature anomaly (cold)
CIL_SX:   tuple[float, float] = (20_000, 80_000) # horizontal extent [m]
CIL_SZ:   tuple[float, float] = (8.0,  25.0)   # vertical extent [m]

# Halocline — Baltic permanent feature ~60–90 m
HALO_Z:   tuple[float, float] = (55.0,  90.0)   # [m]
HALO_DS:  tuple[float, float] = (6.0,  12.0)   # salinity jump [PSU] — min raised for sharper jumps
HALO_TH:  tuple[float, float] = (3.0,   8.0)   # thickness [m] — tightened for frequent sharp haloclines

# Surface S [PSU]
SURF_S:   tuple[float, float] = (5.0,  10.0)
BOTT_S:   tuple[float, float] = (10.0, 20.0)

# Scene-type prior probabilities (§11 of spec)
SCENE_PROBS: dict[ScType, float] = {
    "background":     0.15,
    "internal_waves": 0.20,
    "upwelling":      0.15,
    "downwelling":    0.10,
    "front":          0.15,
    "intrusion":      0.10,
    "jets":           0.15,
}
_SCENE_TYPES: list[ScType] = list(SCENE_PROBS.keys())
_SCENE_W     = np.array([SCENE_PROBS[k] for k in _SCENE_TYPES])


# ---------------------------------------------------------------------------
# DOMAIN
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Domain:
    """Fixed 256×256 grid over 200 km × 200 m."""
    nx:   int   = 256
    nz:   int   = 256
    xmax: float = 200_000.0   # [m]
    zmax: float = 200.0       # [m]

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns x[nx], z[nz] coordinate arrays."""
        return (
            np.linspace(0.0, self.xmax, self.nx),
            np.linspace(0.0, self.zmax, self.nz),
        )


# ---------------------------------------------------------------------------
# MODE SCALE FACTORS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeScale:
    """
    Multiplicative/additive modifiers applied on top of realistic ranges.
      realistic : scale=1.0, extra_slope=1.0, extra_wave=1.0
      extended  : wider T/S ranges, stronger waves & slopes
      stress    : extreme — rare physics pushed to limits
    """
    T_range_scale:   float = 1.0   # multiplies half-width of T surface range
    S_range_scale:   float = 1.0
    slope_scale:     float = 1.0   # multiplies interface slope magnitude
    wave_scale:      float = 1.0   # multiplies wave amplitude
    lens_amp_scale:  float = 1.0   # multiplies lens amplitudes
    n_iface_extra:   int   = 0     # additional interfaces beyond scene default
    bathy_amp_scale: float = 1.0   # bathymetry relief amplitude scale

MODE_SCALES: dict[Mode, ModeScale] = {
    "realistic": ModeScale(),
    "extended":  ModeScale(
        T_range_scale=1.5, S_range_scale=1.5,
        slope_scale=1.8,   wave_scale=1.8,
        lens_amp_scale=1.3, n_iface_extra=1, bathy_amp_scale=1.2,
    ),
    "stress": ModeScale(
        T_range_scale=2.5, S_range_scale=2.5,
        slope_scale=3.0,   wave_scale=2.5,
        lens_amp_scale=2.0, n_iface_extra=2, bathy_amp_scale=1.5,
    ),
}


# ---------------------------------------------------------------------------
# SCENE CONFIG — derived from (mode, season)
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    """
    Full parameter envelope for one scene.
    Constructed by build_config(); consumed by SceneGenerator.generate().
    All ranges are (lo, hi) pairs; sampling is uniform unless noted.
    """
    mode:   Mode
    season: Season

    # T/S background corner values
    T_surf_range: tuple[float, float] = (10.0, 10.0)
    T_bott_range: tuple[float, float] = (4.0,  4.0)
    S_surf_range: tuple[float, float] = (7.0,  7.0)
    S_bott_range: tuple[float, float] = (12.0, 12.0)

    # Halocline
    halo_z_range:  tuple[float, float] = HALO_Z
    halo_ds_range: tuple[float, float] = HALO_DS
    halo_th_range: tuple[float, float] = HALO_TH

    # Interface modifiers
    slope_max:  float = 1e-4   # |slope| ≤ slope_max [m/m]
    wave_amp_max: float = 15.0 # [m]

    # CIL presence (seasons with residual winter water)
    cil_present: bool = False
    cil_amp_range: tuple[float, float] = (-5.0, -2.0)  # ΔT < 0 = cold

    # Lens count range
    n_lens_range: tuple[int, int] = (0, 3)

    # Interface count range (before scene-type override)
    n_iface_range: tuple[int, int] = (1, 3)

    # Jet count range
    n_jet_range: tuple[int, int] = (1, 3)
    jet_strength_max: float = 0.5   # [m/s]


def build_config(mode: Mode, season: Season) -> SceneConfig:
    """
    Constructs SceneConfig by scaling realistic Baltic ranges with ModeScale.

    realistic uses HELCOM climatology directly;
    extended / stress apply ModeScale multipliers to the half-widths,
    ensuring extended ⊃ realistic ⊃ typical, stress ⊃ extended.
    """
    sc    = ModeScale.__new__(ModeScale)   # avoid mutation of frozen defaults
    ms    = MODE_SCALES[mode]
    cfg   = SceneConfig(mode=mode, season=season)

    def _scale_range(lo: float, hi: float, scale: float) -> tuple[float, float]:
        mid = (lo + hi) / 2
        hw  = (hi - lo) / 2 * scale
        return (mid - hw, mid + hw)

    cfg.T_surf_range = _scale_range(*SURF_T[season], ms.T_range_scale)
    cfg.T_bott_range = _scale_range(*BOTT_T[season], ms.T_range_scale)
    cfg.S_surf_range = _scale_range(*SURF_S,          ms.S_range_scale)
    cfg.S_bott_range = _scale_range(*BOTT_S,          ms.S_range_scale)

    cfg.halo_z_range  = HALO_Z
    cfg.halo_ds_range = _scale_range(*HALO_DS, ms.S_range_scale)
    cfg.halo_th_range = HALO_TH

    cfg.slope_max    = 1e-4 * ms.slope_scale
    cfg.wave_amp_max = 15.0 * ms.wave_scale

    # CIL present in winter and spring (residual cold water mass)
    cfg.cil_present = season in ("winter", "spring")
    if mode == "stress":
        cfg.cil_present = True   # stress always includes CIL regardless of season
    cfg.cil_amp_range = _scale_range(*CIL_T, ms.lens_amp_scale)
    cfg.cil_amp_range = (-abs(cfg.cil_amp_range[1]), -abs(cfg.cil_amp_range[0]))

    cfg.n_lens_range  = (0, min(3, 3 + ms.n_iface_extra))
    cfg.n_iface_range = (1, min(4, 3 + ms.n_iface_extra))
    cfg.n_jet_range   = (1, 4)
    cfg.jet_strength_max = 0.5 * (1.0 if mode == "realistic"
                                  else 1.0 if mode == "extended" else 1.5)
    return cfg


# ---------------------------------------------------------------------------
# BATHYMETRY — vectorised Σ-Gaussian
# ---------------------------------------------------------------------------

def generate_bathymetry(
    x: np.ndarray,
    rng: np.random.Generator,
    amp_scale: float = 1.0,
) -> np.ndarray:
    """
    bottom(x) = 80 + Σ_i A_i exp(-(x-x0_i)²/(2σ_i²)),  clipped ∈ [5, 200] m.

    Parameters
    ----------
    x         : (nx,) horizontal grid [m]
    rng       : seeded RNG
    amp_scale : MODE_SCALES.bathy_amp_scale — widens relief in stress mode
    """
    n   = rng.integers(2, 7)
    x0  = rng.uniform(0, x[-1], size=n)              # (n,)
    amp = rng.uniform(-60, 60, size=n) * amp_scale    # (n,)
    sx  = rng.uniform(10_000, 60_000, size=n)         # (n,)

    # vectorised: (nx, n) broadcast
    bottom = 80.0 + ((amp * np.exp(
        -((x[:, None] - x0[None, :]) ** 2) / (2 * sx[None, :] ** 2)
    )).sum(axis=1))

    return np.clip(bottom, 5.0, 200.0)


# ---------------------------------------------------------------------------
# INTERFACES — tanh kernel, vectorised over x
# ---------------------------------------------------------------------------

def render_interfaces(
    T: np.ndarray,
    S: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    params: dict,
) -> None:
    """
    In-place: T += Σ_i A_T_i · tanh((z - z_i(x)) / th_i),  idem S.

    z_i(x) = depth_i + slope_i·x + wave_amp_i · sin(2π·x/λ_i + φ_i)

    params keys (all shape (N,)):
      depth, thickness, A_T, A_S, slopes, wave_amp, wave_len, phase
    """
    zz = z[:, None]          # (nz, 1)

    for i in range(params["depth"].shape[0]):
        zi = (                # (1, nx)
            params["depth"][i]
            + params["slopes"][i] * x
            + params["wave_amp"][i] * np.sin(
                2 * np.pi * x / params["wave_len"][i] + params["phase"][i]
            )
        )[None, :]

        th = params["thickness"][i]
        T += params["A_T"][i] * np.tanh((zz - zi) / th)
        S += params["A_S"][i] * np.tanh((zz - zi) / th)


# ---------------------------------------------------------------------------
# LENSES — 2-D Gaussian kernel
# ---------------------------------------------------------------------------

def render_lenses(
    field: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    L: np.ndarray,
) -> None:
    """
    In-place: field += Σ_i A_i · exp(-((x-x0)²/2σx² + (z-z0)²/2σz²)).

    Parameters
    ----------
    L : (M, 5) array — columns [x0, z0, sx, sz, A]
    """
    if L.shape[0] == 0:
        return

    xx, zz = np.meshgrid(x, z)   # (nz, nx)

    for x0, z0, sx, sz, A in L:
        field += A * np.exp(
            -(
                (xx - x0) ** 2 / (2 * sx ** 2)
                + (zz - z0) ** 2 / (2 * sz ** 2)
            )
        )


# ---------------------------------------------------------------------------
# VELOCITY — layered: independent jets above / below halocline + shear layer
# ---------------------------------------------------------------------------

def _render_jets(
    u: np.ndarray,
    v: np.ndarray,
    xx: np.ndarray,
    zz: np.ndarray,
    J: np.ndarray,
) -> None:
    """In-place u,v += Σ_j Gaussian jet; J shape (K,5): x0,z0,sx,sz,strength."""
    for x0, z0, sx, sz, A in J:
        g = np.exp(
            -(
                (xx - x0) ** 2 / (2 * sx ** 2)
                + (zz - z0) ** 2 / (2 * sz ** 2)
            )
        )
        u += A * g
        v += 0.2 * A * g


def render_layered_velocity(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    J_upper: np.ndarray,
    J_lower: np.ndarray,
    halo_z:  float,
    halo_th: float,
    u0_upper: float,
    u0_lower: float,
    v0_upper: float,
    v0_lower: float,
) -> None:
    """
    Two-layer velocity model separated by the halocline.

    u(x,z) = u_bg(z) + u_jets_upper(x,z) + u_jets_lower(x,z)

    Background shear
    ----------------
      W_u(z) = 0.5·(1 - tanh((z - halo_z)/halo_th))   # upper layer weight
      W_l(z) = 1 - W_u(z)
      u_bg(z) = u0_upper·W_u + u0_lower·W_l             # smooth tanh blend

    The same tanh (depth=halo_z, thickness=halo_th) as the S halocline
    provides a continuous shear layer — no discontinuity.

    Jets
    ----
    J_upper : z0 ∈ (5, halo_z)          — upper layer only
    J_lower : z0 ∈ (halo_z, zmax-5)     — lower layer only
    Both sets add independently without cross-layer coupling.

    Parameters
    ----------
    J_upper, J_lower : (K, 5) arrays — [x0, z0, sx, sz, strength]
    halo_z, halo_th  : halocline geometry shared with S field [m]
    u0_*, v0_*       : independent background velocities per layer [m/s]
    """
    xx, zz = np.meshgrid(x, z)   # (nz, nx)

    W_u = 0.5 * (1.0 - np.tanh((z - halo_z) / halo_th))   # (nz,)
    W_l = 1.0 - W_u

    u += (u0_upper * W_u + u0_lower * W_l)[:, None]
    v += (v0_upper * W_u + v0_lower * W_l)[:, None]

    _render_jets(u, v, xx, zz, J_upper)
    _render_jets(u, v, xx, zz, J_lower)


# ---------------------------------------------------------------------------
# SCENE TYPE → parameter overrides
# ---------------------------------------------------------------------------

def _scene_type_overrides(
    scene_type: ScType,
    cfg: SceneConfig,
    rng: np.random.Generator,
) -> dict:
    """
    Returns per-scene-type overrides merged into interface/jet sampling.

    upwelling   → strong slope, near-shore shallowing of thermocline
    downwelling → inverse slope
    front       → steep lateral gradient, narrow interface
    intrusion   → extra lens, weaker thermocline
    jets        → more jets, stronger currents
    internal_waves → high wave_amp, many interfaces
    background  → default
    """
    overrides: dict = {}

    if scene_type == "upwelling":
        overrides["slope_sign"]  = +1        # thermocline rises toward shore
        overrides["n_iface_min"] = 2
    elif scene_type == "downwelling":
        overrides["slope_sign"]  = -1
        overrides["n_iface_min"] = 2
    elif scene_type == "front":
        overrides["thickness_max"] = 4.0    # sharp front
        overrides["slope_scale"]   = 2.0
    elif scene_type == "intrusion":
        overrides["extra_lenses"]  = 1
    elif scene_type == "jets":
        overrides["n_jet_min"]     = 3
        overrides["jet_scale"]     = 1.5
    elif scene_type == "internal_waves":
        overrides["wave_scale"]    = 2.0
        overrides["n_iface_min"]   = 2

    return overrides


# ---------------------------------------------------------------------------
# SCENE GENERATOR
# ---------------------------------------------------------------------------

class SceneGenerator:
    """
    Generates one 256×256 Baltic hydrophysical section per .generate() call.

    Usage
    -----
    gen  = SceneGenerator(seed=42)
    data = gen.generate(domain, mode="realistic", season="summer")

    Output keys: T, S, u, v, bottom, mask, x, z, meta
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def generate(
        self,
        domain: Domain,
        mode:   Mode   = "realistic",
        season: Season = "summer",
    ) -> dict:
        """
        Full pipeline: bathymetry → background T/S → interfaces →
        halocline → CIL → lenses → jets → mask.

        Returns dict with fields + meta.
        """
        cfg = build_config(mode, season)
        ms  = MODE_SCALES[mode]
        x, z = domain.grid()

        # --- scene type -------------------------------------------------
        scene_type: ScType = self.rng.choice(
            _SCENE_TYPES, p=_SCENE_W / _SCENE_W.sum()
        )
        ov = _scene_type_overrides(scene_type, cfg, self.rng)

        # --- bathymetry -------------------------------------------------
        bottom   = generate_bathymetry(x, self.rng, amp_scale=ms.bathy_amp_scale)
        bottom2d = np.broadcast_to(bottom[None, :], (domain.nz, domain.nx))

        # --- background T/S (linear profile in z) -----------------------
        T_surf = self.rng.uniform(*cfg.T_surf_range)
        T_bott = self.rng.uniform(*cfg.T_bott_range)
        S_surf = self.rng.uniform(*cfg.S_surf_range)
        S_bott = self.rng.uniform(*cfg.S_bott_range)

        # ensure T_surf ≥ T_bott (Baltic summer stratification) in realistic
        if mode == "realistic" and season == "summer":
            T_bott = min(T_bott, T_surf - 1.0)

        z_norm = z / domain.zmax                          # [0,1]
        T = T_surf + (T_bott - T_surf) * z_norm[:, None] * np.ones((1, domain.nx))
        S = S_surf + (S_bott - S_surf) * z_norm[:, None] * np.ones((1, domain.nx))

        # --- interfaces -------------------------------------------------
        # Generate T/S interface amplitudes such that the net density jump
        # is always stable: -α·A_T + β·A_S > 0.
        # Step 1: draw A_T freely (thermocline can be sharp).
        # Step 2: draw A_S independently (halocline can differ).
        # Step 3: if density jump is negative, boost A_S just enough.
        # This guarantees ρ always increases through each interface.
        n_base  = self.rng.integers(*cfg.n_iface_range)
        n_iface = max(n_base, ov.get("n_iface_min", 0))

        slope_scale = ov.get("slope_scale", 1.0)
        wave_scale  = ov.get("wave_scale",  1.0)
        th_max      = ov.get("thickness_max", 20.0)
        slope_sign  = ov.get("slope_sign", 0)    # 0 → random sign

        depths     = self.rng.uniform(20, 120,           size=n_iface)
        thicknesses = self.rng.uniform(2, th_max,         size=n_iface)
        A_T        = self.rng.uniform(-10, 10,            size=n_iface)
        A_S        = self.rng.uniform(-8,  8,             size=n_iface)

        # Enforce stable density: -α·A_T + β·A_S >= min_drho
        from ..eos import min_salinity_amplitude
        A_S = np.maximum(A_S, min_salinity_amplitude(A_T))

        slopes     = (
            self.rng.uniform(0, cfg.slope_max * slope_scale, size=n_iface)
            * (slope_sign or self.rng.choice([-1, 1], size=n_iface))
        )
        wave_amps  = self.rng.uniform(0, cfg.wave_amp_max * wave_scale, size=n_iface)
        wave_lens  = self.rng.uniform(20_000, 120_000,   size=n_iface)
        phases     = self.rng.uniform(0, 2 * np.pi,      size=n_iface)

        iface_params = dict(
            depth=depths, thickness=thicknesses,
            A_T=A_T, A_S=A_S,
            slopes=slopes, wave_amp=wave_amps,
            wave_len=wave_lens, phase=phases,
        )
        render_interfaces(T, S, x, z, iface_params)

        # --- halocline (permanent Baltic feature) -----------------------
        h_depth = self.rng.uniform(*cfg.halo_z_range)
        h_ds    = self.rng.uniform(*cfg.halo_ds_range)
        h_th    = self.rng.uniform(*cfg.halo_th_range)

        zz_h = z[:, None]
        S   += h_ds * (0.5 + 0.5 * np.tanh((zz_h - h_depth) / h_th))

        # --- CIL (Cold Intermediate Layer) — Baltic-specific lens -------
        # CIL is remnant winter water: cold T, slightly fresher S (ice-melt origin).
        # The S anomaly is small (±1 PSU) relative to T anomaly (-2..-5 °C),
        # producing a realistic coupled density anomaly.
        if cfg.cil_present:
            cil_z    = self.rng.uniform(*CIL_Z)
            cil_amp_T = self.rng.uniform(*cfg.cil_amp_range)   # negative = cold
            cil_amp_S = cil_amp_T * self.rng.uniform(0.05, 0.15)  # coupled, same sign (fresher)
            cil_x0   = self.rng.uniform(0, domain.xmax)
            cil_sx   = self.rng.uniform(*CIL_SX)
            cil_sz   = self.rng.uniform(*CIL_SZ)

            L_cil_T = np.array([[cil_x0, cil_z, cil_sx, cil_sz, cil_amp_T]])
            L_cil_S = np.array([[cil_x0, cil_z, cil_sx, cil_sz, cil_amp_S]])
            render_lenses(T, x, z, L_cil_T)
            render_lenses(S, x, z, L_cil_S)

        # --- additional lenses (intrusions / anomalies) -----------------
        # Each lens affects both T and S with correlated amplitudes.
        # T-S ratio follows typical Baltic water mass properties:
        # ΔS/ΔT ≈ 0.3–0.6 PSU/°C (positive: warm = saltier in intrusions).
        n_lens = self.rng.integers(*cfg.n_lens_range) + ov.get("extra_lenses", 0)
        if n_lens > 0:
            x0s  = self.rng.uniform(0,      domain.xmax, n_lens)
            z0s  = self.rng.uniform(20,     150,          n_lens)
            sxs  = self.rng.uniform(5_000,  40_000,       n_lens)
            szs  = self.rng.uniform(5,      30,            n_lens)
            amp_T = (self.rng.uniform(-8, 8, n_lens) * ms.lens_amp_scale)
            amp_S = amp_T * self.rng.uniform(0.2, 0.5, n_lens)
            L_T = np.column_stack([x0s, z0s, sxs, szs, amp_T])
            L_S = np.column_stack([x0s, z0s, sxs, szs, amp_S])
            render_lenses(T, x, z, L_T)
            render_lenses(S, x, z, L_S)

        # --- layered velocity (independent above / below halocline) ----
        # Jets are sampled per layer; z0 constrained to the respective
        # layer so Gaussian kernels stay within [0, halo_z] or [halo_z, zmax].
        # Background u0/v0 pair is drawn independently for each layer and
        # blended through the same tanh as the S halocline → shear layer.
        jet_scale = ov.get("jet_scale", 1.0)
        A_max     = cfg.jet_strength_max * jet_scale

        def _sample_jets(n: int, z_lo: float, z_hi: float) -> np.ndarray:
            """Return (n, 5) jet array with z0 ∈ (z_lo, z_hi)."""
            if n == 0:
                return np.zeros((0, 5))
            z_lo_eff = min(z_lo + 2.0, z_hi - 2.0)   # guard thin layers
            return np.column_stack([
                self.rng.uniform(0,       domain.xmax,  n),
                self.rng.uniform(z_lo_eff, z_hi,        n),
                self.rng.uniform(10_000,  80_000,       n),
                self.rng.uniform(5,       40,           n),   # sz clipped to layer
                self.rng.uniform(-A_max,  A_max,        n),
            ])

        n_total = max(self.rng.integers(*cfg.n_jet_range), ov.get("n_jet_min", 0))
        # split: upper layer gets ~60% of jets (more energetic surface layer)
        n_upper = max(1, int(round(n_total * self.rng.uniform(0.4, 0.8))))
        n_lower = max(1, n_total - n_upper)

        J_upper = _sample_jets(n_upper, 5.0,    h_depth)
        J_lower = _sample_jets(n_lower, h_depth, domain.zmax - 5.0)

        # independent background velocities per layer
        bg_max  = 0.3 * (1.0 if mode == "realistic" else 1.5 if mode == "extended" else 2.0)
        u0_u, v0_u = self.rng.uniform(-bg_max, bg_max, 2)
        u0_l, v0_l = self.rng.uniform(-bg_max, bg_max, 2)

        u = np.zeros((domain.nz, domain.nx))
        v = np.zeros((domain.nz, domain.nx))
        render_layered_velocity(
            u, v, x, z,
            J_upper, J_lower,
            halo_z=h_depth, halo_th=h_th,
            u0_upper=u0_u, u0_lower=u0_l,
            v0_upper=v0_u, v0_lower=v0_l,
        )

        # --- physical clipping (before mask) ----------------------------
        # Prevents unphysical superposition blow-up; limits per mode:
        #   realistic : Baltic observational range
        #   extended  : ×1.5 margins
        #   stress    : hard physical limits (ice-point, hypersaline outlier)
        T_clip, S_clip = {
            "realistic": ((-2.0,  25.0), (3.0,  25.0)),
            "extended":  ((-2.0,  28.0), (2.0,  30.0)),
            "stress":    ((-2.0,  30.0), (1.0,  35.0)),
        }[mode]
        np.clip(T, *T_clip, out=T)
        np.clip(S, *S_clip, out=S)
        np.clip(u, -3.0, 3.0, out=u)
        np.clip(v, -3.0, 3.0, out=v)

        # --- mask (below bathymetry → NaN) ------------------------------
        zz   = z[:, None] * np.ones((1, domain.nx))
        mask = zz < bottom2d
        for arr in (T, S, u, v):
            arr[~mask] = np.nan

        return {
            "T":      T,
            "S":      S,
            "u":      u,
            "v":      v,
            "bottom": bottom,
            "mask":   mask,
            "x":      x,
            "z":      z,
            "meta": {
                "mode":       mode,
                "season":     season,
                "scene_type": scene_type,
            },
        }


# ---------------------------------------------------------------------------
# DATASET GENERATOR — inpainting training mix
# ---------------------------------------------------------------------------

_SEASONS: list[Season] = ["winter", "spring", "summer", "autumn"]
_MODE_WEIGHTS: dict[Mode, float] = {
    "realistic": 0.70,
    "extended":  0.20,
    "stress":    0.10,
}
_MODES: list[Mode] = list(_MODE_WEIGHTS.keys())
_MODE_W = np.array([_MODE_WEIGHTS[m] for m in _MODES])


def generate_dataset(
    n:          int,
    mode_mix:   dict[Mode, float] | None = None,
    seed:       int  = 0,
    domain:     Domain | None = None,
) -> list[dict]:
    """
    Generate n scenes with the given mode mixture (default 70/20/10).

    mode_mix : e.g. {"realistic": 0.7, "extended": 0.2, "stress": 0.1}
               Must sum to 1.0; normalised automatically if not.
    seed     : master seed — each scene gets seed + i for reproducibility.

    Returns list of dicts (same structure as SceneGenerator.generate()).
    """
    domain   = domain or Domain()
    mix      = mode_mix or _MODE_WEIGHTS
    modes    = list(mix.keys())
    weights  = np.array([mix[m] for m in modes], dtype=float)
    weights /= weights.sum()

    master_rng = np.random.default_rng(seed)
    scene_modes: list[Mode]   = master_rng.choice(modes, size=n, p=weights)
    scene_seasons: list[Season] = master_rng.choice(_SEASONS, size=n)

    return [
        SceneGenerator(seed=seed + i).generate(domain, scene_modes[i], scene_seasons[i])
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# STATISTICS SUMMARY
# ---------------------------------------------------------------------------

def dataset_stats(scenes: list[dict]) -> None:
    """
    Prints per-mode T/S summary statistics (nanmean ± nanstd) to stdout.
    Useful for verifying that generated distributions match Baltic climatology.
    """
    from collections import defaultdict

    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"T": [], "S": []})

    for sc in scenes:
        m = sc["meta"]["mode"]
        buckets[m]["T"].append(np.nanmean(sc["T"]))
        buckets[m]["S"].append(np.nanmean(sc["S"]))

    print(f"\n{'Mode':<12} {'N':>5}  {'T̄ [°C]':>12}  {'S̄ [PSU]':>12}")
    print("-" * 48)

    for mode in _MODES:
        if mode not in buckets:
            continue
        T_arr = np.array(buckets[mode]["T"])
        S_arr = np.array(buckets[mode]["S"])
        n     = len(T_arr)
        print(
            f"{mode:<12} {n:>5}"
            f"  {T_arr.mean():>6.2f}±{T_arr.std():>5.2f}"
            f"  {S_arr.mean():>6.2f}±{S_arr.std():>5.2f}"
        )

    print()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    N    = 1_000
    SEED = 0

    print(f"Generating {N} Baltic scenes (seed={SEED}) …")
    t0     = time.perf_counter()
    scenes = generate_dataset(N, seed=SEED)
    dt     = time.perf_counter() - t0
    print(f"Done in {dt:.1f} s  ({dt/N*1000:.1f} ms/scene)")

    # ---------------------------------------------------------------
    # Mode distribution check
    mode_counts: dict[str, int] = {}
    for sc in scenes:
        m = sc["meta"]["mode"]
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print("\nMode distribution:")
    for m, cnt in sorted(mode_counts.items()):
        bar = "█" * (cnt // 10)
        print(f"  {m:<12} {cnt:>4}  {cnt/N*100:>5.1f}%  {bar}")

    # ---------------------------------------------------------------
    # Season distribution
    season_counts: dict[str, int] = {}
    for sc in scenes:
        s = sc["meta"]["season"]
        season_counts[s] = season_counts.get(s, 0) + 1

    print("\nSeason distribution:")
    for s, cnt in sorted(season_counts.items()):
        print(f"  {s:<10} {cnt:>4}  {cnt/N*100:>5.1f}%")

    # ---------------------------------------------------------------
    # T/S statistics per mode
    dataset_stats(scenes)

    # ---------------------------------------------------------------
    # Field shape / NaN sanity check
    sc0 = scenes[0]
    print("Field shapes:")
    for k in ("T", "S", "u", "v", "bottom", "mask"):
        arr  = sc0[k]
        nans = np.isnan(arr).sum() if arr.dtype != bool else 0
        print(f"  {k:<8} shape={arr.shape}  NaNs={nans}")

    # ---------------------------------------------------------------
    # Extreme value check (stress mode)
    stress = [sc for sc in scenes if sc["meta"]["mode"] == "stress"]
    if stress:
        T_max = max(float(np.nanmax(sc["T"])) for sc in stress)
        T_min = min(float(np.nanmin(sc["T"])) for sc in stress)
        S_max = max(float(np.nanmax(sc["S"])) for sc in stress)
        print(f"\nStress extremes:  T ∈ [{T_min:.1f}, {T_max:.1f}] °C"
              f"  S_max={S_max:.1f} PSU")

    # ---------------------------------------------------------------
    # Surface (z[0]) vs near-bottom (z[-20:]) T/S per mode
    print("\nSurface (z=0) vs near-bottom T/S per mode:")
    print(f"  {'Mode':<12}  {'T_surf':>8}  {'T_bott':>8}  {'S_surf':>8}  {'S_bott':>8}")
    for mode in _MODES:
        subs = [sc for sc in scenes if sc["meta"]["mode"] == mode]
        if not subs:
            continue
        ts = np.nanmean([sc["T"][0,   :]    for sc in subs])
        tb = np.nanmean([sc["T"][-20:, :]   for sc in subs])
        ss = np.nanmean([sc["S"][0,   :]    for sc in subs])
        sb = np.nanmean([sc["S"][-20:, :]   for sc in subs])
        print(f"  {mode:<12}  {ts:>8.2f}  {tb:>8.2f}  {ss:>8.2f}  {sb:>8.2f}")

    # ---------------------------------------------------------------
    # Scene-type distribution
    stype_counts: dict[str, int] = {}
    for sc in scenes:
        st = sc["meta"]["scene_type"]
        stype_counts[st] = stype_counts.get(st, 0) + 1

    print("\nScene-type distribution:")
    for st in _SCENE_TYPES:
        cnt = stype_counts.get(st, 0)
        bar = "█" * (cnt // 8)
        print(f"  {st:<18} {cnt:>4}  {cnt/N*100:>5.1f}%  {bar}")

    # ---------------------------------------------------------------
    # Per-season mean T surface (realistic only — climatology check)
    print("\nRealistic mode — surface T̄ per season:")
    for season in _SEASONS:
        subs = [sc for sc in scenes
                if sc["meta"]["mode"] == "realistic"
                and sc["meta"]["season"] == season]
        if not subs:
            continue
        ts = np.nanmean([sc["T"][0, :] for sc in subs])
        print(f"  {season:<8}  {ts:>6.2f} °C  (expect {SURF_T[season]})")
