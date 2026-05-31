# Baltic Sea Synthetic Physics — Generation & Constraints

This document describes the synthetic Baltic Sea data generator, the equation
of state, and the density inversion elimination pipeline.  All physics
expressions live in `hydro_lama_nc/eos.py`; all generation lives in
`hydro_lama_nc/data/synthetic_baltic.py` with post-processing in
`hydro_lama_nc/data/ctd_dropout.py`.

---

## 1. Equation of State (eos.py)

Single source of truth for all density calculations.

### 1.1 Canonical Coefficients

```
ALPHA = 0.04    (thermal expansion)
BETA  = 0.15    (haline contraction)
β / α = 3.75    (haline-dominated — correct for Baltic Sea)
```

Both fields are normalised to [0, 1] before the EOS is applied.
The ratio β/α = 3.75 matches the physical Baltic where salinity
dominates density (β_phys/α_phys ≈ 5, reduced by ΔT/ΔS ≈ 0.75 after
normalisation).

### 1.2 Functions

| Function | Formula | Use |
|---|---|---|
| `linearized_density(T, S)` | ρ = 1 − αT + βS | Forward EOS |
| `density_gradient(T, S, dim)` | ∂ρ/∂x = −α·∂T/∂x + β·∂S/∂x | Geostrophic loss |
| `salinity_from_density(ρ, T)` | S = (ρ − 1 + αT) / β | Inverse EOS (stratification fix) |
| `min_salinity_amplitude(A_T)` | A_S ≥ (α·A_T + min_drho) / β | Interface stability constraint |

---

## 2. Scene Generation Pipeline

### 2.1 Pipeline Overview

```
SceneGenerator.generate(domain, mode, season)
  │
  ├─ 1. Background: linear T(z), S(z) from seasonal climatology
  ├─ 2. Halocline: tanh S increase at 55–90 m
  ├─ 3. Interfaces: tanh perturbations to T and S (density-coupled)
  ├─ 4. CIL: Cold Intermediate Layer lens on T + coupled S
  ├─ 5. Lenses: Gaussian anomalies on T + coupled S
  ├─ 6. Velocity: two-layer jets + background shear
  └─ 7. Bathymetry: Gaussian bottom profile

_normalize_field(T), _normalize_field(S)      → [0, 1]
_enforce_stratification(T, S, below)          → 0% inversions guaranteed
```

### 2.2 Domain

```
Domain(nx=256, nz=256, xmax=200_000 m, zmax=200 m)
```

Horizontal extent 200 km, depth 200 m — typical Baltic cross-section.

### 2.3 Modes

| Mode | T/S range | slopes | waves | lenses | jets | purpose |
|---|---|---|---|---|---|---|
| `realistic` (70%) | 1.0× | 1.0× | 1.0× | 1.0× | 0.5 m/s | typical Baltic |
| `extended` (20%) | 1.5× | 1.8× | 1.8× | 1.3× | 0.5 m/s | unusual conditions |
| `stress` (10%) | 2.5× | 3.0× | 2.5× | 2.0× | 0.75 m/s | extremes, always CIL |

### 2.4 Seasonal Climatology

| Season | T_surf [°C] | T_bott [°C] | S_surf [PSU] | S_bott [PSU] | CIL |
|---|---|---|---|---|---|
| winter | 0 – 4 | 2 – 6 | 5 – 10 | 10 – 20 | yes |
| spring | 2 – 14 | 2 – 6 | 5 – 10 | 10 – 20 | yes |
| summer | 14 – 22 | 2 – 8 | 5 – 10 | 10 – 20 | no |
| autumn | 6 – 16 | 2 – 7 | 5 – 10 | 10 – 20 | no |

Halocline: depth 55–90 m, salinity jump 6–12 PSU, thickness 3–8 m.

### 2.5 Scene Types (sampled by prior)

| Scene | Prob | Description |
|---|---|---|
| background | 0.15 | calm, no anomalies |
| internal_waves | 0.20 | wavy interfaces |
| upwelling | 0.15 | tilted isopycnals upward |
| downwelling | 0.10 | tilted isopycnals downward |
| front | 0.15 | sharp horizontal gradient |
| intrusion | 0.10 | anomalous water mass |
| jets | 0.15 | velocity-dominated |

### 2.6 Physical Clipping

| Mode | T [°C] | S [PSU] | u, v [m/s] |
|---|---|---|---|
| realistic | −2 to 25 | 3 to 25 | −3 to 3 |
| extended | −2 to 28 | 2 to 30 | −3 to 3 |
| stress | −2 to 30 | 1 to 35 | −3 to 3 |

---

## 3. Density Inversion Elimination

### 3.1 The Problem

The synthetic generator creates T and S with independent structures
(interfaces, lenses, CIL).  When density is computed as
ρ = 1 − αT + βS, temperature inversions (warm water below cold) can
produce density inversions (∂ρ/∂z < 0), i.e. closed isopycnals.

Before the fix, **13.6% ± 17.0%** of grid cells had density inversions
across 200 realistic scenes.  The `physics_loss` during training penalises
these inversions in the *model output*, but the training *data* itself
contained them — forcing the model to learn an impossible target.

### 3.2 Root Cause Analysis

| Source | Violations introduced | % of total |
|---|---|---|
| Background (linear profiles) | 0 | 0% |
| **Interfaces** (tanh perturbations) | **523.6** | **~93%** |
| Halocline (tanh S increase) | 6.1 | ~1% |
| CIL (coupled lenses) | 0 | 0% |

The interfaces used independent A_T and A_S amplitudes.  When A_T > 0
(warm below cold), A_S was not guaranteed to compensate.

### 3.3 Three-Part Fix

#### Part 1: Interface Coupling (generation time)

In `synthetic_baltic.py`, interface amplitudes are now coupled:

```python
from ..eos import min_salinity_amplitude
A_S = np.maximum(A_S, min_salinity_amplitude(A_T))
```

This enforces: −α·A_T + β·A_S ≥ min_drho (default 0.005).
Each interface is density-stable by construction.

#### Part 2: CIL and Lens Coupling (generation time)

- **CIL** now affects both T and S: ΔS = ΔT × (0.05 to 0.15), same sign
  (cold + fresh — remnant winter water with ice-melt).
- **Lenses** share geometry (x0, z0, sx, sz) for T and S; amplitudes
  coupled: ΔS/ΔT ≈ 0.2–0.5 PSU/°C.

#### Part 3: Post-processing Stratification Enforcement (after normalisation)

`_enforce_stratification(T, S, below)` in `ctd_dropout.py`:

1. For each water column, compute ρ = linearized_density(T, S).
2. Forward scan: wherever ρ[i] < ρ[i−1] + 1e−4, force ρ[i] = ρ[i−1] + 1e−4.
3. Decompose back: S_new = salinity_from_density(ρ, T).
4. If S_new ∉ [0, 1]: clip S, absorb excess into T via T = (1 + βS − ρ) / α.

**Selective enforcement**: stratification is enforced only at non-CTD columns.
CTD observations contain realistic noise-driven inversions (σ_ctd = 0.05)
that the model should reproduce.  The target at CTD columns is the raw
(noisy) field; the target at non-CTD columns is the enforced (stable) field.

### 3.4 Result

After all three fixes: **0.00% inversions at non-CTD columns** in the
target.  CTD columns preserve **~2.8% inversions** from measurement noise.
`physics_loss` is applied **everywhere** (including CTD) — observation_loss
gradient at CTD is ~100× stronger, so physics only prevents noise
amplification (2% target → model stays near 2%, not 50%).

The `physics_loss` excludes CTD columns from the penalty, so the model
learns to reproduce CTD inversions (driven by `observation_loss`) while
remaining stable elsewhere (driven by `physics_loss`).

---

## 4. Observation Sampling

### 4.1 CTD Profiles

- Random x-positions, full-depth columns (down to bathymetry).
- Noise: Gaussian N(0, 0.05) on T and S, clipped to [0, 1].
- High trust: σ_ctd = 0.05 in observation loss.

### 4.2 CMEMS Observations

- Log-spaced depth levels × linearly-spaced horizontal grid.
- Positions jittered by ±40% of spacing.
- Sampled from *background* (smoothed + shifted truth), not from truth.
- Noise: Gaussian N(0, 0.10), clipped to [0, 1].
- Medium trust: σ_cmems = 0.10 in observation loss.

### 4.3 CMEMS Background

- Gaussian smoothing (σ = 2.0 pixels) of truth.
- Random vertical shift (±6 pixels) simulating thermocline depth bias.
- Represents operational reanalysis: smooth, slightly biased.

### 4.4 Default Dataset Parameters

```
HybridCTDDataset:
  n=400, nx=64, nz=64
  min_ctd=0, max_ctd=7
  n_cmems_x=8, n_cmems_z=10
  noise_ctd=0.05, noise_cmems=0.10
  mode_mix={"realistic": 0.7, "extended": 0.2, "stress": 0.1}
```

---

## 5. Velocity Generation

### 5.1 Two-Layer Model (Baltic generator)

Background velocity uses tanh blending through the halocline:

```
W_upper(z) = 0.5 · (1 − tanh((z − halo_z) / halo_th))
u_bg(z) = u_upper · W_upper + u_lower · (1 − W_upper)
```

Gaussian jets added independently in upper (z < halo_z) and lower layers.
Each jet: `u += A · exp(−((x−x0)²/2σx² + (z−z0)²/2σz²))`, v gets 0.2× of u.

| Mode | background max | jet strength max |
|---|---|---|
| realistic | 0.3 m/s | 0.5 m/s |
| extended | 0.45 m/s | 0.5 m/s |
| stress | 0.6 m/s | 0.75 m/s |

### 5.2 Geostrophic Velocity (simple pipeline)

`synthetic_V_field` generates velocity from signed temperature gradient:

```python
V = dT/dx   # NOT abs(dT/dx) — preserves thermal wind direction
V += exponential_shear + Gaussian_eddy
```

---

## 6. Loss Functions

### 6.1 observation_loss — Value + Structure

```
L = MSE(pred, target, σ_map) at observation points
  + 0.3 · L1(∇pred, ∇cmems) at CMEMS points
```

MSE uses σ² weighting (not NLL — the constant log(σ²) term was removed
because it doesn't affect gradients and distorts Kendall weighting).

σ_map: 0.05 at CTD, 0.10 at CMEMS, 0.25 elsewhere.

### 6.2 physics_loss — Stable Stratification

```
scale = std(∂ρ/∂z) over water domain (detached)
L = mean((ReLU(−∂ρ/∂z) / scale)²)
```

Normalised by the typical density gradient scale (like geostrophic_loss)
so the loss magnitude is O(0.01–0.15) instead of O(1e-5).

Applied **everywhere** (including CTD columns).  observation_loss gradient
at CTD is ~100× stronger than physics, so physics cannot over-smooth CTD
data — it only prevents the model from amplifying noise beyond the 2%
target inversion rate at CTD.

### 6.3 geostrophic_loss — Thermal Wind Balance

```
∂ρ/∂x = density_gradient(T, S, dim=x)    from eos.py
∂v/∂z = gradient(v, dim=z)
L = mean((∂ρ/∂x_n − ∂v/∂z_n)²)
```

In 2D (x, z) only the v-equation is computable:
`∂v/∂z = (g/fρ₀) · ∂ρ/∂x` → dv/dz and dρ/dx should be **equal**
(same sign).  The u-equation requires ∂ρ/∂y which doesn't exist in 2D.

Both sides normalised to unit std (pattern matching, not magnitude).

---

## 7. Input Channel Layout

### 7-channel (HybridCTDDataset)

| Ch | Name | Description |
|---|---|---|
| 0 | T_obs | CMEMS background + CTD overlay |
| 1 | S_obs | same |
| 2 | mask_ctd | 1.0 at CTD columns |
| 3 | mask_cmems | 1.0 at CMEMS grid points |
| 4 | u_lr | zonal velocity (LR upsampled) |
| 5 | v_lr | meridional velocity |
| 6 | bathy_ch | tiled bathymetry |

### 8-channel (make_hydro_synthetic_sample)

Same as above + Ch 7: sigma_obs (per-point uncertainty).

### Target (2-channel)

| Ch | Name | Range |
|---|---|---|
| 0 | T | [0, 1] (normalised) |
| 1 | S | [0, 1] (normalised) |
