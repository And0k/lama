"""
LaMa-inpainting for 2D ocean T/S cross-sections | Baltic Sea synthetic data.

TASK
----
Given a mostly-empty 2D slice (depth × distance), reconstruct full T and S
fields by fusing two observation types:
  • CTD profiles  : vertical lines, high trust (σ≈0.05), sparse in x
  • CMEMS points  : quasi-regular sparse grid with increasing dz toward bottom,
                    low trust (σ≈0.10), need to be nudged toward CTD truth
  • u, v velocity : CMEMS low-res upsampled → geostrophic conditioning channels
  • bathymetry    : continuous bottom depth field ∈ [0,1], not a binary mask

INPUT TENSOR  (8, NZ, NX):
  ch0  T_obs       — observed T values, 0 outside observations
  ch1  S_obs       — observed S values, 0 outside observations
  ch2  mask_ctd    — 1 where CTD profile exists (high-trust vertical line)
  ch3  mask_cmems  — 1 where CMEMS point exists (low-trust sparse point)
  ch4  u_lr        — zonal velocity upsampled from LR CMEMS grid
  ch5  v_lr        — meridional velocity upsampled from LR CMEMS grid
  ch6  bathymetry  — continuous bottom depth ∈ [0,1], broadcast over z-axis
  ch7  sigma_obs   — per-point uncertainty: 0.05 CTD, 0.10 CMEMS, 0 elsewhere
                     (replaces merged binary mask; tells net how much to trust)

OUTPUT TENSOR (2, NZ, NX):
  ch0  T_pred      — reconstructed temperature ∈ [0,1]
  ch1  S_pred      — reconstructed salinity ∈ [0,1]
  T+S predicted jointly so the net learns their physical covariance via ρ(T,S)

WHY NO GAN
----------
PatchGAN discriminator doubles training time and is unstable with N<500 samples.
T/S fields are smooth — adversarial loss gives no quality gain over physical
penalties on scalar oceanographic fields. Add as ablation after this works.

ARCHITECTURE: LaMa-lite with dual-head decoder + T↔S cross-attention
  Encoder → N × FFCResBlock → [Head_T | Head_S] → cross-attention → output
  FFCResBlock = local 3×3 conv + FourierUnit (global receptive field via FFT)
  Cross-attention: channel-wise T↔S interaction before final conv (cheap)

PHYSICAL LOSSES (§1.1 of the report)
  L_stab : penalize ∂ρ(T,S)/∂z < 0  (hydrostatic instability)
  L_bbl  : penalize |∂T/∂z|, |∂S/∂z| near bottom  (BBL homogenization)
  L_grad : isotropic TV regularization  (suppress unphysical artifacts)
  L_geo  : thermal wind balance: ∂u/∂z ~ -∂ρ/∂x  (geostrophic consistency)

PARAM/MEMORY BUDGET (fits free Colab T4, 15 GB VRAM)
  BASE_CH=32, N_BLOCKS=6 → ~1.2M params, ~0.5 GB activations at batch=4
  Reduce BASE_CH=16 or N_BLOCKS=4 if OOM on smaller GPUs
"""

import json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── output directories ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT  = ROOT / "output";      OUT.mkdir(exist_ok=True)
CKPT = ROOT / "checkpoints"; CKPT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# All hyperparameters in one place. Override via CFG.update({...}) in Colab.
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # --- domain ---
    NX = 64,           # horizontal grid points (distance along section)
    NZ = 64,           # vertical grid points   (depth, 0=surface 1=bottom)
    N_CTD    = 5,      # CTD profiles per sample (vertical lines)
    N_CMEMS_X = 8,     # CMEMS columns per sample (sparse x-positions)
    N_CMEMS_Z = 10,    # CMEMS levels per column  (logspace, denser near surf)

    # --- synthetic noise ---
    NOISE_CTD   = 0.05,   # CTD measurement noise  σ  (high trust)
    NOISE_CMEMS = 0.10,   # CMEMS reanalysis noise σ  (low trust, needs nudge)

    # --- dataset ---
    N_TRAIN = 400,
    N_VAL   = 80,

    # --- training ---
    EPOCHS      = 120,
    BATCH       = 4,
    LR          = 2e-4,
    LR_PATIENCE = 15,    # ReduceLROnPlateau: halve LR after this many bad epochs
    ES_PATIENCE = 30,    # early stopping: quit after this many epochs without improvement
    VIZ_EVERY   = 20,    # save comparison plots every N epochs

    # --- loss weights (λ values from §1.1) ---
    W_DATA  = 1.0,   # reconstruction at observed points
    W_STAB  = 0.3,   # hydrostatic stability  ∂ρ/∂z ≥ 0
    W_BBL   = 0.2,   # bottom boundary layer  gradients → 0 near seabed
    W_GRAD  = 0.1,   # TV regularization      suppress artifacts
    W_GEO   = 0.15,  # thermal wind balance   ∂u/∂z ~ -∂ρ/∂x

    # --- architecture ---
    # Memory: ~(BASE_CH² × NZ × NX × BATCH × N_BLOCKS × 4 bytes)
    # BASE_CH=32, N_BLOCKS=6, batch=4, 64×64 → ~0.5 GB activations
    BASE_CH  = 32,
    N_BLOCKS = 6,

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_TS(nx, nz, rng):
    """
    Generate a realistic paired (T, S) cross-section via:
      thermocline/halocline (sigmoid in z) + horizontal gradient + random eddy.
    Both fields share the same eddy center so their spatial structure is correlated,
    mimicking real thermohaline covariance ρ(T,S) in the Baltic.

    Returns T, S as ndarray (nz, nx) normalized to [0,1].
    """
    x = np.linspace(0, 1, nx)
    z = np.linspace(0, 1, nz)   # 0 = surface, 1 = bottom
    XX, ZZ = np.meshgrid(x, z)

    # --- Temperature: warm surface, cold bottom ---
    z_thermo  = rng.uniform(0.15, 0.40)
    sharp_T   = rng.uniform(8, 20)
    T = 1.0 / (1.0 + np.exp(sharp_T * (ZZ - z_thermo)))
    T += rng.uniform(-0.15, 0.15) * XX   # horizontal gradient

    # --- Salinity: fresher surface (Baltic), saltier bottom ---
    z_halo  = rng.uniform(0.20, 0.45)
    sharp_S = rng.uniform(6, 16)
    S = 1.0 / (1.0 + np.exp(-sharp_S * (ZZ - z_halo)))  # inverse of T shape
    S += rng.uniform(-0.10, 0.10) * XX

    # --- Shared eddy: same center for T and S (physical covariance) ---
    cx = rng.uniform(0.2, 0.8)
    cz = rng.uniform(0.1, 0.5)
    rx = rng.uniform(0.1, 0.25)
    rz = rng.uniform(0.05, 0.15)
    eddy = np.exp(-((XX-cx)**2/(2*rx**2) + (ZZ-cz)**2/(2*rz**2)))
    T += rng.uniform(-0.25,  0.25) * eddy
    S += rng.uniform(-0.15,  0.15) * eddy   # weaker S anomaly, same location

    T = (T - T.min()) / (T.max() - T.min() + 1e-8)
    S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    return T.astype(np.float32), S.astype(np.float32)


def _synthetic_uv(nx, nz, T, S, rng):
    """
    Synthetic u, v consistent with thermal wind balance (geostrophic physics):
      ∂u/∂z = -(g / f·ρ₀) · ∂ρ/∂y   [thermal wind, x-component]
    Here we approximate on the 2D (x,z) slice:
      u ∝ -∂ρ/∂x integrated over z  (baroclinic shear)
      v = small random background + edddy contribution
    Both are normalized to [-1, 1] for network input.
    """
    alpha, beta = 0.20, 0.08           # linear EOS coefficients for Baltic
    rho = 1.0 - alpha * T + beta * S  # normalized density (∈ [0.9, 1.1] range)

    # u: geostrophic shear ~ -∂ρ/∂x integrated from surface
    drho_dx = np.gradient(rho, axis=1)          # (nz, nx)
    u = -np.cumsum(drho_dx, axis=0) / max(nz, 1)
    u += rng.uniform(-0.1, 0.1)                  # barotropic background

    # v: weaker cross-section component, partially correlated with eddy
    drho_dz = np.gradient(rho, axis=0)
    v = 0.3 * drho_dz + rng.normal(0, 0.05, (nz, nx)).astype(np.float32)

    # LR simulation: downsample then upsample (mimics CMEMS coarse grid)
    factor = 4
    u_lr = u[::factor, ::factor]
    v_lr = v[::factor, ::factor]
    u_lr = np.kron(u_lr, np.ones((factor, factor)))[:nz, :nx]
    v_lr = np.kron(v_lr, np.ones((factor, factor)))[:nz, :nx]

    def norm11(a):
        m = max(np.abs(a).max(), 1e-8)
        return (a / m).astype(np.float32)

    return norm11(u_lr), norm11(v_lr)


def _sloping_bathy(nx, rng):
    """
    Normalized bottom depth ∈ [0.4, 0.99] per x-column.
    Monotonically deepening with small random walk — typical Baltic shelf slope.
    """
    base  = np.linspace(rng.uniform(0.45, 0.65), rng.uniform(0.75, 0.95), nx)
    walk  = np.cumsum(rng.normal(0, 0.015, nx))
    return np.clip(base + walk, 0.35, 0.99).astype(np.float32)


def _cmems_z_indices(nz, n_levels):
    """
    CMEMS standard depth levels: logspace → denser near surface, coarser at depth.
    Returns sorted unique integer indices into the z-axis.
    """
    raw  = np.logspace(0, np.log10(nz - 1), n_levels)
    return np.unique(np.clip(raw.astype(int), 0, nz - 1))


def make_sample(cfg, rng):
    """
    Build one training sample:
      input  (8, NZ, NX) — network input channels (see module docstring)
      target (2, NZ, NX) — ground truth [T, S]
      bathy  (NX,)        — bottom depth per column (for BBL loss)
      below  (NZ, NX)     — True where below the seabed (ignore in loss)
    """
    nx, nz = cfg['NX'], cfg['NZ']

    T, S   = _synthetic_TS(nx, nz, rng)
    bathy  = _sloping_bathy(nx, rng)          # (nx,) ∈ [0,1]
    u_lr, v_lr = _synthetic_uv(nx, nz, T, S, rng)

    # below-seabed mask: pixels where z > bathy[x]
    z_norm = np.linspace(0, 1, nz)[:, None]   # (nz, 1)
    below  = (z_norm > bathy[None, :])         # (nz, nx) bool
    T[below] = 0.0;  S[below] = 0.0

    # --- channels to be filled ---
    T_obs      = np.zeros((nz, nx), np.float32)
    S_obs      = np.zeros((nz, nx), np.float32)
    mask_ctd   = np.zeros((nz, nx), np.float32)
    mask_cmems = np.zeros((nz, nx), np.float32)
    sigma_obs  = np.zeros((nz, nx), np.float32)

    # CTD profiles: random x-positions, full vertical lines above seabed
    ctd_xs = np.sort(rng.choice(nx, cfg['N_CTD'], replace=False))
    for xi in ctd_xs:
        d = int(bathy[xi] * nz)
        mask_ctd[:d, xi] = 1.0
        sigma_obs[:d, xi] = cfg['NOISE_CTD']
        T_obs[:d, xi] = T[:d, xi] + rng.normal(0, cfg['NOISE_CTD'], d).astype(np.float32)
        S_obs[:d, xi] = S[:d, xi] + rng.normal(0, cfg['NOISE_CTD'], d).astype(np.float32)

    # CMEMS points: logspace z-levels × quasi-regular x-columns
    cmems_zs = _cmems_z_indices(nz, cfg['N_CMEMS_Z'])
    cmems_xs = np.linspace(0, nx-1, cfg['N_CMEMS_X'], dtype=int)
    for zi in cmems_zs:
        for xi in cmems_xs:
            if z_norm[zi, 0] < bathy[xi] and mask_ctd[zi, xi] < 0.5:
                mask_cmems[zi, xi] = 1.0
                sigma_obs[zi, xi]  = cfg['NOISE_CMEMS']
                T_obs[zi, xi] = T[zi, xi] + np.float32(rng.normal(0, cfg['NOISE_CMEMS']))
                S_obs[zi, xi] = S[zi, xi] + np.float32(rng.normal(0, cfg['NOISE_CMEMS']))

    # bathymetry channel: continuous depth value broadcast over z
    bathy_ch = np.tile(bathy[None, :], (nz, 1)).astype(np.float32)  # (nz, nx)

    inp = np.stack([T_obs, S_obs, mask_ctd, mask_cmems,
                    u_lr, v_lr, bathy_ch, sigma_obs], axis=0)   # (8, nz, nx)
    tgt = np.stack([T, S], axis=0)                               # (2, nz, nx)

    return dict(input=inp, target=tgt,
                bathy=bathy, below=below, ctd_xs=ctd_xs)


class OceanDataset(Dataset):
    def __init__(self, n, cfg, seed=0):
        rng = np.random.default_rng(seed)
        self.samples = [make_sample(cfg, rng) for _ in range(n)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return (torch.from_numpy(s['input']),
                torch.from_numpy(s['target']),
                torch.from_numpy(s['bathy']),
                torch.from_numpy(s['below'].astype(np.float32)))


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE: LaMa-lite with dual-head decoder and T↔S cross-attention
# ══════════════════════════════════════════════════════════════════════════════

class FourierUnit(nn.Module):
    """
    Applies a learned linear transform in Fourier frequency space.
    Differentiation in frequency domain = multiplication → global receptive field
    in O(N log N) vs O(N²) for self-attention.
    Operates on real and imaginary parts separately via a 1×1 conv.

    Cost: 2 × (ch × (NZ × NX/2+1)) floats per forward pass (rfft output size).
    For ch=64, 64×64: ~0.5M floats ≈ 2 MB per sample in batch.
    """
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch * 2, ch * 2, 1)
        self.bn   = nn.BatchNorm2d(ch * 2)

    def forward(self, x):
        b, c, h, w = x.shape
        f  = torch.fft.rfft2(x, norm="ortho")            # (b, c, h, w//2+1) complex
        fr = torch.cat([f.real, f.imag], dim=1)          # (b, 2c, h, w//2+1)
        fr = F.relu(self.bn(self.conv(fr)))
        c2 = fr.shape[1] // 2
        f  = torch.complex(fr[:, :c2], fr[:, c2:])
        return torch.fft.irfft2(f, s=(h, w), norm="ortho")


class FFCResBlock(nn.Module):
    """
    Residual block combining:
      local branch  : 3×3 conv (receptive field = kernel size)
      global branch : FourierUnit (receptive field = entire field)
    Both branches run in parallel; their outputs are concatenated and fused
    by a 1×1 conv, then added to the residual.

    Params per block at BASE_CH=32: ~2×(32²×9 + 32²) + 32²×2 ≈ 22K params.
    At N_BLOCKS=6: ~132K params just from blocks (cheap).
    """
    def __init__(self, ch):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.glob = FourierUnit(ch)
        self.fuse = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, x):
        return x + self.fuse(torch.cat([self.local(x), self.glob(x)], 1))


class TSCrossAttention(nn.Module):
    """
    Lightweight channel-wise cross-attention between T and S feature maps.
    T attends to S and vice versa → network learns thermohaline covariance
    (thermocline depth ~ halocline depth in Baltic) before final prediction.

    NOT spatial attention (that would be O(N²) in pixels).
    Channel attention: global avg pool → MLP → scale features.
    Cost: 2 × (ch → ch//8 → ch) MLP = negligible vs conv layers.
    """
    def __init__(self, ch):
        super().__init__()
        r = max(ch // 8, 4)
        self.attn_T = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(ch, r), nn.ReLU(),
                                     nn.Linear(r, ch), nn.Sigmoid())
        self.attn_S = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(ch, r), nn.ReLU(),
                                     nn.Linear(r, ch), nn.Sigmoid())

    def forward(self, fT, fS):
        # T gates S and S gates T — mutual modulation
        wT = self.attn_T(fS).view(fS.shape[0], -1, 1, 1)  # S tells T where to look
        wS = self.attn_S(fT).view(fT.shape[0], -1, 1, 1)  # T tells S where to look
        return fT * wT, fS * wS


class LamaGenerator(nn.Module):
    """
    Full LaMa-lite generator with dual T/S output heads.

    in_ch=8 channels (see module docstring):
      [T_obs, S_obs, mask_ctd, mask_cmems, u_lr, v_lr, bathymetry, sigma_obs]

    Architecture:
      Encoder: 7×7 conv → stride-2 conv  (spatial: NZ×NX → NZ/2×NX/2)
      Body   : N_BLOCKS × FFCResBlock    (spatial unchanged, global context via FFT)
      Decoder: stride-2 transposed conv  (back to NZ×NX)
      Heads  : separate 7×7 conv for T and S
      Cross  : TSCrossAttention before final conv (T↔S covariance)
      Output : Sigmoid → [0,1] for both T and S

    Total params at BASE_CH=32, N_BLOCKS=6: ~1.2M
    At BASE_CH=16, N_BLOCKS=4: ~320K (for very limited GPU)
    """
    def __init__(self, in_ch=8, base_ch=32, n_blocks=6):
        super().__init__()
        ch = base_ch * 2   # channels after encoder stride

        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3), nn.ReLU(),
            nn.Conv2d(base_ch, ch, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[FFCResBlock(ch) for _ in range(n_blocks)])

        self.dec = nn.ConvTranspose2d(ch, base_ch, 4, stride=2, padding=1)

        # separate feature heads for T and S before cross-attention
        self.head_T = nn.Sequential(nn.ReLU(), nn.Conv2d(base_ch, base_ch, 3, padding=1))
        self.head_S = nn.Sequential(nn.ReLU(), nn.Conv2d(base_ch, base_ch, 3, padding=1))

        self.cross = TSCrossAttention(base_ch)

        # final 1-channel predictors
        self.out_T = nn.Sequential(nn.Conv2d(base_ch, 1, 7, padding=3), nn.Sigmoid())
        self.out_S = nn.Sequential(nn.Conv2d(base_ch, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        feat  = self.blocks(self.enc(x))
        dec   = self.dec(feat)
        fT, fS = self.head_T(dec), self.head_S(dec)
        fT, fS = self.cross(fT, fS)
        return torch.cat([self.out_T(fT), self.out_S(fS)], dim=1)  # (B,2,H,W)


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _density(T, S):
    """
    Linearized equation of state for Baltic Sea:
      ρ ≈ ρ₀ (1 - α·T + β·S)
    α=0.20 (thermal expansion), β=0.08 (haline contraction), T,S ∈ [0,1].
    Returns normalized density with same shape as input.
    """
    return 1.0 - 0.20 * T + 0.08 * S


def loss_data(pred, target, inp):
    """
    Weighted MSE at observed points.
    Points with lower sigma (CTD, σ=0.05) get higher weight than CMEMS (σ=0.10).
    weight = 1 / (sigma + ε)² — inverse-variance weighting (maximum likelihood).
    Domain MSE added at 0.1× weight to weakly constrain unobserved interior.
    """
    sigma    = inp[:, 7:8]                          # (B,1,H,W) uncertainty map
    mask_any = (inp[:, 2:3] + inp[:, 3:4]).clamp(0, 1)   # CTD ∪ CMEMS
    weight   = mask_any / (sigma.clamp(min=0.01) ** 2)    # inv-variance

    l_obs = (weight * (pred - target) ** 2).mean()
    l_dom = F.mse_loss(pred, target) * 0.1           # weak interior constraint
    return l_obs + l_dom


def loss_stability(pred, below):
    """
    Hydrostatic stability: ∂ρ/∂z ≥ 0 everywhere (denser water below lighter).
    With linear EOS: ∂ρ/∂z < 0  iff  α·∂T/∂z - β·∂S/∂z > 0.
    Penalize ReLU(∂ρ/∂z_discrete) on interior domain points.
    Central difference: Δρ[i] = ρ[i+1] - ρ[i-1] over z-axis (dim=2).
    """
    T, S   = pred[:, :1], pred[:, 1:]
    rho    = _density(T, S)
    domain = (1.0 - below)  # below is already (B,1,H,W) from compute_loss

    drho_dz  = rho[:, :, 2:, :] - rho[:, :, :-2, :]    # (B,1,H-2,W)
    mask     = domain[:, :, 2:, :] * domain[:, :, :-2, :]
    unstable = F.relu(-drho_dz)                          # only violations
    return (unstable * mask).mean()


def loss_bbl(pred, bathy, nz):
    """
    Bottom Boundary Layer homogenization: ∂T/∂z → 0, ∂S/∂z → 0 near seabed.
    Checks the last 2 grid cells above the bathymetry for each column.
    Cost: O(B × NX) scalar ops — cheap loop, runs on CPU indices.
    """
    b, _, h, w = pred.shape
    total = pred.new_zeros(1)
    n = 0
    for bi in range(b):
        for xi in range(w):
            zi = int(bathy[bi, xi].item() * nz) - 2
            if zi > 1:
                dT = pred[bi, 0, zi, xi] - pred[bi, 0, zi-1, xi]
                dS = pred[bi, 1, zi, xi] - pred[bi, 1, zi-1, xi]
                total = total + dT**2 + dS**2
                n += 1
    return total / max(n, 1)


def loss_grad_tv(pred, below):
    """
    Isotropic Total Variation regularization.
    Suppresses unphysical sharp artifacts between adjacent cells.
    Does NOT flatten the thermocline — TV is weak relative to data loss.
    Applied only within ocean domain (not below seabed).
    """
    domain = 1.0 - below  # already (B,1,H,W)
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) * domain[:, :, 1:, :]
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) * domain[:, :, :, 1:]
    return (dy.abs().mean() + dx.abs().mean()) * 0.5


def loss_geostrophic(pred, inp, below):
    """
    Thermal wind balance on 2D (x, z) slice:
      ∂u/∂z = -(g / f·ρ₀) · ∂ρ/∂x   [geostrophic, x-component]
    Network has u_lr (ch4) as conditioning input.
    Penalize mismatch between predicted ∂ρ/∂x and observed ∂u/∂z.
    This nudges the T/S reconstruction to be consistent with velocity shear.
    Normalized separately so magnitude differences don't dominate.
    """
    T, S   = pred[:, :1], pred[:, 1:]
    rho    = _density(T, S)
    u      = inp[:, 4:5]                              # u_lr channel
    domain = (1.0 - below)  # already (B,1,H,W)

    # central differences
    drho_dx = rho[:, :, :, 2:] - rho[:, :, :, :-2]   # (B,1,H,W-2)
    du_dz   = u[:,  :, 2:, :]  - u[:,  :, :-2, :]    # (B,1,H-2,W)

    # trim to common shape
    h_min = min(drho_dx.shape[2], du_dz.shape[2])
    w_min = min(drho_dx.shape[3], du_dz.shape[3])
    d = drho_dx[:, :, :h_min, :w_min]
    u_shear = du_dz[:, :, :h_min, :w_min]
    m = domain[:, :, :h_min, :w_min]

    # normalize each to unit std before comparing (different physical units)
    d_n = d / (d.std() + 1e-6)
    u_n = u_shear / (u_shear.std() + 1e-6)
    return (m * (d_n + u_n) ** 2).mean()   # should be anti-correlated → sum → 0


def compute_loss(pred, target, inp, bathy, below, cfg):
    """
    Total loss = weighted sum of all physical + data terms.
    Returns (scalar loss tensor, dict of named component values for logging).
    """
    below4 = below.unsqueeze(1)

    ld = loss_data(pred, target, inp)
    ls = loss_stability(pred, below4)
    lb = loss_bbl(pred, bathy, cfg['NZ'])
    lg = loss_grad_tv(pred, below4)
    lp = loss_geostrophic(pred, inp, below4)

    total = (cfg['W_DATA'] * ld + cfg['W_STAB'] * ls
             + cfg['W_BBL']  * lb + cfg['W_GRAD'] * lg
             + cfg['W_GEO']  * lp)

    return total, dict(data=ld.item(), stab=ls.item(), bbl=lb.item(),
                       grad=lg.item(), geo=lp.item(), total=total.item())


# ══════════════════════════════════════════════════════════════════════════════
# OI BASELINE (Optimal Interpolation via scipy.griddata)
# ══════════════════════════════════════════════════════════════════════════════

def optimal_interpolation(inp_np, nx, nz):
    """
    Cubic interpolation from all known points (CTD ∪ CMEMS) onto the full grid.
    Returns T_oi, S_oi as ndarray (nz, nx).
    This is the §1 baseline that LaMa should outperform, especially near:
      • fronts and eddies (OI assumes isotropic correlation)
      • data-sparse regions between CTD profiles
    """
    mask = np.clip(inp_np[2] + inp_np[3], 0, 1)
    pts  = np.argwhere(mask > 0.5)                   # (K, 2): [zi, xi]
    if len(pts) < 4:
        return (np.zeros((nz, nx), np.float32),
                np.zeros((nz, nx), np.float32))

    zi_all, xi_all = np.arange(nz), np.arange(nx)
    ZI, XI = np.meshgrid(zi_all, xi_all, indexing='ij')
    grid   = np.stack([ZI.ravel(), XI.ravel()], axis=1)

    T_oi = griddata(pts, inp_np[0][pts[:,0], pts[:,1]], grid,
                    method='cubic', fill_value=0.0).reshape(nz, nx).astype(np.float32)
    S_oi = griddata(pts, inp_np[1][pts[:,0], pts[:,1]], grid,
                    method='cubic', fill_value=0.0).reshape(nz, nx).astype(np.float32)
    return T_oi, S_oi


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred, target, below):
    """
    MSE, MAE, RMSE, PSNR computed only in ocean domain (not below seabed).
    pred, target, below: ndarray (nz, nx) — single field (T or S separately).
    PSNR reference: dynamic range = 1.0 (both fields normalized to [0,1]).
    """
    mask = ~below
    if mask.sum() == 0:
        return dict(mse=np.nan, mae=np.nan, rmse=np.nan, psnr=np.nan)
    p, t = pred[mask], target[mask]
    mse  = float(np.mean((p - t) ** 2))
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(mse))
    psnr = float(20 * np.log10(1.0 / (rmse + 1e-8)))
    return dict(mse=mse, mae=mae, rmse=rmse, psnr=psnr)


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison(sample, pred_lama, pred_oi, epoch, save_path):
    """
    8-panel figure per epoch:
      Row 0 (T): True T | LaMa T | OI T | LaMa T-error | OI T-error
      Row 1 (S): True S | LaMa S | OI S | LaMa S-error | OI S-error
    CTD profile positions shown as cyan vertical lines.
    Error panels use diverging colormap centered at 0.
    Metrics printed in title for report copy-paste.
    """
    inp    = sample['input']
    tgt    = sample['target']    # (2, nz, nx)
    below  = sample['below']
    ctd_xs = sample['ctd_xs']

    T_true, S_true   = tgt[0], tgt[1]
    T_lama, S_lama   = pred_lama[0], pred_lama[1]
    T_oi,   S_oi     = pred_oi[0],   pred_oi[1]

    def masked(a): v = a.copy(); v[below] = np.nan; return v

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.subplots_adjust(wspace=0.35, hspace=0.35)

    for row, (field_name, true, lama, oi) in enumerate([
            ('T', T_true, T_lama, T_oi),
            ('S', S_true, S_lama, S_oi)]):

        vmin, vmax = np.nanmin(masked(true)), np.nanmax(masked(true))
        emax = max(np.abs((lama - true)[~below]).max(),
                   np.abs((oi   - true)[~below]).max(), 1e-6)
        enorm = TwoSlopeNorm(vmin=-emax, vcenter=0, vmax=emax)

        mL = compute_metrics(lama, true, below)
        mO = compute_metrics(oi,   true, below)

        panels = [
            (f'True {field_name}',      masked(true),       'plasma',  None,  vmin, vmax),
            (f'LaMa {field_name}',      masked(lama),       'plasma',  None,  vmin, vmax),
            (f'OI {field_name}',        masked(oi),         'plasma',  None,  vmin, vmax),
            (f'LaMa err RMSE={mL["rmse"]:.3f}', masked(lama-true), 'RdBu_r', enorm, None, None),
            (f'OI err   RMSE={mO["rmse"]:.3f}', masked(oi  -true), 'RdBu_r', enorm, None, None),
        ]
        for col, (title, arr, cmap, norm, vm, vM) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(arr, origin='upper', aspect='auto', cmap=cmap,
                           norm=norm, vmin=vm, vmax=vM, interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('x (distance)'); ax.set_ylabel('z (depth)')
            for xi in ctd_xs:
                ax.axvline(xi, color='cyan', lw=0.6, alpha=0.5, label='CTD' if xi==ctd_xs[0] else '')

    fig.suptitle(f'Epoch {epoch}  |  cyan lines = CTD profiles  |  '
                 f'sparse dots = CMEMS points', fontsize=10)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(history, save_path):
    """Train/val total loss + val component breakdown, both on log scale."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    for split, color in [('train', 'steelblue'), ('val', 'tomato')]:
        ax1.plot([h['total'] for h in history[split]], label=split, color=color)
    ax1.set(title='Total loss (log)', xlabel='epoch', ylabel='loss')
    ax1.set_yscale('log'); ax1.legend()

    for comp in ['data', 'stab', 'bbl', 'grad', 'geo']:
        ax2.plot([h[comp] for h in history['val']], label=comp)
    ax2.set(title='Val loss components (log)', xlabel='epoch', ylabel='loss')
    ax2.set_yscale('log'); ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _move(batch, device):
    return [t.to(device) for t in batch]


def train_epoch(model, loader, optim, cfg, device):
    model.train()
    acc = dict(total=0, data=0, stab=0, bbl=0, grad=0, geo=0)
    for inp, tgt, bathy, below in loader:
        inp, tgt, bathy, below = _move([inp, tgt, bathy, below], device)
        loss, parts = compute_loss(model(inp), tgt, inp, bathy, below, cfg)
        optim.zero_grad(); loss.backward(); optim.step()
        for k in acc: acc[k] += parts[k]
    n = len(loader)
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def val_epoch(model, loader, cfg, device):
    model.eval()
    acc = dict(total=0, data=0, stab=0, bbl=0, grad=0, geo=0)
    for inp, tgt, bathy, below in loader:
        inp, tgt, bathy, below = _move([inp, tgt, bathy, below], device)
        _, parts = compute_loss(model(inp), tgt, inp, bathy, below, cfg)
        for k in acc: acc[k] += parts[k]
    n = len(loader)
    return {k: v / n for k, v in acc.items()}


@torch.no_grad()
def run_evaluation(model, val_ds, cfg, device, epoch, prefix=""):
    """
    Evaluate LaMa vs OI on first 8 val samples.
    Saves one comparison figure. Returns mean metrics for both methods.
    """
    model.eval()
    rows = dict(T_lama=[], S_lama=[], T_oi=[], S_oi=[])

    for i in range(min(8, len(val_ds))):
        s     = val_ds.samples[i]
        inp_t = torch.from_numpy(s['input']).unsqueeze(0).to(device)
        pred  = model(inp_t)[0].cpu().numpy()    # (2, nz, nx)
        tgt   = s['target']                      # (2, nz, nx)
        below = s['below']

        T_oi, S_oi = optimal_interpolation(s['input'], cfg['NX'], cfg['NZ'])
        pred_oi = np.stack([T_oi, S_oi])

        rows['T_lama'].append(compute_metrics(pred[0],   tgt[0], below))
        rows['S_lama'].append(compute_metrics(pred[1],   tgt[1], below))
        rows['T_oi'  ].append(compute_metrics(pred_oi[0], tgt[0], below))
        rows['S_oi'  ].append(compute_metrics(pred_oi[1], tgt[1], below))

        if i == 0:
            plot_comparison(s, pred, pred_oi, epoch,
                            OUT / f"{prefix}comparison_ep{epoch:04d}.png")

    def mean_m(key):
        return {k: float(np.mean([r[k] for r in rows[key]])) for k in rows[key][0]}

    return {k: mean_m(k) for k in rows}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(cfg=CFG):
    device = cfg['DEVICE']

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"[init] device={device}  NX={cfg['NX']} NZ={cfg['NZ']}")
    print("[data] generating synthetic datasets...")
    train_ds = OceanDataset(cfg['N_TRAIN'], cfg, seed=42)
    val_ds   = OceanDataset(cfg['N_VAL'],   cfg, seed=99)
    train_dl = DataLoader(train_ds, cfg['BATCH'], shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   cfg['BATCH'], shuffle=False, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = LamaGenerator(in_ch=8,
                          base_ch=cfg['BASE_CH'],
                          n_blocks=cfg['N_BLOCKS']).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory estimate: activations dominate at ~4 bytes × batch × ch × H × W × depth
    act_mb = (4 * cfg['BATCH'] * cfg['BASE_CH']*2 *
              cfg['NZ']//2 * cfg['NX']//2 * cfg['N_BLOCKS']) / 1e6
    print(f"[model] params={n_par/1e3:.1f}K  "
          f"estimated activation memory ≈{act_mb:.0f} MB at batch={cfg['BATCH']}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg['LR'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, patience=cfg['LR_PATIENCE'], factor=0.5)

    history  = dict(train=[], val=[])
    best_val = math.inf
    es_count = 0
    eval_table = []
    t0 = time.time()

    # ── training loop ─────────────────────────────────────────────────────────
    for ep in range(1, cfg['EPOCHS'] + 1):
        tr = train_epoch(model, train_dl, optim, cfg, device)
        vl = val_epoch(model, val_dl,   cfg, device)
        history['train'].append(tr)
        history['val'  ].append(vl)
        sched.step(vl['total'])

        improved = vl['total'] < best_val
        if improved:
            best_val = vl['total']
            es_count = 0
            torch.save({'epoch': ep, 'state': model.state_dict(),
                        'cfg': cfg, 'val_loss': best_val}, CKPT / 'best.pt')
        else:
            es_count += 1

        if ep % 10 == 0 or ep == 1:
            lr_now = optim.param_groups[0]['lr']
            elapsed = (time.time() - t0) / 60
            # log format: loss components visible to catch imbalanced λ weights
            print(f"[ep {ep:4d}/{cfg['EPOCHS']}] "
                  f"train={tr['total']:.4f} val={vl['total']:.4f} | "
                  f"data={vl['data']:.4f} stab={vl['stab']:.4f} "
                  f"bbl={vl['bbl']:.4f} grad={vl['grad']:.4f} geo={vl['geo']:.4f} | "
                  f"lr={lr_now:.1e} elapsed={elapsed:.1f}min ES={es_count}/{cfg['ES_PATIENCE']}")

        if ep % cfg['VIZ_EVERY'] == 0 or ep == 1:
            metrics = run_evaluation(model, val_ds, cfg, device, ep)
            mTL = metrics['T_lama']; mTO = metrics['T_oi']
            mSL = metrics['S_lama']; mSO = metrics['S_oi']
            print(f"  [eval ep={ep}]"
                  f"  T: LaMa RMSE={mTL['rmse']:.4f} PSNR={mTL['psnr']:.1f}dB"
                  f"  OI RMSE={mTO['rmse']:.4f} PSNR={mTO['psnr']:.1f}dB"
                  f"  |  S: LaMa RMSE={mSL['rmse']:.4f}  OI RMSE={mSO['rmse']:.4f}")
            eval_table.append(dict(epoch=ep,
                T_lama_rmse=mTL['rmse'], T_lama_psnr=mTL['psnr'],
                T_oi_rmse  =mTO['rmse'], T_oi_psnr  =mTO['psnr'],
                S_lama_rmse=mSL['rmse'], S_lama_psnr=mSL['psnr'],
                S_oi_rmse  =mSO['rmse'], S_oi_psnr  =mSO['psnr']))

        if es_count >= cfg['ES_PATIENCE']:
            print(f"[early stop] no improvement for {cfg['ES_PATIENCE']} epochs → stop at ep={ep}")
            break

    # ── final evaluation on best checkpoint ───────────────────────────────────
    print("[final] loading best checkpoint...")
    model.load_state_dict(torch.load(CKPT / 'best.pt', map_location=device)['state'])
    metrics = run_evaluation(model, val_ds, cfg, device, ep, prefix="final_")
    mTL = metrics['T_lama']; mTO = metrics['T_oi']
    mSL = metrics['S_lama']; mSO = metrics['S_oi']
    eval_table.append(dict(epoch=-1,
        T_lama_rmse=mTL['rmse'], T_lama_psnr=mTL['psnr'],
        T_oi_rmse  =mTO['rmse'], T_oi_psnr  =mTO['psnr'],
        S_lama_rmse=mSL['rmse'], S_lama_psnr=mSL['psnr'],
        S_oi_rmse  =mSO['rmse'], S_oi_psnr  =mSO['psnr']))

    # ── save outputs ──────────────────────────────────────────────────────────
    plot_loss_curves(history, OUT / 'loss_curves.png')
    with open(OUT / 'eval_table.json', 'w') as f:
        json.dump(eval_table, f, indent=2)

    # summary table for report
    hdr = (f"{'Ep':>5} | {'T_LaMa_RMSE':>12} {'T_LaMa_PSNR':>12} "
           f"{'T_OI_RMSE':>10} {'T_OI_PSNR':>10} | "
           f"{'S_LaMa_RMSE':>12} {'S_OI_RMSE':>11}")
    print("\n" + "="*90)
    print(hdr); print("-"*90)
    for r in eval_table:
        tag = "BEST" if r['epoch'] == -1 else str(r['epoch'])
        print(f"{tag:>5} | {r['T_lama_rmse']:>12.4f} {r['T_lama_psnr']:>12.1f} "
              f"{r['T_oi_rmse']:>10.4f} {r['T_oi_psnr']:>10.1f} | "
              f"{r['S_lama_rmse']:>12.4f} {r['S_oi_rmse']:>11.4f}")
    print("="*90)

    total_min = (time.time() - t0) / 60
    print(f"\n[done] total time: {total_min:.1f} min  |  outputs: {OUT}")


if __name__ == "__main__":
    main()
