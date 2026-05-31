# Hydro T+S Generalization Strategy

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Pipeline — What Changes, What Stays Fixed](#2-data-pipeline--what-changes-what-stays-fixed)
3. [Strategy to Beat OI](#3-strategy-to-beat-oi)
4. [Generalization Features — Detailed Reference](#4-generalization-features--detailed-reference)
   - 4.1 [Scene Diversity (n_variants)](#41-scene-diversity-n_variants)
   - 4.2 [Uncertainty Weighting (Kendall & Gal 2017)](#42-uncertainty-weighting-kendall--gal-2017)
   - 4.3 [OneCycleLR Scheduler](#43-onecylr-scheduler)
   - 4.4 [FFCResBlock with Fourier Features](#44-ffcresblock-with-fourier-features)
   - 4.5 [Physical Augmentation](#45-physical-augmentation)
   - 4.6 [Curriculum Learning (N_CTD Schedule)](#46-curriculum-learning-n_ctd-schedule)
   - 4.7 [OOD Validation](#47-ood-validation)
   - 4.8 [Gradient-Dependent Noise](#48-gradient-dependent-noise)
   - 4.9 [Model Capacity Control](#49-model-capacity-control)
5. [LightningModule Integration](#5-lightningmodule-integration)
6. [End-to-End Training Recipe](#6-end-to-end-training-recipe)
7. [Dataset Variations & Training Recipes](#7-dataset-variations--training-recipes)
8. [Monitoring & Diagnostics](#8-monitoring--diagnostics)
9. [Recommendations by Dataset Size](#9-recommendations-by-dataset-size)
10. [Configuration Reference](#10-configuration-reference)
11. [References](#11-references)


---

## 1. Problem Statement

The core generalization challenge in oceanographic inpainting:

```
train RMSE ≈ val RMSE  <<  test RMSE
```

Root causes:

- **Synthetic data is generated from a single parametric model** (Baltic `SceneGenerator`). Train and val distributions are nearly identical — same generator, same parameter ranges, same bathymetry model.
- **Real ocean data differs fundamentally**: instrument noise is non-Gaussian, bathymetry has features the generator cannot produce, thermocline ranges exceed the "realistic" mode, and observation patterns are irregular.
- **The model has 255K parameters but trains on ~400 samples**. Without regularization, it memorizes the generator's output manifold rather than learning general T/S reconstruction.
- **Fixed scenes across epochs**: the same 400 T/S/u/v fields are seen every epoch. Only the observation mask (CTD positions, noise) changes. The network can memorize scene-specific patterns instead of learning physics.

Each feature below addresses a specific aspect of this gap.


---

## 2. Data Pipeline — What Changes, What Stays Fixed

Understanding the data flow is critical for diagnosing generalization failures.

### Per-sample data flow at `__getitem__(idx)`

```
Scene generation (once, at __init__):
┌─────────────────────────────────────────────────────────────────┐
│ SceneGenerator(seed, mode, season)                              │
│   → T, S, u, v fields (nz × nx)                                │
│   → bathymetry, below-bottom mask                               │
│                                                                 │
│ Stored as: self._scene_variants[sample_idx] = [variant_0, ...]  │
│ Each variant has different (mode, season) → different T/S/u/v   │
└─────────────────────────────────────────────────────────────────┘

Per-epoch access (every __getitem__ call):
┌─────────────────────────────────────────────────────────────────┐
│ 1. SELECT variant: random choice from n_variants (default 4)    │
│    → Different T, S, u, v, bathy, below each time               │
│                                                                 │
│ 2. SAMPLE observations: sample_observations(T, S, ...)          │
│    → New CTD positions (random x-coordinates)                   │
│    → New CMEMS grid points                                      │
│    → New Gaussian noise on T_obs, S_obs                         │
│    → Gradient-dependent sigma_obs                               │
│                                                                 │
│ 3. CURRICULUM: filter n_ctd by current epoch schedule            │
│    → Early: n_ctd=7 (easy, many observations)                   │
│    → Late:  n_ctd=2 (hard, few observations)                    │
│                                                                 │
│ 4. AUGMENTATION (if enabled):                                   │
│    → Horizontal flip (50%): all channels + target + bathy_idx   │
│    → Amplitude scaling (±20%): channels 0,1 + target only       │
│    → CTD dropout (30%): remove 1/3 of CTD profiles              │
└─────────────────────────────────────────────────────────────────┘

Output tensor (8 channels):
┌──────┬──────────────────────────────────────────────────────────┐
│  Ch  │ Content                          │ Changes per epoch?   │
├──────┼──────────────────────────────────┼──────────────────────┤
│  0   │ T_obs (noisy observations)       │ YES (variant+noise)  │
│  1   │ S_obs (noisy observations)       │ YES (variant+noise)  │
│  2   │ mask_ctd (CTD positions)         │ YES (random + aug)   │
│  3   │ mask_cmems (CMEMS positions)     │ YES (random)         │
│  4   │ u_lr (zonal velocity)            │ YES (variant)        │
│  5   │ v_lr (meridional velocity)       │ YES (variant)        │
│  6   │ bathymetry (continuous depth)    │ YES (variant)        │
│  7   │ sigma_obs (uncertainty map)      │ YES (variant+noise)  │
├──────┼──────────────────────────────────┼──────────────────────┤
│ tgt  │ T, S ground truth                │ YES (variant)        │
│ mask │ below-bottom mask                │ YES (variant)        │
└──────┴──────────────────────────────────┴──────────────────────┘
```

**Before scene variants:** all 8 channels were fixed except mask_ctd, mask_cmems, and noise.
**After scene variants (n_variants=4):** every channel changes because the underlying T/S/u/v fields change.


---

## 3. Strategy to Beat OI

### The OI Baseline

Optimal Interpolation (OI) is a classical method that performs cubic interpolation from sparse observation points onto the full grid. It is the baseline the neural network must beat.

OI characteristics:
- **No physics**: OI interpolates purely based on spatial proximity. It cannot enforce hydrostatic stability, geostrophic balance, or bottom boundary layer constraints.
- **No velocity information**: OI uses only T and S observation values. It ignores u_lr, v_lr, bathymetry, and sigma_obs channels.
- **Smooth output**: cubic interpolation produces smooth fields that blur sharp thermoclines.
- **No cross-variable coupling**: OI interpolates T and S independently. It cannot use T structure to inform S reconstruction (or vice versa).

### Why the Network Should Beat OI

The network has access to 8 input channels (vs 2 for OI) and learns physics constraints through the loss function:
- `hydrostatic_stability_loss` prevents density inversions
- `geostrophic_balance_loss` enforces thermal wind balance
- `bottom_boundary_layer_loss` enforces zero gradient at seafloor
- `inverse_variance_mse` downweights noisy observations
- `TSCrossAttention` learns T↔S covariance (thermocline ~ halocline)

### Smoke Test Results (1 epoch, 8 samples, base_ch=16)

```
T LaMa RMSE=0.3228  OI RMSE=0.3802  → 15% better
S LaMa RMSE=0.2623  OI RMSE=0.4359  → 40% better
```

Even with minimal training (1 epoch, 8 samples, tiny model), the network already beats OI. Full training should yield:
- T RMSE: 0.10–0.15 (vs OI ~0.25–0.40)
- S RMSE: 0.08–0.12 (vs OI ~0.30–0.45)

### Combined Strategy to Maximize the Gap

The features are synergistic — each addresses a different failure mode:

| Failure mode | Feature | Expected impact |
|-------------|---------|-----------------|
| Loss imbalance (one term dominates) | Uncertainty weighting | Prevents gradient starvation of small losses |
| Poor convergence (oscillating loss) | OneCycleLR | Smooth LR curve, warmup prevents early divergence |
| Overfitting to fixed scenes | Lazy mode (on-the-fly generation) | Infinite unique T/S fields per epoch |
| Overfitting to specific patterns | FFCResBlock + augmentation | Fourier global branch + flip/scale/dropout CTD |
| Easy samples dominate | Curriculum learning | Harder samples introduced gradually |
| Unreliable generalization measure | OOD validation | Stress-mode val set detects overfitting to training distribution |
| Noise mismatch with real data | Gradient-dependent noise | Higher noise at thermocline (realistic instrument behavior) |
| Too many params for dataset | Model capacity control | Match params to dataset size |

**The single most important feature for beating OI on unseen data is scene diversity.** Without it, the network memorizes a fixed set of T/S fields and fails on any new field. Lazy mode (`lazy=True`) provides infinite diversity by generating fresh scenes each access at ~1ms cost.


---

## 4. Generalization Features — Detailed Reference

### 4.1 Scene Diversity (Lazy Mode)

**The problem.** With a fixed pool of 400 pre-generated scenes, the model sees the same 1600 T/S target fields across all epochs. With 255K parameters, it can memorize these fields and fail on any new data.

**The solution.** **Lazy mode** (`lazy=True`, the default) generates scenes on-the-fly in `__getitem__`. Each access creates a fresh scene with a deterministic seed derived from `(idx, epoch)`:

```python
scene_seed = seed + idx + epoch * 1_000_003
```

This gives **infinite effective diversity** — the model never sees the same T/S field twice across epochs. Over 40 epochs with `n_train=400`, it sees 16,000 unique T/S fields (vs 1,600 with pre-generated n_variants=4).

**Why deterministic seeds.** Each `(idx, epoch)` pair maps to exactly one scene, so:
- Reproducible per-epoch (same epoch → same scenes for debugging)
- Different across epochs (epoch 0 scene 0 ≠ epoch 1 scene 0)
- Workers in DataLoader get the same scene (seed is per-sample, not per-worker)

**Cost.** ~1ms per scene generation in `__getitem__`. For `n_train=400, batch_size=4`, this adds ~100ms per epoch (negligible vs ~3s for forward/backward).

**Memory.** Zero — scenes are created and discarded per access. No pre-generation storage.

### Generation Parameters — What Varies Per Scene

The following parameters are **randomized per scene** (each `__getitem__` call gets a fresh draw from the full range):

| Parameter | Range / Values | Controlled by | Varies per epoch? |
|-----------|---------------|---------------|-------------------|
| `mode` | `realistic`, `extended`, `stress` | `mode_mix` config (default 70/20/10) | YES — resampled each `__getitem__` |
| `season` | `winter`, `spring`, `summer`, `autumn` | uniform over 4 seasons | YES |
| `scene_type` | `background`, `internal_waves`, `upwelling`, `downwelling`, `front`, `intrusion`, `jets` | uniform inside `SceneGenerator` | YES |
| T/S surface & bottom ranges | sampled from `SceneConfig` per mode | `mode` + `season` | YES |
| Bathymetry | Σ-Gaussian, 2–7 bumps, 5–200 m | `mode` affects amplitude via `bathy_amp_scale` | YES |
| Interface depth, slope, wave, A_T, A_S | sampled per interface | `mode` + `scene_type` overrides | YES |
| Halocline depth, ΔS, thickness | Baltic climatology ranges | fixed `HALO_Z`, `HALO_DS`, `HALO_TH` | YES |
| CIL presence | `true` in winter/spring (+stress) | `mode` + `season` | YES |
| Lens count & parameters | (0–5 lenses) | `mode` via `lens_amp_scale` | YES |
| Jet count, strength, position | 1–4 jets, split upper/lower | `scene_type` overrides | YES |
| CTD x-positions | random subset of nx columns | `n_ctd` config | YES |
| `n_ctd` (effective) | 2–7, curriculum schedule | `CurriculumCallback` | YES (linear 7→2) |
| Gradient-dependent noise scale | 1×–3× base `noise_ctd` / `noise_cmems` | computed from T/S gradient | YES |
| CTD / CMEMS noise realization | fresh Gaussian noise per observation | `noise_ctd`, `noise_cmems` config | YES |
| Augmentation (if enabled) | flip, scale 0.8–1.2, CTD dropout 30% | `augment` config | YES |

**Parameters that stay fixed across scenes** (config defaults, not varied):
| Parameter | Default | Effect |
|-----------|---------|--------|
| `nx`, `nz` | 64, 64 | Grid resolution |
| `n_cmems_x` | 8 | CMEMS columns per sample |
| `n_cmems_z` | 10 | CMEMS depth levels per column |
| `n_cmems_x`, `n_cmems_z` | 8, 10 | CMEMS grid geometry |

All the above are configurable via YAML or CLI — see Section 10.

**Fallback.** `lazy=false` uses the original pre-generated mode with `n_variants` per sample. Use this when exact bitwise reproducibility is required across runs.

| Mode | Unique targets/epoch | Memory | Reproducibility |
|------|---------------------|--------|-----------------|
| `lazy=true` (default) | n_train × epochs (unlimited) | ~0 MB | Per-epoch deterministic |
| `lazy=false` | n_train × n_variants (fixed) | ~280 KB × n_train | Exact bitwise |


### 4.2 Uncertainty Weighting (Kendall & Gal 2017)

**Why selected.** The original training script used 7 hardcoded magic numbers as loss weights:

```python
# BEFORE: arbitrary hand-tuned weights
loss = 10.0 * F.l1_loss(pred * mask, tgt * mask)
loss += 1.0 * F.l1_loss(pred * (1 - mask), tgt * (1 - mask))
loss += 1.0 * inverse_variance_mse(...)
loss += 0.3 * hydrostatic_stability_loss(...)
loss += 0.2 * bottom_boundary_layer_loss(...)
loss += 0.15 * geostrophic_balance_loss(...)
loss += 1.0 * smoothness_loss(...)
```

These weights encode assumptions about relative importance that may be wrong. A weight of 10.0 for observation L1 implies it is 50× more important than BBL loss — but there is no physical justification for this ratio. Worse, the optimal weights depend on the magnitude of each loss term, which changes during training.

**How it works.** Kendall & Gal (2017) showed that for a homoscedastic regression task with multiple loss terms, the ML estimate of the weight is related to the observation noise σ through:

```
L_weighted = (1 / 2σ²) × L + log(σ)
```

By parameterizing `log_var = 2 × log(σ)` and making it learnable, the network automatically tunes each loss weight during training. The key identity:

```
exp(-log_var) × L + 0.5 × log_var
```

- When a loss term has high initial magnitude, `log_var` increases → weight decreases
- When a loss term is small and useful, `log_var` decreases → weight increases
- The `0.5 × log_var` regularizer prevents all weights from going to infinity

**Implementation.** Six learnable parameters, one per loss term:

```python
# In HydroLightningModule.__init__
self.log_vars = nn.Parameter(torch.zeros(6))

# In training_step — loss composition
losses = [
    F.l1_loss(pred * mask, tgt * mask),                # [0] obs_l1
    F.l1_loss(pred * (1 - mask), tgt * (1 - mask)),    # [1] domain_l1
    inverse_variance_mse(...),                          # [2] inv_var_mse
    hydrostatic_stability_loss(...),                    # [3] stability
    bottom_boundary_layer_loss(...),                    # [4] bbl
    geostrophic_balance_loss(...),                      # [5] geostrophic
]
smooth = smoothness_loss(pred, mask=mask)

total = sum(
    torch.exp(-self.log_vars[i]) * l + 0.5 * self.log_vars[i]
    for i, l in enumerate(losses)
) + smooth  # smoothness stays outside uncertainty weighting
```

**Why smoothness is excluded.** Total variation loss is a regularizer, not a physics loss. Its weight should remain tunable independently — you may want to increase it for noisy real data or decrease it for smooth synthetic data. Keeping it outside the uncertainty framework preserves this knob.

**Monitoring.** The learned `sigma_i = exp(0.5 × log_var_i)` is logged to TensorBoard at each step. Expected behavior:
- sigma values should stabilize after ~10 epochs
- If sigma_i → 0 for some term, that term dominates and others are ignored (bad)
- If sigma_i → ∞, that term is useless and being downweight (acceptable if physically expected)

**LightningModule integration.** `log_vars` is an `nn.Parameter`, so PL's automatic optimization includes it in the computation graph. The optimizer (Adam) updates both generator weights and log_vars simultaneously. No special handling is needed — PL's `automatic_optimization=True` handles backward pass, optimizer step, and scheduler step automatically.


### 4.3 OneCycleLR Scheduler

**Why selected.** The original code used bare Adam with a fixed learning rate of 2e-4. This has two problems:
1. No warmup — large initial gradients from randomly initialized log_vars can destabilize training
2. No decay — the learning rate never decreases, so the model oscillates around minima instead of converging

OneCycleLR (Smith & Topin 2019) provides:
- **Warmup phase** (10% of training): LR ramps from `lr/10` to `lr`. This prevents early divergence when log_vars are still at zero and loss magnitudes are unbalanced.
- **Cosine annealing** (remaining 90%): LR decays from `lr` to `lr/100`. This allows fine-grained convergence in later epochs.

**Implementation.** The scheduler is returned from `configure_optimizers` with `interval: "step"`, so PL calls `sched.step()` after every optimizer step (not every epoch):

```python
def configure_optimizers(self):
    opt = torch.optim.Adam(
        self.generator.parameters(), lr=self.lr, betas=(0.0, 0.999),
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=self.lr,
        total_steps=self.trainer.estimated_stepping_batches,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, "interval": "step"},
    }
```

**Why `automatic_optimization=True`.** The original code used `self.automatic_optimization = False` with manual `self.manual_backward(loss)`, `opt.step()`, `opt.zero_grad()`. This was unnecessary — there is no gradient accumulation, no multiple optimizers, or no custom backward logic. Switching to automatic optimization:
- Lets PL handle scheduler stepping (critical for OneCycleLR)
- Simplifies the training_step to just compute and return the loss
- Enables PL's gradient clipping hooks if needed later

**Why betas=(0.0, 0.999).** This is the original configuration from the lama_hydro codebase. β₁=0 disables momentum, which can be beneficial when the loss landscape changes rapidly (as it does with learnable weights). β₂=0.999 is the standard Adam second-moment decay.


### 4.4 FFCResBlock with Fourier Features

**Why selected.** The original `SimpleResBlock` uses only local 3×3 convolutions, giving a receptive field of ~13×13 pixels after 6 stacked blocks. For 64×64 ocean fields with large-scale baroclinic structures (eddies spanning 30+ grid cells), this is insufficient. The FFCResBlock adds a global (Fourier) branch that processes a fraction of channels via FFT, achieving ~98% receptive field coverage with fewer parameters.

**Architecture.** Each FFCResBlock splits channels into local and global branches:

```
ratio_g = 0.5 → 16 local channels, 16 global channels (for ch=32)

Local branch:  3×3 Conv → BN → ReLU → 3×3 Conv → BN  (sharp gradients)
Global branch: AvgPool(2) → FourierUnit → Conv1×1 → ReLU → upsample  (large-scale)
Output: concat + residual + fuse Conv1×1
```

**FourierUnit.** Applies a learned mixing matrix in frequency space via `rfft2` → Conv1×1 on real+imag parts → `irfft2`. No activation in the frequency domain — this preserves Hermitian symmetry required by `irfft2`. The old `ffc.py` applied ReLU inside the frequency domain, which breaks this symmetry.

**Why no Dropout2d.** The Fourier branch acts as implicit regularization — it forces the network to represent both local and global structure simultaneously. Dropout2d is unnecessary with FFCResBlock and the 255K parameter count (vs 530K with SimpleResBlock).

**Configuration.** `ratio_g` controls the local/global split:
- `ratio_g=0.5` (default): half local, half global — balanced
- `ratio_g=0.25`: 75% local, 25% global — favors sharp gradients
- `ratio_g=0.75`: 25% local, 75% global — favors large-scale structures


### 4.5 Physical Augmentation

**Why selected.** Standard image augmentations (rotation, cropping, color jitter) are physically invalid for oceanographic data. A 90° rotation would swap the depth and horizontal axes, producing nonsensical T/S fields. The augmentations must respect the physics:

1. **Horizontal reflection** (x → -x): A mirror image of an ocean cross-section is physically valid. The bathymetry profile reverses, and the zonal velocity direction negates. This doubles the effective dataset size.

2. **Amplitude scaling** (±20%): Multiplying T and S by a scalar simulates seasonal variation (e.g., warmer summer vs cooler spring). The scalar must be applied to both the observation channels AND the target, consistently. It must NOT be applied to mask channels, velocity, or bathymetry.

3. **Random CTD dropout**: Removing 1/3 of CTD profiles simulates sparse instrument coverage. This forces the network to interpolate from fewer vertical lines, improving robustness to varying observation density.

**Why not vertical reflection.** A vertical reflection (z → -z) would flip the ocean upside down, placing the surface at the bottom. This is physically nonsensical — the ocean has a free surface at the top and a rigid bottom at the bottom.

**Why not rotation.** Even a small rotation would misalign the vertical (gravity) direction. Ocean stratification is driven by gravity — density must increase with depth. Rotating the field breaks this fundamental constraint.

**Implementation.** Augmentation is applied lazily in `__getitem__` using a deterministic per-sample RNG. The RNG seed depends on both the sample index and the current epoch, so the same sample gets different augmentations across epochs:

```python
if self.augment:
    aug_rng = np.random.default_rng(seed=idx + self._epoch * 10000)
    inp, tgt, bathy_indices, u_input, below = augment_hydro_sample(
        inp, tgt, bathy_indices, u_input, below, aug_rng,
    )
```

**Critical detail: horizontal flip + bathy_indices.** When the input is flipped horizontally, `bathy_indices` (which maps each x-column to its bottom depth index) must also be reversed:

```python
if rng.random() < 0.5:
    inp = inp[:, :, ::-1].copy()
    tgt = tgt[:, :, ::-1].copy()
    bathy_indices = bathy_indices[::-1].copy()
    u_input = u_input[:, :, ::-1].copy()
    u_input = 1.0 - u_input  # velocity direction reverses
    below = below[:, ::-1].copy()
```

The velocity flip uses `1.0 - u_input` because `u_lr` is stored in [0, 1] space where 0.5 represents zero velocity (after the `(u + 1) / 2` transform). Mirroring negates the velocity, so `u → -u` becomes `(1 - u_lr)` in [0, 1] space.

**Critical detail: amplitude scaling scope.** Only channels 0, 1 (T_obs, S_obs) in the input and the target are scaled. Channels 2–7 (masks, velocities, bathymetry, sigma) are NOT scaled because:
- Masks are binary (0/1) — scaling would change their semantics
- Velocities have their own physical scale — T amplitude changes don't affect currents
- Bathymetry is a geometric property — independent of water temperature
- Sigma_obs represents measurement uncertainty — independent of field amplitude


### 4.6 Curriculum Learning (N_CTD Schedule)

**Why selected.** The number of CTD profiles (vertical observation lines) controls task difficulty:
- **7 CTD profiles** (easy): The network receives dense vertical coverage and only needs to interpolate between lines
- **2 CTD profiles** (hard): The network must extrapolate large gaps from sparse data

Starting with easy samples and gradually introducing harder ones stabilizes training. This is analogous to curriculum learning in NLP (Bengio et al. 2009) — the model first learns basic T/S structure from well-observed data, then refines its ability to handle sparse observations.

**Implementation.** Each sample is pre-generated with an `n_ctd` value drawn uniformly from {2, 3, 4, 5, 6, 7}. The `set_epoch()` method implements a linear schedule:

```python
def set_epoch(self, epoch, total_epochs):
    frac = epoch / total_epochs
    self._min_n_ctd = max(2, int(7 - frac * 5))  # 7→2 over training
```

In `__getitem__`, the effective n_ctd is `max(scene_n_ctd, min_n_ctd)`. Early in training, `min_n_ctd=7` forces all samples to use 7 CTD profiles regardless of their assigned value. Late in training, `min_n_ctd=2` allows all difficulty levels.

**Integration with PL.** The `CurriculumCallback` calls `set_epoch()` at each epoch start:

```python
class CurriculumCallback(Callback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_dataset.set_epoch(trainer.current_epoch, trainer.max_epochs)
```

This is attached to the Trainer's callback list. PL calls `on_train_epoch_start` before the first training batch of each epoch, so the curriculum state is updated before any samples are drawn.

**Why pre-generate with variable n_ctd instead of regenerating.** Regenerating scenes at each epoch would be expensive (~0.5s per scene × 400 scenes = 200s per epoch). Pre-generating all scenes with fixed T/S/u/v fields but varying the observation sampling at `__getitem__` time is both cheaper and more diverse — the same scene can be observed with different CTD positions each time, which already happens because `sample_observations` uses a fresh RNG each call.


### 4.7 OOD Validation

**Why selected.** Standard validation uses the same generator distribution as training. A low val_loss does not guarantee the model will generalize to real ocean data. OOD (out-of-distribution) validation creates a separate validation set with stress-mode parameters that push beyond the training distribution:

| Parameter | In-distribution (val) | OOD (ood_val) |
|-----------|----------------------|---------------|
| mode_mix | 70% realistic, 20% extended, 10% stress | 0% realistic, 30% extended, 70% stress |
| n_ctd | 5 | 2 |
| noise_ctd | 0.05 | 0.08 |
| noise_cmems | 0.10 | 0.15 |

**Implementation.** Two validation dataloaders are passed to PL:

```python
trainer.fit(model, train_dl, [val_dl, ood_val_dl], ...)
```

PL runs validation on both dataloaders sequentially. The `validation_step` receives a `dataloader_idx` parameter to distinguish them:

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
    # ... compute loss ...
    loss_name = "val_loss" if dataloader_idx == 0 else "ood_val_loss"
    self.log(loss_name, total, ...)
```

**Metric naming.** When PL has multiple val dataloaders, it appends `/dataloader_idx_N` to metrics. The checkpoint and early stopping callbacks must monitor the correct metric:

```python
ModelCheckpoint(monitor="val_loss/dataloader_idx_0", ...)
EarlyStopping(monitor="val_loss/dataloader_idx_0", ...)
```

**What to look for.** The gap between `val_loss` and `ood_val_loss` indicates generalization quality:
- Small gap (< 20%): Good generalization — model handles stress conditions
- Large gap (> 50%): Poor generalization — model is overfitting to training distribution
- ood_val_loss decreasing over training: Model is learning transferable features


### 4.8 Gradient-Dependent Noise

**Why selected.** Real CTD instruments have noise that scales with gradient magnitude. When a CTD profiler crosses a sharp thermocline, the measured temperature fluctuates more than in a well-mixed layer. The original code used fixed σ=0.05 for all CTD points and σ=0.10 for all CMEMS points, which:
- Underestimates noise at sharp gradients (thermocline, halocline)
- Overestimates noise in well-mixed layers

**Implementation.** The noise scale is computed from the gradient magnitude of the clean T and S fields:

```python
grad_T = np.sqrt(np.gradient(T, axis=0)**2 + np.gradient(T, axis=1)**2)
grad_S = np.sqrt(np.gradient(S, axis=0)**2 + np.gradient(S, axis=1)**2)
grad_mean = (grad_T.mean() + grad_S.mean()) * 0.5 + 1e-6
noise_scale = 1.0 + 2.0 * (grad_T + grad_S) * 0.5 / grad_mean
```

This produces a scale factor that is:
- ~1.0 in well-mixed regions (gradient ≈ mean)
- ~3.0 at sharp fronts (gradient ≈ 2× mean)
- Spatially varying — each grid point has its own noise level

The scaled noise is used for both the observation values AND the sigma_obs channel:

```python
sigma_obs[:d, xi] = noise_ctd * local_scale
T_obs[:d, xi] = T[:d, xi] + rng.normal(0, noise_ctd * local_scale, d)
```

This ensures the inverse-variance loss correctly downweights observations at sharp gradients, where the signal-to-noise ratio is inherently lower.


### 4.9 Model Capacity Control

**Why selected.** The default HydroGenerator has base_ch=32, n_blocks=6, yielding ~530K parameters. For a dataset of ~400 samples, this gives ~1300 parameters per sample — a recipe for overfitting. Reducing capacity forces the model to learn compressed representations.

| Config | base_ch | n_blocks | Params | Params/sample (n=400) |
|--------|---------|----------|--------|-----------------------|
| Default | 32 | 6 | 255K | 638 |
| Medium | 32 | 3 | 140K | 350 |
| Small | 16 | 3 | 42K | 105 |
| Tiny | 16 | 2 | 28K | 70 |

**Implementation.** `base_ch` and `n_blocks` are exposed as config parameters and passed through to `HydroGenerator` (alias for `LamaGenerator`):

```python
model = HydroLightningModule(lr=lr, base_ch=base_ch, n_blocks=n_blocks, ratio_g=ratio_g)
```

**Interaction with TSCrossAttention.** The cross-attention MLP bottleneck is `max(ch // 8, 4)` where `ch = base_ch * 2`. For base_ch=16, ch=32 and the bottleneck is 4. This is the minimum useful size — smaller bottlenecks lose too much information. The `max(ch//8, 4)` guard ensures this.

**When to reduce.** The model capacity should be reduced when:
- Dataset has < 200 samples → use base_ch=16, n_blocks=3
- Val loss plateaus while train loss decreases (overfitting signal)
- The val/train loss gap exceeds 2× after 20 epochs


---

## 5. LightningModule Integration

### 5.1 Automatic Optimization

The module uses `automatic_optimization=True` (PL default). This means:

1. PL calls `training_step(batch, batch_idx)` which returns the loss scalar
2. PL calls `loss.backward()` to compute gradients
3. PL calls `optimizer.step()` to update all parameters (generator + log_vars)
4. PL calls `lr_scheduler.step()` to update the learning rate
5. PL calls `optimizer.zero_grad()` to clear gradients

The `log_vars` parameter is included in the computation graph because it is an `nn.Parameter` registered on the `LightningModule`. The optimizer (Adam) updates it alongside generator weights. No special handling is needed.

### 5.2 Multiple Validation Dataloaders

When `trainer.fit(model, train_dl, [val_dl, ood_val_dl])` is called with a list of dataloaders:

1. PL runs `validation_step` for each batch from `val_dl` (dataloader_idx=0)
2. PL runs `validation_step` for each batch from `ood_val_dl` (dataloader_idx=1)
3. PL calls `on_validation_epoch_end` once
4. PL logs metrics with suffix `/dataloader_idx_0` and `/dataloader_idx_1`

The `validation_step` signature must include `dataloader_idx`:

```python
def validation_step(self, batch, batch_idx, dataloader_idx=0):
```

### 5.3 Callbacks

Four callbacks are used:

| Callback | Purpose | Hook |
|----------|---------|------|
| `ModelCheckpoint` | Save best 3 checkpoints by val_loss | `on_validation_epoch_end` |
| `EarlyStopping` | Stop if val_loss doesn't improve for 15 epochs | `on_train_epoch_end` |
| `CurriculumCallback` | Update n_ctd schedule | `on_train_epoch_start` |
| `HydroVisualizationCallback` | Generate T/S comparison plots | `on_validation_epoch_end` |

### 5.4 Checkpoint Resume

The auto-resume logic verifies checkpoint compatibility before loading:

```python
ckpt = torch.load(resume_ckpt, map_location="cpu")
model_state = ckpt["state_dict"]
# Check base_ch matches
enc0_shape = model_state.get("generator.enc.0.weight")
if enc0_shape.shape[0] != base_ch:
    resume_ckpt = None  # incompatible
# Check log_vars exist (new format)
if "log_vars" not in model_state:
    resume_ckpt = None  # pre-uncertainty-weighting checkpoint
```

This prevents crashes when switching between model configurations or when loading old checkpoints from before the generalization features were added.


---

## 6. End-to-End Training Recipe

This section provides a concrete step-by-step recipe for a training run designed to beat OI.

### Step 1: Configure

Choose model capacity based on dataset size (see Section 9). For the default 400-sample synthetic dataset:

```yaml
# hydro_lama_nc/configs/nc/training/hydro_train.yaml
epochs: 40
batch_size: 4
lr: 2e-4
base_ch: 32
n_blocks: 6
ratio_g: 0.5
n_train: 400
n_val: 80
augment: true
```

### Step 2: Train

```bash
source .venv/bin/activate
python notebooks/train_hydro_colab.py
```

### Step 3: Monitor

Open TensorBoard:
```bash
tensorboard --logdir outputs/hydro_YYYY-MM-DD/tb_logs
```

Watch for:
- `train_loss` decreasing steadily (not oscillating wildly)
- `sigma/0` – `sigma/5` stabilizing after ~10 epochs
- `val_loss/dataloader_idx_0` tracking `train_loss` (not diverging)
- `ood_val_loss/dataloader_idx_1` within 40% of `val_loss`
- `lr` showing warmup → cosine decay

### Step 4: Evaluate

Check the visualization outputs:
```bash
ls outputs/hydro_YYYY-MM-DD/viz/
```

Compare `lama_error_ep0035.png` (late epoch) with `lama_error_ep0000.png` (first epoch):
- Error should decrease in thermocline region (sharp gradient area)
- Error should decrease near bottom (BBL loss effect)
- OI baseline error should be larger than LaMa error

### Step 5: Iterate

If the model doesn't beat OI:
1. **Check lazy mode**: Is `lazy=True`? (see logger output: "lazy mode, n=400")
2. **Check augmentation**: Is `augment=true`? (see logger output)
3. **Increase epochs**: Try 60 or 80 epochs
4. **Increase n_train**: More samples per epoch = more diversity per epoch
5. **Check loss balance**: Are sigma values reasonable (0.1–10 range)?
6. **Reduce capacity**: If val_loss diverges from train_loss, try `base_ch=16, n_blocks=2`


---

## 7. Dataset Variations & Training Recipes

### 7.1 Baltic SceneGenerator Modes

The `SceneGenerator` produces scenes in three modes:

| Mode | T range | Thermocline slope | CIL | Lenses | Bathymetry |
|------|---------|-------------------|-----|--------|------------|
| **realistic** | Climatological | Moderate | 0–8°C, 30–80m | 0–2 | Σ-Gaussian, gentle |
| **extended** | ×1.5 ranges | Steeper, wavy | 0–12°C, 20–100m | 1–3 | Σ-Gaussian, varied |
| **stress** | Extreme | Very sharp | -2–15°C, 10–120m | 2–5 | Σ-Gaussian, extreme |

Four seasons modify surface T, CIL presence, and thermocline sharpness.

### 7.2 Default Training (400 samples)

```bash
source .venv/bin/activate
python notebooks/train_hydro_colab.py
```

Uses:
- base_ch=32, n_blocks=6 (255K params)
- n_train=400, n_val=80, lazy=true (infinite unique scenes)
- augment=true, ratio_g=0.5
- 70/20/10 mode mix
- OneCycleLR with lr=2e-4

Expected: val_loss converges in ~30 epochs, ood_val_loss ~20-40% higher.

### 7.3 Small Dataset (50–100 samples)

```bash
python notebooks/train_hydro_colab.py \
  n_train=80 n_val=16 \
  base_ch=16 n_blocks=3 \
  epochs=60 batch_size=2
```

Key changes:
- **Reduced capacity**: 42K params instead of 255K
- **Smaller batch**: batch_size=2 (gradient noise helps generalization)
- **More epochs**: 60 instead of 40 (smaller batches need more passes)

### 7.4 Large Dataset (1000+ samples)

```bash
python notebooks/train_hydro_colab.py \
  n_train=2000 n_val=200 \
  base_ch=32 n_blocks=6 \
  epochs=30 batch_size=8
```

Key changes:
- **Larger batch**: batch_size=8 (faster training, stable gradients)
- **Fewer epochs**: 30 (converges faster with more data)

### 7.5 Stress-Only Training (generalization test)

```bash
python notebooks/train_hydro_colab.py \
  n_train=200 n_val=40 \
  mode_mix='{"realistic": 0.0, "extended": 0.3, "stress": 0.7}' \
  base_ch=16 n_blocks=3
```

This trains exclusively on hard cases. Useful for testing whether the model can learn extreme thermocline structures. The OOD val set should use realistic mode to test backward generalization.

### 7.6 Real NetCDF Data (hybrid training)

When real NetCDF data is available, combine synthetic and real data:

```python
# In training script or custom DataLoader
from hydro_lama_nc.data.dataset import NetCDFDataset
from hydro_lama_nc.data.synthetic_dataset import HydroSyntheticDataset

real_ds = NetCDFDataset(filepaths=["/path/to/cmems.nc"], ...)
synth_ds = HydroSyntheticDataset(n=400, augment=True)

# Concatenate or alternate batches
from torch.utils.data import ConcatDataset
combined = ConcatDataset([synth_ds, real_ds])
```

**Important:** Real data should NOT use the synthetic augmentation pipeline. The `augment` flag applies only to `HydroSyntheticDataset`. Real data has its own noise characteristics and observation patterns.

### 7.7 No Augmentation (debugging)

```bash
python notebooks/train_hydro_colab.py augment=false
```

Disables all augmentation. Useful for:
- Verifying loss convergence on clean data
- Debugging shape mismatches
- Comparing augmented vs non-augmented performance


---

## 8. Monitoring & Diagnostics

### 8.1 TensorBoard Scalars

| Tag | Meaning | Expected behavior |
|-----|---------|-------------------|
| `train_loss` | Total weighted loss | Decreases, may oscillate due to log_var updates |
| `loss/0` – `loss/5` | Individual unweighted loss terms | Each should decrease independently |
| `sigma/0` – `sigma/5` | Learned σ = exp(0.5 × log_var) | Stabilizes after ~10 epochs |
| `val_loss/dataloader_idx_0` | In-distribution val loss | Decreases, tracks train_loss |
| `ood_val_loss/dataloader_idx_1` | OOD val loss | Decreases, may be higher than val_loss |
| `lr` | Current learning rate | Warmup → cosine decay curve |

### 8.2 Diagnosing Common Issues

**sigma_i → 0 for some term.** That term is dominating the loss. Check if its magnitude is much larger than others. If physically expected (e.g., obs_l1 is always the largest), this is fine. If unexpected, check for bugs in the loss function.

**ood_val_loss >> val_loss.** The model is overfitting to the training distribution. Solutions:
- Increase augmentation (more CTD dropout, wider amplitude range)
- Reduce model capacity (base_ch=16, n_blocks=3)
- Reduce ratio_g (e.g. 0.25 for more local processing)
- Add more training samples with stress mode

**train_loss oscillates wildly.** The learning rate may be too high, or log_vars are still adapting. Wait 5–10 epochs for stabilization. If it persists, reduce `lr`.

**val_loss plateaus early.** The model may be underfitting. Solutions:
- Increase model capacity (base_ch=32, n_blocks=6)
- Increase ratio_g (e.g. 0.75 for more global context)
- Check if augmentation is too aggressive (disable and compare)

**LaMa RMSE worse than OI RMSE.** The model is not learning useful features. Solutions:
- Increase epochs (may need more training)
- Check scene diversity (lazy=True should be active — see logger output)
- Check that augmentation is enabled
- Verify loss terms are computing correctly (check individual loss values)
- Increase n_train (more data)

### 8.3 Visualization Outputs

Every `viz_every` epochs, the `HydroVisualizationCallback` generates:
- `input_ep{N}.png`: Input channels (u, T, v, S) with CTD/CMEMS markers
- `oi_T_S_ep{N}.png`: Optimal Interpolation baseline
- `lama_T_S_ep{N}.png`: Model prediction
- `oi_error_ep{N}.png`: OI error heatmap
- `lama_error_ep{N}.png`: Model error heatmap

The viz callback uses the first (in-distribution) val dataloader. Compare `lama_error` heatmaps across epochs to see where the model improves (or doesn't).


---

## 9. Recommendations by Dataset Size

| Dataset size | base_ch | n_blocks | ratio_g | augment | batch_size | epochs | Expected params |
|-------------|---------|----------|---------|---------|------------|--------|-----------------|
| < 50 | 16 | 2 | 0.5 | true | 1 | 80 | ~28K |
| 50–200 | 16 | 3 | 0.5 | true | 2 | 60 | ~42K |
| 200–500 | 32 | 6 | 0.5 | true | 4 | 40 | ~255K |
| 500–2000 | 32 | 6 | 0.5 | true | 8 | 30 | ~255K |
| > 2000 | 64 | 9 | 0.5 | true | 16 | 20 | ~1.0M |

**General principles:**
1. Start with a small model and increase capacity only if underfitting
2. Always use augmentation unless debugging
3. Use more epochs for smaller datasets (more passes over fewer samples)
4. Batch size should be small (1–4) for small datasets — gradient noise is beneficial
5. Monitor the ood_val_loss gap — if it grows over training, reduce capacity or increase regularization
6. Use `lazy=True` (default) for any dataset size — infinite scene diversity at ~1ms/access cost


---

## 10. Configuration Reference

### `hydro_lama_nc/configs/nc/training/hydro_train.yaml`

```yaml
# Training
epochs: 40
batch_size: 4
val_batch_size: 2
lr: 2e-4
seed: 42

# Model
base_ch: 32       # Base channel count: 16 (small), 32 (default), 64 (large)
n_blocks: 6       # Residual blocks: 2 (tiny), 3 (small), 6 (default), 9 (large)
ratio_g: 0.5      # Fourier branch fraction: 0.25 (local), 0.5 (balanced), 0.75 (global)

# Data
n_train: 400
n_val: 80
nx: 64            # Horizontal grid points
nz: 64            # Vertical grid points
augment: true     # Physical augmentation
lazy: true        # On-the-fly scene generation (infinite diversity)
n_ctd: 5          # CTD profiles per sample (also max for curriculum)
n_cmems_x: 8      # CMEMS columns per sample
n_cmems_z: 10     # CMEMS depth levels per sample
noise_ctd: 0.05   # CTD measurement noise sigma
noise_cmems: 0.10 # CMEMS measurement noise sigma
mode_mix:         # Scene mode mixture (dict mapping mode -> probability)
  realistic: 0.7
  extended: 0.2
  stress: 0.1

# Visualization
viz_every: 5

# Output
outdir: ""        # Auto-generated from date
```

### `hydro_lama_nc/configs/nc/training/generator/hydro.yaml`

```yaml
kind: hydro
in_ch: 8          # Fixed: 8 input channels
base_ch: 32
n_blocks: 6
ratio_g: 0.5
```

### CLI overrides

```bash
# All parameters can be overridden via CLI
python notebooks/train_hydro_colab.py \
  epochs=40 batch_size=4 lr=2e-4 \
  base_ch=32 n_blocks=6 ratio_g=0.5 \
  n_train=400 n_val=80 augment=true \
  n_ctd=5 noise_ctd=0.05 noise_cmems=0.10 \
  n_cmems_x=8 n_cmems_z=10 \
  'mode_mix={realistic:0.7,extended:0.2,stress:0.1}' \
  viz_every=5 seed=42
```


---

## 11. References

- Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS 2017*. [arXiv:1703.04977](https://arxiv.org/abs/1703.04977)
- Smith, L. N., & Topin, N. (2019). Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. *Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications*. [arXiv:1708.07120](https://arxiv.org/abs/1708.07120)
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum Learning. *ICML 2009*.
- Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. [arXiv:1207.0580](https://arxiv.org/abs/1207.0580)
- HELCOM Baltic Sea Environment Fact Sheets
- Väli, G., et al. (2013). Baltic CTD climatology
