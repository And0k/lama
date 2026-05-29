# Next Steps: Improving Beyond Optimal Interpolation

This document outlines what is needed for the LaMa NetCDF pipeline to produce
reconstructions better than the standard Ocean Data Assimilation (ODI) /
Optimal Interpolation (OI) baseline used by CMEMS.

## Current State

The pipeline can train on CMEMS vertical slices and produce inpainted ocean
fields. However, several gaps remain before the model can be validated against
OI products.

## Open Problems and Concerns

### 1. Training Data Quality

**Problem:** The current training data is a single 3-month Baltic inflow
segment (Oct–Dec 2023, 62 time steps). OI products are validated against
independent CTD casts and Argo profiles across seasons and regions.

**What to do:**
- Use at least one full year of CMEMS data to capture seasonal variability
- Train on multiple regions (Baltic, North Sea, Mediterranean) to test
  generalization vs region-specific OI
- Consider multi-file aggregation (`MultiFileNetCDFDataset`) for longer time series

**Concern:** OI products themselves are trained on the same in-situ data
distributed globally. If LaMa sees only the Baltic, it cannot outperform OI
on held-out regions.

### 2. Evaluation Against OI

**Problem:** There is no ground truth for ocean fields — both OI and LaMa
interpolate from sparse observations. The comparison must be done via
cross-validation: hold out a fraction of observations, reconstruct, and
compare against the held-out data.

**What to do:**
- Implement a hold-out evaluation protocol: mask N% of in-situ profiles,
  run LaMa reconstruction, compare RMSE/MAE against held-out values
- Compare against CMEMS OI product interpolated to the same locations
- Report: LaMa RMSE vs OI RMSE per variable, per depth bin, per region

**Concern:** OI uses Gaussian process regression with physically informed
covariance functions. LaMa uses a data-driven generative model. The
comparison is not straightforward — OI provides uncertainty estimates, LaMa
does not.

### 3. Fill Value and NaN Propagation

**Problem:** `|V| = sqrt(uo² + vo²)` propagates NaN from coastal masks,
giving ~95% fill ratio. The `fill_ratio_threshold: 1.0` bypasses this,
but the model still sees mostly NaN-filled data.

**What to do:**
- Test training on `thetao` and `so` only (without `|V|`) to reduce NaN
- Implement `fill_ratio_threshold` with per-variable thresholds
- Consider using a sea-land mask to pre-exclude coastal cells entirely

**Concern:** If the model learns to reconstruct NaN regions as constant
values, it will look good on loss metrics but produce unphysical results
where OI would have used real data.

### 4. Vertical Consistency

**Problem:** Each sample is an independent 2D vertical slice. The model
has no constraint that ensures vertical consistency — neighboring depth
levels may produce contradictory reconstructions.

**What to do:**
- Consider 3D convolutions or axial attention for depth coherence
- Add a vertical consistency loss: penalize large second derivatives in
  the depth dimension
- Test sequential reconstruction: use outputs at one depth as input for
  the next

**Concern:** OI naturally produces vertically consistent fields because it
operates in 3D with physical correlation lengths. LaMa's 2D approach
fundamentally lacks this.

### 5. Physical Conservation Laws

**Problem:** The domain losses (smoothness, bounds, gradient) are soft
constraints. Ocean fields must satisfy conservation of mass, heat, and
salt — these are not enforced.

**What to do:**
- Add a divergence-free constraint for velocity fields: penalize
  `∂u/∂x + ∂v/∂y` in the horizontal plane
- Add a temperature-salinity consistency check (T-S diagrams should be
  physically plausible)
- Consider a post-processing step that enforces conservation laws

**Concern:** OI uses physical covariance functions that implicitly respect
these constraints. A purely pixel-level loss will not enforce them.

### 6. Temporal Coherence

**Problem:** The model processes each time step independently. OI uses
temporal correlation (adjacent time steps are correlated).

**What to do:**
- Use `MultiFileNetCDFDataset` with `sequence_length > 1` to train on
  temporal sequences
- Add a temporal consistency loss: penalize large frame-to-frame changes
  that are not physically justified
- Test autoregressive inference: use previous reconstruction as input for
  the next time step

**Concern:** Ocean temporal variability has different timescales at different
depths (surface: days, deep: months). The model needs to learn these
timescales from data.

### 7. Uncertainty Quantification

**Problem:** OI provides formal uncertainty estimates (analysis error
covariance). LaMa produces point estimates only.

**What to do:**
- Implement MC-Dropout or ensemble-based uncertainty estimation
- Report prediction intervals alongside point estimates
- Compare LaMa uncertainty against OI uncertainty on the same data

**Concern:** Without uncertainty, LaMa reconstructions cannot replace OI
in operational oceanography where forecast confidence matters.

## Proposed Evaluation Protocol

1. **Data split:** Hold out 20% of time steps for testing, 10% for
   validation, train on the rest
2. **Cross-validation:** For each test time step, mask 30% of spatial
   cells, reconstruct, compare against CMEMS OI
3. **Metrics:** RMSE, MAE, SSIM, correlation per variable, per depth bin
4. **Baseline:** CMEMS OI product interpolated to the same grid
5. **Statistical significance:** Paired t-test on per-pixel errors (LaMa
   vs OI)

## Priority Order

1. **Data quality** (full year, multi-region) — highest impact
2. **Evaluation protocol** (hold-out, comparison vs OI) — validates approach
3. **Vertical consistency loss** — addresses fundamental limitation
4. **Temporal coherence** — uses existing `MultiFileNetCDFDataset`
5. **Physical conservation** — long-term, high complexity
6. **Uncertainty quantification** — operational requirement
