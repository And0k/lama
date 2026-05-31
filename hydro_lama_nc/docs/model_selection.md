# Model Selection: FFCResBlock vs SimpleResBlock for Hydro Inpainting

## Decision

Adopt the corrected LaMa-lite architecture (originally prototyped in `bin/todo/model.py`,
now self-contained in `hydro_lama_nc/model.py`) with
FFCResBlock (Fourier feature mixing) replacing SimpleResBlock (local 3×3 only).

## Architecture Comparison

| Component | Old (SimpleResBlock) | New (FFCResBlock) |
|-----------|---------------------|-------------------|
| Body block | Local 3×3 conv only | Local 3×3 + FFT global branch |
| FourierUnit | N/A | BN → Conv1×1 on real+imag, no activation in freq domain |
| SpectralTransform | N/A | AvgPool(2) → FourierUnit → Conv1×1 → ReLU → upsample |
| BBL loss | Python for-loop, scalar indexing | `torch.gather`, vectorized, no loops |
| Channel handling | N/A | Explicit local/global split via `ratio_g` |
| Params (B=32, N=6) | 530.8K | 254.9K |
| RF coverage (full) | ~13×13 px (local only) | ~98% (global via FFT) |

## Why Not Patch ffc.py

The original `ffc.py` has structural bugs:
1. FourierUnit applies ReLU inside frequency domain — breaks Hermitian symmetry
2. SpectralTransform has multi-scale LFU complexity not needed for 64×64
3. FFC_BN_ACT passes `(x_l, x_g)` tuples through entire pipeline — fragile
4. FFCResnetBlock expects tuple inputs from upstream — forces encoder changes

`hydro_lama_nc/model.py` is a clean rewrite: explicit channel split, no tuple passing,
no activation in frequency domain. Self-contained in one file, no `ffc.py` dependency.

## Measured Results

### Param Count (base_ch=32, n_blocks=6, ratio_g=0.5)

```
encoder        :  31,168
blocks (×6)    : 168,576  (28,096 each)
decoder        :  32,832
heads T+S      :  18,560
cross-attention:     584
out T+S        :   3,138
TOTAL          : 254,858
```

### Receptive Field Coverage (n=5 random inputs, threshold=mean×0.01)

```
FourierUnit(16)           :  96.5%  [min=93.6% max=98.9%]
SpectralTransform(16)     :  98.7%  [min=98.4% max=98.8%]
LamaGenerator full        :  98.2%  [min=97.9% max=98.5%]
```

### Forward/Backward Timing (CPU, batch=4, 64×64)

```
LamaGenerator FFC : fwd= 71.5ms  bwd= 260.3ms  params=255K
6×SimpleResBlock  : fwd=207.3ms  bwd= 612.4ms  params=476K
```

FFC is **2.9× faster forward** despite global RF, because 255K < 476K params.

### Gradient Flow

- Simple loss (`pred.mean()`): 114/117 params receive gradient
- BBL loss: 111/117 params receive gradient
- Dead params: `cross.gate_S.2.{weight,bias}`, `cross.gate_S.4.weight`
  (and symmetrically `gate_T` for BBL) — known initialization issue,
  fixable with non-zero gate bias init

## Known Issues

### 3 Dead Gradients in Cross-Attention

`cross.gate_S.2.weight`, `cross.gate_S.2.bias`, `cross.gate_S.4.weight` get
zero gradients at initialization. This is because T→S gating path:
`fS * gate_S(fT)` — if `fT` is constant w.r.t. these params, they get no
gradient. Same issue exists in both old and new architecture. Can be fixed
later by initializing gate_S with non-zero bias.

## Configuration

### `hydro_lama_nc/configs/nc/training/generator/hydro.yaml`

```yaml
kind: hydro
in_ch: 8
base_ch: 32
n_blocks: 6
ratio_g: 0.5
```

### `hydro_lama_nc/configs/nc/training/hydro_train.yaml`

```yaml
base_ch: 32
n_blocks: 6
ratio_g: 0.5
```

### Key Parameter: `ratio_g`

`ratio_g` controls the fraction of channels processed by the global (Fourier)
branch. Default 0.5 means half local, half global. Lower values (e.g. 0.25)
favor local detail; higher values (e.g. 0.75) favor large-scale structures.

## Tests

25 tests in `hydro_lama_nc/tests/test_hydro_model.py` covering:
- No inplace ReLU audit
- Param count per component (within 10% of reference)
- Output shape (B,2,H,W) and Sigmoid range (0,1)
- Gradient flow through simple and BBL losses
- FourierUnit output is real (irfft2 guarantees)
- SpectralTransform spatial preservation (including non-square 32×17)
- FFCResBlock channel split: lch + gch == ch
- Receptive field coverage >90% for full generator
- Forward pass timing <200ms (batch=4, 64×64, CPU)
- HydroGenerator = LamaGenerator alias
- Factory import compatibility
- HydroLightningModule forward pass
- BBL loss with real bathy_indices
