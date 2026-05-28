# From Smoke Test to Real Training — Next Steps

This document walks through moving from a successful 2-batch smoke test to a
full training run on CMEMS oceanographic data.

## 1. Verify the Smoke Test Passed

A passing smoke test means:
- Config resolves correctly (OmegaConf resolvers, Hydra composition)
- Dataset loads and returns valid samples (27×224 depth×lon slices)
- Generator forward pass works (padding to div-8, then trimming)
- Loss backward works (L1, L2, adversarial, domain losses)
- Validation loop completes (evaluator guards handle empty states)

Look for this in the log:
```
Epoch 0: 100%|██████████| 2/2 [00:02<00:00,  0.73it/s]
```

## 2. Load Pretrained big-lama Weights

The pretrained generator (`input_nc=4, output_nc=3`) is architecture-compatible.
Loading pretrained weights gives the model a strong starting point for texture
and structure — much better than random initialization.

### Script: `scripts/load_pretrained_nc.py`

```python
#!/usr/bin/env python3
"""Load pretrained big-lama generator weights into the NC training model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Register resolvers
OmegaConf.register_new_resolver("slice", lambda *a: slice(*[None if x == "None" else int(x) for x in a]), replace=True)
OmegaConf.register_new_resolver("indices", lambda *a: [int(x) for x in a], replace=True)

PRETRAINED_PATH = "LaMa_models/big-lama/models/best.ckpt"
OUTPUT_PATH = "LaMa_models/big-lama/nc_init_generator.pth"

# Load pretrained checkpoint
ckpt = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
gen_state = {k.replace("generator.", ""): v
             for k, v in ckpt["state_dict"].items()
             if k.startswith("generator.")}

# Load into new model
from saicinpainting.training.trainers import make_training_model
cfg = OmegaConf.load("configs/training/cmems_vertical_l1_l2_domain.yaml")
# Merge with default data cfg...
model = make_training_model(cfg)

# Transfer weights (skip first/last conv if channel count differs)
missing, unexpected = model.generator.load_state_dict(gen_state, strict=False)
print(f"Loaded {len(gen_state)} keys. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing keys (first 5): {missing[:5]}")

torch.save(model.generator.state_dict(), OUTPUT_PATH)
print(f"Saved NC-initialized generator to {OUTPUT_PATH}")
```

### Loading into Training

Override the generator init in the training config or load manually:

```bash
# Option A: Use Hydra to override
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  +pretrained_generator_path=LaMa_models/big-lama/nc_init_generator.pth
```

```python
# Option B: Load in code before training
model.generator.load_state_dict(torch.load("LaMa_models/big-lama/nc_init_generator.pth"))
```

### What Transfers Well
- All FFC-ResNet blocks (most of the network) — learned texture priors
- FourierUnit layers — frequency-domain features
- BatchNorm statistics (partially — ocean data has different distribution)

### What Doesn't Transfer
- First conv layer (`input_nc=4` mask channel may differ in semantics)
- Last conv layer (output channels: RGB → physical fields)
- These are loaded with `strict=False` and retrained from scratch

## 3. Start Real Training

### Recommended First Run: 5 Epochs, Full Dataset

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.max_epochs=5 \
  trainer.kwargs.limit_train_batches=4836 \
  trainer.kwargs.val_check_interval=4836
```

This processes all 4,836 samples per epoch (one full pass through the data).
Set `limit_train_batches` to the dataset size to avoid infinite iteration.

### Full Training (40 epochs)

```bash
.venv/bin/python bin/train.py --config-name=cmems_vertical_l1_l2_domain
```

Default config runs 40 epochs × 25,000 batches. With 4,836 samples and
batch_size=2, one epoch = 2,418 steps. The config's `limit_train_batches=25000`
means it runs ~10 epochs worth per `max_epochs` counter.

**Recommended override for real training:**

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.limit_train_batches=2418 \
  trainer.kwargs.val_check_interval=2418
```

This makes `max_epochs=40` mean exactly 40 passes through the data.

## 4. Monitor Training

### TensorBoard

```bash
tensorboard --logdir /data/tb_logs --port 6006
```

### Key Metrics to Watch

| Metric | Where | What It Tells You |
|--------|-------|-------------------|
| `train_gen_l1` | TB / stdout | L1 reconstruction loss (should decrease steadily) |
| `train_gen_l2` | TB | L2 loss (should decrease; faster = more large errors fixed) |
| `train_gen_adv` | TB | Adversarial loss (should oscillate, not collapse to 0) |
| `train_gen_smooth` | TB | Smoothness penalty (should decrease then stabilize) |
| `train_gen_bounds` | TB | Bounds violation (should approach 0 quickly) |
| `train_gen_grad` | TB | Gradient consistency (should decrease) |
| `train_discr_adv` | TB | Discriminator loss (should stabilize, not diverge) |
| `val_ssim` | TB / checkpoint | Structural similarity on validation (should increase) |
| `val_fid` | TB / checkpoint | FID score (should decrease — lower = better) |
| `val_ssim_fid100_f1_total_mean` | checkpoint | Combined metric used for best checkpoint selection |

### What Healthy Training Looks Like

**First 100 steps:**
- `gen_l1` drops rapidly (model learns basic reconstruction)
- `gen_bounds` drops to near 0 (physical constraints easy to satisfy)
- `gen_smooth` drops (model learns spatial continuity)

**Steps 100–1000:**
- `gen_l1` continues decreasing at a slower rate
- `gen_adv` starts increasing (discriminator learns to detect fakes)
- `discr_adv` stabilizes around 0.5–1.0
- Generator and discriminator reach equilibrium

**Steps 1000+:**
- Losses stabilize with small oscillations
- SSIM increases, FID decreases on validation
- Visual quality improves: sharper reconstructions, fewer artifacts

### Red Flags

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `gen_l1` not decreasing | Learning rate too low, or bug | Check data loading, increase lr |
| `gen_adv` → 0 instantly | Discriminator too weak | Increase discr capacity or lr |
| `discr_adv` → 0 instantly | Discriminator too strong | Decrease discr lr or weight |
| NaN in losses | Numerical instability | Lower lr, check scaling ranges |
| `gen_smooth` dominates | Smoothness weight too high | Decrease `domain.smoothness.weight` |
| Validation SSIM = 0.1 | Model outputs constant | Check data pipeline, augmentation |
| FID = NaN or very large | sqrtm fails on small batches | Increase val batch count |

## 5. Tune Loss Weights

### Strategy: Start Conservative, Add Complexity

**Phase 1: Reconstruction only (epochs 1–5)**
```yaml
losses:
  l1:
    weight_missing: 1.0
    weight_known: 10.0
  l2:
    weight_missing: 0.5
    weight_known: 5.0
  adversarial:
    weight: 0          # Disable GAN initially
  domain:
    smoothness:
      weight: 0.5      # Light regularization
    bounds:
      weight: 0.5
    gradient:
      weight: 0.05
```

This lets the model learn reconstruction without adversarial instability.

**Phase 2: Add adversarial (epochs 5–15)**
```yaml
  adversarial:
    weight: 5           # Start low
    gp_coef: 0.001
```

**Phase 3: Full training (epochs 15+)**
```yaml
  adversarial:
    weight: 10          # Full weight
  domain:
    smoothness:
      weight: 1.0
    bounds:
      weight: 0.5
    gradient:
      weight: 0.1
```

### Domain Loss Tuning Guide

**Smoothness weight too high:** Over-smoothed predictions, loss of fine detail.
Look for: blurry reconstructions, `gen_smooth` >> `gen_l1`.

**Smoothness weight too low:** Noisy or jagged predictions.
Look for: checkerboard artifacts in predicted fields.

**Bounds weight too high:** Model clips to [0,1] aggressively, losing contrast.
Look for: predictions clustered near 0 or 1.

**Bounds weight too low:** Some predictions slightly outside [0,1].
Look for: `gen_bounds` > 0.01 after initial convergence.

**Gradient weight too high:** Over-regularized Laplacian, model avoids sharp gradients.
Look for: loss of mesoscale features (eddies, fronts).

**Gradient weight too low:** Gradient structure ignored, model focuses only on pixel values.
Look for: visual comparison shows structural mismatch in gradients.

### Recommended Starting Weights

```yaml
losses:
  l1:    { weight_missing: 1.0, weight_known: 10.0 }
  l2:    { weight_missing: 0.5, weight_known: 5.0 }
  adversarial: { weight: 10, gp_coef: 0.001 }
  feature_matching: { weight: 100 }
  domain:
    smoothness: { weight: 1.0 }
    bounds:     { weight: 0.5 }
    gradient:   { weight: 0.1 }
```

These are the defaults in `cmems_vertical_l1_l2_domain.yaml`.

## 6. Hyperparameter Adjustments

### Learning Rate

- Default: `1e-4` for both generator and discriminator
- If training diverges: try `5e-5`
- If training stalls: try `2e-4`
- Use separate LRs for gen/discr if one dominates:

```yaml
optimizers:
  generator:
    lr: 1e-4
  discriminator:
    lr: 5e-5     # Lower if discriminator too strong
```

### Batch Size

- Default: `2` (CPU training)
- With GPU: try `4` or `8` for more stable gradients
- Larger batch = smoother loss curves, but needs more memory

### Generator Architecture

Current: FFC-ResNet with 75% global ratio, 9 blocks, 64 base channels.
- `n_blocks: 9` → `n_blocks: 13` for more capacity (slower, better quality)
- `ngf: 64` → `ngf: 128` for wider network (2× params)
- `n_downsampling: 3` requires input divisible by 8 (handled by auto-padding)

### Mask Density

Current: 15 vertical lines with log-distributed model points.
- `n_lines: 15` → fewer = easier task, more = harder
- `y_law: log` → denser near surface (physically realistic)
- `y_law: uniform` → equal distribution across depths

## 7. Input Padding

The FFC-ResNet generator requires spatial dimensions divisible by 8. The
current sample size is 27×224:
- 27 is not divisible by 8 → padded to 32 with reflect mode
- 224 = 28×8 → no padding needed
- Output is trimmed back to 27×224 after the generator

This is handled automatically in `default.py:forward()`. No config changes needed.

## 8. Pre-filtering and NaN Handling

### fill_ratio_threshold

Current: `1.0` (accept all samples).

This is because `|V|` (computed from uo/vo) propagates NaN from the coastal
mask, giving ~95% fill ratio even in the best ocean cells. A threshold < 1.0
would filter out most samples.

NaN values are handled:
- In the dataset: `nan_to_num` replaces NaN with `vmin` during scaling
- In the mask: the vertical mask generator creates masks independent of NaN
- In losses: the mask (1 = hole, 0 = known) ensures losses only apply where
  the model should predict

### If You Change the Data Region

1. Analyze NaN patterns: `python bin/demo_nc.py` with different lat/lon ranges
2. Check fill ratio: the prefilter log shows `Prefiltered X/Y samples`
3. Adjust `fill_ratio_threshold` if too many/few samples are filtered
4. Adjust `lat_indices` and `lon_indices` using `${slice:start,stop}` syntax

## 9. Validation and Checkpointing

### What Gets Saved

- `checkpoints/last.ckpt` — latest checkpoint
- `checkpoints/epoch=X-step=Y.ckpt` — top 5 by `val_ssim_fid100_f1_total_mean`
- `samples/` — visualization images every `visualize_each_iters=500` steps

### Validation Frequency

Default: `val_check_interval=25000` (once per ~10 epochs).
Recommended for real training: `val_check_interval=2418` (once per epoch).

Validation runs through all 4,836 samples at `batch_size=1`, taking ~80 min
on CPU. Reduce with `limit_val_batches=N` for faster feedback:

```bash
trainer.kwargs.limit_val_batches=100  # Only 100 val samples (~2 min)
```

### Checkpoint Selection

Best checkpoint is selected by `val_ssim_fid100_f1_total_mean` (mode=max).
This composite metric balances SSIM (higher=better) and FID (lower=better):
```
score = SSIM + (100 - FID/10) / 100
```

## 10. Troubleshooting

### `RuntimeError: element 0 of tensors does not require grad`

Cause: Generator parameters lost `requires_grad` between training steps
(discriminator step sets `set_requires_grad(generator, False)`).

Fix: Ensure `set_requires_grad(self.generator, True)` is called BEFORE
the forward pass, not after. This is already fixed in `base.py`.

### `torch.cat(): expected a non-empty list of Tensors`

Cause: All validation samples returned dummy data (e.g., due to indexing error),
so evaluator has no states to concatenate.

Fix: Guard `evaluation_end()` calls with `if states:` checks.
Already fixed in `base.py:on_validation_epoch_end()`.

### `val_check_interval must be less than or equal to training batches`

Cause: Config sets `val_check_interval=25000` but dataset has only 2,418
samples per epoch.

Fix: Set `val_check_interval` to dataset size:
```bash
trainer.kwargs.val_check_interval=2418
```

### `invalid indexer array, does not have integer dtype`

Cause: OmegaConf serialized `slice(141, 365)` as a string instead of a
Python slice object.

Fix: Use `${slice:141,365}` resolver syntax in YAML configs.

### FID `sqrtm` Warning

`Matrix is singular` or `invalid value in scalar divide` during FID computation.
This happens with very small validation sets (< 10 samples). Increase
`limit_val_batches` or ignore — FID is unreliable with few samples.

### Data Range Warning

`DirectoryVisualizer target image must be in 0..1 range, but it ranges -0.04..0.88`

Normal. The visualizer warning fires because some samples have values slightly
outside [0,1] after augmentation noise. Not a training issue.

## 11. Scaling to Larger Datasets

### Multiple NetCDF Files

```yaml
dataset:
  filepaths:
    - "/path/to/file_2023-10.nc"
    - "/path/to/file_2023-11.nc"
    - "/path/to/file_2023-12.nc"
```

The dataset concatenates samples across files automatically.

### Larger Region

Change `lat_indices` and `lon_indices` to cover more area:

```yaml
lat_indices: ${slice:0,298}      # All latitudes
lon_indices: ${slice:0,390}      # All longitudes
```

More samples = longer epochs but better geographic coverage.

### GPU Training

```bash
.venv/bin/python bin/train.py \
  --config-name=cmems_vertical_l1_l2_domain \
  trainer.kwargs.accelerator=gpu \
  trainer.kwargs.devices=1 \
  trainer.kwargs.precision=16 \
  data.nc.data.batch_size=4 \
  data.nc.data.dataloader_kwargs.batch_size=4
```

Mixed precision (`precision=16`) gives ~2× speedup on modern GPUs.
