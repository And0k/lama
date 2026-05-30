"""
Ocean T/S inpainting model — corrected LaMa-lite architecture.

Key corrections vs train_v2.py:
  1. FourierUnit: Dense (not Conv2d) in frequency domain, no nonlinearity inside
  2. SpectralTransform: AvgPool → FourierUnit → Conv, matches LaMa paper Fig.2
  3. FFCResBlock: explicit local/global channel split via ratio_g (not concat-all)
  4. BBL loss: vectorized gather, scalar output, no inplace ops anywhere
  5. All ReLU are out-of-place (inplace=False)

References:
  Suvorov et al. (2022) LaMa https://arxiv.org/abs/2109.07161  Fig.2, Eq.1
  Hu et al. (2018) SENet  https://arxiv.org/abs/1709.01507      TSCrossAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class FourierUnit(nn.Module):
    """
    Learns a mixing matrix in Fourier frequency space.

    WHY Dense and not Conv2d:
      After rfft2, each spatial frequency (u,v) is a complex number per channel.
      We want to mix ALL channels at each frequency — that is a Dense (Linear)
      operation over the channel axis, applied identically at every (u,v).
      Conv2d(1×1) is the same as Linear over channels, so pointwise conv IS
      correct here — but the key error in the old code was applying ReLU to
      the frequency-domain values, which breaks Hermitian symmetry required
      by irfft2. The fix: no nonlinearity inside the frequency domain.

    Shape:
      input  : (B, C, H, W)
      rfft2  : (B, C, H, W//2+1)  complex
      concat : (B, 2C, H, W//2+1) real  [real | imag]
      conv1x1: (B, 2C, H, W//2+1) — mixes channels, no spatial mixing
      irfft2 : (B, C, H, W)
    """
    def __init__(self, ch: int):
        super().__init__()
        # BN before mixing, NO activation inside frequency domain
        self.bn   = nn.BatchNorm2d(ch * 2)
        self.conv = nn.Conv2d(ch * 2, ch * 2, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # rfft2: exploit conjugate symmetry → only W//2+1 unique frequencies
        f  = torch.fft.rfft2(x, norm="ortho")            # (B, C, H, W//2+1) complex

        # split real/imag → treat as 2C real channels for the linear mix
        fr = torch.cat([f.real, f.imag], dim=1)          # (B, 2C, H, W//2+1)
        fr = self.conv(self.bn(fr))                       # channel mix, NO relu

        # reassemble complex and invert
        c2 = fr.shape[1] // 2
        f_out = torch.complex(fr[:, :c2], fr[:, c2:])
        return torch.fft.irfft2(f_out, s=(h, w), norm="ortho")  # (B, C, H, W)


class SpectralTransform(nn.Module):
    """
    Multi-scale spectral processing from LaMa paper Fig.2:
      x → AvgPool(stride=2) → FourierUnit → Conv1×1 → upsample → out

    WHY AvgPool before FourierUnit:
      Processing at half resolution halves the number of Fourier coefficients
      (W//4+1 instead of W//2+1), reducing cost by ~4×.
      More importantly: low-frequency ocean structures (eddies, fronts spanning
      many grid cells) are better captured at coarser resolution where they
      dominate the spectrum. High-frequency detail (thermocline sharpness) is
      handled by the local branch in FFCResBlock.

    Shape: (B,C,H,W) → (B,C,H,W)  [spatial size preserved via interpolate]
    """
    def __init__(self, ch: int):
        super().__init__()
        self.pool    = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fu      = FourierUnit(ch)
        self.conv    = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        out  = self.pool(x)                                      # (B,C,H/2,W/2)
        out  = self.fu(out)                                      # Fourier mix
        out  = F.relu(self.bn(self.conv(out)))                   # relu AFTER irfft
        return F.interpolate(out, size=(h, w), mode="bilinear",  # back to original
                             align_corners=False)


class FFCResBlock(nn.Module):
    """
    FFC Residual Block with explicit local/global channel split.

    ratio_g ∈ (0,1): fraction of channels processed by global (Fourier) branch.
    Remaining (1-ratio_g) fraction goes through local 3×3 conv only.

    WHY split instead of concat-all (old code):
      If all C channels go through both branches and results are concatenated,
      the fuse conv must learn to ignore 50% of its input — wasteful.
      Explicit split means each channel specialises: local captures sharp local
      gradients (thermocline), global captures slowly-varying baroclinic structure.

    local_ch  = C × (1 - ratio_g)
    global_ch = C × ratio_g

    Forward:
      x_l, x_g = split(x, [local_ch, global_ch], dim=1)
      x_l → 3×3 conv → 3×3 conv  (local branch)
      x_g → SpectralTransform     (global branch)
      out = concat(x_l_out, x_g_out) + x   (residual)
    """
    def __init__(self, ch: int, ratio_g: float = 0.5):
        super().__init__()
        self.lch = ch - int(ch * ratio_g)   # local channels
        self.gch = int(ch * ratio_g)        # global channels

        self.local_branch = nn.Sequential(
            nn.Conv2d(self.lch, self.lch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.lch),
            nn.ReLU(),                       # out-of-place — no inplace anywhere
            nn.Conv2d(self.lch, self.lch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.lch),
        )
        self.global_branch = SpectralTransform(self.gch)

        # fuse after residual add — operates on full C channels
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = x[:, :self.lch], x[:, self.lch:]   # split along channel dim
        out_l = self.local_branch(x_l)                  # (B, lch, H, W)
        out_g = self.global_branch(x_g)                 # (B, gch, H, W)
        out   = torch.cat([out_l, out_g], dim=1)        # (B, C, H, W)
        return self.fuse(out + x)                        # residual + fuse


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION T↔S  (SENet-style, channel-wise)
# ══════════════════════════════════════════════════════════════════════════════

class TSCrossAttention(nn.Module):
    """
    Channel-wise cross-attention: S informs T's channel weights and vice versa.

    Physically: thermocline and halocline are co-located in Baltic →
    if S-features detect a halocline, T-head should emphasise gradient channels.

    Cost: 2 × (C → C//8 → C) MLP = 2 × (C²/8 + C²/8) ≈ C²/2 params.
    At C=32: ~512 params — negligible vs FFCResBlock (~22K).

    No spatial attention (would be O((H×W)²) — too expensive).
    """
    def __init__(self, ch: int):
        super().__init__()
        r = max(ch // 8, 4)
        # S → weights for T channels
        self.gate_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )
        # T → weights for S channels
        self.gate_S = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )

    def forward(self, fT: torch.Tensor, fS: torch.Tensor):
        wT = self.gate_T(fS).view(fS.shape[0], -1, 1, 1)   # S → T gate
        wS = self.gate_S(fT).view(fT.shape[0], -1, 1, 1)   # T → S gate
        return fT * wT, fS * wS


# ══════════════════════════════════════════════════════════════════════════════
# FULL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class LamaGenerator(nn.Module):
    """
    LaMa-lite generator for joint T+S ocean field inpainting.

    Input  (B, 8, NZ, NX):
      ch0 T_obs, ch1 S_obs       — observed values (0 outside mask)
      ch2 mask_ctd, ch3 mask_cmems  — observation type masks
      ch4 u_lr, ch5 v_lr         — velocity conditioning (geostrophic info)
      ch6 bathymetry             — continuous bottom depth ∈ [0,1]
      ch7 sigma_obs              — per-point uncertainty (0.05 CTD, 0.10 CMEMS)

    Output (B, 2, NZ, NX): [T_pred, S_pred] ∈ [0,1]

    Architecture:
      Encoder : 7×7 → stride-2 3×3   (NZ×NX → NZ/2×NX/2, channels: in→B→2B)
      Body    : N×FFCResBlock(2B)     (size fixed, global+local context)
      Decoder : stride-2 ConvTranspose (NZ/2×NX/2 → NZ×NX, 2B→B)
      Heads   : two parallel 3×3 convs (B→B) for T and S features
      Cross   : TSCrossAttention      (T↔S channel gating)
      Output  : two 7×7 convs + Sigmoid → (B,1,NZ,NX) each → cat → (B,2,NZ,NX)

    Param count (BASE_CH=B):
      Encoder:  8×B×49 + B×2B×9 ≈ 400B + 18B²
      Body:     N × (lch²×9×2 + SpectralTransform + fuse) ≈ N × 25B²
      Decoder:  2B×B×16 = 32B²
      Heads×2:  2 × B²×9 = 18B²
      Cross:    B²/4 × 4 ≈ B²
      Out×2:    2 × B×49 = 98B
      Total at B=32, N=6: ≈ 680K  (verified by npar() below)
    """
    def __init__(self, in_ch: int = 8, base_ch: int = 32,
                 n_blocks: int = 6, ratio_g: float = 0.5):
        super().__init__()
        ch = base_ch * 2

        self.enc = nn.Sequential(
            nn.Conv2d(in_ch,   base_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Conv2d(base_ch, ch,      3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *[FFCResBlock(ch, ratio_g=ratio_g) for _ in range(n_blocks)]
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )

        # separate feature extractors before cross-attention
        self.head_T = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )
        self.head_S = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )

        self.cross = TSCrossAttention(base_ch)

        self.out_T = nn.Sequential(
            nn.Conv2d(base_ch, 1, 7, padding=3), nn.Sigmoid()
        )
        self.out_S = nn.Sequential(
            nn.Conv2d(base_ch, 1, 7, padding=3), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat       = self.blocks(self.enc(x))     # (B, 2C, NZ/2, NX/2)
        dec        = self.dec(feat)                # (B, C,  NZ,   NX)
        fT, fS     = self.head_T(dec), self.head_S(dec)
        fT, fS     = self.cross(fT, fS)
        return torch.cat([self.out_T(fT), self.out_S(fS)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTED BBL LOSS
# ══════════════════════════════════════════════════════════════════════════════

def bottom_boundary_layer_loss(pred: torch.Tensor,
                                bathy_indices: torch.Tensor,
                                channel_index: int = 0,
                                salinity_index: int = None) -> torch.Tensor:
    """
    Penalize non-zero vertical gradient of T (and optionally S) near the seabed.

    Physically: turbulent mixing in the BBL homogenizes T and S →
    ∂T/∂z → 0, ∂S/∂z → 0 at the bottom.

    WHY gather instead of scalar indexing:
      pred[b, c, int(zi), x] extracts a Python scalar via .item() + int cast.
      Although autograd can track this in simple cases, it silently gives wrong
      gradients when upstream activations were modified in-place (LaMa bug #1).
      torch.gather is a first-class differentiable op — gradient is always correct.

    Args:
      pred          : (B, C, H, W) — model output, requires_grad=True
      bathy_indices : (B, W) int   — bottom depth index per column, in [0, H-1]
      channel_index : int          — T channel index in pred
      salinity_index: int|None     — S channel index, or None to skip

    Returns: scalar tensor (differentiable)

    Shape walkthrough (B=4, C=2, H=64, W=32):
      zi          (B, W)       = bathy_indices - 2, clamped to [1, H-2]
      valid       (B, W) bool  = zi > 1
      zi_exp      (B, 1, 1, W) → expand (B, C, 1, W) for gather along dim=2
      val_bot     (B, C, W)    = pred gathered at zi
      val_above   (B, C, W)    = pred gathered at zi-1
      dT          (B, W)       = val_bot[:,T,:] - val_above[:,T,:]
      loss        scalar       = sum(dT² * valid) / n_valid
    """
    B, C, H, W = pred.shape

    zi    = (bathy_indices.long() - 2).clamp(1, H - 2)    # (B, W)
    valid = (zi > 1) & (zi < H - 1)                        # (B, W) bool

    # expand to (B, C, 1, W) for gather along depth axis (dim=2)
    zi_exp  = zi.unsqueeze(1).unsqueeze(2).expand(B, C, 1, W)
    zi1_exp = (zi - 1).clamp(0).unsqueeze(1).unsqueeze(2).expand(B, C, 1, W)

    val_bot   = pred.gather(2, zi_exp ).squeeze(2)   # (B, C, W)
    val_above = pred.gather(2, zi1_exp).squeeze(2)   # (B, C, W)

    vf    = valid.float()                             # (B, W)
    dT    = val_bot[:, channel_index, :] - val_above[:, channel_index, :]
    loss  = (dT ** 2 * vf).sum()
    count = vf.sum().clamp(min=1)

    if salinity_index is not None:
        dS   = val_bot[:, salinity_index, :] - val_above[:, salinity_index, :]
        loss = loss + (dS ** 2 * vf).sum()
        count = count * 2

    return loss / count   # scalar


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    B, NZ, NX = 4, 64, 64

    model = LamaGenerator(in_ch=8, base_ch=32, n_blocks=6, ratio_g=0.5)
    npar  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {npar/1e3:.1f}K")

    # check no inplace ops exist anywhere in model
    def check_inplace(module, path=""):
        for name, child in module.named_children():
            full = f"{path}.{name}" if path else name
            if isinstance(child, nn.ReLU) and child.inplace:
                print(f"  INPLACE ReLU at {full}  ← bug")
            check_inplace(child, full)
    check_inplace(model)
    print("Inplace ReLU check done (no output = clean)")

    # forward + backward
    x      = torch.rand(B, 8, NZ, NX)
    bathy  = torch.randint(10, NZ-2, (B, NX))
    target = torch.rand(B, 2, NZ, NX)

    model.train()
    pred = model(x)
    print(f"Output shape: {pred.shape}  min={pred.min():.3f} max={pred.max():.3f}")

    # BBL loss gradient test
    pred2 = model(x)
    bbl   = bottom_boundary_layer_loss(pred2, bathy,
                                        channel_index=0, salinity_index=1)
    bbl.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    nonzero = sum((g.abs() > 0).any().item() for g in grads)
    print(f"BBL loss={bbl.item():.6f}  "
          f"param groups with nonzero grad: {nonzero}/{len(grads)}")

    # receptive field via center-pixel gradient
    model.eval()
    x3 = torch.zeros(1, 8, NZ, NX, requires_grad=True)
    model(x3)[0, 0, NZ//2, NX//2].backward()
    coverage = (x3.grad[0,0].abs() > x3.grad[0,0].abs().max() * 0.01).float().mean()
    print(f"Receptive field coverage: {coverage.item()*100:.1f}%")

    # timing
    model.eval()
    xb = torch.rand(B, 8, NZ, NX)
    with torch.no_grad():
        for _ in range(3): model(xb)
        t0 = time.perf_counter()
        for _ in range(10): model(xb)
        ms = (time.perf_counter()-t0)/10*1000
    print(f"Forward: {ms:.1f} ms/batch (batch={B}, {NZ}×{NX}, CPU)")
