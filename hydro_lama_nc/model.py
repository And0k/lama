"""LaMa-lite generator with Fourier feature mixing for oceanographic inpainting.

Corrected architecture from bin/todo/model.py — self-contained, no ffc.py dependency.

Key design choices:
  1. FourierUnit: Conv1×1 on real+imag parts, NO activation in frequency domain
     (preserves Hermitian symmetry required by irfft2)
  2. SpectralTransform: AvgPool(2) → FourierUnit → Conv1×1 → ReLU → upsample
  3. FFCResBlock: explicit local/global channel split via ratio_g
  4. TSCrossAttention: SENet-style channel gating between T and S heads

Param count (base_ch=32, n_blocks=6, ratio_g=0.5): 254,858

References:
  Suvorov et al. (2022) LaMa https://arxiv.org/abs/2109.07161
  Hu et al. (2018) SENet  https://arxiv.org/abs/1709.01507
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class FourierUnit(nn.Module):
    """Learn a mixing matrix in Fourier frequency space.

    After rfft2 each spatial frequency (u,v) is a complex number per channel.
    We mix channels via Conv1×1 on concatenated real+imag parts.
    NO nonlinearity inside the frequency domain — that would break the
    Hermitian symmetry required by irfft2.

    Shape: (B,C,H,W) → rfft2 → (B,2C,H,W//2+1) real → Conv1×1 → irfft2 → (B,C,H,W)
    """

    def __init__(self, ch: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch * 2)
        self.conv = nn.Conv2d(ch * 2, ch * 2, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = torch.fft.rfft2(x, norm="ortho")
        fr = torch.cat([f.real, f.imag], dim=1)
        fr = self.conv(self.bn(fr))
        c2 = fr.shape[1] // 2
        f_out = torch.complex(fr[:, :c2], fr[:, c2:])
        return torch.fft.irfft2(f_out, s=(h, w), norm="ortho")


class SpectralTransform(nn.Module):
    """Multi-scale spectral processing: AvgPool → FourierUnit → Conv → upsample.

    Processing at half resolution halves Fourier coefficients (cost ~4×).
    Low-frequency ocean structures are captured at coarser resolution;
    high-frequency detail is handled by the local branch in FFCResBlock.

    Shape: (B,C,H,W) → (B,C,H,W)  [spatial size preserved]
    """

    def __init__(self, ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fu = FourierUnit(ch)
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        out = self.pool(x)
        out = self.fu(out)
        out = F.relu(self.bn(self.conv(out)))
        return F.interpolate(out, size=(h, w), mode="bilinear",
                             align_corners=False)


class FFCResBlock(nn.Module):
    """FFC Residual Block with explicit local/global channel split.

    ratio_g: fraction of channels processed by global (Fourier) branch.
    Remaining fraction goes through local 3×3 conv only.

    Forward:
      x_l, x_g = split(x, [local_ch, global_ch], dim=1)
      x_l → 3×3 conv → 3×3 conv  (local branch)
      x_g → SpectralTransform     (global branch)
      out = concat(x_l_out, x_g_out) + x   (residual + fuse)
    """

    def __init__(self, ch: int, ratio_g: float = 0.5):
        super().__init__()
        self.lch = ch - int(ch * ratio_g)
        self.gch = int(ch * ratio_g)

        self.local_branch = nn.Sequential(
            nn.Conv2d(self.lch, self.lch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.lch),
            nn.ReLU(),
            nn.Conv2d(self.lch, self.lch, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.lch),
        )
        self.global_branch = SpectralTransform(self.gch)

        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l, x_g = x[:, :self.lch], x[:, self.lch:]
        out_l = self.local_branch(x_l)
        out_g = self.global_branch(x_g)
        out = torch.cat([out_l, out_g], dim=1)
        return self.fuse(out + x)


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION T↔S  (SENet-style, channel-wise)
# ══════════════════════════════════════════════════════════════════════════════

class TSCrossAttention(nn.Module):
    """Channel-wise cross-attention: S informs T's channel weights and vice versa.

    Physically: thermocline and halocline are co-located in Baltic —
    if S-features detect a halocline, T-head should emphasise gradient channels.
    """

    def __init__(self, ch: int):
        super().__init__()
        r = max(ch // 8, 4)
        self.gate_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )
        self.gate_S = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )

    def forward(self, fT: torch.Tensor, fS: torch.Tensor):
        wT = self.gate_T(fS).view(fS.shape[0], -1, 1, 1)
        wS = self.gate_S(fT).view(fT.shape[0], -1, 1, 1)
        return fT * wT, fS * wS


# ══════════════════════════════════════════════════════════════════════════════
# FULL GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class LamaGenerator(nn.Module):
    """LaMa-lite generator for joint T+S ocean field inpainting.

    Input  (B, in_ch, NZ, NX): 7 channels [T_obs, S_obs, mask_ctd, mask_cmems,
                                          u_lr, v_lr, bathymetry]
    Output (B, 2, NZ, NX): [T_pred, S_pred] ∈ [0,1]

    Architecture:
      Encoder : 7×7 → stride-2 3×3   (NZ×NX → NZ/2×NX/2)
      Body    : N×FFCResBlock(2B)     (global+local context)
      Decoder : stride-2 ConvTranspose (NZ/2×NX/2 → NZ×NX)
      Heads   : two parallel 3×3 convs for T and S features
      Cross   : TSCrossAttention      (T↔S channel gating)
      Output  : two 7×7 convs + Sigmoid → (B,2,NZ,NX)
    """

    def __init__(self, in_ch: int = 7, base_ch: int = 32,
                 n_blocks: int = 6, ratio_g: float = 0.5):
        super().__init__()
        ch = base_ch * 2

        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.Conv2d(base_ch, ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *[FFCResBlock(ch, ratio_g=ratio_g) for _ in range(n_blocks)]
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
        )

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
        feat = self.blocks(self.enc(x))
        dec = self.dec(feat)
        fT, fS = self.head_T(dec), self.head_S(dec)
        fT, fS = self.cross(fT, fS)
        return torch.cat([self.out_T(fT), self.out_S(fS)], dim=1)


HydroGenerator = LamaGenerator
