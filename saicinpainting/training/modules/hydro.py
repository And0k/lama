"""Hydro-generator with T↔S cross-attention for oceanographic inpainting.

Implements the architecture from bin/lama_hydro_simple_end2end/hydro_attention.py:
    Encoder → N × ResBlock → [Head_T | Head_S] → cross-attention → output

The T↔S cross-attention modulates feature maps channel-wise so the network
learns thermohaline covariance (thermocline depth ~ halocline depth).

Uses SimpleResBlock (no in-place activations) instead of FFCResnetBlock
to avoid autograd issues. FFC Fourier features are disabled in the original
config (ratio_gin=0, ratio_gout=0) so this is functionally equivalent.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TSCrossAttention(nn.Module):
    """Channel-wise cross-attention between T and S feature maps.

    T attends to S and vice versa — mutual modulation of channel
    activations before the final prediction heads. This lets the network
    learn thermohaline covariance without expensive spatial attention.

    Cost: 2 × (ch → ch//8 → ch) MLP — negligible vs conv layers.
    """

    def __init__(self, ch):
        super().__init__()
        r = max(ch // 8, 4)
        self.attn_T = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )
        self.attn_S = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, r), nn.ReLU(),
            nn.Linear(r, ch), nn.Sigmoid(),
        )

    def forward(self, fT, fS):
        wT = self.attn_T(fS).view(fS.shape[0], -1, 1, 1)
        wS = self.attn_S(fT).view(fT.shape[0], -1, 1, 1)
        return fT * wT, fS * wS


class SimpleResBlock(nn.Module):
    """Residual block: BN → ReLU → Conv → BN → ReLU → Conv + skip.

    No in-place activations — safe for autograd with physical losses
    that index into the feature tensor (e.g. BBL loss).
    """

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class HydroGenerator(nn.Module):
    """LaMa-lite generator with dual T/S output heads + cross-attention.

    Architecture:
        Encoder: 7×7 conv → stride-2 conv (NZ×NX → NZ/2 × NX/2)
        Body:    N_BLOCKS × SimpleResBlock (no in-place activations)
        Decoder: ConvTranspose2d (back to NZ×NX)
        Heads:   separate 3×3 conv for T and S features
        Cross:   TSCrossAttention (mutual channel modulation)
        Output:  separate 7×7 conv → Sigmoid for T and S

    Args:
        input_nc:  Number of input channels (e.g. 8 for hydro).
        output_nc: Number of output channels (e.g. 2 for T+S).
        base_ch:   Base channel count (default 32).
        n_blocks:  Number of resblock layers (default 6).
    """

    def __init__(self, input_nc=8, output_nc=2, base_ch=32, n_blocks=6):
        super().__init__()
        ch = base_ch * 2

        self.enc = nn.Sequential(
            nn.Conv2d(input_nc, base_ch, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(base_ch, ch, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[SimpleResBlock(ch) for _ in range(n_blocks)]
        )

        self.dec = nn.ConvTranspose2d(ch, base_ch, 4, stride=2, padding=1)

        self.head_T = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )
        self.head_S = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )

        self.cross = TSCrossAttention(base_ch)

        n_out = output_nc // 2 if output_nc >= 2 else 1
        self.out_T = nn.Sequential(
            nn.Conv2d(base_ch, n_out, 7, padding=3),
            nn.Sigmoid(),
        )
        self.out_S = nn.Sequential(
            nn.Conv2d(base_ch, n_out, 7, padding=3),
            nn.Sigmoid(),
        ) if output_nc >= 2 else None

    def forward(self, x):
        feat = self.enc(x)
        feat = self.blocks(feat)
        dec = self.dec(feat)

        fT = self.head_T(dec)
        fS = self.head_S(dec)
        fT, fS = self.cross(fT, fS)

        out_T = self.out_T(fT)
        if self.out_S is not None:
            out_S = self.out_S(fS)
            return torch.cat([out_T, out_S], dim=1)
        return out_T
