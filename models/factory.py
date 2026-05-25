from typing import Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig


class _SimpleConvGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def build_model(in_channels: int = 3, num_classes: int = 3, generator_cfg: Optional[DictConfig] = None):
    """Build a model. If `generator_cfg` is provided and contains a Hydra `_target_`, try to
    instantiate it (passing `in_channels` if supported). Otherwise return a small conv net.
    """
    if generator_cfg is not None and isinstance(generator_cfg, DictConfig) and "_target_" in generator_cfg:
        try:
            # try passing in_channels to the target
            return instantiate(generator_cfg, in_channels=in_channels)
        except Exception:
            try:
                return instantiate(generator_cfg)
            except Exception:
                pass

    # fallback simple generator for testing
    model = _SimpleConvGenerator(in_channels=in_channels, out_channels=num_classes)
    # wrap as a simple LightningModule-like object if needed by Trainer
    class LitWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)

    return LitWrapper(model)
