"""Load and compose NC training configs from YAML files.

Uses OmegaConf.load() directly — no Hydra runtime required.
YAML ``@package`` comments are ignored by the YAML parser.
``defaults`` keys (Hydra-only) are stripped after loading.
"""

import logging
import os

from omegaconf import DictConfig, OmegaConf

import netcdf.resolvers  # noqa: F401 — registers ${slice:…}, ${indices:…}, ${env:…}

log = logging.getLogger(__name__)


def _load(relpath: str, configs_dir: str) -> DictConfig:
    """Load a single YAML, strip Hydra-only ``defaults`` key."""
    path = os.path.join(configs_dir, relpath)
    cfg = OmegaConf.load(path)
    cfg.pop("defaults", None)
    return cfg


def load_nc_training_config(
    configs_dir: str,
    scaling_name: str = "cmems_baltic",
) -> DictConfig:
    """Compose the full NC training config from YAML files.

    Mirrors the Hydra ``defaults`` composition of
    ``configs/nc/training/cmems_vertical_l1_l2_domain.yaml`` without
    requiring a Hydra runtime.  Each sub-config is placed at the key
    that ``base.py`` / ``make_training_model()`` expects.

    Args:
        configs_dir: Absolute path to ``configs/`` directory.
        scaling_name: Name of the scaling YAML under ``configs/nc/scaling/``
            (without extension).  Default ``"cmems_baltic"`` loads
            ``configs/nc/scaling/cmems_baltic.yaml`` as the single
            source of truth for normalization ranges.

    Returns:
        Composed OmegaConf DictConfig ready for ``make_training_model()``.
    """
    # 1. Main config: run_title, training_model, losses
    config = _load("nc/training/cmems_vertical_l1_l2_domain.yaml", configs_dir)

    # 2. Sub-configs placed at top-level keys
    config.generator = _load("training/generator/ffc_resnet_075.yaml", configs_dir)
    config.discriminator = _load("training/discriminator/pix2pixhd_nlayer.yaml", configs_dir)

    # NC optimizers YAML has an "optimizers:" wrapper key — unwrap it.
    config.optimizers = _load("nc/training/optimizers/default.yaml", configs_dir).optimizers

    # Data config nested under data.nc.data (matches base.py access pattern)
    data_cfg = _load("nc/data/cmems_vertical.yaml", configs_dir)
    data_cfg.pop("dataloader_kwargs", None)   # references ${nc.data.batch_size} — wrong path outside Hydra

    # Load scaling from the single-source-of-truth YAML and inject into
    # dataset config.  This overrides any inline scaling in the data YAML
    # so that training and inference always use the same ranges.
    scaling_cfg = _load(f"nc/scaling/{scaling_name}.yaml", configs_dir)
    data_cfg.dataset.scaling = scaling_cfg
    log.info("Scaling loaded from nc/scaling/%s.yaml", scaling_name)

    config.data = OmegaConf.create({"nc": {"data": data_cfg}})

    config.visualizer = _load("training/visualizer/directory.yaml", configs_dir)
    config.evaluator = _load("training/evaluator/default_inpainted.yaml", configs_dir)
    config.trainer = _load("training/trainer/any_gpu_large_ssim_ddp_final.yaml", configs_dir)
    config.location = _load("training/location/docker.yaml", configs_dir)

    # 3. Fix circular self-reference in generator YAML:
    #    downsample_conv_kwargs.ratio_gout references itself.
    config.generator.downsample_conv_kwargs.ratio_gout = 0

    log.info(
        "Composed config from %d YAML files under %s",
        9, configs_dir,
    )
    return config


def detect_precision() -> str:
    """Pick the fastest mixed-precision mode for the current hardware.

    Returns:
        ``"bf16-mixed"`` on Ampere+ GPUs (A100, RTX 30xx, L4, H100),
        ``"16-mixed"`` on Volta/Turing GPUs (V100, T4),
        ``"32"`` on CPU or older hardware.
    """
    import torch

    if not torch.cuda.is_available():
        return "32"

    cc = torch.cuda.get_device_capability()
    if cc[0] >= 8:       # Ampere+
        return "bf16-mixed"
    if cc[0] >= 7:       # Volta / Turing
        return "16-mixed"
    return "32"
