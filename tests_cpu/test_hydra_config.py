import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate


def test_instantiate_lama_small_from_yaml():
    cfg = OmegaConf.load("configs/model/lama_small.yaml")
    model = instantiate(cfg)
    assert hasattr(model, "forward")

    x = torch.randn(1, int(cfg.in_channels), 16, 16, device="cpu")
    out = model(x)
    assert out.shape[0] == 1
    assert out.shape[1] == int(cfg.num_classes)


def test_modify_config_and_instantiate():
    cfg = OmegaConf.load("configs/model/lama_small.yaml")
    # change channels and classes
    cfg.in_channels = 1
    cfg.num_classes = 2
    model = instantiate(cfg)

    x = torch.randn(1, 1, 8, 8, device="cpu")
    out = model(x)
    assert out.shape == (1, int(cfg.num_classes), 8, 8)
