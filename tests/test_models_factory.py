import torch

from models.factory import build_model


def test_build_model_default():
    model = build_model(in_channels=3, num_classes=2)
    assert model is not None

    x = torch.randn(1, 3, 8, 8)
    y = model(x)
    assert y.shape == (1, 2, 8, 8)


def test_build_model_returns_torch_module():
    model = build_model(in_channels=1, num_classes=1)
    assert hasattr(model, "forward")
    x = torch.randn(1, 1, 4, 4)
    y = model(x)
    assert y.shape == (1, 1, 4, 4)
