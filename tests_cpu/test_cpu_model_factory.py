import torch

from models.factory import build_model


def test_build_model_uses_cpu():
    model = build_model(in_channels=3, num_classes=3)
    inputs = torch.randn(1, 3, 16, 16, device="cpu")
    outputs = model(inputs)

    assert outputs.shape == (1, 3, 16, 16)
    assert outputs.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())
