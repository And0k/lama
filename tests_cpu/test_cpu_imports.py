import torch

from saicinpainting.utils import get_shape


def test_cpu_tensor_shape():
    tensor = torch.randn(2, 3, 8, 8, device="cpu")
    assert tensor.device.type == "cpu"
    assert get_shape(tensor) == (2, 3, 8, 8)


def test_cpu_simple_cpu_operation():
    tensor = torch.arange(6, dtype=torch.float32, device="cpu").reshape(2, 3)
    assert tensor.sum().item() == 15.0
    assert tensor.mean().item() == 2.5
