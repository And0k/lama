"""Step 5 — Model: build model and run forward pass."""

import numpy as np
import pytest
import torch

from models.factory import build_model


class TestBuildModel:

    def test_build_default_model(self):
        model = build_model(in_channels=1, num_classes=1)
        assert model is not None

    def test_forward_single_channel(self):
        model = build_model(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 16, 20)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 16, 20)

    def test_forward_multi_channel(self):
        model = build_model(in_channels=3, num_classes=3)
        x = torch.randn(1, 3, 16, 20)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 16, 20)

    def test_output_is_tensor(self):
        model = build_model(in_channels=1, num_classes=1)
        x = torch.randn(2, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_output_dtype(self):
        model = build_model(in_channels=1, num_classes=1)
        x = torch.randn(1, 1, 16, 20)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float32

    def test_batch_dimension_preserved(self):
        model = build_model(in_channels=2, num_classes=2)
        for bs in [1, 2, 4]:
            x = torch.randn(bs, 2, 16, 20)
            with torch.no_grad():
                out = model(x)
            assert out.shape[0] == bs

    def test_model_eval_mode(self):
        model = build_model(in_channels=1, num_classes=1)
        model.eval()
        x = torch.randn(1, 1, 16, 20)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
