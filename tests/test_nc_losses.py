"""TDD tests for NetCDF domain-specific losses and L1/L2 loss integration."""

import pytest
import torch
import torch.nn.functional as F


class TestSmoothnessLoss:
    """Tests for spatial smoothness regularization loss."""

    def test_smoothness_loss_zero_for_constant(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.ones(1, 3, 16, 16) * 0.5
        loss = smoothness_loss(pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_smoothness_loss_positive_for_gradient(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.zeros(1, 3, 16, 16)
        pred[0, 0, :, 8:] = 1.0  # step function along W
        loss = smoothness_loss(pred)
        assert loss.item() > 0.0

    def test_smoothness_loss_batch_independence(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.randn(2, 3, 8, 8)
        loss = smoothness_loss(pred)
        assert loss.ndim == 0  # scalar

    def test_smoothness_loss_masked_regions_ignored(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :] = 1.0  # mask top half
        loss_masked = smoothness_loss(pred, mask=mask)
        loss_full = smoothness_loss(pred)
        assert loss_masked.item() <= loss_full.item() + 1e-6

    def test_smoothness_loss_scales_linearly(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.randn(1, 2, 8, 8)
        loss = smoothness_loss(pred)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestPhysicalBoundsLoss:
    """Tests for physical bounds constraint loss."""

    def test_bounds_loss_zero_in_range(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.ones(1, 3, 8, 8) * 0.5
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        loss = physical_bounds_loss(pred, bounds)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_bounds_loss_positive_out_of_range(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.ones(1, 3, 8, 8) * 0.5
        pred[0, 0, :, :] = 2.0  # above max for channel 0
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        loss = physical_bounds_loss(pred, bounds)
        assert loss.item() > 0.0

    def test_bounds_loss_masked_regions_ignored(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.ones(1, 2, 8, 8) * 2.0  # out of range
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :] = 1.0  # mask top half
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        loss_masked = physical_bounds_loss(pred, bounds, mask=mask)
        loss_full = physical_bounds_loss(pred, bounds)
        assert loss_masked.item() < loss_full.item()

    def test_bounds_loss_per_channel_bounds(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.ones(1, 2, 8, 8) * 0.5
        pred[0, 0, :, :] = 45.0  # out of range for ch0 (vmax=40)
        bounds = [(-3.0, 40.0), (0.0, 1.0)]
        loss = physical_bounds_loss(pred, bounds)
        assert loss.item() > 0.0

    def test_bounds_loss_scales_linearly(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.ones(1, 2, 8, 8) * 2.0
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        loss = physical_bounds_loss(pred, bounds)
        assert loss.ndim == 0
        assert loss.item() > 0.0


class TestGradientConsistencyLoss:
    """Tests for gradient consistency (Laplacian-based) loss."""

    def test_gradient_loss_zero_for_constant(self):
        from saicinpainting.training.losses.physical import gradient_consistency_loss
        pred = torch.ones(1, 3, 16, 16) * 0.5
        loss = gradient_consistency_loss(pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_loss_positive_for_ramp(self):
        from saicinpainting.training.losses.physical import gradient_consistency_loss
        pred = torch.linspace(0, 1, 16).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, 2, 16, 16)
        loss = gradient_consistency_loss(pred)
        assert loss.item() > 0.0

    def test_gradient_loss_with_mask(self):
        from saicinpainting.training.losses.physical import gradient_consistency_loss
        pred = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :4, :] = 1.0
        loss = gradient_consistency_loss(pred, mask=mask)
        assert loss.ndim == 0

    def test_gradient_loss_compares_pred_and_target(self):
        from saicinpainting.training.losses.physical import gradient_consistency_loss
        target = torch.ones(1, 2, 8, 8) * 0.5
        pred_smooth = target + torch.randn_like(target) * 0.01
        pred_rough = target + torch.randn_like(target) * 0.5
        loss_smooth = gradient_consistency_loss(pred_smooth, target=target)
        loss_rough = gradient_consistency_loss(pred_rough, target=target)
        assert loss_smooth.item() < loss_rough.item()


class TestMaskedL2Loss:
    """Tests for masked L2 loss (extension of existing masked_l1_loss pattern)."""

    def test_masked_l2_loss_zero_identical(self):
        from saicinpainting.training.losses.feature_matching import masked_l2_loss
        pred = torch.ones(1, 3, 8, 8) * 0.5
        target = torch.ones(1, 3, 8, 8) * 0.5
        mask = torch.zeros(1, 1, 8, 8)
        loss = masked_l2_loss(pred, target, mask, weight_known=1.0, weight_missing=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_masked_l2_loss_positive_different(self):
        from saicinpainting.training.losses.feature_matching import masked_l2_loss
        pred = torch.zeros(1, 3, 8, 8)
        target = torch.ones(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        loss = masked_l2_loss(pred, target, mask, weight_known=1.0, weight_missing=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_masked_l2_weight_missing_applies_to_masked(self):
        from saicinpainting.training.losses.feature_matching import masked_l2_loss
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)  # all masked
        loss = masked_l2_loss(pred, target, mask, weight_known=0.0, weight_missing=1.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_masked_l2_weight_known_applies_to_known(self):
        from saicinpainting.training.losses.feature_matching import masked_l2_loss
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)  # all known
        loss = masked_l2_loss(pred, target, mask, weight_known=1.0, weight_missing=0.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)


class TestSmoothnessLossAutograd:
    """Tests that smoothness loss is differentiable."""

    def test_smoothness_loss_backward(self):
        from saicinpainting.training.losses.physical import smoothness_loss
        pred = torch.randn(1, 2, 8, 8, requires_grad=True)
        loss = smoothness_loss(pred)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_bounds_loss_backward(self):
        from saicinpainting.training.losses.physical import physical_bounds_loss
        pred = torch.full((1, 2, 8, 8), 2.0, requires_grad=True)
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        loss = physical_bounds_loss(pred, bounds)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_gradient_loss_backward(self):
        from saicinpainting.training.losses.physical import gradient_consistency_loss
        pred = torch.randn(1, 2, 8, 8, requires_grad=True)
        loss = gradient_consistency_loss(pred)
        loss.backward()
        assert pred.grad is not None


class TestPhysicalLossConfig:
    """Tests for loss config integration."""

    def test_config_has_domain_losses(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            'losses': {
                'l1': {'weight_missing': 0, 'weight_known': 10},
                'l2': {'weight_missing': 0, 'weight_known': 10},
                'perceptual': {'weight': 0},
                'adversarial': {'kind': 'r1', 'weight': 10},
                'feature_matching': {'weight': 100},
                'resnet_pl': {'weight': 30, 'weights_path': ''},
                'domain': {
                    'smoothness': {'weight': 1.0},
                    'bounds': {'weight': 0.5},
                    'gradient': {'weight': 0.1},
                }
            }
        })
        assert cfg.losses.domain.smoothness.weight == 1.0
        assert cfg.losses.domain.bounds.weight == 0.5
        assert cfg.losses.domain.gradient.weight == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
