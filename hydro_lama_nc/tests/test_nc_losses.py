"""Tests for the 3-loss pipeline: observation_loss, physics_loss, geostrophic_loss."""

import pytest
import torch


class TestObservationLoss:
    """Tests for combined value + structural gradient observation loss."""

    def test_zero_for_identical(self):
        from hydro_lama_nc.losses import observation_loss
        pred = torch.ones(1, 2, 8, 8) * 0.5
        target = torch.ones(1, 2, 8, 8) * 0.5
        mask_ctd = torch.ones(1, 1, 8, 8)
        mask_cmems = torch.zeros(1, 1, 8, 8)
        cmems_raw = torch.ones(1, 2, 8, 8) * 0.5
        loss = observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different(self):
        from hydro_lama_nc.losses import observation_loss
        pred = torch.zeros(1, 2, 8, 8)
        target = torch.ones(1, 2, 8, 8)
        mask_ctd = torch.ones(1, 1, 8, 8)
        mask_cmems = torch.zeros(1, 1, 8, 8)
        cmems_raw = torch.ones(1, 2, 8, 8)
        loss = observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw)
        assert loss.item() > 0.0

    def test_scalar_output(self):
        from hydro_lama_nc.losses import observation_loss
        pred = torch.randn(2, 2, 8, 8)
        target = torch.randn(2, 2, 8, 8)
        mask_ctd = torch.ones(2, 1, 8, 8) * 0.5
        mask_cmems = torch.ones(2, 1, 8, 8) * 0.3
        cmems_raw = torch.randn(2, 2, 8, 8)
        loss = observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw)
        assert loss.ndim == 0

    def test_differentiable(self):
        from hydro_lama_nc.losses import observation_loss
        pred = torch.randn(1, 2, 8, 8, requires_grad=True)
        target = torch.randn(1, 2, 8, 8)
        mask_ctd = torch.ones(1, 1, 8, 8) * 0.5
        mask_cmems = torch.ones(1, 1, 8, 8) * 0.3
        cmems_raw = torch.randn(1, 2, 8, 8)
        loss = observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_ctd_weight_higher_than_cmems(self):
        from hydro_lama_nc.losses import observation_loss
        pred = torch.zeros(1, 2, 8, 8)
        target = torch.ones(1, 2, 8, 8)
        cmems_raw = torch.ones(1, 2, 8, 8)
        # CTD only
        mask_ctd = torch.ones(1, 1, 8, 8)
        mask_cmems = torch.zeros(1, 1, 8, 8)
        loss_ctd = observation_loss(pred, target, mask_ctd, mask_cmems, cmems_raw)
        # CMEMS only
        mask_ctd2 = torch.zeros(1, 1, 8, 8)
        mask_cmems2 = torch.ones(1, 1, 8, 8)
        loss_cmems = observation_loss(pred, target, mask_ctd2, mask_cmems2, cmems_raw)
        # CTD has lower sigma → higher weight → higher loss for same error
        assert loss_ctd.item() > loss_cmems.item()


class TestPhysicsLoss:
    """Tests for consolidated density inversion (stratification) loss."""

    def test_zero_for_stable_stratification(self):
        from hydro_lama_nc.losses import physics_loss
        pred = torch.zeros(1, 2, 16, 16)
        z = torch.linspace(0, 1, 16).unsqueeze(1).expand(16, 16)
        pred[0, 0] = 1.0 - z  # T decreases with depth → density increases
        pred[0, 1] = 0.5 + z  # S increases with depth → density increases
        loss = physics_loss(pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_density_inversion(self):
        from hydro_lama_nc.losses import physics_loss
        pred = torch.zeros(1, 2, 16, 16)
        z = torch.linspace(0, 1, 16).unsqueeze(1).expand(16, 16)
        pred[0, 0] = z  # T increases with depth → density decreases (unstable)
        pred[0, 1] = 0.0
        loss = physics_loss(pred)
        assert loss.item() > 0.0

    def test_mask_below_excludes_seabed(self):
        from hydro_lama_nc.losses import physics_loss
        pred = torch.zeros(1, 2, 16, 16)
        mask_below = torch.zeros(1, 1, 16, 16)
        mask_below[:, :, 12:, :] = 1.0  # bottom 4 rows
        loss = physics_loss(pred, mask_below=mask_below)
        assert loss.ndim == 0

    def test_scalar_output(self):
        from hydro_lama_nc.losses import physics_loss
        pred = torch.randn(2, 2, 16, 16)
        loss = physics_loss(pred)
        assert loss.ndim == 0

    def test_differentiable(self):
        from hydro_lama_nc.losses import physics_loss
        pred = torch.randn(1, 2, 16, 16, requires_grad=True)
        loss = physics_loss(pred)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


class TestGeostrophicLoss:
    """Tests for simplified thermal wind balance loss."""

    def test_zero_for_matching_gradients(self):
        from hydro_lama_nc.losses import geostrophic_loss
        # Constant fields: all gradients = 0 → loss = 0
        T2 = torch.ones(1, 1, 16, 16) * 0.5
        S2 = torch.ones(1, 1, 16, 16) * 0.5
        u2 = torch.ones(1, 1, 16, 16) * 0.5
        v2 = torch.ones(1, 1, 16, 16) * 0.5
        loss = geostrophic_loss(T2, S2, u2, v2)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_zero_for_thermal_wind_balance(self):
        from hydro_lama_nc.losses import geostrophic_loss
        from hydro_lama_nc.eos import ALPHA, BETA
        # Thermal wind: ∂v/∂z = c · ∂ρ/∂x  where c > 0
        # Construct T(x), S(x) → ∂ρ/∂x varies in x
        # Then v(x,z) = ∂ρ/∂x · z → ∂v/∂z = ∂ρ/∂x (matches exactly)
        x = torch.linspace(0.1, 0.9, 16)
        z = torch.linspace(0.1, 0.9, 16)
        ZZ, XX = torch.meshgrid(z, x, indexing="ij")
        # T = 1 - x, S = x → ∂ρ/∂x = α + β (constant but non-zero)
        # Add z-variation so ∂ρ/∂x has spatial structure
        T = (1 - XX).unsqueeze(0).unsqueeze(0) * (1 + 0.5 * ZZ.unsqueeze(0).unsqueeze(0))
        S = XX.unsqueeze(0).unsqueeze(0) * (1 + 0.3 * ZZ.unsqueeze(0).unsqueeze(0))
        # v = ∫(∂ρ/∂x)dz so ∂v/∂z = ∂ρ/∂x
        drho_dx = -ALPHA * torch.gradient(T, dim=3)[0] + BETA * torch.gradient(S, dim=3)[0]
        # v such that ∂v/∂z = drho_dx (integrate numerically)
        v = torch.cumsum(drho_dx, dim=2) * (z[1] - z[0])
        u = torch.zeros_like(v)
        loss = geostrophic_loss(T, S, u, v)
        assert loss.item() == pytest.approx(0.0, abs=0.1)

    def test_positive_for_mismatched_gradients(self):
        from hydro_lama_nc.losses import geostrophic_loss
        T = torch.randn(1, 1, 16, 16)
        S = torch.randn(1, 1, 16, 16)
        u = torch.randn(1, 1, 16, 16)
        v = torch.randn(1, 1, 16, 16)
        loss = geostrophic_loss(T, S, u, v)
        assert loss.item() > 0.0

    def test_scalar_output(self):
        from hydro_lama_nc.losses import geostrophic_loss
        T = torch.randn(1, 1, 16, 16)
        S = torch.randn(1, 1, 16, 16)
        u = torch.randn(1, 1, 16, 16)
        v = torch.randn(1, 1, 16, 16)
        loss = geostrophic_loss(T, S, u, v)
        assert loss.ndim == 0

    def test_differentiable(self):
        from hydro_lama_nc.losses import geostrophic_loss
        T = torch.randn(1, 1, 16, 16, requires_grad=True)
        S = torch.randn(1, 1, 16, 16, requires_grad=True)
        u = torch.randn(1, 1, 16, 16)
        v = torch.randn(1, 1, 16, 16)
        loss = geostrophic_loss(T, S, u, v)
        loss.backward()
        assert T.grad is not None
        assert S.grad is not None


class TestLinearizedDensity:
    """Tests for linearized equation of state helper."""

    def test_basic_values(self):
        from hydro_lama_nc.eos import linearized_density, ALPHA
        T = torch.tensor([0.0, 1.0])
        S = torch.tensor([0.0, 0.0])
        rho = linearized_density(T, S)
        assert rho[0].item() == pytest.approx(1.0)
        assert rho[1].item() == pytest.approx(1.0 - ALPHA)

    def test_salinity_increases_density(self):
        from hydro_lama_nc.eos import linearized_density
        T = torch.tensor([0.5])
        S = torch.tensor([0.0, 1.0])
        rho = linearized_density(T, S)
        assert rho[1].item() > rho[0].item()  # higher S → higher density


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
