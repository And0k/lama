"""TDD tests for corrected LaMa-lite hydro architecture (bin/todo/model.py).

13 tests matching 10 verification procedures from model_checking_results.md
plus 3 hydro-specific integration tests.

These tests should FAIL against current hydro.py (SimpleResBlock, no Fourier)
and PASS after replacing hydro.py with model.py components.
"""

import time

import pytest
import torch
import torch.nn as nn


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def generator():
    torch.manual_seed(42)
    from saicinpainting.training.modules.hydro import HydroGenerator
    return HydroGenerator(in_ch=8, base_ch=32, n_blocks=6, ratio_g=0.5)


@pytest.fixture
def batch():
    torch.manual_seed(123)
    B, NZ, NX = 4, 64, 64
    return {
        "image": torch.rand(B, 8, NZ, NX),
        "target": torch.rand(B, 2, NZ, NX),
        "mask": torch.zeros(B, 1, NZ, NX),
        "bathy_indices": torch.randint(10, NZ - 2, (B, NX)),
    }


# ── Test 1: No inplace ReLU ─────────────────────────────────────────────

class TestNoInplaceReLU:

    def test_no_inplace_relu_in_model(self, generator):
        for name, m in generator.named_modules():
            if isinstance(m, nn.ReLU):
                assert not m.inplace, f"inplace ReLU found at {name}"


# ── Test 2: Param count per component ────────────────────────────────────

class TestParamCount:

    def test_total_params_within_range(self, generator):
        npar = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        assert 200_000 < npar < 300_000, f"Expected ~255K params, got {npar}"

    def test_encoder_params(self, generator):
        npar = sum(p.numel() for p in generator.enc.parameters() if p.requires_grad)
        assert 25_000 < npar < 40_000, f"Expected ~31K encoder params, got {npar}"

    def test_blocks_params(self, generator):
        npar = sum(p.numel() for p in generator.blocks.parameters() if p.requires_grad)
        assert 150_000 < npar < 200_000, f"Expected ~168K block params, got {npar}"

    def test_decoder_params(self, generator):
        npar = sum(p.numel() for p in generator.dec.parameters() if p.requires_grad)
        assert 25_000 < npar < 40_000, f"Expected ~33K decoder params, got {npar}"

    def test_cross_attention_params(self, generator):
        npar = sum(p.numel() for p in generator.cross.parameters() if p.requires_grad)
        assert 400 < npar < 800, f"Expected ~584 cross-attn params, got {npar}"


# ── Test 3: Output shape and Sigmoid range ───────────────────────────────

class TestOutputShape:

    def test_output_shape_b2hw(self, generator, batch):
        generator.train()
        pred = generator(batch["image"])
        B, NZ, NX = batch["image"].shape[0], batch["image"].shape[2], batch["image"].shape[3]
        assert pred.shape == (B, 2, NZ, NX)

    def test_sigmoid_range(self, generator, batch):
        generator.train()
        pred = generator(batch["image"])
        assert (pred > 0).all(), "Sigmoid output has values <= 0"
        assert (pred < 1).all(), "Sigmoid output has values >= 1"


# ── Test 4: Gradient flow (simple loss) ──────────────────────────────────

class TestGradientFlow:

    def test_all_params_receive_grad(self, generator, batch):
        generator.train()
        pred = generator(batch["image"])
        loss = pred.mean()
        loss.backward()
        grads = [(name, p.grad) for name, p in generator.named_parameters()
                 if p.requires_grad]
        zero_grad = [(n, g) for n, g in grads if g is None or g.abs().max() == 0]
        known_dead = {"cross.gate_S.2.weight", "cross.gate_S.2.bias",
                      "cross.gate_S.4.weight"}
        real_dead = [(n, g) for n, g in zero_grad if n not in known_dead]
        assert len(real_dead) == 0, f"Dead gradients (excluding known): {real_dead}"


# ── Test 5: BBL loss gradient flow ───────────────────────────────────────

class TestBBLLoss:

    def test_bbl_is_scalar(self, generator, batch):
        from saicinpainting.training.losses.physical import bottom_boundary_layer_loss
        generator.train()
        pred = generator(batch["image"])
        bbl = bottom_boundary_layer_loss(
            pred, bathy_indices=batch["bathy_indices"],
            channel_index=0, salinity_index=1,
        )
        assert bbl.shape == torch.Size([]), "BBL loss is not scalar"

    def test_bbl_loss_gradients(self, generator, batch):
        from saicinpainting.training.losses.physical import bottom_boundary_layer_loss
        generator.train()
        pred = generator(batch["image"])
        bbl = bottom_boundary_layer_loss(
            pred, bathy_indices=batch["bathy_indices"],
            channel_index=0, salinity_index=1,
        )
        bbl.backward()
        grads = [(n, p.grad) for n, p in generator.named_parameters()
                 if p.requires_grad]
        zero_grad = [(n, g) for n, g in grads if g is None or g.abs().max() == 0]
        known_dead = {"cross.gate_S.2.weight", "cross.gate_S.2.bias",
                      "cross.gate_S.4.weight",
                      "cross.gate_T.2.weight", "cross.gate_T.2.bias",
                      "cross.gate_T.4.weight"}
        real_dead = [(n, g) for n, g in zero_grad if n not in known_dead]
        assert len(real_dead) == 0, f"BBL dead gradients: {real_dead}"


# ── Test 6: FourierUnit output is real ───────────────────────────────────

class TestFourierUnit:

    def test_output_is_real_tensor(self):
        from saicinpainting.training.modules.hydro import FourierUnit
        fu = FourierUnit(8)
        fu.eval()
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            out = fu(x)
        assert not out.is_complex(), "FourierUnit output should be real tensor"

    def test_output_shape_preserved(self):
        from saicinpainting.training.modules.hydro import FourierUnit
        fu = FourierUnit(16)
        fu.eval()
        x = torch.randn(2, 16, 32, 32)
        with torch.no_grad():
            out = fu(x)
        assert out.shape == x.shape


# ── Test 7: SpectralTransform preserves spatial size ─────────────────────

class TestSpectralTransform:

    def test_square_input_preserved(self):
        from saicinpainting.training.modules.hydro import SpectralTransform
        st = SpectralTransform(8)
        st.eval()
        x = torch.randn(2, 8, 32, 32)
        with torch.no_grad():
            out = st(x)
        assert out.shape == x.shape

    def test_non_square_input_preserved(self):
        from saicinpainting.training.modules.hydro import SpectralTransform
        st = SpectralTransform(8)
        st.eval()
        x = torch.randn(2, 8, 32, 17)
        with torch.no_grad():
            out = st(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


# ── Test 8: FFCResBlock channel split ────────────────────────────────────

class TestFFCResBlock:

    def test_channel_split_sum(self):
        from saicinpainting.training.modules.hydro import FFCResBlock
        blk = FFCResBlock(32, ratio_g=0.5)
        assert blk.lch + blk.gch == 32

    def test_shape_preserved(self):
        from saicinpainting.training.modules.hydro import FFCResBlock
        blk = FFCResBlock(32, ratio_g=0.5)
        blk.eval()
        x = torch.randn(2, 32, 16, 16)
        with torch.no_grad():
            out = blk(x)
        assert out.shape == x.shape


# ── Test 9: Receptive field coverage >90% ────────────────────────────────

class TestReceptiveField:

    def test_ffcresblock_has_global_branch(self):
        from saicinpainting.training.modules.hydro import FFCResBlock
        blk = FFCResBlock(32, ratio_g=0.5)
        assert hasattr(blk, 'global_branch')
        assert hasattr(blk, 'local_branch')
        assert blk.gch == 16
        assert blk.lch == 16

    def test_generator_coverage(self, generator):
        mean_cov = self._measure_rf(generator, (1, 8, 64, 64))
        assert mean_cov > 90.0, f"Generator RF coverage {mean_cov:.1f}% < 90%"

    @staticmethod
    def _measure_rf(module, in_shape, n_trials=5, threshold_factor=0.01):
        coverages = []
        for seed in range(n_trials):
            torch.manual_seed(seed)
            x = torch.randn(*in_shape, requires_grad=True)
            module.train()
            out = module(x)
            cx, cy = out.shape[-1] // 2, out.shape[-2] // 2
            out[0, 0, cy, cx].backward()
            g = x.grad[0, 0].abs()
            thr = g.mean() * threshold_factor
            coverages.append((g > thr).float().mean().item() * 100)
            if x.grad is not None:
                x.grad.zero_()
        return sum(coverages) / len(coverages)


# ── Test 10: Forward pass timing <200ms ──────────────────────────────────

class TestTiming:

    def test_forward_pass_under_200ms(self, generator):
        generator.eval()
        x = torch.rand(4, 8, 64, 64)
        with torch.no_grad():
            for _ in range(3):
                generator(x)
            t0 = time.perf_counter()
            for _ in range(10):
                generator(x)
            ms = (time.perf_counter() - t0) / 10 * 1000
        assert ms < 200, f"Forward pass {ms:.1f}ms > 200ms"


# ── Test 11: HydroGenerator accepts 8ch input, outputs 2ch ──────────────

class TestHydroGenerator:

    def test_hydrogenerator_alias(self):
        from saicinpainting.training.modules.hydro import HydroGenerator, LamaGenerator
        assert HydroGenerator is LamaGenerator

    def test_factory_import(self):
        from saicinpainting.training.modules import make_generator
        gen = make_generator(None, "hydro", in_ch=8, base_ch=32, n_blocks=6,
                             ratio_g=0.5)
        assert gen.__class__.__name__ == "LamaGenerator"


# ── Test 12: HydroLightningModule forward pass ──────────────────────────

class TestHydroLightningModule:

    def test_forward_with_batch_dict(self, batch):
        from notebooks.train_hydro_colab import HydroLightningModule
        model = HydroLightningModule(lr=1e-3, base_ch=16, n_blocks=2, ratio_g=0.5)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert "predicted_image" in out
        assert out["predicted_image"].shape == batch["target"].shape


# ── Test 13: BBL loss with bathy_indices from dataset ────────────────────

class TestBBLWithDataset:

    def test_bbl_with_real_indices(self, batch):
        from saicinpainting.training.losses.physical import bottom_boundary_layer_loss
        pred = torch.rand_like(batch["target"])
        bbl = bottom_boundary_layer_loss(
            pred, bathy_indices=batch["bathy_indices"],
            channel_index=0, salinity_index=1,
        )
        assert bbl.ndim == 0
        assert bbl.item() >= 0.0

    def test_bbl_differentiable(self, batch):
        from saicinpainting.training.losses.physical import bottom_boundary_layer_loss
        pred = torch.rand_like(batch["target"], requires_grad=True)
        bbl = bottom_boundary_layer_loss(
            pred, bathy_indices=batch["bathy_indices"],
            channel_index=0, salinity_index=1,
        )
        bbl.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
