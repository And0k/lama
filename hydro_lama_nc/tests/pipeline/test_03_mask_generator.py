"""Step 3 — Mask Generator: produce binary masks for inpainting."""

import numpy as np
import pytest

from hydro_lama_nc.mask_generator import generate_mask, generate_masks_for_batch


class TestGenerateMask:

    def test_shape(self, mask_params):
        mask = generate_mask((64, 64), mask_params)
        assert mask.shape == (1, 64, 64)

    def test_dtype(self, mask_params):
        mask = generate_mask((64, 64), mask_params)
        assert mask.dtype == np.uint8

    def test_binary_values(self, mask_params):
        mask = generate_mask((64, 64), mask_params)
        unique = set(np.unique(mask))
        assert unique <= {0, 1}

    def test_nonzero_coverage(self, mask_params):
        mask = generate_mask((64, 64), mask_params)
        assert mask.sum() > 0, "mask should have at least some masked pixels"

    def test_different_sizes(self, mask_params):
        for h, w in [(32, 40), (64, 80), (128, 128)]:
            mask = generate_mask((h, w), mask_params)
            assert mask.shape == (1, h, w)


class TestGenerateMasksForBatch:

    def test_batch_shape(self, mask_params):
        masks = generate_masks_for_batch((4, 1, 64, 64), mask_params)
        assert masks.shape == (4, 1, 64, 64)

    def test_batch_binary(self, mask_params):
        masks = generate_masks_for_batch((4, 1, 64, 64), mask_params)
        unique = set(np.unique(masks))
        assert unique <= {0, 1}
