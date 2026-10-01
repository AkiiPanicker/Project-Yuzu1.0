"""Tests and numerical checks for probability and loss primitives."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    cross_entropy_with_logits,
    logsumexp,
    softmax,
)


class StableNumericsTests(unittest.TestCase):
    def test_softmax_is_finite_and_normalized_for_large_logits(self) -> None:
        logits = np.array([[10_000.0, 10_001.0, 9_999.0]], dtype=np.float64)
        probabilities = softmax(logits)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-12)

    def test_softmax_is_invariant_to_constant_shift(self) -> None:
        logits = np.array([[0.2, -0.7, 1.1], [3.0, 2.0, -4.0]])
        np.testing.assert_allclose(softmax(logits), softmax(logits + 50_000.0))

    def test_logsumexp_matches_direct_calculation_at_safe_scale(self) -> None:
        values = np.array([[0.5, -0.2, 1.7], [-2.0, 0.0, 2.0]])
        expected = np.log(np.sum(np.exp(values), axis=-1))
        np.testing.assert_allclose(logsumexp(values), expected, rtol=1e-12)

    def test_uniform_cross_entropy_equals_log_vocabulary_size(self) -> None:
        logits = np.zeros((4, 7), dtype=np.float64)
        targets = np.array([0, 2, 4, 6], dtype=np.int64)
        loss, gradient = cross_entropy_with_logits(logits, targets)
        self.assertAlmostEqual(float(loss), float(np.log(7.0)), places=12)
        self.assertEqual(gradient.shape, logits.shape)
        np.testing.assert_allclose(gradient.sum(axis=-1), 0.0, atol=1e-12)

    def test_cross_entropy_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(20261001)
        logits = rng.normal(size=(2, 3, 5)).astype(np.float64)
        targets = np.array([[0, 3, 1], [4, 2, 0]], dtype=np.int64)
        _, analytical = cross_entropy_with_logits(logits, targets, reduction="mean")

        def loss_function(candidate: np.ndarray) -> float:
            loss, _ = cross_entropy_with_logits(candidate, targets, reduction="mean")
            return float(loss)

        result = check_gradient(loss_function, logits, analytical)
        self.assertTrue(result.passed, msg=str(result))
        self.assertLess(result.max_absolute_error, 1e-9)

    def test_invalid_target_is_rejected(self) -> None:
        logits = np.zeros((2, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "targets must be in"):
            cross_entropy_with_logits(logits, np.array([0, 3]))

    def test_nonfinite_logits_are_rejected(self) -> None:
        logits = np.array([[0.0, np.inf]])
        with self.assertRaisesRegex(ValueError, "finite"):
            softmax(logits)


if __name__ == "__main__":
    unittest.main()

