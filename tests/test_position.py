"""Forward, structural, and gradient tests for rotary position embeddings."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import check_gradient, rope_backward, rope_forward  # noqa: E402


class RoPETests(unittest.TestCase):
    def test_position_zero_is_identity(self) -> None:
        rng = np.random.default_rng(53)
        inputs = rng.normal(size=(2, 4, 6))
        output, _ = rope_forward(inputs)
        np.testing.assert_allclose(output[:, 0, :], inputs[:, 0, :])

    def test_forward_matches_explicit_pair_rotation(self) -> None:
        inputs = np.arange(12, dtype=np.float64).reshape(1, 3, 4)
        base = 10.0
        output, _ = rope_forward(inputs, base=base)
        pair_dimensions = np.arange(0, 4, 2, dtype=np.float64)
        frequencies = np.power(base, -pair_dimensions / 4)
        angles = np.arange(3, dtype=np.float64)[:, None] * frequencies[None, :]
        cosine = np.cos(angles)
        sine = np.sin(angles)
        expected = np.empty_like(inputs)
        expected[..., 0::2] = (
            inputs[..., 0::2] * cosine - inputs[..., 1::2] * sine
        )
        expected[..., 1::2] = (
            inputs[..., 0::2] * sine + inputs[..., 1::2] * cosine
        )
        np.testing.assert_allclose(output, expected)

    def test_rotation_preserves_each_pair_norm(self) -> None:
        rng = np.random.default_rng(59)
        inputs = rng.normal(size=(2, 3, 5, 8))
        output, _ = rope_forward(inputs)
        input_pair_norm = inputs[..., 0::2] ** 2 + inputs[..., 1::2] ** 2
        output_pair_norm = output[..., 0::2] ** 2 + output[..., 1::2] ** 2
        np.testing.assert_allclose(output_pair_norm, input_pair_norm, atol=1e-12)

    def test_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(61)
        inputs = rng.normal(size=(2, 3, 4))
        gradient_output = rng.normal(size=(2, 3, 4))
        _, cache = rope_forward(inputs, base=100.0, position_offset=2)
        analytical = rope_backward(gradient_output, cache)

        def objective(candidate: np.ndarray) -> float:
            output, _ = rope_forward(candidate, base=100.0, position_offset=2)
            return float(np.sum(output * gradient_output))

        result = check_gradient(objective, inputs, analytical)
        self.assertTrue(result.passed, msg=str(result))

    def test_offset_matches_a_slice_of_full_sequence(self) -> None:
        rng = np.random.default_rng(67)
        inputs = rng.normal(size=(2, 6, 8))
        full_output, _ = rope_forward(inputs)
        sliced_output, _ = rope_forward(inputs[:, 3:, :], position_offset=3)
        np.testing.assert_allclose(sliced_output, full_output[:, 3:, :])

    def test_invalid_width_base_and_offset_are_rejected(self) -> None:
        with self.subTest(case="odd width"):
            with self.assertRaisesRegex(ValueError, "must be even"):
                rope_forward(np.zeros((2, 3, 5)))
        with self.subTest(case="base"):
            with self.assertRaisesRegex(ValueError, "base"):
                rope_forward(np.zeros((2, 3, 4)), base=0.0)
        with self.subTest(case="offset"):
            with self.assertRaisesRegex(ValueError, "position_offset"):
                rope_forward(np.zeros((2, 3, 4)), position_offset=-1)


if __name__ == "__main__":
    unittest.main()

