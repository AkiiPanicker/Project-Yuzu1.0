"""Forward and finite-difference tests for RMSNorm and LayerNorm."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    layer_norm_backward,
    layer_norm_forward,
    rms_norm_backward,
    rms_norm_forward,
)


class RMSNormTests(unittest.TestCase):
    def test_forward_matches_definition(self) -> None:
        inputs = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]])
        scale = np.array([1.0, 0.5, 2.0])
        epsilon = 1e-5
        output, _ = rms_norm_forward(inputs, scale, epsilon=epsilon)
        expected = inputs / np.sqrt(
            np.mean(inputs * inputs, axis=-1, keepdims=True) + epsilon
        )
        expected *= scale
        np.testing.assert_allclose(output, expected)

    def test_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(37)
        inputs = rng.normal(size=(2, 3, 4))
        scale = rng.normal(size=(4,))
        gradient_output = rng.normal(size=(2, 3, 4))
        _, cache = rms_norm_forward(inputs, scale)
        gradient_inputs, gradient_scale = rms_norm_backward(gradient_output, cache)

        def input_objective(candidate: np.ndarray) -> float:
            output, _ = rms_norm_forward(candidate, scale)
            return float(np.sum(output * gradient_output))

        def scale_objective(candidate: np.ndarray) -> float:
            output, _ = rms_norm_forward(inputs, candidate)
            return float(np.sum(output * gradient_output))

        checks = {
            "inputs": check_gradient(input_objective, inputs, gradient_inputs),
            "scale": check_gradient(scale_objective, scale, gradient_scale),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))


class LayerNormTests(unittest.TestCase):
    def test_forward_matches_definition(self) -> None:
        inputs = np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]])
        scale = np.array([1.0, 0.5, 2.0])
        bias = np.array([-1.0, 0.25, 3.0])
        epsilon = 1e-5
        output, _ = layer_norm_forward(inputs, scale, bias, epsilon=epsilon)
        centered = inputs - np.mean(inputs, axis=-1, keepdims=True)
        expected = centered / np.sqrt(
            np.mean(centered * centered, axis=-1, keepdims=True) + epsilon
        )
        expected = expected * scale + bias
        np.testing.assert_allclose(output, expected)

    def test_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(41)
        inputs = rng.normal(size=(2, 3, 4))
        scale = rng.normal(size=(4,))
        bias = rng.normal(size=(4,))
        gradient_output = rng.normal(size=(2, 3, 4))
        _, cache = layer_norm_forward(inputs, scale, bias)
        gradient_inputs, gradient_scale, gradient_bias = layer_norm_backward(
            gradient_output, cache
        )
        self.assertIsNotNone(gradient_bias)

        def input_objective(candidate: np.ndarray) -> float:
            output, _ = layer_norm_forward(candidate, scale, bias)
            return float(np.sum(output * gradient_output))

        def scale_objective(candidate: np.ndarray) -> float:
            output, _ = layer_norm_forward(inputs, candidate, bias)
            return float(np.sum(output * gradient_output))

        def bias_objective(candidate: np.ndarray) -> float:
            output, _ = layer_norm_forward(inputs, scale, candidate)
            return float(np.sum(output * gradient_output))

        checks = {
            "inputs": check_gradient(input_objective, inputs, gradient_inputs),
            "scale": check_gradient(scale_objective, scale, gradient_scale),
            "bias": check_gradient(bias_objective, bias, gradient_bias),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_constant_input_remains_finite(self) -> None:
        inputs = np.full((2, 3, 4), 7.0)
        scale = np.ones(4)
        bias = np.zeros(4)
        output, cache = layer_norm_forward(inputs, scale, bias)
        gradient_inputs, gradient_scale, gradient_bias = layer_norm_backward(
            np.ones_like(output), cache
        )
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.all(np.isfinite(gradient_inputs)))
        self.assertTrue(np.all(np.isfinite(gradient_scale)))
        self.assertTrue(np.all(np.isfinite(gradient_bias)))
        np.testing.assert_allclose(output, 0.0)

    def test_invalid_scale_and_epsilon_are_rejected(self) -> None:
        inputs = np.ones((2, 3), dtype=np.float64)
        with self.subTest(case="scale"):
            with self.assertRaisesRegex(ValueError, "scale must have shape"):
                rms_norm_forward(inputs, np.ones(2))
        with self.subTest(case="epsilon"):
            with self.assertRaisesRegex(ValueError, "epsilon"):
                layer_norm_forward(inputs, np.ones(3), epsilon=0.0)


if __name__ == "__main__":
    unittest.main()

