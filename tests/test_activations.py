"""Forward and finite-difference tests for SiLU and SwiGLU."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    silu_backward,
    silu_forward,
    stable_sigmoid,
    swiglu_backward,
    swiglu_forward,
)


class SiLUTests(unittest.TestCase):
    def test_forward_matches_definition(self) -> None:
        inputs = np.array([[-2.0, -0.5, 0.0, 1.5, 3.0]])
        output, _ = silu_forward(inputs)
        expected = inputs / (1.0 + np.exp(-inputs))
        np.testing.assert_allclose(output, expected)

    def test_extreme_inputs_remain_finite(self) -> None:
        inputs = np.array([-10_000.0, 0.0, 10_000.0])
        sigmoid = stable_sigmoid(inputs)
        output, _ = silu_forward(inputs)
        self.assertTrue(np.all(np.isfinite(sigmoid)))
        self.assertTrue(np.all(np.isfinite(output)))
        np.testing.assert_allclose(sigmoid, np.array([0.0, 0.5, 1.0]))

    def test_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(43)
        inputs = rng.normal(size=(2, 3, 4))
        gradient_output = rng.normal(size=(2, 3, 4))
        _, cache = silu_forward(inputs)
        analytical = silu_backward(gradient_output, cache)

        def objective(candidate: np.ndarray) -> float:
            output, _ = silu_forward(candidate)
            return float(np.sum(output * gradient_output))

        result = check_gradient(objective, inputs, analytical)
        self.assertTrue(result.passed, msg=str(result))


class SwiGLUTests(unittest.TestCase):
    def test_forward_matches_definition(self) -> None:
        gate = np.array([[-1.0, 0.0, 2.0]])
        value = np.array([[3.0, -4.0, 0.5]])
        output, _ = swiglu_forward(gate, value)
        expected = gate * stable_sigmoid(gate) * value
        np.testing.assert_allclose(output, expected)

    def test_backward_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(47)
        gate = rng.normal(size=(2, 3, 4))
        value = rng.normal(size=(2, 3, 4))
        gradient_output = rng.normal(size=(2, 3, 4))
        _, cache = swiglu_forward(gate, value)
        gradient_gate, gradient_value = swiglu_backward(gradient_output, cache)

        def gate_objective(candidate: np.ndarray) -> float:
            output, _ = swiglu_forward(candidate, value)
            return float(np.sum(output * gradient_output))

        def value_objective(candidate: np.ndarray) -> float:
            output, _ = swiglu_forward(gate, candidate)
            return float(np.sum(output * gradient_output))

        checks = {
            "gate": check_gradient(gate_objective, gate, gradient_gate),
            "value": check_gradient(value_objective, value, gradient_value),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_mismatched_shapes_are_rejected(self) -> None:
        gate = np.zeros((2, 3), dtype=np.float64)
        value = np.zeros((2, 4), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            swiglu_forward(gate, value)


if __name__ == "__main__":
    unittest.main()

