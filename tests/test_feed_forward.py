"""Composition and gradient tests for the complete SwiGLU feed-forward network."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    feed_forward_backward,
    feed_forward_forward,
    linear_backward,
    linear_forward,
    swiglu_backward,
    swiglu_forward,
)


class FeedForwardTests(unittest.TestCase):
    def test_forward_matches_explicit_primitive_composition(self) -> None:
        rng = np.random.default_rng(107)
        inputs = rng.normal(size=(2, 3, 4))
        gate_weight = rng.normal(size=(4, 6))
        value_weight = rng.normal(size=(4, 6))
        output_weight = rng.normal(size=(6, 4))
        output, _ = feed_forward_forward(
            inputs,
            gate_weight,
            value_weight,
            output_weight,
        )
        gate, _ = linear_forward(inputs, gate_weight)
        value, _ = linear_forward(inputs, value_weight)
        hidden, _ = swiglu_forward(gate, value)
        expected, _ = linear_forward(hidden, output_weight)
        np.testing.assert_allclose(output, expected)

    def test_arbitrary_leading_dimensions_are_preserved(self) -> None:
        rng = np.random.default_rng(109)
        inputs = rng.normal(size=(2, 3, 5, 4))
        output, cache = feed_forward_forward(
            inputs,
            rng.normal(size=(4, 7)),
            rng.normal(size=(4, 7)),
            rng.normal(size=(7, 4)),
        )
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(cache.activation.activated_gate.shape, (2, 3, 5, 7))

    def test_zero_value_projection_produces_zero_output(self) -> None:
        rng = np.random.default_rng(113)
        inputs = rng.normal(size=(2, 3, 4))
        output, _ = feed_forward_forward(
            inputs,
            rng.normal(size=(4, 6)),
            np.zeros((4, 6)),
            rng.normal(size=(6, 4)),
        )
        np.testing.assert_allclose(output, 0.0)

    def test_all_input_and_weight_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(127)
        inputs = rng.normal(scale=0.3, size=(1, 2, 3))
        gate_weight = rng.normal(scale=0.2, size=(3, 4))
        value_weight = rng.normal(scale=0.2, size=(3, 4))
        output_weight = rng.normal(scale=0.2, size=(4, 3))
        gradient_output = rng.normal(size=(1, 2, 3))
        _, cache = feed_forward_forward(
            inputs,
            gate_weight,
            value_weight,
            output_weight,
        )
        gradient_inputs, gradient_weights = feed_forward_backward(
            gradient_output,
            cache,
        )

        def objective(
            candidate_inputs: np.ndarray,
            candidate_gate: np.ndarray,
            candidate_value: np.ndarray,
            candidate_output: np.ndarray,
        ) -> float:
            output, _ = feed_forward_forward(
                candidate_inputs,
                candidate_gate,
                candidate_value,
                candidate_output,
            )
            return float(np.sum(output * gradient_output))

        checks = {
            "inputs": check_gradient(
                lambda candidate: objective(
                    candidate, gate_weight, value_weight, output_weight
                ),
                inputs,
                gradient_inputs,
            ),
            "gate_weight": check_gradient(
                lambda candidate: objective(
                    inputs, candidate, value_weight, output_weight
                ),
                gate_weight,
                gradient_weights.gate,
            ),
            "value_weight": check_gradient(
                lambda candidate: objective(
                    inputs, gate_weight, candidate, output_weight
                ),
                value_weight,
                gradient_weights.value,
            ),
            "output_weight": check_gradient(
                lambda candidate: objective(
                    inputs, gate_weight, value_weight, candidate
                ),
                output_weight,
                gradient_weights.output,
            ),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_input_gradient_is_sum_of_gate_and_value_branches(self) -> None:
        rng = np.random.default_rng(131)
        inputs = rng.normal(size=(1, 2, 3))
        gate_weight = rng.normal(size=(3, 4))
        value_weight = rng.normal(size=(3, 4))
        output_weight = rng.normal(size=(4, 3))
        gradient_output = rng.normal(size=(1, 2, 3))
        _, cache = feed_forward_forward(
            inputs,
            gate_weight,
            value_weight,
            output_weight,
        )
        gradient_inputs, _ = feed_forward_backward(gradient_output, cache)
        gradient_hidden, _, _ = linear_backward(
            gradient_output,
            cache.output_linear,
        )
        gradient_gate, gradient_value = swiglu_backward(
            gradient_hidden,
            cache.activation,
        )
        gradient_input_gate, _, _ = linear_backward(
            gradient_gate,
            cache.gate_linear,
        )
        gradient_input_value, _, _ = linear_backward(
            gradient_value,
            cache.value_linear,
        )
        np.testing.assert_allclose(
            gradient_inputs,
            gradient_input_gate + gradient_input_value,
        )

    def test_invalid_weight_shapes_are_rejected(self) -> None:
        inputs = np.zeros((1, 2, 3), dtype=np.float64)
        gate = np.zeros((3, 4))
        value = np.zeros((3, 4))
        output = np.zeros((4, 3))
        with self.subTest(case="gate"):
            with self.assertRaisesRegex(ValueError, "gate weight"):
                feed_forward_forward(inputs, gate[:2], value, output)
        with self.subTest(case="value"):
            with self.assertRaisesRegex(ValueError, "value weight"):
                feed_forward_forward(inputs, gate, value[:, :3], output)
        with self.subTest(case="output"):
            with self.assertRaisesRegex(ValueError, "output weight"):
                feed_forward_forward(inputs, gate, value, output[:, :2])


if __name__ == "__main__":
    unittest.main()

