"""Deterministic forward and finite-difference tests for trainable layers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
)


class LinearTests(unittest.TestCase):
    def test_forward_matches_matrix_multiplication_and_bias(self) -> None:
        inputs = np.array([[1.0, 2.0], [-3.0, 0.5]])
        weight = np.array([[2.0, -1.0, 0.0], [0.5, 3.0, -2.0]])
        bias = np.array([0.25, -0.5, 1.5])
        output, _ = linear_forward(inputs, weight, bias)
        np.testing.assert_allclose(output, inputs @ weight + bias)

    def test_all_backward_outputs_match_finite_differences(self) -> None:
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(2, 2, 3))
        weight = rng.normal(size=(3, 4))
        bias = rng.normal(size=(4,))
        gradient_output = rng.normal(size=(2, 2, 4))
        _, cache = linear_forward(inputs, weight, bias)
        gradient_inputs, gradient_weight, gradient_bias = linear_backward(
            gradient_output, cache
        )
        self.assertIsNotNone(gradient_bias)

        def input_objective(candidate: np.ndarray) -> float:
            output, _ = linear_forward(candidate, weight, bias)
            return float(np.sum(output * gradient_output))

        def weight_objective(candidate: np.ndarray) -> float:
            output, _ = linear_forward(inputs, candidate, bias)
            return float(np.sum(output * gradient_output))

        def bias_objective(candidate: np.ndarray) -> float:
            output, _ = linear_forward(inputs, weight, candidate)
            return float(np.sum(output * gradient_output))

        checks = {
            "inputs": check_gradient(input_objective, inputs, gradient_inputs),
            "weight": check_gradient(weight_objective, weight, gradient_weight),
            "bias": check_gradient(bias_objective, bias, gradient_bias),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))


class EmbeddingTests(unittest.TestCase):
    def test_forward_selects_requested_rows(self) -> None:
        weight = np.arange(15, dtype=np.float64).reshape(5, 3)
        token_ids = np.array([[3, 1], [0, 3]], dtype=np.int64)
        output, _ = embedding_forward(token_ids, weight)
        np.testing.assert_array_equal(output, weight[token_ids])

    def test_backward_accumulates_repeated_token_ids(self) -> None:
        weight = np.zeros((4, 2), dtype=np.float64)
        token_ids = np.array([1, 2, 1, 1], dtype=np.int64)
        _, cache = embedding_forward(token_ids, weight)
        gradient_output = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        )
        gradient_weight = embedding_backward(gradient_output, cache)
        expected = np.array(
            [[0.0, 0.0], [13.0, 16.0], [3.0, 4.0], [0.0, 0.0]]
        )
        np.testing.assert_array_equal(gradient_weight, expected)

    def test_weight_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(29)
        weight = rng.normal(size=(5, 3))
        token_ids = np.array([[1, 3], [1, 4]], dtype=np.int64)
        gradient_output = rng.normal(size=(2, 2, 3))
        _, cache = embedding_forward(token_ids, weight)
        analytical = embedding_backward(gradient_output, cache)

        def objective(candidate: np.ndarray) -> float:
            output, _ = embedding_forward(token_ids, candidate)
            return float(np.sum(output * gradient_output))

        result = check_gradient(objective, weight, analytical)
        self.assertTrue(result.passed, msg=str(result))

    def test_invalid_token_id_is_rejected(self) -> None:
        weight = np.zeros((3, 2), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "token_ids must be in"):
            embedding_forward(np.array([0, 3]), weight)


if __name__ == "__main__":
    unittest.main()

