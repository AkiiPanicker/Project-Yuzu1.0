"""Structural and finite-difference tests for causal self-attention."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    causal_attention_backward,
    causal_attention_forward,
    check_gradient,
)


class CausalAttentionTests(unittest.TestCase):
    def test_probabilities_are_normalized_and_strictly_causal(self) -> None:
        rng = np.random.default_rng(71)
        query = rng.normal(size=(2, 3, 4, 5))
        key = rng.normal(size=(2, 3, 4, 5))
        value = rng.normal(size=(2, 3, 4, 5))
        _, cache = causal_attention_forward(query, key, value)
        np.testing.assert_allclose(
            np.sum(cache.probabilities, axis=-1),
            1.0,
            atol=1e-12,
        )
        future_positions = np.triu(np.ones((4, 4), dtype=bool), k=1)
        self.assertTrue(np.all(cache.probabilities[..., future_positions] == 0.0))

    def test_zero_queries_and_keys_produce_prefix_means(self) -> None:
        query = np.zeros((1, 4, 2), dtype=np.float64)
        key = np.zeros_like(query)
        value = np.array([[[1.0, 5.0], [3.0, 1.0], [8.0, 0.0], [4.0, 6.0]]])
        output, _ = causal_attention_forward(query, key, value)
        expected = np.array(
            [[[1.0, 5.0], [2.0, 3.0], [4.0, 2.0], [4.0, 3.0]]]
        )
        np.testing.assert_allclose(output, expected)

    def test_future_key_and_value_changes_do_not_affect_earlier_outputs(self) -> None:
        rng = np.random.default_rng(73)
        query = rng.normal(size=(1, 5, 4))
        key = rng.normal(size=(1, 5, 4))
        value = rng.normal(size=(1, 5, 4))
        original, _ = causal_attention_forward(query, key, value)
        changed_key = key.copy()
        changed_value = value.copy()
        changed_key[:, 3:, :] += 1_000.0
        changed_value[:, 3:, :] -= 1_000.0
        changed, _ = causal_attention_forward(query, changed_key, changed_value)
        np.testing.assert_allclose(changed[:, :3, :], original[:, :3, :])

    def test_all_backward_outputs_match_finite_differences(self) -> None:
        rng = np.random.default_rng(79)
        query = rng.normal(size=(1, 3, 2))
        key = rng.normal(size=(1, 3, 2))
        value = rng.normal(size=(1, 3, 2))
        gradient_output = rng.normal(size=(1, 3, 2))
        _, cache = causal_attention_forward(query, key, value)
        gradient_query, gradient_key, gradient_value = causal_attention_backward(
            gradient_output, cache
        )

        def query_objective(candidate: np.ndarray) -> float:
            output, _ = causal_attention_forward(candidate, key, value)
            return float(np.sum(output * gradient_output))

        def key_objective(candidate: np.ndarray) -> float:
            output, _ = causal_attention_forward(query, candidate, value)
            return float(np.sum(output * gradient_output))

        def value_objective(candidate: np.ndarray) -> float:
            output, _ = causal_attention_forward(query, key, candidate)
            return float(np.sum(output * gradient_output))

        checks = {
            "query": check_gradient(query_objective, query, gradient_query),
            "key": check_gradient(key_objective, key, gradient_key),
            "value": check_gradient(value_objective, value, gradient_value),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_single_token_attention_is_value_identity(self) -> None:
        query = np.array([[[2.0, -1.0]]])
        key = np.array([[[0.5, 3.0]]])
        value = np.array([[[4.0, 7.0]]])
        output, cache = causal_attention_forward(query, key, value)
        np.testing.assert_allclose(output, value)
        gradient_output = np.array([[[3.0, -2.0]]])
        gradient_query, gradient_key, gradient_value = causal_attention_backward(
            gradient_output, cache
        )
        np.testing.assert_allclose(gradient_query, 0.0)
        np.testing.assert_allclose(gradient_key, 0.0)
        np.testing.assert_allclose(gradient_value, gradient_output)

    def test_invalid_shapes_and_scale_are_rejected(self) -> None:
        valid = np.zeros((1, 3, 4), dtype=np.float64)
        with self.subTest(case="shape"):
            with self.assertRaisesRegex(ValueError, "shapes must match"):
                causal_attention_forward(valid, valid[:, :2, :], valid)
        with self.subTest(case="scale"):
            with self.assertRaisesRegex(ValueError, "scale"):
                causal_attention_forward(valid, valid, valid, scale=0.0)


if __name__ == "__main__":
    unittest.main()

