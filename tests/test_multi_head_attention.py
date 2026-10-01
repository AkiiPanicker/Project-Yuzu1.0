"""Composition and gradient tests for full multi-head causal attention."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    causal_attention_forward,
    check_gradient,
    multi_head_attention_backward,
    multi_head_attention_forward,
    rope_forward,
)


class MultiHeadAttentionTests(unittest.TestCase):
    def test_one_head_identity_projections_match_primitives(self) -> None:
        rng = np.random.default_rng(83)
        inputs = rng.normal(size=(2, 4, 4))
        identity = np.eye(4)
        output, _ = multi_head_attention_forward(
            inputs,
            identity,
            identity,
            identity,
            identity,
            n_heads=1,
        )
        rotated_query, _ = rope_forward(inputs)
        rotated_key, _ = rope_forward(inputs)
        expected, _ = causal_attention_forward(rotated_query, rotated_key, inputs)
        np.testing.assert_allclose(output, expected)

    def test_multiple_heads_have_expected_output_and_probability_shapes(self) -> None:
        rng = np.random.default_rng(89)
        inputs = rng.normal(size=(2, 5, 8))
        weights = [rng.normal(size=(8, 8)) for _ in range(4)]
        output, cache = multi_head_attention_forward(
            inputs,
            *weights,
            n_heads=2,
        )
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(cache.attention.probabilities.shape, (2, 2, 5, 5))

    def test_future_inputs_do_not_affect_earlier_outputs(self) -> None:
        rng = np.random.default_rng(97)
        inputs = rng.normal(size=(1, 5, 4))
        weights = [rng.normal(scale=0.2, size=(4, 4)) for _ in range(4)]
        original, _ = multi_head_attention_forward(inputs, *weights, n_heads=2)
        changed_inputs = inputs.copy()
        changed_inputs[:, 3:, :] += 1_000.0
        changed, _ = multi_head_attention_forward(
            changed_inputs,
            *weights,
            n_heads=2,
        )
        np.testing.assert_allclose(changed[:, :3, :], original[:, :3, :])

    def test_all_input_and_weight_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(101)
        inputs = rng.normal(scale=0.3, size=(1, 3, 4))
        query_weight = rng.normal(scale=0.2, size=(4, 4))
        key_weight = rng.normal(scale=0.2, size=(4, 4))
        value_weight = rng.normal(scale=0.2, size=(4, 4))
        output_weight = rng.normal(scale=0.2, size=(4, 4))
        gradient_output = rng.normal(size=(1, 3, 4))
        _, cache = multi_head_attention_forward(
            inputs,
            query_weight,
            key_weight,
            value_weight,
            output_weight,
            n_heads=2,
        )
        gradient_inputs, gradient_weights = multi_head_attention_backward(
            gradient_output,
            cache,
        )

        def objective(
            candidate_inputs: np.ndarray,
            candidate_query: np.ndarray,
            candidate_key: np.ndarray,
            candidate_value: np.ndarray,
            candidate_output: np.ndarray,
        ) -> float:
            output, _ = multi_head_attention_forward(
                candidate_inputs,
                candidate_query,
                candidate_key,
                candidate_value,
                candidate_output,
                n_heads=2,
            )
            return float(np.sum(output * gradient_output))

        checks = {
            "inputs": check_gradient(
                lambda candidate: objective(
                    candidate,
                    query_weight,
                    key_weight,
                    value_weight,
                    output_weight,
                ),
                inputs,
                gradient_inputs,
            ),
            "query_weight": check_gradient(
                lambda candidate: objective(
                    inputs,
                    candidate,
                    key_weight,
                    value_weight,
                    output_weight,
                ),
                query_weight,
                gradient_weights.query,
            ),
            "key_weight": check_gradient(
                lambda candidate: objective(
                    inputs,
                    query_weight,
                    candidate,
                    value_weight,
                    output_weight,
                ),
                key_weight,
                gradient_weights.key,
            ),
            "value_weight": check_gradient(
                lambda candidate: objective(
                    inputs,
                    query_weight,
                    key_weight,
                    candidate,
                    output_weight,
                ),
                value_weight,
                gradient_weights.value,
            ),
            "output_weight": check_gradient(
                lambda candidate: objective(
                    inputs,
                    query_weight,
                    key_weight,
                    value_weight,
                    candidate,
                ),
                output_weight,
                gradient_weights.output,
            ),
        }
        for name, result in checks.items():
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_zero_value_projection_produces_zero_output(self) -> None:
        rng = np.random.default_rng(103)
        inputs = rng.normal(size=(2, 3, 4))
        identity = np.eye(4)
        zero = np.zeros((4, 4))
        output, _ = multi_head_attention_forward(
            inputs,
            identity,
            identity,
            zero,
            identity,
            n_heads=2,
        )
        np.testing.assert_allclose(output, 0.0)

    def test_invalid_heads_and_weight_shapes_are_rejected(self) -> None:
        inputs = np.zeros((1, 3, 4), dtype=np.float64)
        identity = np.eye(4)
        with self.subTest(case="not divisible"):
            with self.assertRaisesRegex(ValueError, "divisible"):
                multi_head_attention_forward(
                    inputs, identity, identity, identity, identity, n_heads=3
                )
        with self.subTest(case="odd head width"):
            with self.assertRaisesRegex(ValueError, "even head"):
                multi_head_attention_forward(
                    inputs, identity, identity, identity, identity, n_heads=4
                )
        with self.subTest(case="weight shape"):
            with self.assertRaisesRegex(ValueError, "weight must have shape"):
                multi_head_attention_forward(
                    inputs,
                    identity[:, :3],
                    identity,
                    identity,
                    identity,
                    n_heads=2,
                )


if __name__ == "__main__":
    unittest.main()

