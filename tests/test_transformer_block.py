"""Composition and gradient tests for one pre-norm transformer block."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    check_gradient,
    feed_forward_forward,
    multi_head_attention_forward,
    rms_norm_forward,
    transformer_block_backward,
    transformer_block_forward,
)


class TransformerBlockTests(unittest.TestCase):
    def test_forward_matches_explicit_primitive_composition(self) -> None:
        rng = np.random.default_rng(137)
        inputs = rng.normal(scale=0.3, size=(2, 3, 4))
        attention_norm_scale = rng.normal(size=4)
        query_weight = rng.normal(scale=0.2, size=(4, 4))
        key_weight = rng.normal(scale=0.2, size=(4, 4))
        attention_value_weight = rng.normal(scale=0.2, size=(4, 4))
        attention_output_weight = rng.normal(scale=0.2, size=(4, 4))
        feed_forward_norm_scale = rng.normal(size=4)
        gate_weight = rng.normal(scale=0.2, size=(4, 6))
        feed_forward_value_weight = rng.normal(scale=0.2, size=(4, 6))
        feed_forward_output_weight = rng.normal(scale=0.2, size=(6, 4))

        output, _ = transformer_block_forward(
            inputs,
            attention_norm_scale,
            query_weight,
            key_weight,
            attention_value_weight,
            attention_output_weight,
            feed_forward_norm_scale,
            gate_weight,
            feed_forward_value_weight,
            feed_forward_output_weight,
            n_heads=2,
        )

        normalized_attention, _ = rms_norm_forward(
            inputs,
            attention_norm_scale,
        )
        attention_output, _ = multi_head_attention_forward(
            normalized_attention,
            query_weight,
            key_weight,
            attention_value_weight,
            attention_output_weight,
            n_heads=2,
        )
        post_attention = inputs + attention_output
        normalized_feed_forward, _ = rms_norm_forward(
            post_attention,
            feed_forward_norm_scale,
        )
        feed_forward_output, _ = feed_forward_forward(
            normalized_feed_forward,
            gate_weight,
            feed_forward_value_weight,
            feed_forward_output_weight,
        )
        expected = post_attention + feed_forward_output
        np.testing.assert_allclose(output, expected)

    def test_arbitrary_leading_dimensions_are_preserved(self) -> None:
        rng = np.random.default_rng(139)
        inputs = rng.normal(size=(2, 3, 5, 4))
        attention_weights = tuple(
            rng.normal(scale=0.2, size=(4, 4)) for _ in range(4)
        )
        output, cache = transformer_block_forward(
            inputs,
            np.ones(4),
            *attention_weights,
            np.ones(4),
            rng.normal(scale=0.2, size=(4, 6)),
            rng.normal(scale=0.2, size=(4, 6)),
            rng.normal(scale=0.2, size=(6, 4)),
            n_heads=2,
        )
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(
            cache.attention.attention.probabilities.shape,
            (2, 3, 2, 5, 5),
        )

    def test_future_inputs_do_not_affect_earlier_outputs(self) -> None:
        rng = np.random.default_rng(149)
        inputs = rng.normal(scale=0.3, size=(1, 5, 4))
        attention_weights = tuple(
            rng.normal(scale=0.2, size=(4, 4)) for _ in range(4)
        )
        parameters = (
            np.ones(4),
            *attention_weights,
            np.ones(4),
            rng.normal(scale=0.2, size=(4, 6)),
            rng.normal(scale=0.2, size=(4, 6)),
            rng.normal(scale=0.2, size=(6, 4)),
        )
        original, _ = transformer_block_forward(
            inputs,
            *parameters,
            n_heads=2,
        )
        changed_inputs = inputs.copy()
        changed_inputs[:, 3:, :] += 1_000.0
        changed, _ = transformer_block_forward(
            changed_inputs,
            *parameters,
            n_heads=2,
        )
        np.testing.assert_allclose(changed[:, :3, :], original[:, :3, :])

    def test_all_input_and_weight_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(151)
        inputs = rng.normal(scale=0.3, size=(1, 2, 4))
        attention_norm_scale = rng.normal(scale=0.2, size=4)
        query_weight = rng.normal(scale=0.2, size=(4, 4))
        key_weight = rng.normal(scale=0.2, size=(4, 4))
        attention_value_weight = rng.normal(scale=0.2, size=(4, 4))
        attention_output_weight = rng.normal(scale=0.2, size=(4, 4))
        feed_forward_norm_scale = rng.normal(scale=0.2, size=4)
        gate_weight = rng.normal(scale=0.2, size=(4, 5))
        feed_forward_value_weight = rng.normal(scale=0.2, size=(4, 5))
        feed_forward_output_weight = rng.normal(scale=0.2, size=(5, 4))
        gradient_output = rng.normal(size=(1, 2, 4))

        _, cache = transformer_block_forward(
            inputs,
            attention_norm_scale,
            query_weight,
            key_weight,
            attention_value_weight,
            attention_output_weight,
            feed_forward_norm_scale,
            gate_weight,
            feed_forward_value_weight,
            feed_forward_output_weight,
            n_heads=2,
        )
        gradient_inputs, gradients = transformer_block_backward(
            gradient_output,
            cache,
        )

        def objective(
            candidate_inputs: np.ndarray,
            candidate_attention_norm: np.ndarray,
            candidate_query: np.ndarray,
            candidate_key: np.ndarray,
            candidate_attention_value: np.ndarray,
            candidate_attention_output: np.ndarray,
            candidate_feed_forward_norm: np.ndarray,
            candidate_gate: np.ndarray,
            candidate_feed_forward_value: np.ndarray,
            candidate_feed_forward_output: np.ndarray,
        ) -> float:
            output, _ = transformer_block_forward(
                candidate_inputs,
                candidate_attention_norm,
                candidate_query,
                candidate_key,
                candidate_attention_value,
                candidate_attention_output,
                candidate_feed_forward_norm,
                candidate_gate,
                candidate_feed_forward_value,
                candidate_feed_forward_output,
                n_heads=2,
            )
            return float(np.sum(output * gradient_output))

        parameters = (
            inputs,
            attention_norm_scale,
            query_weight,
            key_weight,
            attention_value_weight,
            attention_output_weight,
            feed_forward_norm_scale,
            gate_weight,
            feed_forward_value_weight,
            feed_forward_output_weight,
        )
        analytical_gradients = (
            gradient_inputs,
            gradients.attention_norm,
            gradients.attention.query,
            gradients.attention.key,
            gradients.attention.value,
            gradients.attention.output,
            gradients.feed_forward_norm,
            gradients.feed_forward.gate,
            gradients.feed_forward.value,
            gradients.feed_forward.output,
        )
        names = (
            "inputs",
            "attention_norm_scale",
            "query_weight",
            "key_weight",
            "attention_value_weight",
            "attention_output_weight",
            "feed_forward_norm_scale",
            "gate_weight",
            "feed_forward_value_weight",
            "feed_forward_output_weight",
        )
        for parameter_index, (name, values, analytical) in enumerate(
            zip(names, parameters, analytical_gradients, strict=True)
        ):
            def parameter_objective(
                candidate: np.ndarray,
                index: int = parameter_index,
            ) -> float:
                candidates = list(parameters)
                candidates[index] = candidate
                return objective(*candidates)

            result = check_gradient(
                parameter_objective,
                values,
                analytical,
                absolute_tolerance=2e-7,
            )
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_zero_sublayers_reduce_to_residual_identity(self) -> None:
        rng = np.random.default_rng(157)
        inputs = rng.normal(size=(1, 3, 4))
        gradient_output = rng.normal(size=inputs.shape)
        zero_attention = np.zeros((4, 4))
        output, cache = transformer_block_forward(
            inputs,
            np.ones(4),
            zero_attention,
            zero_attention,
            zero_attention,
            zero_attention,
            np.ones(4),
            np.zeros((4, 6)),
            np.zeros((4, 6)),
            np.zeros((6, 4)),
            n_heads=2,
        )
        gradient_inputs, gradients = transformer_block_backward(
            gradient_output,
            cache,
        )
        np.testing.assert_allclose(output, inputs)
        np.testing.assert_allclose(gradient_inputs, gradient_output)
        np.testing.assert_allclose(gradients.attention_norm, 0.0)
        np.testing.assert_allclose(gradients.feed_forward_norm, 0.0)

    def test_invalid_norm_attention_and_feed_forward_shapes_are_rejected(self) -> None:
        inputs = np.zeros((1, 3, 4), dtype=np.float64)
        identity = np.eye(4)
        gate = np.zeros((4, 6))
        value = np.zeros((4, 6))
        output = np.zeros((6, 4))
        with self.subTest(case="attention norm"):
            with self.assertRaisesRegex(ValueError, "scale must have shape"):
                transformer_block_forward(
                    inputs,
                    np.ones(3),
                    identity,
                    identity,
                    identity,
                    identity,
                    np.ones(4),
                    gate,
                    value,
                    output,
                    n_heads=2,
                )
        with self.subTest(case="attention"):
            with self.assertRaisesRegex(ValueError, "divisible"):
                transformer_block_forward(
                    inputs,
                    np.ones(4),
                    identity,
                    identity,
                    identity,
                    identity,
                    np.ones(4),
                    gate,
                    value,
                    output,
                    n_heads=3,
                )
        with self.subTest(case="feed-forward"):
            with self.assertRaisesRegex(ValueError, "output weight"):
                transformer_block_forward(
                    inputs,
                    np.ones(4),
                    identity,
                    identity,
                    identity,
                    identity,
                    np.ones(4),
                    gate,
                    value,
                    output[:, :3],
                    n_heads=2,
                )


if __name__ == "__main__":
    unittest.main()
