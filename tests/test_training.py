"""Tests for one complete train step and the read-only evaluation path."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ByteBatch,
    ModelConfig,
    adamw_step,
    cross_entropy_with_logits,
    evaluate_batch,
    initialize_adamw_state,
    initialize_parameters,
    language_model_backward,
    language_model_forward,
    named_gradients,
    named_parameters,
    train_step,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        context_length=4,
        d_model=4,
        n_layers=1,
        n_heads=2,
        d_ff=6,
        seed=241,
    )


def _batch() -> ByteBatch:
    return ByteBatch(
        inputs=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        targets=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        start_indices=np.array([0, 3], dtype=np.int64),
    )


class TrainingCompositionTests(unittest.TestCase):
    def test_evaluation_matches_direct_mean_cross_entropy(self) -> None:
        config = _tiny_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        batch = _batch()
        logits, _ = language_model_forward(batch.inputs, parameters, config)
        expected_loss, _ = cross_entropy_with_logits(logits, batch.targets)

        metrics = evaluate_batch(batch, parameters, config)

        self.assertAlmostEqual(metrics.loss, float(expected_loss))
        self.assertAlmostEqual(metrics.perplexity, float(np.exp(expected_loss)))
        self.assertEqual(metrics.token_count, batch.targets.size)

    def test_train_step_matches_explicit_primitive_composition(self) -> None:
        config = _tiny_config()
        batch = _batch()
        parameters = initialize_parameters(config, dtype=np.float64)
        expected_parameters = initialize_parameters(config, dtype=np.float64)
        state = initialize_adamw_state(named_parameters(parameters))
        expected_state = initialize_adamw_state(
            named_parameters(expected_parameters)
        )
        keyword_arguments = {
            "learning_rate": 0.002,
            "beta1": 0.8,
            "beta2": 0.95,
            "epsilon": 1e-7,
            "weight_decay": 0.03,
            "max_gradient_norm": 0.4,
        }

        logits, cache = language_model_forward(
            batch.inputs,
            expected_parameters,
            config,
        )
        expected_loss, gradient_logits = cross_entropy_with_logits(
            logits,
            batch.targets,
        )
        expected_gradients = language_model_backward(gradient_logits, cache)
        expected_state, expected_optimizer_stats = adamw_step(
            named_parameters(expected_parameters),
            named_gradients(expected_gradients),
            expected_state,
            **keyword_arguments,
        )

        actual_state, actual_stats = train_step(
            batch,
            parameters,
            state,
            config,
            **keyword_arguments,
        )

        for (actual_name, actual), (expected_name, expected) in zip(
            named_parameters(parameters),
            named_parameters(expected_parameters),
            strict=True,
        ):
            self.assertEqual(actual_name, expected_name)
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(
            actual_state.first_moments,
            expected_state.first_moments,
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(
            actual_state.second_moments,
            expected_state.second_moments,
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(actual_stats.loss, float(expected_loss))
        self.assertEqual(actual_stats.step, expected_optimizer_stats.step)
        self.assertAlmostEqual(
            actual_stats.gradient_norm,
            expected_optimizer_stats.gradient_norm,
        )
        self.assertAlmostEqual(
            actual_stats.clip_coefficient,
            expected_optimizer_stats.clip_coefficient,
        )

    def test_train_step_updates_parameters_and_increments_state(self) -> None:
        config = _tiny_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        before = tuple(values.copy() for _, values in named_parameters(parameters))
        state = initialize_adamw_state(named_parameters(parameters))

        next_state, stats = train_step(
            _batch(),
            parameters,
            state,
            config,
            learning_rate=0.001,
        )

        self.assertEqual(state.step, 0)
        self.assertEqual(next_state.step, 1)
        self.assertEqual(stats.step, 1)
        self.assertTrue(np.isfinite(stats.loss))
        self.assertTrue(np.isfinite(stats.perplexity))
        self.assertEqual(stats.token_count, 6)
        self.assertTrue(
            any(
                not np.array_equal(previous, current)
                for previous, (_, current) in zip(
                    before,
                    named_parameters(parameters),
                    strict=True,
                )
            )
        )

    def test_evaluation_does_not_mutate_parameters(self) -> None:
        config = _tiny_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        before = tuple(values.copy() for _, values in named_parameters(parameters))

        first = evaluate_batch(_batch(), parameters, config)
        second = evaluate_batch(_batch(), parameters, config)

        self.assertEqual(first, second)
        for previous, (_, current) in zip(
            before,
            named_parameters(parameters),
            strict=True,
        ):
            np.testing.assert_array_equal(current, previous)

    def test_clipping_diagnostics_are_exposed_by_the_training_step(self) -> None:
        config = _tiny_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        state = initialize_adamw_state(named_parameters(parameters))

        _, stats = train_step(
            _batch(),
            parameters,
            state,
            config,
            learning_rate=0.001,
            max_gradient_norm=1e-10,
        )

        self.assertGreater(stats.gradient_norm, 1e-10)
        self.assertGreater(stats.clip_coefficient, 0.0)
        self.assertLess(stats.clip_coefficient, 1.0)

    def test_invalid_inputs_fail_without_parameter_mutation(self) -> None:
        config = _tiny_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        state = initialize_adamw_state(named_parameters(parameters))
        before = tuple(values.copy() for _, values in named_parameters(parameters))
        invalid_batches = (
            ByteBatch(
                inputs=np.array([0, 1]),
                targets=np.array([1, 2]),
                start_indices=np.array([0]),
            ),
            ByteBatch(
                inputs=np.array([[0, 1]]),
                targets=np.array([[1]]),
                start_indices=np.array([0]),
            ),
            ByteBatch(
                inputs=np.array([[0.0, 1.0]]),
                targets=np.array([[1, 2]]),
                start_indices=np.array([0]),
            ),
            ByteBatch(
                inputs=np.array([[0, 8]]),
                targets=np.array([[1, 2]]),
                start_indices=np.array([0]),
            ),
        )
        for batch in invalid_batches:
            with self.subTest(batch=batch):
                with self.assertRaises((TypeError, ValueError)):
                    train_step(
                        batch,
                        parameters,
                        state,
                        config,
                        learning_rate=0.001,
                    )
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            train_step(
                _batch(),
                parameters,
                state,
                config,
                learning_rate=0.0,
            )
        for previous, (_, current) in zip(
            before,
            named_parameters(parameters),
            strict=True,
        ):
            np.testing.assert_array_equal(current, previous)


if __name__ == "__main__":
    unittest.main()
