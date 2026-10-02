"""Tests for named gradients, global clipping, and AdamW updates."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    adamw_step,
    clip_gradients_by_global_norm,
    global_gradient_norm,
    initialize_adamw_state,
    initialize_parameters,
    language_model_backward,
    language_model_forward,
    named_gradients,
    named_parameters,
)


class OptimizerTests(unittest.TestCase):
    def test_named_gradients_match_parameter_names_order_and_shapes(self) -> None:
        config = ModelConfig(
            vocab_size=7,
            context_length=5,
            d_model=4,
            n_layers=2,
            n_heads=2,
            d_ff=5,
            seed=211,
        )
        parameters = initialize_parameters(config, dtype=np.float64)
        token_ids = np.array([[0, 2, 1], [3, 6, 4]], dtype=np.int64)
        logits, cache = language_model_forward(token_ids, parameters, config)
        gradients = language_model_backward(np.ones_like(logits), cache)

        parameter_items = list(named_parameters(parameters))
        gradient_items = list(named_gradients(gradients))
        self.assertEqual(
            [name for name, _ in gradient_items],
            [name for name, _ in parameter_items],
        )
        self.assertEqual(
            [values.shape for _, values in gradient_items],
            [values.shape for _, values in parameter_items],
        )

    def test_global_norm_clipping_uses_one_joint_l2_norm(self) -> None:
        first = np.array([3.0, 4.0], dtype=np.float64)
        second = np.array([0.0, 12.0], dtype=np.float64)
        first_before = first.copy()
        second_before = second.copy()

        result = clip_gradients_by_global_norm(
            (("first", first), ("second", second)),
            max_norm=6.5,
            epsilon=1e-12,
        )
        expected_coefficient = 6.5 / (13.0 + 1e-12)
        self.assertAlmostEqual(result.global_norm, 13.0)
        self.assertAlmostEqual(result.clip_coefficient, expected_coefficient)
        clipped = dict(result.gradients)
        np.testing.assert_allclose(clipped["first"], first * expected_coefficient)
        np.testing.assert_allclose(clipped["second"], second * expected_coefficient)
        self.assertLessEqual(
            global_gradient_norm(result.gradients),
            6.5,
        )
        np.testing.assert_array_equal(first, first_before)
        np.testing.assert_array_equal(second, second_before)

    def test_gradients_below_threshold_are_copied_without_scaling(self) -> None:
        gradient = np.array([0.3, -0.4], dtype=np.float32)
        result = clip_gradients_by_global_norm(
            (("gradient", gradient),),
            max_norm=1.0,
        )
        self.assertAlmostEqual(result.global_norm, 0.5, places=6)
        self.assertEqual(result.clip_coefficient, 1.0)
        np.testing.assert_array_equal(result.gradients[0][1], gradient)
        self.assertFalse(np.shares_memory(result.gradients[0][1], gradient))

    def test_first_adamw_step_matches_definition_and_decay_exclusions(self) -> None:
        matrix = np.array([[1.0, -2.0]], dtype=np.float64)
        norm_scale = np.array([1.5, 0.5], dtype=np.float64)
        matrix_before = matrix.copy()
        norm_before = norm_scale.copy()
        matrix_gradient = np.array([[0.5, -0.25]], dtype=np.float64)
        norm_gradient = np.array([0.2, -0.4], dtype=np.float64)
        parameters = (("matrix", matrix), ("norm_scale", norm_scale))
        gradients = (
            ("matrix", matrix_gradient),
            ("norm_scale", norm_gradient),
        )
        state = initialize_adamw_state(parameters)
        expected_gradient_norm = float(
            np.sqrt(
                np.sum(matrix_gradient * matrix_gradient)
                + np.sum(norm_gradient * norm_gradient)
            )
        )
        expected_clip_coefficient = 0.35 / (expected_gradient_norm + 1e-6)
        clipped_matrix_gradient = matrix_gradient * expected_clip_coefficient
        clipped_norm_gradient = norm_gradient * expected_clip_coefficient

        next_state, stats = adamw_step(
            parameters,
            gradients,
            state,
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.2,
            max_gradient_norm=0.35,
        )
        expected_matrix_update = clipped_matrix_gradient / (
            np.abs(clipped_matrix_gradient) + 1e-8
        )
        expected_norm_update = clipped_norm_gradient / (
            np.abs(clipped_norm_gradient) + 1e-8
        )
        np.testing.assert_allclose(
            matrix,
            matrix_before
            - 0.1 * expected_matrix_update
            - 0.1 * 0.2 * matrix_before,
        )
        np.testing.assert_allclose(
            norm_scale,
            norm_before - 0.1 * expected_norm_update,
        )
        np.testing.assert_allclose(
            next_state.first_moments[0],
            0.1 * clipped_matrix_gradient,
        )
        np.testing.assert_allclose(
            next_state.second_moments[0],
            0.001 * clipped_matrix_gradient * clipped_matrix_gradient,
        )
        self.assertEqual(next_state.step, 1)
        self.assertEqual(stats.step, 1)
        self.assertEqual(stats.learning_rate, 0.1)
        self.assertAlmostEqual(stats.gradient_norm, expected_gradient_norm)
        self.assertAlmostEqual(
            stats.clip_coefficient,
            expected_clip_coefficient,
        )

    def test_second_step_uses_accumulated_moments_and_bias_correction(self) -> None:
        parameter = np.array([[1.0]], dtype=np.float64)
        parameters = (("weight", parameter),)
        initial_state = initialize_adamw_state(parameters)
        first_gradient = np.array([[2.0]], dtype=np.float64)
        second_gradient = np.array([[4.0]], dtype=np.float64)

        first_state, _ = adamw_step(
            parameters,
            (("weight", first_gradient),),
            initial_state,
            learning_rate=0.05,
            beta1=0.5,
            beta2=0.25,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=100.0,
        )
        expected_after_first = 1.0 - 0.05 * (2.0 / (2.0 + 1e-8))
        second_state, _ = adamw_step(
            parameters,
            (("weight", second_gradient),),
            first_state,
            learning_rate=0.05,
            beta1=0.5,
            beta2=0.25,
            epsilon=1e-8,
            weight_decay=0.0,
            max_gradient_norm=100.0,
        )

        expected_first_moment = 0.5 * 1.0 + 0.5 * 4.0
        expected_second_moment = 0.25 * 3.0 + 0.75 * 16.0
        corrected_first = expected_first_moment / (1.0 - 0.5**2)
        corrected_second = expected_second_moment / (1.0 - 0.25**2)
        expected_parameter = expected_after_first - 0.05 * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        np.testing.assert_allclose(parameter, [[expected_parameter]])
        np.testing.assert_allclose(
            second_state.first_moments[0],
            [[expected_first_moment]],
        )
        np.testing.assert_allclose(
            second_state.second_moments[0],
            [[expected_second_moment]],
        )
        self.assertEqual(second_state.step, 2)
        self.assertEqual(initial_state.step, 0)
        np.testing.assert_array_equal(
            initial_state.first_moments[0],
            np.zeros_like(parameter),
        )

    def test_invalid_contracts_fail_before_parameters_are_modified(self) -> None:
        parameter = np.array([[1.0, -2.0]], dtype=np.float64)
        parameters = (("weight", parameter),)
        state = initialize_adamw_state(parameters)
        before = parameter.copy()

        cases = (
            (
                "name",
                (("different", np.ones_like(parameter)),),
                {"learning_rate": 0.1},
                ValueError,
            ),
            (
                "shape",
                (("weight", np.ones(3, dtype=np.float64)),),
                {"learning_rate": 0.1},
                ValueError,
            ),
            (
                "nonfinite",
                (("weight", np.array([[np.nan, 1.0]])),),
                {"learning_rate": 0.1},
                FloatingPointError,
            ),
            (
                "learning rate",
                (("weight", np.ones_like(parameter)),),
                {"learning_rate": 0.0},
                ValueError,
            ),
            (
                "beta",
                (("weight", np.ones_like(parameter)),),
                {"learning_rate": 0.1, "beta1": 1.0},
                ValueError,
            ),
            (
                "clip norm",
                (("weight", np.ones_like(parameter)),),
                {"learning_rate": 0.1, "max_gradient_norm": 0.0},
                ValueError,
            ),
        )
        for name, gradients, keyword_arguments, error_type in cases:
            with self.subTest(case=name):
                with self.assertRaises(error_type):
                    adamw_step(
                        parameters,
                        gradients,
                        state,
                        **keyword_arguments,
                    )
                np.testing.assert_array_equal(parameter, before)


if __name__ == "__main__":
    unittest.main()
