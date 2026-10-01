"""Tests for deterministic full-model parameter initialization."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    initialize_parameters,
    named_parameters,
)


def _small_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "context_length": 16,
        "d_model": 16,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 32,
        "seed": 197,
    }
    values.update(overrides)
    return ModelConfig(**values)


class InitializationTests(unittest.TestCase):
    def test_same_seed_produces_identical_parameters(self) -> None:
        config = _small_config()
        first = list(named_parameters(initialize_parameters(config)))
        second = list(named_parameters(initialize_parameters(config)))
        self.assertEqual([name for name, _ in first], [name for name, _ in second])
        for (name, first_values), (_, second_values) in zip(
            first,
            second,
            strict=True,
        ):
            with self.subTest(name=name):
                np.testing.assert_array_equal(first_values, second_values)

    def test_seed_override_changes_random_weights(self) -> None:
        config = _small_config()
        first = initialize_parameters(config, seed=1)
        second = initialize_parameters(config, seed=2)
        self.assertFalse(np.array_equal(first.token_embedding, second.token_embedding))
        np.testing.assert_array_equal(
            first.blocks[0].attention_norm_scale,
            second.blocks[0].attention_norm_scale,
        )

    def test_default_shapes_names_and_parameter_count_match_configuration(self) -> None:
        config = ModelConfig()
        parameters = initialize_parameters(config)
        named = list(named_parameters(parameters))
        names = [name for name, _ in named]
        self.assertEqual(len(names), 2 + 9 * config.n_layers)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "token_embedding")
        self.assertEqual(names[-1], "final_norm_scale")
        self.assertEqual(
            sum(values.size for _, values in named),
            config.parameter_count(),
        )
        self.assertEqual(
            parameters.token_embedding.shape,
            (config.vocab_size, config.d_model),
        )
        self.assertEqual(
            parameters.blocks[0].gate_weight.shape,
            (config.d_model, config.d_ff),
        )

    def test_norm_scales_dtype_and_finiteness_are_correct(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        for name, values in named_parameters(parameters):
            with self.subTest(name=name):
                self.assertEqual(values.dtype, np.dtype(np.float64))
                self.assertTrue(np.all(np.isfinite(values)))
                if name.endswith("norm_scale"):
                    np.testing.assert_array_equal(values, 1.0)

    def test_residual_output_projections_use_depth_scaled_standard_deviation(
        self,
    ) -> None:
        config = _small_config(d_model=128, d_ff=256, n_layers=8, n_heads=8)
        parameters = initialize_parameters(config, dtype=np.float64)
        ordinary = np.concatenate(
            [block.query_weight.ravel() for block in parameters.blocks]
        )
        residual = np.concatenate(
            [
                values.ravel()
                for block in parameters.blocks
                for values in (
                    block.attention_output_weight,
                    block.feed_forward_output_weight,
                )
            ]
        )
        expected_residual_std = config.initializer_std / np.sqrt(
            2.0 * config.n_layers
        )
        self.assertAlmostEqual(
            float(np.std(ordinary)),
            config.initializer_std,
            delta=config.initializer_std * 0.03,
        )
        self.assertAlmostEqual(
            float(np.std(residual)),
            expected_residual_std,
            delta=expected_residual_std * 0.03,
        )

    def test_invalid_initialization_settings_are_rejected(self) -> None:
        with self.subTest(case="dtype"):
            with self.assertRaisesRegex(TypeError, "floating-point"):
                initialize_parameters(_small_config(), dtype=np.int64)
        with self.subTest(case="seed"):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                initialize_parameters(_small_config(), seed=-1)
        with self.subTest(case="initializer std"):
            with self.assertRaisesRegex(ValueError, "initializer_std"):
                _small_config(initializer_std=0.0)
        with self.subTest(case="untied"):
            with self.assertRaisesRegex(ValueError, "tied embeddings"):
                initialize_parameters(_small_config(tie_embeddings=False))
        with self.subTest(case="bias"):
            with self.assertRaisesRegex(ValueError, "bias-free"):
                initialize_parameters(_small_config(use_bias=True))


if __name__ == "__main__":
    unittest.main()
