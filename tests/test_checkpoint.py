"""Tests for exact, versioned, non-pickle training checkpoints."""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    LanguageModelParameters,
    ModelConfig,
    adamw_step,
    initialize_adamw_state,
    initialize_parameters,
    load_checkpoint,
    named_parameters,
    save_checkpoint,
)


def _small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        context_length=8,
        d_model=4,
        n_layers=2,
        n_heads=2,
        d_ff=6,
        seed=223,
    )


def _gradient_items(
    parameters: LanguageModelParameters,
    *,
    scale: float,
) -> tuple[tuple[str, np.ndarray], ...]:
    return tuple(
        (name, np.full_like(values, scale * (index + 1)))
        for index, (name, values) in enumerate(named_parameters(parameters))
    )


def _assert_named_arrays_equal(
    test: unittest.TestCase,
    first: tuple[tuple[str, np.ndarray], ...],
    second: tuple[tuple[str, np.ndarray], ...],
) -> None:
    test.assertEqual(
        [name for name, _ in first],
        [name for name, _ in second],
    )
    for (name, first_values), (_, second_values) in zip(
        first,
        second,
        strict=True,
    ):
        with test.subTest(name=name):
            test.assertEqual(first_values.dtype, second_values.dtype)
            np.testing.assert_array_equal(first_values, second_values)


class CheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_every_array_config_step_and_rng_state(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
        gradients = _gradient_items(parameters, scale=0.003)
        optimizer_state, _ = adamw_step(
            named_parameters(parameters),
            gradients,
            optimizer_state,
            learning_rate=0.002,
            max_gradient_norm=10.0,
        )
        rng = np.random.default_rng(227)
        rng.normal(size=7)
        rng_state = rng.bit_generator.state

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.npz"
            returned_path = save_checkpoint(
                path,
                parameters,
                optimizer_state,
                config,
                rng_state=rng_state,
            )
            loaded = load_checkpoint(path)

        self.assertEqual(returned_path, path)
        self.assertEqual(loaded.config, config)
        _assert_named_arrays_equal(
            self,
            tuple(named_parameters(parameters)),
            tuple(named_parameters(loaded.parameters)),
        )
        self.assertEqual(loaded.optimizer_state.step, optimizer_state.step)
        self.assertEqual(
            loaded.optimizer_state.parameter_names,
            optimizer_state.parameter_names,
        )
        for loaded_values, expected_values in zip(
            loaded.optimizer_state.first_moments,
            optimizer_state.first_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(loaded_values, expected_values)
        for loaded_values, expected_values in zip(
            loaded.optimizer_state.second_moments,
            optimizer_state.second_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(loaded_values, expected_values)
        self.assertEqual(loaded.rng_state, rng_state)
        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = loaded.rng_state
        np.testing.assert_array_equal(
            restored_rng.integers(0, 1_000_000, size=12),
            rng.integers(0, 1_000_000, size=12),
        )

    def test_restored_state_produces_the_identical_next_adamw_update(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
        optimizer_state, _ = adamw_step(
            named_parameters(parameters),
            _gradient_items(parameters, scale=0.002),
            optimizer_state,
            learning_rate=0.001,
            beta2=0.95,
            max_gradient_norm=5.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.npz"
            save_checkpoint(path, parameters, optimizer_state, config)
            loaded = load_checkpoint(path)

        next_gradients = _gradient_items(parameters, scale=-0.0015)
        expected_state, expected_stats = adamw_step(
            named_parameters(parameters),
            next_gradients,
            optimizer_state,
            learning_rate=0.001,
            beta2=0.95,
            max_gradient_norm=5.0,
        )
        loaded_state, loaded_stats = adamw_step(
            named_parameters(loaded.parameters),
            next_gradients,
            loaded.optimizer_state,
            learning_rate=0.001,
            beta2=0.95,
            max_gradient_norm=5.0,
        )

        _assert_named_arrays_equal(
            self,
            tuple(named_parameters(parameters)),
            tuple(named_parameters(loaded.parameters)),
        )
        self.assertEqual(loaded_stats, expected_stats)
        self.assertEqual(loaded_state.step, expected_state.step)
        for loaded_values, expected_values in zip(
            loaded_state.first_moments,
            expected_state.first_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(loaded_values, expected_values)
        for loaded_values, expected_values in zip(
            loaded_state.second_moments,
            expected_state.second_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(loaded_values, expected_values)

    def test_archive_contains_only_non_object_arrays(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safe_arrays.npz"
            save_checkpoint(path, parameters, optimizer_state, config)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["manifest"].dtype, np.dtype(np.uint8))
                for name in archive.files:
                    with self.subTest(name=name):
                        self.assertNotEqual(archive[name].dtype, np.dtype(object))

    def test_unknown_checkpoint_version_is_rejected(self) -> None:
        manifest = {
            "format": "numpy-gpt-from-scratch",
            "version": 999,
            "config": {},
            "optimizer_step": 0,
            "parameter_names": [],
            "rng_state": None,
        }
        manifest_bytes = np.frombuffer(
            json.dumps(manifest).encode("utf-8"),
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future_version.npz"
            np.savez(path, manifest=manifest_bytes)
            with self.assertRaisesRegex(ValueError, "version"):
                load_checkpoint(path)

    def test_parameter_shape_tampering_is_rejected(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.npz"
            save_checkpoint(path, parameters, optimizer_state, config)
            with np.load(path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                }
            arrays["parameter_0000"] = arrays["parameter_0000"].reshape(-1)
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_checkpoint(path)

    def test_invalid_state_cannot_overwrite_an_existing_checkpoint(self) -> None:
        config = _small_config()
        parameters = initialize_parameters(config)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
        invalid_state = replace(
            optimizer_state,
            parameter_names=("wrong",) + optimizer_state.parameter_names[1:],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected.npz"
            save_checkpoint(path, parameters, optimizer_state, config)
            original_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "state names"):
                save_checkpoint(path, parameters, invalid_state, config)
            self.assertEqual(path.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
