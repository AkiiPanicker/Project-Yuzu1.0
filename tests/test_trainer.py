"""Tests for terminal-visible, logged, and resumable multi-step training."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    TrainingLogRecord,
    TrainingLoopConfig,
    initialize_adamw_state,
    initialize_parameters,
    load_checkpoint,
    named_parameters,
    run_training,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        context_length=4,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=6,
        seed=251,
    )


def _tokens(offset: int = 0) -> np.ndarray:
    return ((np.arange(40, dtype=np.int64) + offset) % 16).astype(np.uint8)


def _loop_config(max_steps: int) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        max_steps=max_steps,
        batch_size=2,
        learning_rate=0.002,
        validation_interval=1,
        checkpoint_interval=2,
        seed=257,
    )


class TrainingLoopTests(unittest.TestCase):
    def test_loop_configuration_rejects_invalid_values(self) -> None:
        valid = _loop_config(3)
        cases = (
            ("max_steps", 0),
            ("batch_size", 0),
            ("learning_rate", 0.0),
            ("validation_interval", 0),
            ("log_interval", 0),
            ("checkpoint_interval", 0),
            ("beta1", 1.0),
            ("beta2", -0.1),
            ("adam_epsilon", 0.0),
            ("weight_decay", -0.1),
            ("max_gradient_norm", 0.0),
            ("seed", -1),
        )
        for field, value in cases:
            with self.subTest(field=field):
                with self.assertRaises((TypeError, ValueError)):
                    replace(valid, **{field: value})

    def test_log_record_contains_every_required_terminal_field(self) -> None:
        record = TrainingLogRecord(
            step=7,
            epoch=1.25,
            train_loss=2.5,
            validation_loss=2.75,
            validation_perplexity=15.64,
            learning_rate=0.001,
            gradient_norm=0.9,
            clip_coefficient=1.0,
            tokens_per_second=123.4,
            elapsed_seconds=4.2,
        )

        terminal = record.to_terminal()
        for field in (
            "step=",
            "epoch=",
            "train_loss=",
            "validation_loss=",
            "lr=",
            "grad_norm=",
            "tok/s=",
            "elapsed=",
        ):
            self.assertIn(field, terminal)
        self.assertEqual(record.to_dict()["event"], "metrics")

    def test_run_writes_metrics_checkpoint_and_terminal_events(self) -> None:
        messages: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            result = run_training(
                _tokens(),
                _tokens(1),
                _model_config(),
                _loop_config(2),
                directory,
                output=messages.append,
            )

            self.assertEqual(result.optimizer_state.step, 2)
            self.assertEqual([record.step for record in result.records], [1, 2])
            self.assertEqual(len(result.checkpoint_paths), 1)
            self.assertTrue(result.checkpoint_paths[0].is_file())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], "numpy-gpt-training-run")
            self.assertEqual(manifest["train_tokens"]["count"], 40)
            lines = [
                json.loads(line)
                for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [line["event"] for line in lines],
                ["metrics", "metrics", "checkpoint"],
            )
            self.assertTrue(any("validation_loss=" in line for line in messages))
            self.assertTrue(any(line.startswith("checkpoint ") for line in messages))

    def test_final_step_is_logged_and_checkpointed_between_intervals(self) -> None:
        loop_config = replace(
            _loop_config(1),
            log_interval=50,
            validation_interval=50,
            checkpoint_interval=50,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_training(
                _tokens(),
                _tokens(1),
                _model_config(),
                loop_config,
                directory,
                output=None,
            )
            restored = load_checkpoint(result.checkpoint_paths[-1])

        self.assertEqual([record.step for record in result.records], [1])
        self.assertEqual(restored.optimizer_state.step, 1)
        self.assertIsNotNone(restored.rng_state)

    def test_checkpoint_resume_matches_uninterrupted_training(self) -> None:
        config = _model_config()
        train_tokens = _tokens()
        validation_tokens = _tokens(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = run_training(
                train_tokens,
                validation_tokens,
                config,
                _loop_config(3),
                root / "uninterrupted",
                output=None,
            )
            partial = run_training(
                train_tokens,
                validation_tokens,
                config,
                _loop_config(2),
                root / "resumed",
                output=None,
            )
            loaded = load_checkpoint(partial.checkpoint_paths[-1])
            restored_rng = np.random.default_rng()
            restored_rng.bit_generator.state = loaded.rng_state
            resumed = run_training(
                train_tokens,
                validation_tokens,
                loaded.config,
                _loop_config(3),
                root / "resumed",
                parameters=loaded.parameters,
                optimizer_state=loaded.optimizer_state,
                rng=restored_rng,
                output=None,
            )

        for (_, expected), (_, actual) in zip(
            named_parameters(uninterrupted.parameters),
            named_parameters(resumed.parameters),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(
            uninterrupted.optimizer_state.first_moments,
            resumed.optimizer_state.first_moments,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(resumed.optimizer_state.step, 3)

        with tempfile.TemporaryDirectory() as directory:
            initial = run_training(
                train_tokens,
                validation_tokens,
                config,
                _loop_config(1),
                directory,
                output=None,
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                run_training(
                    _tokens(2),
                    validation_tokens,
                    config,
                    _loop_config(2),
                    directory,
                    parameters=initial.parameters,
                    optimizer_state=initial.optimizer_state,
                    output=None,
                )

    def test_invalid_session_inputs_fail_before_training(self) -> None:
        config = _model_config()
        parameters = initialize_parameters(config)
        state = initialize_adamw_state(named_parameters(parameters))
        cases = (
            (_tokens().reshape(2, 20), _tokens(1), parameters, state),
            (_tokens(), np.array([0, 256, 1, 2, 3]), parameters, state),
            (_tokens(), _tokens(1), parameters, None),
        )
        for index, (
            train_tokens,
            validation_tokens,
            candidate_parameters,
            candidate_state,
        ) in enumerate(cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((TypeError, ValueError)):
                    run_training(
                        train_tokens,
                        validation_tokens,
                        config,
                        _loop_config(1),
                        directory,
                        parameters=candidate_parameters,
                        optimizer_state=candidate_state,
                        output=None,
                    )


if __name__ == "__main__":
    unittest.main()
