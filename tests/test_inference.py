"""Tests for checkpoint-backed and terminal-safe one-shot generation."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    CheckpointGenerationResult,
    GenerationResult,
    ModelConfig,
    encode_utf8,
    format_generation_result,
    generate_from_checkpoint,
    generate_tokens,
    initialize_adamw_state,
    initialize_parameters,
    load_checkpoint,
    named_parameters,
    save_checkpoint,
)


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        context_length=8,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=6,
        seed=281,
    )


def _checkpoint(directory: str) -> Path:
    config = _config()
    parameters = initialize_parameters(config, dtype=np.float64)
    optimizer_state = initialize_adamw_state(named_parameters(parameters))
    return save_checkpoint(
        Path(directory) / "model.npz",
        parameters,
        optimizer_state,
        config,
    )


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


class CheckpointInferenceTests(unittest.TestCase):
    def test_greedy_checkpoint_generation_matches_direct_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            result = generate_from_checkpoint(
                checkpoint_path,
                "A",
                max_new_tokens=3,
                greedy=True,
                clock=_clock(10.0, 12.0),
            )
            loaded = load_checkpoint(checkpoint_path)
            expected = generate_tokens(
                encode_utf8("A"),
                loaded.parameters,
                loaded.config,
                max_new_tokens=3,
                greedy=True,
            )

        np.testing.assert_array_equal(
            result.generation.token_ids,
            expected.token_ids,
        )
        self.assertEqual(result.checkpoint_step, 0)
        self.assertEqual(result.elapsed_seconds, 2.0)
        self.assertEqual(result.tokens_per_second, 1.5)

    def test_stochastic_checkpoint_generation_repeats_with_the_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            first = generate_from_checkpoint(
                checkpoint_path,
                "A",
                max_new_tokens=4,
                temperature=0.8,
                top_k=16,
                seed=283,
            )
            second = generate_from_checkpoint(
                checkpoint_path,
                "A",
                max_new_tokens=4,
                temperature=0.8,
                top_k=16,
                seed=283,
            )

        np.testing.assert_array_equal(
            first.generation.generated_tokens,
            second.generation.generated_tokens,
        )
        self.assertEqual(first.generated_text, second.generated_text)

    def test_context_stop_and_timing_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            result = generate_from_checkpoint(
                checkpoint_path,
                "1234567",
                max_new_tokens=5,
                greedy=True,
                clock=_clock(4.0, 6.0),
            )
            full_context_result = generate_from_checkpoint(
                checkpoint_path,
                "12345678",
                max_new_tokens=5,
                greedy=True,
                clock=_clock(8.0, 8.0),
            )

        self.assertEqual(result.generation.stop_reason, "context_limit")
        self.assertEqual(result.generation.generated_tokens.size, 1)
        self.assertEqual(result.tokens_per_second, 0.5)
        self.assertEqual(result.context_length, 8)
        self.assertEqual(full_context_result.generation.stop_reason, "context_limit")
        self.assertEqual(full_context_result.generation.generated_tokens.size, 0)
        self.assertEqual(full_context_result.tokens_per_second, 0.0)

    def test_utf8_prompt_counts_bytes_and_preserves_the_prompt_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            result = generate_from_checkpoint(
                checkpoint_path,
                "é",
                max_new_tokens=1,
                greedy=True,
            )

        self.assertEqual(result.prompt_text, "é")
        self.assertEqual(result.generation.prompt_tokens.size, 2)
        np.testing.assert_array_equal(
            result.generation.prompt_tokens,
            encode_utf8("é"),
        )

    def test_terminal_format_escapes_control_characters(self) -> None:
        generation = GenerationResult(
            prompt_tokens=np.array([65], dtype=np.int64),
            generated_tokens=np.array([27, 10], dtype=np.int64),
            token_ids=np.array([65, 27, 10], dtype=np.int64),
            stop_reason="max_new_tokens",
        )
        result = CheckpointGenerationResult(
            checkpoint_path=Path("model.npz"),
            checkpoint_step=20,
            context_length=16,
            prompt_text="A",
            generated_text="\x1b\n",
            full_text="A\x1b\n",
            generation=generation,
            elapsed_seconds=0.5,
            tokens_per_second=4.0,
        )

        formatted = format_generation_result(result)

        self.assertIn("generated='\\x1b\\n'", formatted)
        self.assertNotIn("\x1b", formatted)
        self.assertIn("stop_reason=max_new_tokens", formatted)
        self.assertIn("generated_tokens=2", formatted)

    def test_invalid_prompt_seed_clock_and_result_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            with self.assertRaises(TypeError):
                generate_from_checkpoint(
                    checkpoint_path,
                    123,
                    max_new_tokens=1,
                    greedy=True,
                )
            with self.assertRaises(ValueError):
                generate_from_checkpoint(
                    checkpoint_path,
                    "A",
                    max_new_tokens=1,
                    greedy=True,
                    seed=-1,
                )
            with self.assertRaisesRegex(ValueError, "clock"):
                generate_from_checkpoint(
                    checkpoint_path,
                    "A",
                    max_new_tokens=1,
                    greedy=True,
                    clock=_clock(2.0, 1.0),
                )
        with self.assertRaises(TypeError):
            format_generation_result(None)


if __name__ == "__main__":
    unittest.main()
