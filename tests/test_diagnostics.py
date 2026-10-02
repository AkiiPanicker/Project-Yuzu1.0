"""Tests for read-only teacher-forced checkpoint diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    encode_utf8,
    format_continuation_probe,
    initialize_adamw_state,
    initialize_parameters,
    language_model_forward,
    load_checkpoint,
    logsumexp,
    named_parameters,
    probe_checkpoint_continuation,
    save_checkpoint,
    softmax,
)


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        context_length=8,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=6,
        seed=307,
    )


def _checkpoint(directory: str, *, zero_parameters: bool = False) -> Path:
    config = _config()
    parameters = initialize_parameters(config, dtype=np.float64)
    if zero_parameters:
        for _, values in named_parameters(parameters):
            values.fill(0.0)
    optimizer_state = initialize_adamw_state(named_parameters(parameters))
    return save_checkpoint(
        Path(directory) / "model.npz",
        parameters,
        optimizer_state,
        config,
    )


class ContinuationProbeTests(unittest.TestCase):
    def test_teacher_forced_steps_match_direct_model_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            result = probe_checkpoint_continuation(
                checkpoint_path,
                "A",
                "BC",
                top_k=4,
            )
            loaded = load_checkpoint(checkpoint_path)
            manual_losses = []
            for offset, expected_token in enumerate(encode_utf8("BC")):
                visible = encode_utf8("A" + "BC"[:offset]).astype(np.int64)
                logits, _ = language_model_forward(
                    visible[None, :],
                    loaded.parameters,
                    loaded.config,
                )
                scores = logits[0, -1].astype(np.float64)
                probabilities = softmax(scores)
                token_ids = np.arange(scores.size)
                ranked = np.lexsort((token_ids, -scores))
                step = result.steps[offset]
                expected_id = int(expected_token)
                loss = float(logsumexp(scores) - scores[expected_id])
                manual_losses.append(loss)
                self.assertEqual(step.predicted_token_id, int(ranked[0]))
                self.assertEqual(
                    step.expected_rank,
                    int(np.flatnonzero(ranked == expected_id)[0]) + 1,
                )
                self.assertAlmostEqual(
                    step.expected_probability,
                    float(probabilities[expected_id]),
                )
                self.assertAlmostEqual(step.negative_log_likelihood, loss)
                self.assertEqual(
                    tuple(candidate.token_id for candidate in step.candidates),
                    tuple(int(value) for value in ranked[:4]),
                )

        self.assertAlmostEqual(result.mean_loss, float(np.mean(manual_losses)))
        self.assertAlmostEqual(result.perplexity, float(np.exp(result.mean_loss)))

    def test_equal_logits_use_token_id_as_the_rank_tie_breaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory, zero_parameters=True)
            result = probe_checkpoint_continuation(
                checkpoint_path,
                "A",
                "\x05\x06",
                top_k=3,
            )

        first, second = result.steps
        self.assertEqual(first.predicted_token_id, 0)
        self.assertEqual(first.expected_rank, 6)
        self.assertEqual(
            tuple(candidate.token_id for candidate in first.candidates),
            (0, 1, 2),
        )
        self.assertAlmostEqual(first.expected_probability, 1.0 / 256.0)
        self.assertEqual(second.prefix_token_count, 2)
        self.assertEqual(second.expected_rank, 7)

    def test_utf8_text_is_scored_as_bytes_with_gold_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            result = probe_checkpoint_continuation(
                checkpoint_path,
                "é",
                "ß",
            )
            full_context_result = probe_checkpoint_continuation(
                checkpoint_path,
                "12345678",
                "9",
            )

        self.assertEqual(result.prompt_token_count, 2)
        self.assertEqual(result.expected_token_count, 2)
        self.assertEqual(tuple(step.offset for step in result.steps), (0, 1))
        self.assertEqual(
            tuple(step.prefix_token_count for step in result.steps),
            (2, 3),
        )
        self.assertEqual(full_context_result.steps[0].prefix_token_count, 8)

    def test_formatter_escapes_control_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory, zero_parameters=True)
            result = probe_checkpoint_continuation(
                checkpoint_path,
                "A",
                "\x1b\n",
                top_k=2,
            )

        formatted = format_continuation_probe(result)
        self.assertIn("expected_text='\\x1b\\n'", formatted)
        self.assertNotIn("\x1b", formatted)
        self.assertIn("teacher_forced_top1=", formatted)
        self.assertIn("rank=", formatted)

    def test_invalid_text_top_k_and_context_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            invalid_calls = (
                lambda: probe_checkpoint_continuation(checkpoint_path, "", "B"),
                lambda: probe_checkpoint_continuation(checkpoint_path, "A", ""),
                lambda: probe_checkpoint_continuation(
                    checkpoint_path,
                    "A",
                    "B",
                    top_k=0,
                ),
                lambda: probe_checkpoint_continuation(
                    checkpoint_path,
                    "123456789",
                    "0",
                ),
            )
            for call in invalid_calls:
                with self.subTest(call=call):
                    with self.assertRaises(ValueError):
                        call()
            with self.assertRaises(TypeError):
                probe_checkpoint_continuation(checkpoint_path, 1, "B")
            with self.assertRaises(TypeError):
                probe_checkpoint_continuation(checkpoint_path, "A", 2)
            with self.assertRaises(TypeError):
                format_continuation_probe(None)

    def test_probe_is_repeatable_and_does_not_modify_the_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            before = checkpoint_path.read_bytes()
            first = probe_checkpoint_continuation(checkpoint_path, "A", "BC")
            second = probe_checkpoint_continuation(checkpoint_path, "A", "BC")
            after = checkpoint_path.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
