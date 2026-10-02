"""Tests for byte encoding, split integrity, and deterministic batching."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    decode_utf8,
    encode_utf8,
    sample_next_token_batch,
    split_byte_tokens,
)


class ByteDataTests(unittest.TestCase):
    def test_utf8_text_round_trips_through_the_fixed_byte_vocabulary(self) -> None:
        text = "Yuvika / Yuzu — नमस्ते 🌱"
        tokens = encode_utf8(text)
        self.assertEqual(tokens.dtype, np.dtype(np.uint8))
        self.assertTrue(tokens.flags.writeable)
        np.testing.assert_array_equal(
            tokens,
            np.frombuffer(text.encode("utf-8"), dtype=np.uint8),
        )
        self.assertEqual(decode_utf8(tokens, errors="strict"), text)

    def test_generated_invalid_utf8_is_visible_and_bad_ids_are_rejected(self) -> None:
        tokens = np.array([0xFF, ord("A")], dtype=np.int64)
        self.assertEqual(decode_utf8(tokens), "\ufffdA")
        self.assertEqual(decode_utf8(tokens, errors="ignore"), "A")

        cases = (
            (np.array([-1], dtype=np.int64), ValueError),
            (np.array([256], dtype=np.int64), ValueError),
            (np.array([1.5], dtype=np.float64), TypeError),
            (np.array([[65]], dtype=np.int64), ValueError),
        )
        for values, error_type in cases:
            with self.subTest(values=values):
                with self.assertRaises(error_type):
                    decode_utf8(values)
        with self.assertRaisesRegex(ValueError, "errors"):
            decode_utf8(np.array([65]), errors="invented")

    def test_train_validation_split_is_contiguous_disjoint_and_copied(self) -> None:
        tokens = np.arange(100, dtype=np.uint8)
        split = split_byte_tokens(
            tokens,
            validation_fraction=0.2,
            context_length=8,
        )
        self.assertEqual(split.split_index, 80)
        np.testing.assert_array_equal(split.train, tokens[:80])
        np.testing.assert_array_equal(split.validation, tokens[80:])
        np.testing.assert_array_equal(
            np.concatenate((split.train, split.validation)),
            tokens,
        )
        self.assertFalse(np.shares_memory(split.train, tokens))
        self.assertFalse(np.shares_memory(split.validation, tokens))
        self.assertEqual(int(split.train[-1]) + 1, int(split.validation[0]))

    def test_batch_inputs_and_targets_are_exactly_one_byte_apart(self) -> None:
        tokens = np.arange(30, dtype=np.uint8)
        batch = sample_next_token_batch(
            tokens,
            batch_size=7,
            context_length=4,
            rng=np.random.default_rng(229),
        )
        self.assertEqual(batch.inputs.shape, (7, 4))
        self.assertEqual(batch.targets.shape, (7, 4))
        self.assertEqual(batch.inputs.dtype, np.dtype(np.int64))
        self.assertEqual(batch.targets.dtype, np.dtype(np.int64))
        for row, start in enumerate(batch.start_indices):
            start_index = int(start)
            np.testing.assert_array_equal(
                batch.inputs[row],
                tokens[start_index : start_index + 4],
            )
            np.testing.assert_array_equal(
                batch.targets[row],
                tokens[start_index + 1 : start_index + 5],
            )
        self.assertLessEqual(int(np.max(batch.start_indices)), 25)

    def test_restored_rng_state_reproduces_the_exact_next_batch(self) -> None:
        tokens = np.arange(80, dtype=np.uint8)
        original_rng = np.random.default_rng(233)
        sample_next_token_batch(
            tokens,
            batch_size=3,
            context_length=6,
            rng=original_rng,
        )
        saved_state = copy.deepcopy(original_rng.bit_generator.state)
        expected = sample_next_token_batch(
            tokens,
            batch_size=5,
            context_length=6,
            rng=original_rng,
        )

        restored_rng = np.random.default_rng()
        restored_rng.bit_generator.state = saved_state
        actual = sample_next_token_batch(
            tokens,
            batch_size=5,
            context_length=6,
            rng=restored_rng,
        )
        np.testing.assert_array_equal(actual.start_indices, expected.start_indices)
        np.testing.assert_array_equal(actual.inputs, expected.inputs)
        np.testing.assert_array_equal(actual.targets, expected.targets)

    def test_invalid_split_and_batch_contracts_are_rejected(self) -> None:
        valid = np.arange(40, dtype=np.uint8)
        split_cases = (
            ({"validation_fraction": 0.0, "context_length": 4}, ValueError),
            ({"validation_fraction": 1.0, "context_length": 4}, ValueError),
            ({"validation_fraction": 0.1, "context_length": 10}, ValueError),
            ({"validation_fraction": 0.2, "context_length": 0}, ValueError),
        )
        for keyword_arguments, error_type in split_cases:
            with self.subTest(split=keyword_arguments):
                with self.assertRaises(error_type):
                    split_byte_tokens(valid, **keyword_arguments)

        batch_cases = (
            ({"batch_size": 0, "context_length": 4}, ValueError),
            ({"batch_size": 2, "context_length": 0}, ValueError),
            ({"batch_size": 2, "context_length": 40}, ValueError),
        )
        for keyword_arguments, error_type in batch_cases:
            with self.subTest(batch=keyword_arguments):
                with self.assertRaises(error_type):
                    sample_next_token_batch(
                        valid,
                        rng=np.random.default_rng(239),
                        **keyword_arguments,
                    )
        with self.assertRaisesRegex(TypeError, "Generator"):
            sample_next_token_batch(
                valid,
                batch_size=2,
                context_length=4,
                rng=None,
            )


if __name__ == "__main__":
    unittest.main()
