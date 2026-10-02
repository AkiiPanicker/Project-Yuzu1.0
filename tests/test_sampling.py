"""Tests for greedy, stochastic, top-k, and autoregressive token generation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    ModelConfig,
    generate_tokens,
    initialize_parameters,
    language_model_forward,
    select_next_token,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=8,
        context_length=5,
        d_model=4,
        n_layers=1,
        n_heads=1,
        d_ff=6,
        seed=269,
    )


class SamplingTests(unittest.TestCase):
    def test_greedy_selection_uses_the_first_maximum_without_rng(self) -> None:
        logits = np.array([-1.0, 3.0, 3.0, 0.5], dtype=np.float64)
        self.assertEqual(select_next_token(logits, greedy=True), 1)
        self.assertEqual(
            select_next_token(
                logits,
                greedy=True,
                temperature=0.25,
                top_k=2,
            ),
            1,
        )

    def test_seeded_stochastic_selection_is_reproducible(self) -> None:
        logits = np.array([-2.0, -0.5, 0.0, 0.5, 1.5], dtype=np.float64)
        first_rng = np.random.default_rng(271)
        second_rng = np.random.default_rng(271)
        first = [
            select_next_token(
                logits,
                temperature=0.7,
                top_k=4,
                rng=first_rng,
            )
            for _ in range(30)
        ]
        second = [
            select_next_token(
                logits,
                temperature=0.7,
                top_k=4,
                rng=second_rng,
            )
            for _ in range(30)
        ]
        self.assertEqual(first, second)

    def test_top_k_sampling_never_selects_an_excluded_token(self) -> None:
        logits = np.arange(6, dtype=np.float64)
        rng = np.random.default_rng(277)
        selected = {
            select_next_token(logits, top_k=2, rng=rng) for _ in range(200)
        }
        self.assertTrue(selected)
        self.assertLessEqual(selected, {4, 5})
        for _ in range(20):
            self.assertEqual(select_next_token(logits, top_k=1, rng=rng), 5)

    def test_greedy_generation_matches_manual_full_context_recomputation(self) -> None:
        config = _model_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        prompt = np.array([1, 3], dtype=np.int64)
        expected = prompt.tolist()
        for _ in range(2):
            logits, _ = language_model_forward(
                np.asarray(expected, dtype=np.int64)[None, :],
                parameters,
                config,
            )
            expected.append(int(np.argmax(logits[0, -1])))

        result = generate_tokens(
            prompt,
            parameters,
            config,
            max_new_tokens=2,
            greedy=True,
        )

        np.testing.assert_array_equal(result.prompt_tokens, prompt)
        np.testing.assert_array_equal(result.token_ids, expected)
        np.testing.assert_array_equal(result.generated_tokens, expected[2:])
        self.assertEqual(result.stop_reason, "max_new_tokens")

    def test_generation_stops_at_context_limit_without_mutating_prompt(self) -> None:
        config = _model_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        prompt = np.array([0, 1, 2, 3], dtype=np.int64)
        before = prompt.copy()

        result = generate_tokens(
            prompt,
            parameters,
            config,
            max_new_tokens=4,
            greedy=True,
        )

        np.testing.assert_array_equal(prompt, before)
        self.assertEqual(result.generated_tokens.size, 1)
        self.assertEqual(result.token_ids.size, config.context_length)
        self.assertEqual(result.stop_reason, "context_limit")

    def test_invalid_sampling_and_generation_contracts_are_rejected(self) -> None:
        valid_logits = np.array([0.0, 1.0], dtype=np.float64)
        selection_cases = (
            ({"logits": np.array([[0.0, 1.0]]), "greedy": True}, ValueError),
            ({"logits": np.array([0, 1]), "greedy": True}, TypeError),
            ({"logits": np.array([0.0, np.nan]), "greedy": True}, ValueError),
            ({"logits": valid_logits, "greedy": True, "temperature": 0.0}, ValueError),
            ({"logits": valid_logits, "greedy": True, "top_k": 0}, ValueError),
            ({"logits": valid_logits}, TypeError),
        )
        for keyword_arguments, error_type in selection_cases:
            with self.subTest(selection=keyword_arguments):
                with self.assertRaises(error_type):
                    select_next_token(**keyword_arguments)

        config = _model_config()
        parameters = initialize_parameters(config, dtype=np.float64)
        generation_cases = (
            (np.array([], dtype=np.int64), 1, ValueError),
            (np.array([8], dtype=np.int64), 1, ValueError),
            (np.array([1.5]), 1, TypeError),
            (np.array([1], dtype=np.int64), 0, ValueError),
        )
        for prompt, max_new_tokens, error_type in generation_cases:
            with self.subTest(prompt=prompt, max_new_tokens=max_new_tokens):
                with self.assertRaises(error_type):
                    generate_tokens(
                        prompt,
                        parameters,
                        config,
                        max_new_tokens=max_new_tokens,
                        greedy=True,
                    )


if __name__ == "__main__":
    unittest.main()
