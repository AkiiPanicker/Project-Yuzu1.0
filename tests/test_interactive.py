"""Tests for the stateless interactive checkpoint diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy_gpt.interactive as interactive_module  # noqa: E402
from numpy_gpt import (  # noqa: E402
    GenerationResult,
    InteractiveTurnResult,
    ModelConfig,
    encode_utf8,
    format_interactive_turn,
    generate_tokens,
    initialize_adamw_state,
    initialize_parameters,
    load_checkpoint,
    named_parameters,
    run_checkpoint_diagnostic,
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
        seed=307,
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


def _reader(*responses: str | BaseException):
    iterator = iter(responses)

    def read_prompt(_: str) -> str:
        response = next(iterator)
        if isinstance(response, BaseException):
            raise response
        return response

    return read_prompt


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


class InteractiveCheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_once_logs_turns_and_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            log_path = Path(directory) / "interactive.jsonl"
            original_bytes = checkpoint_path.read_bytes()
            output: list[str] = []
            with mock.patch.object(
                interactive_module,
                "load_checkpoint",
                wraps=interactive_module.load_checkpoint,
            ) as loader:
                result = run_checkpoint_diagnostic(
                    checkpoint_path,
                    max_new_tokens=1,
                    greedy=True,
                    read_prompt=_reader("A", "B", "/exit"),
                    write_line=output.append,
                    clock=_clock(1.0, 2.0, 3.0, 5.0),
                    log_path=log_path,
                )
            records = [json.loads(line) for line in log_path.read_text().splitlines()]

            self.assertEqual(loader.call_count, 1)
            self.assertEqual(checkpoint_path.read_bytes(), original_bytes)
            self.assertEqual(result.exit_reason, "command")
            self.assertEqual(len(result.turns), 2)
            self.assertEqual(result.turns[0].tokens_per_second, 1.0)
            self.assertEqual(result.turns[1].tokens_per_second, 0.5)
            self.assertEqual(
                [record["event"] for record in records],
                ["session_start", "turn", "turn", "session_end"],
            )
            self.assertEqual(len({record["session_id"] for record in records}), 1)
            self.assertIn("not a chatbot", output[0])

    def test_greedy_turn_matches_direct_generation_and_fake_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            loaded = load_checkpoint(checkpoint_path)
            expected = generate_tokens(
                encode_utf8("A"),
                loaded.parameters,
                loaded.config,
                max_new_tokens=3,
                greedy=True,
            )
            result = run_checkpoint_diagnostic(
                checkpoint_path,
                max_new_tokens=3,
                greedy=True,
                read_prompt=_reader("A", "/exit"),
                write_line=lambda _: None,
                clock=_clock(10.0, 12.0),
            )

        turn = result.turns[0]
        np.testing.assert_array_equal(turn.generation.token_ids, expected.token_ids)
        self.assertEqual(turn.elapsed_seconds, 2.0)
        self.assertEqual(turn.tokens_per_second, 1.5)

    def test_stochastic_rng_advances_across_turns_and_repeats_by_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            loaded = load_checkpoint(checkpoint_path)
            expected_rng = np.random.default_rng(311)
            expected = tuple(
                generate_tokens(
                    encode_utf8(prompt),
                    loaded.parameters,
                    loaded.config,
                    max_new_tokens=2,
                    temperature=0.8,
                    top_k=16,
                    rng=expected_rng,
                )
                for prompt in ("A", "B")
            )

            def run_session():
                return run_checkpoint_diagnostic(
                    checkpoint_path,
                    max_new_tokens=2,
                    temperature=0.8,
                    top_k=16,
                    seed=311,
                    read_prompt=_reader("A", "B", "/exit"),
                    write_line=lambda _: None,
                )

            first = run_session()
            second = run_session()

        for turn, direct in zip(first.turns, expected, strict=True):
            np.testing.assert_array_equal(
                turn.generation.generated_tokens,
                direct.generated_tokens,
            )
        for first_turn, second_turn in zip(first.turns, second.turns, strict=True):
            np.testing.assert_array_equal(
                first_turn.generation.generated_tokens,
                second_turn.generation.generated_tokens,
            )

    def test_turns_are_stateless_and_do_not_inherit_prior_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_checkpoint_diagnostic(
                _checkpoint(directory),
                max_new_tokens=1,
                greedy=True,
                read_prompt=_reader("first", "B", "/exit"),
                write_line=lambda _: None,
            )

        np.testing.assert_array_equal(
            result.turns[0].generation.prompt_tokens,
            encode_utf8("first"),
        )
        np.testing.assert_array_equal(
            result.turns[1].generation.prompt_tokens,
            encode_utf8("B"),
        )
        self.assertEqual(result.turns[1].prompt_text, "B")

    def test_help_exit_eof_and_interrupt_do_not_generate(self) -> None:
        cases = (
            ((_reader("/help", "/exit")), "command", "help:"),
            ((_reader(EOFError())), "eof", "session_end reason=eof"),
            ((_reader(KeyboardInterrupt())), "interrupt", "session_end reason=interrupt"),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            for reader, expected_reason, expected_output in cases:
                with self.subTest(reason=expected_reason):
                    output: list[str] = []
                    with mock.patch.object(interactive_module, "generate_tokens") as generate:
                        result = run_checkpoint_diagnostic(
                            checkpoint_path,
                            max_new_tokens=1,
                            greedy=True,
                            read_prompt=reader,
                            write_line=output.append,
                        )
                    generate.assert_not_called()
                    self.assertEqual(result.exit_reason, expected_reason)
                    self.assertTrue(
                        any(expected_output in line for line in output),
                        output,
                    )

    def test_invalid_prompts_recover_before_a_valid_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output: list[str] = []
            result = run_checkpoint_diagnostic(
                _checkpoint(directory),
                max_new_tokens=1,
                greedy=True,
                read_prompt=_reader("", "123456789", "A", "/exit"),
                write_line=output.append,
            )

        self.assertEqual(len(result.turns), 1)
        self.assertEqual(result.turns[0].prompt_text, "A")
        self.assertTrue(any("error=empty_prompt" in line for line in output))
        self.assertTrue(any("error=prompt_too_long" in line for line in output))

    def test_full_context_uses_utf8_byte_count_and_generates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_checkpoint_diagnostic(
                _checkpoint(directory),
                max_new_tokens=5,
                greedy=True,
                read_prompt=_reader("é123456", "/exit"),
                write_line=lambda _: None,
                clock=_clock(4.0, 4.0),
            )

        turn = result.turns[0]
        self.assertEqual(turn.generation.prompt_tokens.size, 8)
        self.assertEqual(turn.generation.generated_tokens.size, 0)
        self.assertEqual(turn.generation.stop_reason, "context_limit")
        self.assertEqual(turn.tokens_per_second, 0.0)

    def test_formatter_escapes_control_bytes_and_options_are_validated(self) -> None:
        generation = GenerationResult(
            prompt_tokens=np.array([65], dtype=np.int64),
            generated_tokens=np.array([27, 10], dtype=np.int64),
            token_ids=np.array([65, 27, 10], dtype=np.int64),
            stop_reason="max_new_tokens",
        )
        turn = InteractiveTurnResult(
            turn_index=1,
            prompt_text="A",
            generated_text="\x1b\n",
            full_text="A\x1b\n",
            generation=generation,
            elapsed_seconds=0.5,
            tokens_per_second=4.0,
        )
        formatted = format_interactive_turn(turn)

        self.assertIn("generated='\\x1b\\n'", formatted)
        self.assertNotIn("\x1b", formatted)
        self.assertIn("generated_token_ids=[27, 10]", formatted)
        with self.assertRaises(TypeError):
            format_interactive_turn(None)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = _checkpoint(directory)
            base = {
                "checkpoint_path": checkpoint_path,
                "max_new_tokens": 1,
                "greedy": True,
                "read_prompt": _reader("/exit"),
                "write_line": lambda _: None,
            }
            invalid_cases = (
                ({"max_new_tokens": 0}, ValueError),
                ({"greedy": "yes"}, TypeError),
                ({"temperature": 0.0}, ValueError),
                ({"top_k": 257}, ValueError),
                ({"seed": -1}, ValueError),
                ({"read_prompt": None}, TypeError),
                ({"write_line": None}, TypeError),
                ({"clock": None}, TypeError),
                ({"log_path": checkpoint_path}, ValueError),
            )
            for overrides, error_type in invalid_cases:
                with self.subTest(overrides=overrides):
                    options = {**base, **overrides}
                    with self.assertRaises(error_type):
                        run_checkpoint_diagnostic(**options)


if __name__ == "__main__":
    unittest.main()
