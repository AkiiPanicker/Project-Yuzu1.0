"""Stateless, terminal-safe interactive inspection of one loaded checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable, Literal, TextIO, TypeAlias
from uuid import uuid4

import numpy as np

from .byte_data import decode_utf8, encode_utf8
from .checkpoint import load_checkpoint
from .sampling import GenerationResult, generate_tokens


PromptReader: TypeAlias = Callable[[str], str]
LineWriter: TypeAlias = Callable[[str], None]
ClockFunction: TypeAlias = Callable[[], float]
ExitReason: TypeAlias = Literal["command", "eof", "interrupt"]


@dataclass(frozen=True, slots=True)
class InteractiveTurnResult:
    """One independent prompt, its bounded continuation, and generation timing."""

    turn_index: int
    prompt_text: str
    generated_text: str
    full_text: str
    generation: GenerationResult
    elapsed_seconds: float
    tokens_per_second: float


@dataclass(frozen=True, slots=True)
class InteractiveSessionResult:
    """Completed diagnostic session metadata and all successful turns."""

    checkpoint_path: Path
    checkpoint_step: int
    context_length: int
    turns: tuple[InteractiveTurnResult, ...]
    exit_reason: ExitReason
    log_path: Path | None


def _validated_options(
    *,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> tuple[bool, float, int | None, int]:
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(greedy, (bool, np.bool_)):
        raise TypeError("greedy must be a boolean")
    if isinstance(temperature, (bool, np.bool_)):
        raise TypeError("temperature must be a real number")
    try:
        resolved_temperature = float(temperature)
    except (TypeError, ValueError) as error:
        raise TypeError("temperature must be a real number") from error
    if not np.isfinite(resolved_temperature) or resolved_temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer or None")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return bool(greedy), resolved_temperature, top_k, seed


def _write_event(handle: TextIO | None, event: str, **fields: object) -> None:
    if handle is None:
        return
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    handle.flush()


def _help_lines(context_length: int) -> tuple[str, ...]:
    return (
        "help: this is a checkpoint diagnostic, not a chatbot.",
        f"help: each independent prompt is limited to {context_length} UTF-8 bytes.",
        "help: no role labels or conversation history are added.",
        "help: commands=/help,/exit",
    )


def format_interactive_turn(result: InteractiveTurnResult) -> str:
    """Render a turn without emitting generated terminal control characters."""

    if not isinstance(result, InteractiveTurnResult):
        raise TypeError("result must be an InteractiveTurnResult")
    generated_count = int(result.generation.generated_tokens.size)
    prompt_count = int(result.generation.prompt_tokens.size)
    generated_ids = result.generation.generated_tokens.tolist()
    return "\n".join(
        (
            f"turn={result.turn_index:04d} prompt={result.prompt_text!r} "
            f"prompt_tokens={prompt_count}",
            f"generated={result.generated_text!r}",
            f"generated_token_ids={generated_ids}",
            f"full_text={result.full_text!r}",
            f"stop_reason={result.generation.stop_reason} "
            f"generated_tokens={generated_count} "
            f"tok/s={result.tokens_per_second:.1f} "
            f"elapsed={result.elapsed_seconds:.4f}s",
        )
    )


def run_checkpoint_diagnostic(
    checkpoint_path: str | Path,
    *,
    max_new_tokens: int,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 1337,
    read_prompt: PromptReader = input,
    write_line: LineWriter = print,
    clock: ClockFunction = time.perf_counter,
    log_path: str | Path | None = None,
) -> InteractiveSessionResult:
    """Run independent prompts against one checkpoint loaded once per session."""

    resolved_greedy, resolved_temperature, resolved_top_k, resolved_seed = (
        _validated_options(
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
    )
    if not callable(read_prompt):
        raise TypeError("read_prompt must be callable")
    if not callable(write_line):
        raise TypeError("write_line must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")

    source = Path(checkpoint_path)
    destination = None if log_path is None else Path(log_path)
    if destination is not None and source.resolve() == destination.resolve():
        raise ValueError("log_path must not overwrite the checkpoint")

    loaded = load_checkpoint(source)
    if resolved_top_k is not None and resolved_top_k > loaded.config.vocab_size:
        raise ValueError(
            f"top_k must be in [1, {loaded.config.vocab_size}] or None"
        )
    rng = None if resolved_greedy else np.random.default_rng(resolved_seed)
    mode = "greedy" if resolved_greedy else "stochastic"
    session_id = uuid4().hex
    turns: list[InteractiveTurnResult] = []
    log_handle: TextIO | None = None

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        log_handle = destination.open("a", encoding="utf-8", newline="\n")

    try:
        _write_event(
            log_handle,
            "session_start",
            session_id=session_id,
            checkpoint=str(source),
            checkpoint_step=loaded.optimizer_state.step,
            context_length=loaded.config.context_length,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            seed=resolved_seed,
        )
        write_line("NumPy GPT interactive checkpoint diagnostic - not a chatbot")
        write_line(
            f"checkpoint={source} step={loaded.optimizer_state.step:06d} "
            f"context={loaded.config.context_length} mode={mode}"
        )
        write_line(f"session_log={destination if destination is not None else 'disabled'}")
        write_line(
            "Each turn is an independent prompt; no conversation history is retained."
        )
        write_line("commands=/help,/exit")

        exit_reason: ExitReason
        while True:
            try:
                prompt_text = read_prompt("prompt> ")
            except EOFError:
                exit_reason = "eof"
                break
            except KeyboardInterrupt:
                exit_reason = "interrupt"
                break

            if not isinstance(prompt_text, str):
                raise TypeError("read_prompt must return a string")
            if prompt_text == "/exit":
                exit_reason = "command"
                break
            if prompt_text == "/help":
                for line in _help_lines(loaded.config.context_length):
                    write_line(line)
                _write_event(log_handle, "help", session_id=session_id)
                continue
            if prompt_text == "":
                message = "error=empty_prompt enter at least one character"
                write_line(message)
                _write_event(
                    log_handle,
                    "input_error",
                    session_id=session_id,
                    reason="empty_prompt",
                )
                continue

            prompt_tokens = encode_utf8(prompt_text).astype(np.int64, copy=False)
            prompt_count = int(prompt_tokens.size)
            if prompt_count > loaded.config.context_length:
                message = (
                    "error=prompt_too_long "
                    f"prompt_tokens={prompt_count} "
                    f"context={loaded.config.context_length}"
                )
                write_line(message)
                _write_event(
                    log_handle,
                    "input_error",
                    session_id=session_id,
                    reason="prompt_too_long",
                    prompt=prompt_text,
                    prompt_tokens=prompt_count,
                    context_length=loaded.config.context_length,
                )
                continue

            start_time = float(clock())
            if not np.isfinite(start_time):
                raise ValueError("clock must return finite values")
            generation = generate_tokens(
                prompt_tokens,
                loaded.parameters,
                loaded.config,
                max_new_tokens=max_new_tokens,
                greedy=resolved_greedy,
                temperature=resolved_temperature,
                top_k=resolved_top_k,
                rng=rng,
            )
            elapsed_seconds = float(clock()) - start_time
            if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
                raise ValueError("clock must advance with finite values")
            generated_count = int(generation.generated_tokens.size)
            safe_elapsed = max(elapsed_seconds, np.finfo(np.float64).eps)
            tokens_per_second = (
                generated_count / safe_elapsed if generated_count else 0.0
            )
            turn = InteractiveTurnResult(
                turn_index=len(turns) + 1,
                prompt_text=prompt_text,
                generated_text=decode_utf8(generation.generated_tokens),
                full_text=decode_utf8(generation.token_ids),
                generation=generation,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
            )
            turns.append(turn)
            write_line(format_interactive_turn(turn))
            _write_event(
                log_handle,
                "turn",
                session_id=session_id,
                turn=turn.turn_index,
                prompt=turn.prompt_text,
                prompt_token_ids=turn.generation.prompt_tokens.tolist(),
                generated=turn.generated_text,
                generated_token_ids=turn.generation.generated_tokens.tolist(),
                full_text=turn.full_text,
                stop_reason=turn.generation.stop_reason,
                elapsed_seconds=turn.elapsed_seconds,
                tokens_per_second=turn.tokens_per_second,
            )

        write_line(f"session_end reason={exit_reason} turns={len(turns)}")
        _write_event(
            log_handle,
            "session_end",
            session_id=session_id,
            reason=exit_reason,
            turns=len(turns),
        )
        return InteractiveSessionResult(
            checkpoint_path=source,
            checkpoint_step=loaded.optimizer_state.step,
            context_length=loaded.config.context_length,
            turns=tuple(turns),
            exit_reason=exit_reason,
            log_path=destination,
        )
    finally:
        if log_handle is not None:
            log_handle.close()
