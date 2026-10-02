"""Checkpoint-backed, terminal-safe one-shot text generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, TypeAlias

import numpy as np

from .byte_data import decode_utf8, encode_utf8
from .checkpoint import load_checkpoint
from .sampling import GenerationResult, generate_tokens


ClockFunction: TypeAlias = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CheckpointGenerationResult:
    """Decoded generation, timing, checkpoint step, and raw token result."""

    checkpoint_path: Path
    checkpoint_step: int
    context_length: int
    prompt_text: str
    generated_text: str
    full_text: str
    generation: GenerationResult
    elapsed_seconds: float
    tokens_per_second: float


def _validated_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def generate_from_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    max_new_tokens: int,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 1337,
    clock: ClockFunction = time.perf_counter,
) -> CheckpointGenerationResult:
    """Load a checkpoint and time one bounded autoregressive generation."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    resolved_seed = _validated_seed(seed)
    if not callable(clock):
        raise TypeError("clock must be callable")

    source = Path(checkpoint_path)
    loaded = load_checkpoint(source)
    prompt_tokens = encode_utf8(prompt).astype(np.int64, copy=False)
    rng = None if greedy else np.random.default_rng(resolved_seed)
    start_time = float(clock())
    if not np.isfinite(start_time):
        raise ValueError("clock must return finite values")
    generation = generate_tokens(
        prompt_tokens,
        loaded.parameters,
        loaded.config,
        max_new_tokens=max_new_tokens,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        rng=rng,
    )
    elapsed_seconds = float(clock()) - start_time
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError("clock must advance with finite values")
    generated_count = int(generation.generated_tokens.size)
    safe_elapsed = max(elapsed_seconds, np.finfo(np.float64).eps)
    tokens_per_second = generated_count / safe_elapsed if generated_count else 0.0
    return CheckpointGenerationResult(
        checkpoint_path=source,
        checkpoint_step=loaded.optimizer_state.step,
        context_length=loaded.config.context_length,
        prompt_text=prompt,
        generated_text=decode_utf8(generation.generated_tokens),
        full_text=decode_utf8(generation.token_ids),
        generation=generation,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=tokens_per_second,
    )


def format_generation_result(result: CheckpointGenerationResult) -> str:
    """Return terminal-safe one-line representations plus generation metrics."""

    if not isinstance(result, CheckpointGenerationResult):
        raise TypeError("result must be a CheckpointGenerationResult")
    generated_count = int(result.generation.generated_tokens.size)
    prompt_count = int(result.generation.prompt_tokens.size)
    return "\n".join(
        (
            f"checkpoint={result.checkpoint_path} step={result.checkpoint_step:06d} "
            f"context={result.context_length}",
            f"prompt={result.prompt_text!r} prompt_tokens={prompt_count}",
            f"generated={result.generated_text!r}",
            f"full_text={result.full_text!r}",
            f"stop_reason={result.generation.stop_reason} "
            f"generated_tokens={generated_count} "
            f"tok/s={result.tokens_per_second:.1f} "
            f"elapsed={result.elapsed_seconds:.4f}s",
        )
    )
