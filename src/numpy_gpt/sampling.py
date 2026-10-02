"""Deterministic greedy and stochastic autoregressive byte-token sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .model import LanguageModelParameters, language_model_forward


FloatArray: TypeAlias = NDArray[np.floating]
IntegerArray: TypeAlias = NDArray[np.integer]
StopReason: TypeAlias = Literal["max_new_tokens", "context_limit"]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Copied prompt/generated tokens plus the reason decoding stopped."""

    prompt_tokens: IntegerArray
    generated_tokens: IntegerArray
    token_ids: IntegerArray
    stop_reason: StopReason


def _positive_temperature(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("temperature must be a real number")
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("temperature must be a real number") from error
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return temperature


def _validated_top_k(top_k: int | None, vocabulary_size: int) -> int:
    if top_k is None:
        return vocabulary_size
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer or None")
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError(f"top_k must be in [1, {vocabulary_size}]")
    return top_k


def select_next_token(
    logits: FloatArray,
    *,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Select one token from one-dimensional finite logits."""

    scores = np.asarray(logits)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("logits must be a nonempty one-dimensional array")
    if not np.issubdtype(scores.dtype, np.floating):
        raise TypeError("logits must have a floating-point dtype")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits must contain only finite values")
    if not isinstance(greedy, (bool, np.bool_)):
        raise TypeError("greedy must be a boolean")
    resolved_temperature = _positive_temperature(temperature)
    resolved_top_k = _validated_top_k(top_k, scores.size)

    if greedy:
        return int(np.argmax(scores))
    if not isinstance(rng, np.random.Generator):
        raise TypeError("stochastic sampling requires a numpy.random.Generator")

    token_indices = np.arange(scores.size)
    ranked_indices = np.lexsort((token_indices, -scores.astype(np.float64)))
    candidate_indices = ranked_indices[:resolved_top_k]
    candidate_scores = scores[candidate_indices].astype(np.float64, copy=False)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        shifted = (candidate_scores - np.max(candidate_scores)) / resolved_temperature
        weights = np.exp(shifted)
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise FloatingPointError("sampling probabilities could not be normalized")
    probabilities = weights / weight_sum
    return int(rng.choice(candidate_indices, p=probabilities))


def _validated_prompt(
    prompt_token_ids: IntegerArray,
    config: ModelConfig,
) -> IntegerArray:
    prompt = np.asarray(prompt_token_ids)
    if prompt.ndim != 1 or prompt.size == 0:
        raise ValueError("prompt_token_ids must be a nonempty one-dimensional array")
    if not np.issubdtype(prompt.dtype, np.integer):
        raise TypeError("prompt_token_ids must contain integers")
    if prompt.size > config.context_length:
        raise ValueError("prompt length exceeds the configured context length")
    if np.any(prompt < 0) or np.any(prompt >= config.vocab_size):
        raise ValueError(
            f"prompt_token_ids values must be in [0, {config.vocab_size})"
        )
    return prompt.astype(np.int64, copy=True)


def generate_tokens(
    prompt_token_ids: IntegerArray,
    parameters: LanguageModelParameters,
    config: ModelConfig,
    *,
    max_new_tokens: int,
    greedy: bool = False,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
) -> GenerationResult:
    """Generate bytes by recomputing the complete visible context each step."""

    prompt = _validated_prompt(prompt_token_ids, config)
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    _positive_temperature(temperature)
    _validated_top_k(top_k, config.vocab_size)
    if not isinstance(greedy, (bool, np.bool_)):
        raise TypeError("greedy must be a boolean")
    if not greedy and not isinstance(rng, np.random.Generator):
        raise TypeError("stochastic generation requires a numpy.random.Generator")

    available_tokens = config.context_length - prompt.size
    tokens_to_generate = min(max_new_tokens, available_tokens)
    token_ids = prompt.copy()
    generated: list[int] = []
    for _ in range(tokens_to_generate):
        logits, _ = language_model_forward(token_ids[None, :], parameters, config)
        next_token = select_next_token(
            logits[0, -1],
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        generated.append(next_token)
        token_ids = np.append(token_ids, next_token)

    stop_reason: StopReason = "max_new_tokens"
    if tokens_to_generate < max_new_tokens:
        stop_reason = "context_limit"
    return GenerationResult(
        prompt_tokens=prompt,
        generated_tokens=np.asarray(generated, dtype=np.int64),
        token_ids=token_ids.astype(np.int64, copy=False),
        stop_reason=stop_reason,
    )
