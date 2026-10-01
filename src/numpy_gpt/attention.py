"""Causal scaled-dot-product self-attention with explicit backward equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]


def _require_floating(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype, got {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class CausalAttentionCache:
    """Values required to differentiate scaled-dot-product attention."""

    query: FloatArray
    key: FloatArray
    value: FloatArray
    probabilities: FloatArray
    causal_mask: NDArray[np.bool_]
    scale: float


def causal_attention_forward(
    query: FloatArray,
    key: FloatArray,
    value: FloatArray,
    *,
    scale: float | None = None,
) -> tuple[FloatArray, CausalAttentionCache]:
    """Apply causal self-attention over the penultimate sequence dimension.

    Query, key, and value must share shape ``(..., sequence, head_dimension)``.
    The leading dimensions may represent a batch, attention heads, or both.
    """

    q = _require_floating(query, name="query")
    k = _require_floating(key, name="key")
    v = _require_floating(value, name="value")
    if q.ndim < 2:
        raise ValueError("query, key, and value must have at least two dimensions")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "query, key, and value shapes must match, "
            f"got {q.shape}, {k.shape}, and {v.shape}"
        )

    attention_scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else float(scale)
    if not np.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("scale must be finite and positive")

    scores = (q @ np.swapaxes(k, -1, -2)) * attention_scale
    sequence_length = q.shape[-2]
    causal_mask = np.tri(sequence_length, sequence_length, dtype=np.bool_)
    masked_scores = np.where(causal_mask, scores, -np.inf)
    row_maximum = np.max(masked_scores, axis=-1, keepdims=True)
    exponentials = np.exp(masked_scores - row_maximum)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    output = probabilities @ v
    cache = CausalAttentionCache(
        query=q,
        key=k,
        value=v,
        probabilities=probabilities,
        causal_mask=causal_mask,
        scale=attention_scale,
    )
    return output, cache


def causal_attention_backward(
    gradient_output: FloatArray,
    cache: CausalAttentionCache,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Differentiate causal attention with respect to query, key, and value."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.query.shape:
        raise ValueError(
            f"gradient_output must have shape {cache.query.shape}, got {gradient.shape}"
        )

    gradient_probabilities = gradient @ np.swapaxes(cache.value, -1, -2)
    gradient_value = np.swapaxes(cache.probabilities, -1, -2) @ gradient

    probability_projection = np.sum(
        gradient_probabilities * cache.probabilities,
        axis=-1,
        keepdims=True,
    )
    gradient_scores = cache.probabilities * (
        gradient_probabilities - probability_projection
    )
    gradient_scores = np.where(cache.causal_mask, gradient_scores, 0.0)

    gradient_query = (gradient_scores @ cache.key) * cache.scale
    gradient_key = (
        np.swapaxes(gradient_scores, -1, -2) @ cache.query
    ) * cache.scale
    return gradient_query, gradient_key, gradient_value

