"""Full bias-free multi-head causal self-attention composed from verified parts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .attention import (
    CausalAttentionCache,
    causal_attention_backward,
    causal_attention_forward,
)
from .layers import LinearCache, linear_backward, linear_forward
from .position import RoPECache, rope_backward, rope_forward


FloatArray: TypeAlias = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class MultiHeadAttentionWeightGradients:
    """Gradients for the four bias-free attention projection matrices."""

    query: FloatArray
    key: FloatArray
    value: FloatArray
    output: FloatArray


@dataclass(frozen=True, slots=True)
class MultiHeadAttentionCache:
    """Composition caches needed for multi-head attention backward."""

    query_linear: LinearCache
    key_linear: LinearCache
    value_linear: LinearCache
    query_rope: RoPECache
    key_rope: RoPECache
    attention: CausalAttentionCache
    output_linear: LinearCache
    n_heads: int
    model_width: int


def _split_heads(projected: FloatArray, n_heads: int) -> FloatArray:
    sequence_length = projected.shape[-2]
    model_width = projected.shape[-1]
    head_dimension = model_width // n_heads
    split_shape = projected.shape[:-2] + (
        sequence_length,
        n_heads,
        head_dimension,
    )
    return np.swapaxes(projected.reshape(split_shape), -3, -2)


def _merge_heads(heads: FloatArray) -> FloatArray:
    sequence_length = heads.shape[-2]
    model_width = heads.shape[-3] * heads.shape[-1]
    merged_shape = heads.shape[:-3] + (sequence_length, model_width)
    return np.swapaxes(heads, -3, -2).reshape(merged_shape)


def _validate_configuration(
    inputs: FloatArray,
    weights: tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    n_heads: int,
) -> tuple[int, int]:
    x = np.asarray(inputs)
    if x.ndim < 2:
        raise ValueError("inputs must include sequence and model dimensions")
    model_width = x.shape[-1]
    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads <= 0:
        raise ValueError("n_heads must be a positive integer")
    if model_width % n_heads != 0:
        raise ValueError("model width must be divisible by n_heads")
    head_dimension = model_width // n_heads
    if head_dimension % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    expected_weight_shape = (model_width, model_width)
    weight_names = ("query", "key", "value", "output")
    for name, weight in zip(weight_names, weights, strict=True):
        if np.asarray(weight).shape != expected_weight_shape:
            raise ValueError(
                f"{name} weight must have shape {expected_weight_shape}, "
                f"got {np.asarray(weight).shape}"
            )
    return model_width, head_dimension


def multi_head_attention_forward(
    inputs: FloatArray,
    query_weight: FloatArray,
    key_weight: FloatArray,
    value_weight: FloatArray,
    output_weight: FloatArray,
    *,
    n_heads: int,
    rope_base: float = 10_000.0,
    position_offset: int = 0,
) -> tuple[FloatArray, MultiHeadAttentionCache]:
    """Project, rotate, attend, merge, and project a self-attention input."""

    model_width, _ = _validate_configuration(
        inputs,
        (query_weight, key_weight, value_weight, output_weight),
        n_heads,
    )
    query, query_linear = linear_forward(inputs, query_weight)
    key, key_linear = linear_forward(inputs, key_weight)
    value, value_linear = linear_forward(inputs, value_weight)

    query_heads = _split_heads(query, n_heads)
    key_heads = _split_heads(key, n_heads)
    value_heads = _split_heads(value, n_heads)
    rotated_query, query_rope = rope_forward(
        query_heads,
        base=rope_base,
        position_offset=position_offset,
    )
    rotated_key, key_rope = rope_forward(
        key_heads,
        base=rope_base,
        position_offset=position_offset,
    )
    attended_heads, attention = causal_attention_forward(
        rotated_query,
        rotated_key,
        value_heads,
    )
    merged = _merge_heads(attended_heads)
    output, output_linear = linear_forward(merged, output_weight)
    cache = MultiHeadAttentionCache(
        query_linear=query_linear,
        key_linear=key_linear,
        value_linear=value_linear,
        query_rope=query_rope,
        key_rope=key_rope,
        attention=attention,
        output_linear=output_linear,
        n_heads=n_heads,
        model_width=model_width,
    )
    return output, cache


def multi_head_attention_backward(
    gradient_output: FloatArray,
    cache: MultiHeadAttentionCache,
) -> tuple[FloatArray, MultiHeadAttentionWeightGradients]:
    """Backpropagate through every multi-head attention composition step."""

    gradient_merged, gradient_output_weight, _ = linear_backward(
        gradient_output,
        cache.output_linear,
    )
    gradient_attended_heads = _split_heads(gradient_merged, cache.n_heads)
    gradient_rotated_query, gradient_rotated_key, gradient_value_heads = (
        causal_attention_backward(gradient_attended_heads, cache.attention)
    )
    gradient_query_heads = rope_backward(
        gradient_rotated_query,
        cache.query_rope,
    )
    gradient_key_heads = rope_backward(
        gradient_rotated_key,
        cache.key_rope,
    )
    gradient_query = _merge_heads(gradient_query_heads)
    gradient_key = _merge_heads(gradient_key_heads)
    gradient_value = _merge_heads(gradient_value_heads)

    gradient_input_query, gradient_query_weight, _ = linear_backward(
        gradient_query,
        cache.query_linear,
    )
    gradient_input_key, gradient_key_weight, _ = linear_backward(
        gradient_key,
        cache.key_linear,
    )
    gradient_input_value, gradient_value_weight, _ = linear_backward(
        gradient_value,
        cache.value_linear,
    )
    gradient_inputs = (
        gradient_input_query + gradient_input_key + gradient_input_value
    )
    gradients = MultiHeadAttentionWeightGradients(
        query=gradient_query_weight,
        key=gradient_key_weight,
        value=gradient_value_weight,
        output=gradient_output_weight,
    )
    return gradient_inputs, gradients

