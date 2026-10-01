"""Trainable neural-network primitives with explicit forward/backward caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]
IntegerArray: TypeAlias = NDArray[np.integer]


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
class LinearCache:
    """Values needed to differentiate an affine projection."""

    inputs: FloatArray
    weight: FloatArray
    has_bias: bool


def linear_forward(
    inputs: FloatArray,
    weight: FloatArray,
    bias: FloatArray | None = None,
) -> tuple[FloatArray, LinearCache]:
    """Apply ``inputs @ weight + bias`` over the final input dimension."""

    x = _require_floating(inputs, name="inputs")
    w = _require_floating(weight, name="weight")
    if x.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if w.ndim != 2:
        raise ValueError(f"weight must be two-dimensional, got shape {w.shape}")
    if x.shape[-1] != w.shape[0]:
        raise ValueError(
            f"input width {x.shape[-1]} does not match weight width {w.shape[0]}"
        )

    output = x @ w
    has_bias = bias is not None
    if bias is not None:
        b = _require_floating(bias, name="bias")
        if b.shape != (w.shape[1],):
            raise ValueError(
                f"bias must have shape {(w.shape[1],)}, got shape {b.shape}"
            )
        output = output + b
    return output, LinearCache(inputs=x, weight=w, has_bias=has_bias)


def linear_backward(
    gradient_output: FloatArray,
    cache: LinearCache,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Differentiate a linear projection with respect to input, weight, and bias."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    expected_shape = cache.inputs.shape[:-1] + (cache.weight.shape[1],)
    if gradient.shape != expected_shape:
        raise ValueError(
            f"gradient_output must have shape {expected_shape}, got {gradient.shape}"
        )

    input_width, output_width = cache.weight.shape
    flat_inputs = cache.inputs.reshape(-1, input_width)
    flat_gradient = gradient.reshape(-1, output_width)
    gradient_inputs = gradient @ cache.weight.T
    gradient_weight = flat_inputs.T @ flat_gradient
    gradient_bias = np.sum(flat_gradient, axis=0) if cache.has_bias else None
    return gradient_inputs, gradient_weight, gradient_bias


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    """Discrete indices and table metadata needed for embedding backward."""

    token_ids: IntegerArray
    weight_shape: tuple[int, int]
    weight_dtype: np.dtype


def embedding_forward(
    token_ids: IntegerArray,
    weight: FloatArray,
) -> tuple[FloatArray, EmbeddingCache]:
    """Look up token vectors from a ``(vocabulary, width)`` embedding table."""

    ids = np.asarray(token_ids)
    table = _require_floating(weight, name="weight")
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"token_ids must have an integer dtype, got {ids.dtype}")
    if ids.size == 0:
        raise ValueError("token_ids must not be empty")
    if table.ndim != 2:
        raise ValueError(f"weight must be two-dimensional, got shape {table.shape}")
    if np.any(ids < 0) or np.any(ids >= table.shape[0]):
        raise ValueError(f"token_ids must be in [0, {table.shape[0]})")

    output = table[ids]
    cache = EmbeddingCache(
        token_ids=ids.copy(),
        weight_shape=table.shape,
        weight_dtype=table.dtype,
    )
    return output, cache


def embedding_backward(
    gradient_output: FloatArray,
    cache: EmbeddingCache,
) -> FloatArray:
    """Accumulate output gradients into the rows of an embedding table."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    expected_shape = cache.token_ids.shape + (cache.weight_shape[1],)
    if gradient.shape != expected_shape:
        raise ValueError(
            f"gradient_output must have shape {expected_shape}, got {gradient.shape}"
        )

    gradient_weight = np.zeros(cache.weight_shape, dtype=cache.weight_dtype)
    np.add.at(
        gradient_weight,
        cache.token_ids.reshape(-1),
        gradient.reshape(-1, cache.weight_shape[1]),
    )
    return gradient_weight

