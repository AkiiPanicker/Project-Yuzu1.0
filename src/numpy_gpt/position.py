"""Rotary position embeddings with an explicit inverse-rotation backward pass."""

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


def _rope_cosine_sine(
    sequence_length: int,
    head_dimension: int,
    *,
    base: float,
    position_offset: int,
    dtype: np.dtype,
) -> tuple[FloatArray, FloatArray]:
    pair_dimensions = np.arange(0, head_dimension, 2, dtype=np.float64)
    inverse_frequencies = np.power(base, -pair_dimensions / head_dimension)
    positions = np.arange(
        position_offset,
        position_offset + sequence_length,
        dtype=np.float64,
    )
    angles = positions[:, None] * inverse_frequencies[None, :]
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


@dataclass(frozen=True, slots=True)
class RoPECache:
    """Trigonometric rotations and input shape needed for RoPE backward."""

    cosine: FloatArray
    sine: FloatArray
    input_shape: tuple[int, ...]


def rope_forward(
    inputs: FloatArray,
    *,
    base: float = 10_000.0,
    position_offset: int = 0,
) -> tuple[FloatArray, RoPECache]:
    """Rotate adjacent feature pairs using positions on the penultimate axis.

    Expected shapes include ``(batch, sequence, head_dimension)`` and
    ``(batch, heads, sequence, head_dimension)``. The final dimension must be
    even because each adjacent pair forms one two-dimensional rotation plane.
    """

    x = _require_floating(inputs, name="inputs")
    if x.ndim < 2:
        raise ValueError("inputs must include sequence and feature dimensions")
    if x.shape[-1] % 2 != 0:
        raise ValueError("the final feature dimension must be even")
    if not np.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive")
    if (
        not isinstance(position_offset, int)
        or isinstance(position_offset, bool)
        or position_offset < 0
    ):
        raise ValueError("position_offset must be a nonnegative integer")

    cosine, sine = _rope_cosine_sine(
        x.shape[-2],
        x.shape[-1],
        base=base,
        position_offset=position_offset,
        dtype=x.dtype,
    )
    even = x[..., 0::2]
    odd = x[..., 1::2]
    output = np.empty_like(x)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output, RoPECache(cosine=cosine, sine=sine, input_shape=x.shape)


def rope_backward(
    gradient_output: FloatArray,
    cache: RoPECache,
) -> FloatArray:
    """Apply the inverse rotation to obtain the RoPE input gradient."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.input_shape:
        raise ValueError(
            f"gradient_output must have shape {cache.input_shape}, got {gradient.shape}"
        )
    gradient_even = gradient[..., 0::2]
    gradient_odd = gradient[..., 1::2]
    gradient_inputs = np.empty_like(gradient)
    gradient_inputs[..., 0::2] = (
        gradient_even * cache.cosine + gradient_odd * cache.sine
    )
    gradient_inputs[..., 1::2] = (
        -gradient_even * cache.sine + gradient_odd * cache.cosine
    )
    return gradient_inputs

