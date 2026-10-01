"""Numerically stable SiLU and SwiGLU activation primitives."""

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


def stable_sigmoid(inputs: FloatArray) -> FloatArray:
    """Compute sigmoid without overflowing for large-magnitude finite inputs."""

    x = _require_floating(inputs, name="inputs")
    result = np.empty_like(x)
    nonnegative = x >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    result[~nonnegative] = exp_x / (1.0 + exp_x)
    return result


@dataclass(frozen=True, slots=True)
class SiLUCache:
    """Values needed for the SiLU backward pass."""

    inputs: FloatArray
    sigmoid: FloatArray


def silu_forward(inputs: FloatArray) -> tuple[FloatArray, SiLUCache]:
    """Apply the SiLU/Swish activation ``x * sigmoid(x)``."""

    x = _require_floating(inputs, name="inputs")
    sigmoid = stable_sigmoid(x)
    output = x * sigmoid
    return output, SiLUCache(inputs=x, sigmoid=sigmoid)


def silu_backward(
    gradient_output: FloatArray,
    cache: SiLUCache,
) -> FloatArray:
    """Differentiate SiLU with respect to its input."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.inputs.shape:
        raise ValueError(
            f"gradient_output must have shape {cache.inputs.shape}, got {gradient.shape}"
        )
    derivative = cache.sigmoid * (
        1.0 + cache.inputs * (1.0 - cache.sigmoid)
    )
    return gradient * derivative


@dataclass(frozen=True, slots=True)
class SwiGLUCache:
    """Values needed to differentiate the elementwise SwiGLU core."""

    silu_cache: SiLUCache
    activated_gate: FloatArray
    value: FloatArray


def swiglu_forward(
    gate: FloatArray,
    value: FloatArray,
) -> tuple[FloatArray, SwiGLUCache]:
    """Return ``SiLU(gate) * value`` after the two input projections."""

    gate_array = _require_floating(gate, name="gate")
    value_array = _require_floating(value, name="value")
    if gate_array.shape != value_array.shape:
        raise ValueError(
            f"gate and value shapes must match, got {gate_array.shape} and "
            f"{value_array.shape}"
        )
    activated_gate, silu_cache = silu_forward(gate_array)
    output = activated_gate * value_array
    cache = SwiGLUCache(
        silu_cache=silu_cache,
        activated_gate=activated_gate,
        value=value_array,
    )
    return output, cache


def swiglu_backward(
    gradient_output: FloatArray,
    cache: SwiGLUCache,
) -> tuple[FloatArray, FloatArray]:
    """Differentiate SwiGLU with respect to gate and value projections."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.activated_gate.shape:
        raise ValueError(
            "gradient_output must match the cached activation shape, "
            f"got {gradient.shape} and {cache.activated_gate.shape}"
        )
    gradient_gate = silu_backward(gradient * cache.value, cache.silu_cache)
    gradient_value = gradient * cache.activated_gate
    return gradient_gate, gradient_value

