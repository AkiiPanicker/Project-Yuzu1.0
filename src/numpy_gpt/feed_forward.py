"""Complete bias-free SwiGLU feed-forward network and backward composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .activations import SwiGLUCache, swiglu_backward, swiglu_forward
from .layers import LinearCache, linear_backward, linear_forward


FloatArray: TypeAlias = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class FeedForwardWeightGradients:
    """Gradients for the gate, value, and output projection matrices."""

    gate: FloatArray
    value: FloatArray
    output: FloatArray


@dataclass(frozen=True, slots=True)
class FeedForwardCache:
    """Composition caches needed for the feed-forward backward pass."""

    gate_linear: LinearCache
    value_linear: LinearCache
    activation: SwiGLUCache
    output_linear: LinearCache


def _validate_weights(
    inputs: FloatArray,
    gate_weight: FloatArray,
    value_weight: FloatArray,
    output_weight: FloatArray,
) -> tuple[int, int]:
    x = np.asarray(inputs)
    if x.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    model_width = x.shape[-1]
    gate_shape = np.asarray(gate_weight).shape
    value_shape = np.asarray(value_weight).shape
    output_shape = np.asarray(output_weight).shape
    if len(gate_shape) != 2 or gate_shape[0] != model_width:
        raise ValueError(
            "gate weight must have shape (model_width, hidden_width), "
            f"got {gate_shape}"
        )
    if value_shape != gate_shape:
        raise ValueError(
            f"value weight must have shape {gate_shape}, got {value_shape}"
        )
    hidden_width = gate_shape[1]
    expected_output_shape = (hidden_width, model_width)
    if output_shape != expected_output_shape:
        raise ValueError(
            f"output weight must have shape {expected_output_shape}, "
            f"got {output_shape}"
        )
    return model_width, hidden_width


def feed_forward_forward(
    inputs: FloatArray,
    gate_weight: FloatArray,
    value_weight: FloatArray,
    output_weight: FloatArray,
) -> tuple[FloatArray, FeedForwardCache]:
    """Apply two input projections, SwiGLU gating, and an output projection."""

    _validate_weights(inputs, gate_weight, value_weight, output_weight)
    gate, gate_linear = linear_forward(inputs, gate_weight)
    value, value_linear = linear_forward(inputs, value_weight)
    hidden, activation = swiglu_forward(gate, value)
    output, output_linear = linear_forward(hidden, output_weight)
    cache = FeedForwardCache(
        gate_linear=gate_linear,
        value_linear=value_linear,
        activation=activation,
        output_linear=output_linear,
    )
    return output, cache


def feed_forward_backward(
    gradient_output: FloatArray,
    cache: FeedForwardCache,
) -> tuple[FloatArray, FeedForwardWeightGradients]:
    """Backpropagate through output projection, SwiGLU, and both input branches."""

    gradient_hidden, gradient_output_weight, _ = linear_backward(
        gradient_output,
        cache.output_linear,
    )
    gradient_gate, gradient_value = swiglu_backward(
        gradient_hidden,
        cache.activation,
    )
    gradient_input_gate, gradient_gate_weight, _ = linear_backward(
        gradient_gate,
        cache.gate_linear,
    )
    gradient_input_value, gradient_value_weight, _ = linear_backward(
        gradient_value,
        cache.value_linear,
    )
    gradient_inputs = gradient_input_gate + gradient_input_value
    gradients = FeedForwardWeightGradients(
        gate=gradient_gate_weight,
        value=gradient_value_weight,
        output=gradient_output_weight,
    )
    return gradient_inputs, gradients

