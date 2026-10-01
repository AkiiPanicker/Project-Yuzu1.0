"""RMSNorm and LayerNorm with explicit, hand-written backward passes."""

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


def _validate_inputs_and_scale(
    inputs: FloatArray,
    scale: FloatArray,
    epsilon: float,
) -> tuple[FloatArray, FloatArray]:
    x = _require_floating(inputs, name="inputs")
    gamma = _require_floating(scale, name="scale")
    if x.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if gamma.shape != (x.shape[-1],):
        raise ValueError(
            f"scale must have shape {(x.shape[-1],)}, got {gamma.shape}"
        )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return x, gamma


def _scale_gradient(
    gradient_output: FloatArray,
    normalized: FloatArray,
) -> FloatArray:
    width = normalized.shape[-1]
    return (gradient_output * normalized).reshape(-1, width).sum(axis=0)


@dataclass(frozen=True, slots=True)
class RMSNormCache:
    """Values needed for the RMSNorm backward pass."""

    normalized: FloatArray
    inverse_rms: FloatArray
    scale: FloatArray


def rms_norm_forward(
    inputs: FloatArray,
    scale: FloatArray,
    *,
    epsilon: float = 1e-5,
) -> tuple[FloatArray, RMSNormCache]:
    """Normalize by root-mean-square over the final dimension."""

    x, gamma = _validate_inputs_and_scale(inputs, scale, epsilon)
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    inverse_rms = 1.0 / np.sqrt(mean_square + epsilon)
    normalized = x * inverse_rms
    output = normalized * gamma
    cache = RMSNormCache(
        normalized=normalized,
        inverse_rms=inverse_rms,
        scale=gamma,
    )
    return output, cache


def rms_norm_backward(
    gradient_output: FloatArray,
    cache: RMSNormCache,
) -> tuple[FloatArray, FloatArray]:
    """Differentiate RMSNorm with respect to inputs and scale."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.normalized.shape:
        raise ValueError(
            "gradient_output must match the cached normalized shape, "
            f"got {gradient.shape} and {cache.normalized.shape}"
        )
    scaled_gradient = gradient * cache.scale
    projection = np.mean(
        scaled_gradient * cache.normalized,
        axis=-1,
        keepdims=True,
    )
    gradient_inputs = cache.inverse_rms * (
        scaled_gradient - cache.normalized * projection
    )
    gradient_scale = _scale_gradient(gradient, cache.normalized)
    return gradient_inputs, gradient_scale


@dataclass(frozen=True, slots=True)
class LayerNormCache:
    """Values needed for the LayerNorm backward pass."""

    normalized: FloatArray
    inverse_standard_deviation: FloatArray
    scale: FloatArray
    has_bias: bool


def layer_norm_forward(
    inputs: FloatArray,
    scale: FloatArray,
    bias: FloatArray | None = None,
    *,
    epsilon: float = 1e-5,
) -> tuple[FloatArray, LayerNormCache]:
    """Center and variance-normalize over the final dimension."""

    x, gamma = _validate_inputs_and_scale(inputs, scale, epsilon)
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    inverse_standard_deviation = 1.0 / np.sqrt(variance + epsilon)
    normalized = centered * inverse_standard_deviation
    output = normalized * gamma
    has_bias = bias is not None
    if bias is not None:
        beta = _require_floating(bias, name="bias")
        if beta.shape != gamma.shape:
            raise ValueError(f"bias must have shape {gamma.shape}, got {beta.shape}")
        output = output + beta
    cache = LayerNormCache(
        normalized=normalized,
        inverse_standard_deviation=inverse_standard_deviation,
        scale=gamma,
        has_bias=has_bias,
    )
    return output, cache


def layer_norm_backward(
    gradient_output: FloatArray,
    cache: LayerNormCache,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Differentiate LayerNorm with respect to inputs, scale, and optional bias."""

    gradient = _require_floating(gradient_output, name="gradient_output")
    if gradient.shape != cache.normalized.shape:
        raise ValueError(
            "gradient_output must match the cached normalized shape, "
            f"got {gradient.shape} and {cache.normalized.shape}"
        )
    scaled_gradient = gradient * cache.scale
    gradient_mean = np.mean(scaled_gradient, axis=-1, keepdims=True)
    projected_gradient = np.mean(
        scaled_gradient * cache.normalized,
        axis=-1,
        keepdims=True,
    )
    gradient_inputs = cache.inverse_standard_deviation * (
        scaled_gradient
        - gradient_mean
        - cache.normalized * projected_gradient
    )
    gradient_scale = _scale_gradient(gradient, cache.normalized)
    gradient_bias = (
        gradient.reshape(-1, gradient.shape[-1]).sum(axis=0)
        if cache.has_bias
        else None
    )
    return gradient_inputs, gradient_scale, gradient_bias

