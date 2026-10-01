"""Numerically stable probability and language-model loss primitives."""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]
Reduction: TypeAlias = Literal["none", "sum", "mean"]


def _floating_array(values: FloatArray, *, name: str) -> FloatArray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype, got {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def logsumexp(
    values: FloatArray,
    axis: int | tuple[int, ...] = -1,
    *,
    keepdims: bool = False,
) -> FloatArray:
    """Compute ``log(sum(exp(values)))`` without avoidable overflow."""

    array = _floating_array(values, name="values")
    maximum = np.max(array, axis=axis, keepdims=True)
    shifted = array - maximum
    result = maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    if keepdims:
        return result
    return np.squeeze(result, axis=axis)


def softmax(values: FloatArray, axis: int = -1) -> FloatArray:
    """Return a stable softmax over ``axis``."""

    array = _floating_array(values, name="values")
    maximum = np.max(array, axis=axis, keepdims=True)
    exponentials = np.exp(array - maximum)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def cross_entropy_with_logits(
    logits: FloatArray,
    targets: NDArray[np.integer],
    *,
    reduction: Reduction = "mean",
) -> tuple[FloatArray | np.floating, FloatArray]:
    """Return categorical cross-entropy and its exact gradient with respect to logits.

    ``logits`` has shape ``(..., vocabulary)`` and ``targets`` must have shape
    ``logits.shape[:-1]``. For ``reduction='none'``, each row of the returned
    gradient is the gradient of the corresponding unreduced loss.
    """

    scores = _floating_array(logits, name="logits")
    labels = np.asarray(targets)
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"targets must have an integer dtype, got {labels.dtype}")
    if labels.shape != scores.shape[:-1]:
        raise ValueError(
            "targets shape must equal logits.shape[:-1], "
            f"got targets={labels.shape} and logits={scores.shape}"
        )
    if labels.size == 0:
        raise ValueError("targets must not be empty")
    vocabulary_size = scores.shape[-1]
    if np.any(labels < 0) or np.any(labels >= vocabulary_size):
        raise ValueError(f"targets must be in [0, {vocabulary_size})")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError(f"unsupported reduction: {reduction!r}")

    normalizers = logsumexp(scores, axis=-1, keepdims=True)
    selected = np.take_along_axis(scores, labels[..., None], axis=-1)[..., 0]
    losses = normalizers[..., 0] - selected

    gradient = np.exp(scores - normalizers)
    flat_gradient = gradient.reshape(-1, vocabulary_size)
    flat_labels = labels.reshape(-1)
    flat_gradient[np.arange(flat_labels.size), flat_labels] -= 1.0

    if reduction == "none":
        return losses, gradient
    if reduction == "sum":
        return np.sum(losses), gradient
    return np.mean(losses), gradient / labels.size

