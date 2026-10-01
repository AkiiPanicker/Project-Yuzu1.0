"""Finite-difference tools for verifying hand-written backward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Float64Array = NDArray[np.float64]
ScalarFunction = Callable[[Float64Array], float | np.floating]


@dataclass(frozen=True, slots=True)
class GradientCheckResult:
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    worst_index: tuple[int, ...]


def finite_difference_gradient(
    function: ScalarFunction,
    values: NDArray[np.floating],
    *,
    epsilon: float = 1e-6,
) -> Float64Array:
    """Estimate a scalar function's gradient using centered finite differences."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    point = np.asarray(values, dtype=np.float64).copy()
    if point.size == 0:
        raise ValueError("values must not be empty")
    gradient = np.empty_like(point)

    for index in np.ndindex(point.shape):
        original = point[index]
        point[index] = original + epsilon
        positive = float(function(point))
        point[index] = original - epsilon
        negative = float(function(point))
        point[index] = original
        gradient[index] = (positive - negative) / (2.0 * epsilon)

    return gradient


def check_gradient(
    function: ScalarFunction,
    values: NDArray[np.floating],
    analytical_gradient: NDArray[np.floating],
    *,
    epsilon: float = 1e-6,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-5,
) -> GradientCheckResult:
    """Compare an analytical gradient with a numerical gradient."""

    analytical = np.asarray(analytical_gradient, dtype=np.float64)
    numerical = finite_difference_gradient(function, values, epsilon=epsilon)
    if analytical.shape != numerical.shape:
        raise ValueError(
            f"gradient shapes differ: analytical={analytical.shape}, "
            f"numerical={numerical.shape}"
        )
    if not np.all(np.isfinite(analytical)):
        raise ValueError("analytical_gradient must contain only finite values")

    absolute_error = np.abs(analytical - numerical)
    scale = np.maximum(np.abs(analytical) + np.abs(numerical), 1e-12)
    relative_error = absolute_error / scale
    worst_flat_index = int(np.argmax(relative_error))
    worst_index = tuple(np.unravel_index(worst_flat_index, relative_error.shape))
    passed = bool(
        np.all(
            (absolute_error <= absolute_tolerance)
            | (relative_error <= relative_tolerance)
        )
    )
    return GradientCheckResult(
        passed=passed,
        max_absolute_error=float(np.max(absolute_error)),
        max_relative_error=float(np.max(relative_error)),
        worst_index=worst_index,
    )

