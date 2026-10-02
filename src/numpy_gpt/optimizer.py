"""Global-norm gradient clipping and a transparent AdamW optimizer step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.floating]
NamedArrays: TypeAlias = Iterable[tuple[str, FloatArray]]


@dataclass(frozen=True, slots=True)
class GradientClippingResult:
    """Copied gradients plus the norm and scale used to clip them."""

    gradients: tuple[tuple[str, FloatArray], ...]
    global_norm: float
    clip_coefficient: float


@dataclass(frozen=True, slots=True)
class AdamWState:
    """Step counter and moment arrays in deterministic parameter order."""

    step: int
    parameter_names: tuple[str, ...]
    first_moments: tuple[FloatArray, ...]
    second_moments: tuple[FloatArray, ...]


@dataclass(frozen=True, slots=True)
class AdamWStepStats:
    """Values needed for terminal-visible training diagnostics."""

    step: int
    learning_rate: float
    gradient_norm: float
    clip_coefficient: float


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _positive_float(value: float, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _materialize_named_arrays(
    named_arrays: NamedArrays,
    *,
    kind: str,
    require_writable: bool = False,
) -> tuple[tuple[str, FloatArray], ...]:
    items: list[tuple[str, FloatArray]] = []
    observed_names: set[str] = set()
    for name, values in named_arrays:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{kind} names must be nonempty strings")
        if name in observed_names:
            raise ValueError(f"duplicate {kind} name: {name}")
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{kind} {name} must be floating-point")
        if array.size == 0:
            raise ValueError(f"{kind} {name} must not be empty")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{kind} {name} contains nonfinite values")
        if require_writable and not array.flags.writeable:
            raise ValueError(f"{kind} {name} must be writable")
        observed_names.add(name)
        items.append((name, array))
    if not items:
        raise ValueError(f"at least one {kind} array is required")
    return tuple(items)


def _global_norm(items: tuple[tuple[str, FloatArray], ...]) -> float:
    maximum = max(float(np.max(np.abs(values))) for _, values in items)
    if maximum == 0.0:
        return 0.0
    scaled_square_sum = 0.0
    for _, values in items:
        scaled = values.astype(np.float64, copy=False) / maximum
        scaled_square_sum += float(np.sum(scaled * scaled, dtype=np.float64))
    norm = maximum * float(np.sqrt(scaled_square_sum))
    if not np.isfinite(norm):
        raise FloatingPointError("global gradient norm is nonfinite")
    return norm


def global_gradient_norm(named_gradients: NamedArrays) -> float:
    """Return one overflow-resistant L2 norm across every gradient element."""

    gradients = _materialize_named_arrays(named_gradients, kind="gradient")
    return _global_norm(gradients)


def clip_gradients_by_global_norm(
    named_gradients: NamedArrays,
    max_norm: float,
    *,
    epsilon: float = 1e-6,
) -> GradientClippingResult:
    """Copy and jointly scale gradients whose global L2 norm is too large."""

    resolved_max_norm = _positive_float(max_norm, "max_norm")
    resolved_epsilon = _positive_float(epsilon, "epsilon")
    gradients = _materialize_named_arrays(named_gradients, kind="gradient")
    norm = _global_norm(gradients)
    coefficient = min(1.0, resolved_max_norm / (norm + resolved_epsilon))
    clipped = tuple(
        (
            name,
            (values * coefficient).astype(values.dtype, copy=False),
        )
        for name, values in gradients
    )
    return GradientClippingResult(
        gradients=clipped,
        global_norm=norm,
        clip_coefficient=coefficient,
    )


def initialize_adamw_state(named_parameters: NamedArrays) -> AdamWState:
    """Allocate zero-valued first and second moments for all parameters."""

    parameters = _materialize_named_arrays(named_parameters, kind="parameter")
    return AdamWState(
        step=0,
        parameter_names=tuple(name for name, _ in parameters),
        first_moments=tuple(np.zeros_like(values) for _, values in parameters),
        second_moments=tuple(np.zeros_like(values) for _, values in parameters),
    )


def _validate_state(
    state: AdamWState,
    parameters: tuple[tuple[str, FloatArray], ...],
) -> None:
    if (
        not isinstance(state.step, int)
        or isinstance(state.step, bool)
        or state.step < 0
    ):
        raise ValueError("optimizer step must be a nonnegative integer")
    names = tuple(name for name, _ in parameters)
    if state.parameter_names != names:
        raise ValueError("optimizer state parameter names do not match parameters")
    if len(state.first_moments) != len(parameters) or len(
        state.second_moments
    ) != len(parameters):
        raise ValueError("optimizer state moment count does not match parameters")
    for (name, parameter), first, second in zip(
        parameters,
        state.first_moments,
        state.second_moments,
        strict=True,
    ):
        for moment_name, moment in (("first", first), ("second", second)):
            array = np.asarray(moment)
            if array.shape != parameter.shape:
                raise ValueError(
                    f"{moment_name} moment for {name} must have shape "
                    f"{parameter.shape}, got {array.shape}"
                )
            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(
                    f"{moment_name} moment for {name} must be floating-point"
                )
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    f"{moment_name} moment for {name} contains nonfinite values"
                )


def adamw_step(
    named_parameter_values: NamedArrays,
    named_gradient_values: NamedArrays,
    state: AdamWState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.1,
    max_gradient_norm: float = 1.0,
) -> tuple[AdamWState, AdamWStepStats]:
    """Apply one atomic AdamW step, excluding one-dimensional norm scales."""

    resolved_learning_rate = _positive_float(learning_rate, "learning_rate")
    resolved_beta1 = _finite_float(beta1, "beta1")
    resolved_beta2 = _finite_float(beta2, "beta2")
    if not 0.0 <= resolved_beta1 < 1.0:
        raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
    if not 0.0 <= resolved_beta2 < 1.0:
        raise ValueError("beta2 must satisfy 0 <= beta2 < 1")
    resolved_epsilon = _positive_float(epsilon, "epsilon")
    resolved_weight_decay = _finite_float(weight_decay, "weight_decay")
    if resolved_weight_decay < 0.0:
        raise ValueError("weight_decay must be nonnegative")

    parameters = _materialize_named_arrays(
        named_parameter_values,
        kind="parameter",
        require_writable=True,
    )
    clipping = clip_gradients_by_global_norm(
        named_gradient_values,
        max_gradient_norm,
    )
    parameter_names = tuple(name for name, _ in parameters)
    gradient_names = tuple(name for name, _ in clipping.gradients)
    if gradient_names != parameter_names:
        raise ValueError("gradient names and order must exactly match parameters")
    for (name, parameter), (_, gradient) in zip(
        parameters,
        clipping.gradients,
        strict=True,
    ):
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"gradient for {name} must have shape {parameter.shape}, "
                f"got {gradient.shape}"
            )
    _validate_state(state, parameters)

    next_step = state.step + 1
    first_correction = 1.0 - resolved_beta1**next_step
    second_correction = 1.0 - resolved_beta2**next_step
    next_first_moments: list[FloatArray] = []
    next_second_moments: list[FloatArray] = []
    next_parameter_values: list[FloatArray] = []

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for (_, parameter), (_, gradient), first, second in zip(
            parameters,
            clipping.gradients,
            state.first_moments,
            state.second_moments,
            strict=True,
        ):
            next_first = (
                resolved_beta1 * first + (1.0 - resolved_beta1) * gradient
            ).astype(first.dtype, copy=False)
            next_second = (
                resolved_beta2 * second
                + (1.0 - resolved_beta2) * gradient * gradient
            ).astype(second.dtype, copy=False)
            corrected_first = next_first / first_correction
            corrected_second = next_second / second_correction
            adaptive_update = corrected_first / (
                np.sqrt(corrected_second) + resolved_epsilon
            )
            decay = resolved_weight_decay if parameter.ndim >= 2 else 0.0
            next_parameter = (
                parameter
                - resolved_learning_rate * adaptive_update
                - resolved_learning_rate * decay * parameter
            ).astype(parameter.dtype, copy=False)
            for description, values in (
                ("first moment", next_first),
                ("second moment", next_second),
                ("updated parameter", next_parameter),
            ):
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError(f"{description} contains nonfinite values")
            next_first_moments.append(next_first)
            next_second_moments.append(next_second)
            next_parameter_values.append(next_parameter)

    for (_, parameter), next_parameter in zip(
        parameters,
        next_parameter_values,
        strict=True,
    ):
        parameter[...] = next_parameter

    next_state = AdamWState(
        step=next_step,
        parameter_names=parameter_names,
        first_moments=tuple(next_first_moments),
        second_moments=tuple(next_second_moments),
    )
    stats = AdamWStepStats(
        step=next_step,
        learning_rate=resolved_learning_rate,
        gradient_norm=clipping.global_norm,
        clip_coefficient=clipping.clip_coefficient,
    )
    return next_state, stats
