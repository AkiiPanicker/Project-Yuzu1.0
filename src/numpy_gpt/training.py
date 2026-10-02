"""One complete language-model training step and read-only evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .byte_data import ByteBatch
from .config import ModelConfig
from .model import (
    LanguageModelParameters,
    language_model_backward,
    language_model_forward,
    named_gradients,
    named_parameters,
)
from .numerics import cross_entropy_with_logits
from .optimizer import AdamWState, adamw_step


IntegerArray: TypeAlias = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Mean next-token loss and derived statistics for one immutable batch."""

    loss: float
    perplexity: float
    token_count: int


@dataclass(frozen=True, slots=True)
class TrainingStepStats:
    """Loss and optimizer values intended for terminal-visible diagnostics."""

    step: int
    loss: float
    perplexity: float
    token_count: int
    learning_rate: float
    gradient_norm: float
    clip_coefficient: float


def _validated_batch(
    batch: ByteBatch,
    config: ModelConfig,
) -> tuple[IntegerArray, IntegerArray]:
    if not isinstance(batch, ByteBatch):
        raise TypeError("batch must be a ByteBatch")
    inputs = np.asarray(batch.inputs)
    targets = np.asarray(batch.targets)
    for name, values in (("inputs", inputs), ("targets", targets)):
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"batch {name} must contain integers")
        if values.ndim != 2:
            raise ValueError(f"batch {name} must be two-dimensional")
        if values.size == 0:
            raise ValueError(f"batch {name} must not be empty")
        if np.any(values < 0) or np.any(values >= config.vocab_size):
            raise ValueError(
                f"batch {name} values must be in [0, {config.vocab_size})"
            )
    if inputs.shape != targets.shape:
        raise ValueError(
            "batch inputs and targets must have the same shape, "
            f"got {inputs.shape} and {targets.shape}"
        )
    if inputs.shape[1] > config.context_length:
        raise ValueError(
            f"batch sequence length {inputs.shape[1]} exceeds context length "
            f"{config.context_length}"
        )
    return inputs, targets


def _perplexity(loss: float) -> float:
    with np.errstate(over="ignore"):
        return float(np.exp(loss))


def evaluate_batch(
    batch: ByteBatch,
    parameters: LanguageModelParameters,
    config: ModelConfig,
) -> EvaluationMetrics:
    """Measure a batch without backward propagation or parameter mutation."""

    inputs, targets = _validated_batch(batch, config)
    logits, _ = language_model_forward(inputs, parameters, config)
    loss, _ = cross_entropy_with_logits(logits, targets, reduction="mean")
    resolved_loss = float(loss)
    return EvaluationMetrics(
        loss=resolved_loss,
        perplexity=_perplexity(resolved_loss),
        token_count=int(targets.size),
    )


def train_step(
    batch: ByteBatch,
    parameters: LanguageModelParameters,
    optimizer_state: AdamWState,
    config: ModelConfig,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.1,
    max_gradient_norm: float = 1.0,
) -> tuple[AdamWState, TrainingStepStats]:
    """Run mean loss, full backward propagation, clipping, and one AdamW update."""

    inputs, targets = _validated_batch(batch, config)
    logits, cache = language_model_forward(inputs, parameters, config)
    loss, gradient_logits = cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
    )
    gradients = language_model_backward(gradient_logits, cache)
    next_state, optimizer_stats = adamw_step(
        named_parameters(parameters),
        named_gradients(gradients),
        optimizer_state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        weight_decay=weight_decay,
        max_gradient_norm=max_gradient_norm,
    )
    resolved_loss = float(loss)
    stats = TrainingStepStats(
        step=optimizer_stats.step,
        loss=resolved_loss,
        perplexity=_perplexity(resolved_loss),
        token_count=int(targets.size),
        learning_rate=optimizer_stats.learning_rate,
        gradient_norm=optimizer_stats.gradient_norm,
        clip_coefficient=optimizer_stats.clip_coefficient,
    )
    return next_state, stats
