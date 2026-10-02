"""Observable multi-step training with metrics logs and resumable checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable, Mapping, TextIO, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .byte_data import sample_next_token_batch
from .checkpoint import save_checkpoint
from .config import ModelConfig
from .initialization import initialize_parameters
from .model import LanguageModelParameters, named_parameters
from .optimizer import AdamWState, initialize_adamw_state
from .training import evaluate_batch, train_step


IntegerArray: TypeAlias = NDArray[np.integer]
OutputFunction: TypeAlias = Callable[[str], None]
ClockFunction: TypeAlias = Callable[[], float]
RUN_FORMAT = "numpy-gpt-training-run"
RUN_VERSION = 1


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    """Hyperparameters and output intervals for one resumable training run."""

    max_steps: int
    batch_size: int
    learning_rate: float
    validation_interval: int = 10
    log_interval: int = 1
    checkpoint_interval: int = 100
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    seed: int = 1337

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "batch_size",
            "validation_interval",
            "log_interval",
            "checkpoint_interval",
        ):
            _positive_integer(getattr(self, name), name)
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if _finite_float(self.learning_rate, "learning_rate") <= 0.0:
            raise ValueError("learning_rate must be positive")
        beta1 = _finite_float(self.beta1, "beta1")
        beta2 = _finite_float(self.beta2, "beta2")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must satisfy 0 <= beta2 < 1")
        if _finite_float(self.adam_epsilon, "adam_epsilon") <= 0.0:
            raise ValueError("adam_epsilon must be positive")
        if _finite_float(self.weight_decay, "weight_decay") < 0.0:
            raise ValueError("weight_decay must be nonnegative")
        if _finite_float(self.max_gradient_norm, "max_gradient_norm") <= 0.0:
            raise ValueError("max_gradient_norm must be positive")


@dataclass(frozen=True, slots=True)
class TrainingLogRecord:
    """One terminal and JSON-lines metrics record."""

    step: int
    epoch: float
    train_loss: float
    validation_loss: float
    validation_perplexity: float
    learning_rate: float
    gradient_norm: float
    clip_coefficient: float
    tokens_per_second: float
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {"event": "metrics", **asdict(self)}

    def to_terminal(self) -> str:
        return (
            f"step={self.step:06d} epoch={self.epoch:.3f} "
            f"train_loss={self.train_loss:.6f} "
            f"validation_loss={self.validation_loss:.6f} "
            f"validation_ppl={self.validation_perplexity:.3f} "
            f"lr={self.learning_rate:.3e} "
            f"grad_norm={self.gradient_norm:.6f} "
            f"clip={self.clip_coefficient:.4f} "
            f"tok/s={self.tokens_per_second:.1f} "
            f"elapsed={self.elapsed_seconds:.2f}s"
        )


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Final in-memory state and artifacts produced by a training run."""

    parameters: LanguageModelParameters
    optimizer_state: AdamWState
    records: tuple[TrainingLogRecord, ...]
    checkpoint_paths: tuple[Path, ...]
    metrics_path: Path
    manifest_path: Path


def _validated_tokens(
    token_ids: IntegerArray,
    config: ModelConfig,
    name: str,
) -> IntegerArray:
    values = np.asarray(token_ids)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if values.size < config.context_length + 1:
        raise ValueError(f"{name} must contain at least context_length + 1 tokens")
    if np.any(values < 0) or np.any(values >= config.vocab_size):
        raise ValueError(f"{name} values must be in [0, {config.vocab_size})")
    return values


def _write_json_line(stream: TextIO, values: Mapping[str, object]) -> None:
    stream.write(json.dumps(values, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def _token_digest(values: IntegerArray) -> str:
    canonical = values.astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _manifest_values(
    training_values: IntegerArray,
    validation_values: IntegerArray,
    model_config: ModelConfig,
    loop_config: TrainingLoopConfig,
) -> dict[str, object]:
    return {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "model_config": model_config.to_dict(),
        "training_hyperparameters": {
            "batch_size": loop_config.batch_size,
            "learning_rate": float(loop_config.learning_rate),
            "beta1": float(loop_config.beta1),
            "beta2": float(loop_config.beta2),
            "adam_epsilon": float(loop_config.adam_epsilon),
            "weight_decay": float(loop_config.weight_decay),
            "max_gradient_norm": float(loop_config.max_gradient_norm),
            "seed": loop_config.seed,
        },
        "train_tokens": {
            "count": int(training_values.size),
            "sha256": _token_digest(training_values),
        },
        "validation_tokens": {
            "count": int(validation_values.size),
            "sha256": _token_digest(validation_values),
        },
    }


def _prepare_manifest(path: Path, values: Mapping[str, object]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing run manifest is unreadable") from error
        if existing != values:
            raise ValueError("existing run manifest does not match this run")
        return

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run_training(
    train_tokens: IntegerArray,
    validation_tokens: IntegerArray,
    model_config: ModelConfig,
    loop_config: TrainingLoopConfig,
    run_directory: str | Path,
    *,
    parameters: LanguageModelParameters | None = None,
    optimizer_state: AdamWState | None = None,
    rng: np.random.Generator | None = None,
    output: OutputFunction | None = print,
    clock: ClockFunction = time.perf_counter,
) -> TrainingRunResult:
    """Train to an absolute step target with terminal, JSONL, and checkpoints."""

    training_values = _validated_tokens(train_tokens, model_config, "train_tokens")
    validation_values = _validated_tokens(
        validation_tokens,
        model_config,
        "validation_tokens",
    )
    if (parameters is None) != (optimizer_state is None):
        raise ValueError("parameters and optimizer_state must be supplied together")
    if parameters is None:
        parameters = initialize_parameters(model_config)
        optimizer_state = initialize_adamw_state(named_parameters(parameters))
    assert optimizer_state is not None
    if (
        not isinstance(optimizer_state.step, int)
        or isinstance(optimizer_state.step, bool)
        or optimizer_state.step < 0
    ):
        raise ValueError("optimizer step must be a nonnegative integer")
    if optimizer_state.step >= loop_config.max_steps:
        raise ValueError("max_steps must be greater than the optimizer step")
    if rng is None:
        rng = np.random.default_rng(loop_config.seed)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if output is not None and not callable(output):
        raise TypeError("output must be callable or None")
    if not callable(clock):
        raise TypeError("clock must be callable")

    destination = Path(run_directory)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    manifest_path = destination / "run_manifest.json"
    if (
        optimizer_state.step == 0
        and metrics_path.exists()
        and metrics_path.stat().st_size
    ):
        raise FileExistsError(
            "a new run cannot append to a nonempty metrics.jsonl file"
        )
    _prepare_manifest(
        manifest_path,
        _manifest_values(
            training_values,
            validation_values,
            model_config,
            loop_config,
        ),
    )

    validation_rng = np.random.default_rng(loop_config.seed + 1)
    validation_batch = sample_next_token_batch(
        validation_values,
        batch_size=loop_config.batch_size,
        context_length=model_config.context_length,
        rng=validation_rng,
    )
    validation_metrics = evaluate_batch(
        validation_batch,
        parameters,
        model_config,
    )
    start_step = optimizer_state.step
    start_time = float(clock())
    if not np.isfinite(start_time):
        raise ValueError("clock must return finite values")
    records: list[TrainingLogRecord] = []
    checkpoint_paths: list[Path] = []

    with metrics_path.open("a", encoding="utf-8", newline="\n") as metrics_stream:
        while optimizer_state.step < loop_config.max_steps:
            batch = sample_next_token_batch(
                training_values,
                batch_size=loop_config.batch_size,
                context_length=model_config.context_length,
                rng=rng,
            )
            optimizer_state, step_stats = train_step(
                batch,
                parameters,
                optimizer_state,
                model_config,
                learning_rate=loop_config.learning_rate,
                beta1=loop_config.beta1,
                beta2=loop_config.beta2,
                epsilon=loop_config.adam_epsilon,
                weight_decay=loop_config.weight_decay,
                max_gradient_norm=loop_config.max_gradient_norm,
            )
            step = optimizer_state.step
            is_final_step = step == loop_config.max_steps
            if step % loop_config.validation_interval == 0 or is_final_step:
                validation_metrics = evaluate_batch(
                    validation_batch,
                    parameters,
                    model_config,
                )

            if step % loop_config.log_interval == 0 or is_final_step:
                elapsed = float(clock()) - start_time
                if not np.isfinite(elapsed) or elapsed < 0.0:
                    raise ValueError("clock must advance with finite values")
                completed_steps = step - start_step
                processed_tokens = completed_steps * step_stats.token_count
                safe_elapsed = max(elapsed, np.finfo(np.float64).eps)
                total_tokens = step * step_stats.token_count
                record = TrainingLogRecord(
                    step=step,
                    epoch=total_tokens / max(1, training_values.size - 1),
                    train_loss=step_stats.loss,
                    validation_loss=validation_metrics.loss,
                    validation_perplexity=validation_metrics.perplexity,
                    learning_rate=step_stats.learning_rate,
                    gradient_norm=step_stats.gradient_norm,
                    clip_coefficient=step_stats.clip_coefficient,
                    tokens_per_second=processed_tokens / safe_elapsed,
                    elapsed_seconds=elapsed,
                )
                records.append(record)
                _write_json_line(metrics_stream, record.to_dict())
                if output is not None:
                    output(record.to_terminal())

            if step % loop_config.checkpoint_interval == 0 or is_final_step:
                checkpoint_path = destination / f"checkpoint-step-{step:08d}.npz"
                save_checkpoint(
                    checkpoint_path,
                    parameters,
                    optimizer_state,
                    model_config,
                    rng_state=rng.bit_generator.state,
                )
                checkpoint_paths.append(checkpoint_path)
                checkpoint_event = {
                    "event": "checkpoint",
                    "path": str(checkpoint_path),
                    "step": step,
                }
                _write_json_line(metrics_stream, checkpoint_event)
                if output is not None:
                    output(f"checkpoint step={step:06d} path={checkpoint_path}")

    return TrainingRunResult(
        parameters=parameters,
        optimizer_state=optimizer_state,
        records=tuple(records),
        checkpoint_paths=tuple(checkpoint_paths),
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )
