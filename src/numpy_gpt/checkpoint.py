"""Versioned, non-pickle checkpoints for exact NumPy training continuation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .config import ModelConfig
from .model import (
    LanguageModelParameters,
    TransformerBlockParameters,
    named_parameters,
)
from .optimizer import AdamWState


CHECKPOINT_FORMAT = "numpy-gpt-from-scratch"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """All state required to resume deterministic model updates."""

    config: ModelConfig
    parameters: LanguageModelParameters
    optimizer_state: AdamWState
    rng_state: dict[str, Any] | None


def _parameter_specs(
    config: ModelConfig,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    width = config.d_model
    hidden = config.d_ff
    specs: list[tuple[str, tuple[int, ...]]] = [
        ("token_embedding", (config.vocab_size, width))
    ]
    for index in range(config.n_layers):
        prefix = f"blocks.{index}"
        specs.extend(
            (
                (f"{prefix}.attention_norm_scale", (width,)),
                (f"{prefix}.query_weight", (width, width)),
                (f"{prefix}.key_weight", (width, width)),
                (f"{prefix}.attention_value_weight", (width, width)),
                (f"{prefix}.attention_output_weight", (width, width)),
                (f"{prefix}.feed_forward_norm_scale", (width,)),
                (f"{prefix}.gate_weight", (width, hidden)),
                (f"{prefix}.feed_forward_value_weight", (width, hidden)),
                (f"{prefix}.feed_forward_output_weight", (hidden, width)),
            )
        )
    specs.append(("final_norm_scale", (width,)))
    return tuple(specs)


def _validated_training_arrays(
    parameters: LanguageModelParameters,
    optimizer_state: AdamWState,
    config: ModelConfig,
) -> tuple[
    tuple[tuple[str, np.ndarray], ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    parameter_items = tuple(
        (name, np.asarray(values)) for name, values in named_parameters(parameters)
    )
    specs = _parameter_specs(config)
    names = tuple(name for name, _ in parameter_items)
    expected_names = tuple(name for name, _ in specs)
    if names != expected_names:
        raise ValueError("parameter names do not match the checkpoint configuration")
    for (name, values), (_, expected_shape) in zip(
        parameter_items,
        specs,
        strict=True,
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"parameter {name} must have shape {expected_shape}, "
                f"got {values.shape}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError(f"parameter {name} must be floating-point")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"parameter {name} contains nonfinite values")

    if (
        not isinstance(optimizer_state.step, int)
        or isinstance(optimizer_state.step, bool)
        or optimizer_state.step < 0
    ):
        raise ValueError("optimizer step must be a nonnegative integer")
    if optimizer_state.parameter_names != names:
        raise ValueError("optimizer state names do not match model parameters")
    first_moments = tuple(np.asarray(values) for values in optimizer_state.first_moments)
    second_moments = tuple(
        np.asarray(values) for values in optimizer_state.second_moments
    )
    if len(first_moments) != len(parameter_items) or len(second_moments) != len(
        parameter_items
    ):
        raise ValueError("optimizer moment count does not match model parameters")
    for (name, parameter), first, second in zip(
        parameter_items,
        first_moments,
        second_moments,
        strict=True,
    ):
        for moment_name, moment in (("first", first), ("second", second)):
            if moment.shape != parameter.shape:
                raise ValueError(
                    f"{moment_name} moment for {name} must have shape "
                    f"{parameter.shape}, got {moment.shape}"
                )
            if moment.dtype != parameter.dtype:
                raise TypeError(
                    f"{moment_name} moment for {name} must have dtype "
                    f"{parameter.dtype}, got {moment.dtype}"
                )
            if not np.all(np.isfinite(moment)):
                raise FloatingPointError(
                    f"{moment_name} moment for {name} contains nonfinite values"
                )
    return parameter_items, first_moments, second_moments


def _encode_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("checkpoint JSON values must be finite")
        return value
    if isinstance(value, np.generic):
        return _encode_json_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("checkpoint JSON arrays must not contain objects")
        return {
            "__numpy_gpt_type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _encode_json_value(value.tolist()),
        }
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint JSON object keys must be strings")
            encoded[key] = _encode_json_value(item)
        return encoded
    if isinstance(value, (list, tuple)):
        return [_encode_json_value(item) for item in value]
    raise TypeError(f"unsupported checkpoint JSON value: {type(value).__name__}")


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("__numpy_gpt_type__") == "ndarray":
            if set(value) != {
                "__numpy_gpt_type__",
                "dtype",
                "shape",
                "values",
            }:
                raise ValueError("invalid encoded NumPy array in checkpoint")
            if not isinstance(value["dtype"], str):
                raise ValueError("encoded NumPy array dtype must be a string")
            shape = value["shape"]
            if not isinstance(shape, list) or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension < 0
                for dimension in shape
            ):
                raise ValueError("encoded NumPy array shape is invalid")
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise TypeError("encoded NumPy arrays must not contain objects")
            array = np.asarray(value["values"], dtype=dtype)
            expected_shape = tuple(shape)
            if array.shape != expected_shape:
                raise ValueError("encoded NumPy array shape does not match values")
            return array
        return {key: _decode_json_value(item) for key, item in value.items()}
    return value


def _manifest_bytes(manifest: Mapping[str, Any]) -> np.ndarray:
    encoded = json.dumps(
        _encode_json_value(manifest),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def save_checkpoint(
    path: str | Path,
    parameters: LanguageModelParameters,
    optimizer_state: AdamWState,
    config: ModelConfig,
    *,
    rng_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer, configuration, and optional RNG state."""

    parameter_items, first_moments, second_moments = _validated_training_arrays(
        parameters,
        optimizer_state,
        config,
    )
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": config.to_dict(),
        "optimizer_step": optimizer_state.step,
        "parameter_names": [name for name, _ in parameter_items],
        "rng_state": None if rng_state is None else dict(rng_state),
    }
    arrays: dict[str, np.ndarray] = {"manifest": _manifest_bytes(manifest)}
    for index, (_, parameter) in enumerate(parameter_items):
        arrays[f"parameter_{index:04d}"] = parameter
        arrays[f"first_moment_{index:04d}"] = first_moments[index]
        arrays[f"second_moment_{index:04d}"] = second_moments[index]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _read_manifest(archive: Any) -> dict[str, Any]:
    if "manifest" not in archive.files:
        raise ValueError("checkpoint is missing its manifest")
    manifest_array = np.asarray(archive["manifest"])
    if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
        raise ValueError("checkpoint manifest must be a one-dimensional byte array")
    try:
        decoded = json.loads(manifest_array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest is not valid UTF-8 JSON") from error
    manifest = _decode_json_value(decoded)
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    required_keys = {
        "format",
        "version",
        "config",
        "optimizer_step",
        "parameter_names",
        "rng_state",
    }
    if set(manifest) != required_keys:
        raise ValueError("checkpoint manifest fields do not match this format")
    if manifest["format"] != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint format identifier is not supported")
    version = manifest["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != CHECKPOINT_VERSION
    ):
        raise ValueError("checkpoint version is not supported")
    return manifest


def _rebuild_parameters(
    arrays: Mapping[str, np.ndarray],
    config: ModelConfig,
) -> LanguageModelParameters:
    blocks: list[TransformerBlockParameters] = []
    for index in range(config.n_layers):
        prefix = f"blocks.{index}"
        blocks.append(
            TransformerBlockParameters(
                attention_norm_scale=arrays[f"{prefix}.attention_norm_scale"],
                query_weight=arrays[f"{prefix}.query_weight"],
                key_weight=arrays[f"{prefix}.key_weight"],
                attention_value_weight=arrays[
                    f"{prefix}.attention_value_weight"
                ],
                attention_output_weight=arrays[
                    f"{prefix}.attention_output_weight"
                ],
                feed_forward_norm_scale=arrays[
                    f"{prefix}.feed_forward_norm_scale"
                ],
                gate_weight=arrays[f"{prefix}.gate_weight"],
                feed_forward_value_weight=arrays[
                    f"{prefix}.feed_forward_value_weight"
                ],
                feed_forward_output_weight=arrays[
                    f"{prefix}.feed_forward_output_weight"
                ],
            )
        )
    return LanguageModelParameters(
        token_embedding=arrays["token_embedding"],
        blocks=tuple(blocks),
        final_norm_scale=arrays["final_norm_scale"],
    )


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Load and strictly validate a checkpoint without enabling pickle."""

    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read checkpoint: {source}") from error
    if isinstance(archive, np.ndarray):
        raise ValueError("checkpoint must be a NumPy .npz archive")

    with archive:
        manifest = _read_manifest(archive)
        config_values = manifest["config"]
        if not isinstance(config_values, dict):
            raise ValueError("checkpoint configuration must be a JSON object")
        try:
            config = ModelConfig(**config_values)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint configuration is invalid") from error
        specs = _parameter_specs(config)
        expected_names = tuple(name for name, _ in specs)
        stored_names = manifest["parameter_names"]
        if not isinstance(stored_names, list) or tuple(stored_names) != expected_names:
            raise ValueError("checkpoint parameter names do not match configuration")

        expected_archive_keys = {"manifest"}
        for index in range(len(specs)):
            expected_archive_keys.update(
                {
                    f"parameter_{index:04d}",
                    f"first_moment_{index:04d}",
                    f"second_moment_{index:04d}",
                }
            )
        if set(archive.files) != expected_archive_keys:
            raise ValueError("checkpoint array fields are missing or unexpected")

        parameter_arrays: dict[str, np.ndarray] = {}
        first_moments: list[np.ndarray] = []
        second_moments: list[np.ndarray] = []
        for index, (name, expected_shape) in enumerate(specs):
            parameter = np.array(archive[f"parameter_{index:04d}"], copy=True)
            first = np.array(archive[f"first_moment_{index:04d}"], copy=True)
            second = np.array(archive[f"second_moment_{index:04d}"], copy=True)
            for description, values in (
                ("parameter", parameter),
                ("first moment", first),
                ("second moment", second),
            ):
                if values.shape != expected_shape:
                    raise ValueError(
                        f"{description} {name} must have shape {expected_shape}, "
                        f"got {values.shape}"
                    )
                if not np.issubdtype(values.dtype, np.floating):
                    raise TypeError(f"{description} {name} must be floating-point")
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError(
                        f"{description} {name} contains nonfinite values"
                    )
            if first.dtype != parameter.dtype or second.dtype != parameter.dtype:
                raise TypeError(f"optimizer moments for {name} must match its dtype")
            parameter_arrays[name] = parameter
            first_moments.append(first)
            second_moments.append(second)

        optimizer_step = manifest["optimizer_step"]
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or optimizer_step < 0
        ):
            raise ValueError("checkpoint optimizer step must be nonnegative")
        rng_state = manifest["rng_state"]
        if rng_state is not None and not isinstance(rng_state, dict):
            raise ValueError("checkpoint RNG state must be an object or null")

        parameters = _rebuild_parameters(parameter_arrays, config)
        optimizer_state = AdamWState(
            step=optimizer_step,
            parameter_names=expected_names,
            first_moments=tuple(first_moments),
            second_moments=tuple(second_moments),
        )
        _validated_training_arrays(parameters, optimizer_state, config)
        return LoadedCheckpoint(
            config=config,
            parameters=parameters,
            optimizer_state=optimizer_state,
            rng_state=rng_state,
        )
