"""Deterministic initialization for the complete NumPy language model."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .config import ModelConfig
from .model import LanguageModelParameters, TransformerBlockParameters


DTypeLike: TypeAlias = np.dtype | type[np.floating]


def _normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    standard_deviation: float,
    dtype: np.dtype,
) -> np.ndarray:
    return rng.normal(
        loc=0.0,
        scale=standard_deviation,
        size=shape,
    ).astype(dtype, copy=False)


def initialize_parameters(
    config: ModelConfig,
    *,
    dtype: DTypeLike = np.float32,
    seed: int | None = None,
) -> LanguageModelParameters:
    """Create all model arrays from one reproducible random-number stream."""

    resolved_dtype = np.dtype(dtype)
    if not np.issubdtype(resolved_dtype, np.floating):
        raise TypeError(f"dtype must be floating-point, got {resolved_dtype}")
    resolved_seed = config.seed if seed is None else seed
    if (
        not isinstance(resolved_seed, int)
        or isinstance(resolved_seed, bool)
        or resolved_seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    if not config.tie_embeddings:
        raise ValueError("initialization requires tied embeddings")
    if config.use_bias:
        raise ValueError("initialization currently requires bias-free projections")

    rng = np.random.default_rng(resolved_seed)
    width = config.d_model
    hidden = config.d_ff
    base_std = config.initializer_std
    residual_std = base_std / np.sqrt(2.0 * config.n_layers)

    token_embedding = _normal(
        rng,
        (config.vocab_size, width),
        base_std,
        resolved_dtype,
    )
    blocks: list[TransformerBlockParameters] = []
    for _ in range(config.n_layers):
        blocks.append(
            TransformerBlockParameters(
                attention_norm_scale=np.ones(width, dtype=resolved_dtype),
                query_weight=_normal(
                    rng,
                    (width, width),
                    base_std,
                    resolved_dtype,
                ),
                key_weight=_normal(
                    rng,
                    (width, width),
                    base_std,
                    resolved_dtype,
                ),
                attention_value_weight=_normal(
                    rng,
                    (width, width),
                    base_std,
                    resolved_dtype,
                ),
                attention_output_weight=_normal(
                    rng,
                    (width, width),
                    residual_std,
                    resolved_dtype,
                ),
                feed_forward_norm_scale=np.ones(width, dtype=resolved_dtype),
                gate_weight=_normal(
                    rng,
                    (width, hidden),
                    base_std,
                    resolved_dtype,
                ),
                feed_forward_value_weight=_normal(
                    rng,
                    (width, hidden),
                    base_std,
                    resolved_dtype,
                ),
                feed_forward_output_weight=_normal(
                    rng,
                    (hidden, width),
                    residual_std,
                    resolved_dtype,
                ),
            )
        )
    return LanguageModelParameters(
        token_embedding=token_embedding,
        blocks=tuple(blocks),
        final_norm_scale=np.ones(width, dtype=resolved_dtype),
    )
