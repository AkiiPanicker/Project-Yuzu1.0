"""Stacked decoder-only byte language model with tied token embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .config import ModelConfig
from .layers import (
    EmbeddingCache,
    LinearCache,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
)
from .normalization import RMSNormCache, rms_norm_backward, rms_norm_forward
from .transformer_block import (
    TransformerBlockCache,
    TransformerBlockWeightGradients,
    transformer_block_backward,
    transformer_block_forward,
)


FloatArray: TypeAlias = NDArray[np.floating]
IntegerArray: TypeAlias = NDArray[np.integer]


@dataclass(frozen=True, slots=True)
class TransformerBlockParameters:
    """The two norm scales and seven projection matrices in one block."""

    attention_norm_scale: FloatArray
    query_weight: FloatArray
    key_weight: FloatArray
    attention_value_weight: FloatArray
    attention_output_weight: FloatArray
    feed_forward_norm_scale: FloatArray
    gate_weight: FloatArray
    feed_forward_value_weight: FloatArray
    feed_forward_output_weight: FloatArray


@dataclass(frozen=True, slots=True)
class LanguageModelParameters:
    """All trainable arrays in the tied, bias-free language model."""

    token_embedding: FloatArray
    blocks: tuple[TransformerBlockParameters, ...]
    final_norm_scale: FloatArray


@dataclass(frozen=True, slots=True)
class LanguageModelWeightGradients:
    """Gradients matching the nested language-model parameter structure."""

    token_embedding: FloatArray
    blocks: tuple[TransformerBlockWeightGradients, ...]
    final_norm_scale: FloatArray


@dataclass(frozen=True, slots=True)
class LanguageModelCache:
    """Caches required to reverse the embedding, block stack, and output head."""

    embedding: EmbeddingCache
    blocks: tuple[TransformerBlockCache, ...]
    final_norm: RMSNormCache
    output_linear: LinearCache


def _require_shape(values: FloatArray, expected: tuple[int, ...], name: str) -> None:
    actual = np.asarray(values).shape
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")


def _validate_parameters(
    parameters: LanguageModelParameters,
    config: ModelConfig,
) -> None:
    if not config.tie_embeddings:
        raise ValueError("language model requires tied embeddings")
    if config.use_bias:
        raise ValueError("language model currently requires bias-free projections")
    if len(parameters.blocks) != config.n_layers:
        raise ValueError(
            f"expected {config.n_layers} transformer blocks, "
            f"got {len(parameters.blocks)}"
        )

    model_shape = (config.d_model, config.d_model)
    feed_forward_input_shape = (config.d_model, config.d_ff)
    feed_forward_output_shape = (config.d_ff, config.d_model)
    _require_shape(
        parameters.token_embedding,
        (config.vocab_size, config.d_model),
        "token_embedding",
    )
    _require_shape(
        parameters.final_norm_scale,
        (config.d_model,),
        "final_norm_scale",
    )
    for index, block in enumerate(parameters.blocks):
        prefix = f"blocks[{index}]"
        _require_shape(
            block.attention_norm_scale,
            (config.d_model,),
            f"{prefix}.attention_norm_scale",
        )
        _require_shape(block.query_weight, model_shape, f"{prefix}.query_weight")
        _require_shape(block.key_weight, model_shape, f"{prefix}.key_weight")
        _require_shape(
            block.attention_value_weight,
            model_shape,
            f"{prefix}.attention_value_weight",
        )
        _require_shape(
            block.attention_output_weight,
            model_shape,
            f"{prefix}.attention_output_weight",
        )
        _require_shape(
            block.feed_forward_norm_scale,
            (config.d_model,),
            f"{prefix}.feed_forward_norm_scale",
        )
        _require_shape(
            block.gate_weight,
            feed_forward_input_shape,
            f"{prefix}.gate_weight",
        )
        _require_shape(
            block.feed_forward_value_weight,
            feed_forward_input_shape,
            f"{prefix}.feed_forward_value_weight",
        )
        _require_shape(
            block.feed_forward_output_weight,
            feed_forward_output_shape,
            f"{prefix}.feed_forward_output_weight",
        )


def language_model_forward(
    token_ids: IntegerArray,
    parameters: LanguageModelParameters,
    config: ModelConfig,
    *,
    position_offset: int = 0,
) -> tuple[FloatArray, LanguageModelCache]:
    """Map byte-token IDs to next-token logits through the complete stack."""

    ids = np.asarray(token_ids)
    if ids.ndim < 1:
        raise ValueError("token_ids must include a sequence dimension")
    if (
        not isinstance(position_offset, int)
        or isinstance(position_offset, bool)
        or position_offset < 0
    ):
        raise ValueError("position_offset must be a nonnegative integer")
    sequence_end = position_offset + ids.shape[-1]
    if sequence_end > config.context_length:
        raise ValueError(
            f"sequence end {sequence_end} exceeds context length "
            f"{config.context_length}"
        )
    _validate_parameters(parameters, config)

    hidden, embedding = embedding_forward(ids, parameters.token_embedding)
    block_caches: list[TransformerBlockCache] = []
    for block in parameters.blocks:
        hidden, block_cache = transformer_block_forward(
            hidden,
            block.attention_norm_scale,
            block.query_weight,
            block.key_weight,
            block.attention_value_weight,
            block.attention_output_weight,
            block.feed_forward_norm_scale,
            block.gate_weight,
            block.feed_forward_value_weight,
            block.feed_forward_output_weight,
            n_heads=config.n_heads,
            epsilon=config.norm_epsilon,
            rope_base=config.rope_base,
            position_offset=position_offset,
        )
        block_caches.append(block_cache)

    normalized, final_norm = rms_norm_forward(
        hidden,
        parameters.final_norm_scale,
        epsilon=config.norm_epsilon,
    )
    logits, output_linear = linear_forward(
        normalized,
        parameters.token_embedding.T,
    )
    cache = LanguageModelCache(
        embedding=embedding,
        blocks=tuple(block_caches),
        final_norm=final_norm,
        output_linear=output_linear,
    )
    return logits, cache


def language_model_backward(
    gradient_logits: FloatArray,
    cache: LanguageModelCache,
) -> LanguageModelWeightGradients:
    """Backpropagate logits and add both uses of the tied embedding gradient."""

    gradient_hidden, gradient_output_weight, _ = linear_backward(
        gradient_logits,
        cache.output_linear,
    )
    gradient_hidden, gradient_final_norm = rms_norm_backward(
        gradient_hidden,
        cache.final_norm,
    )

    reversed_block_gradients: list[TransformerBlockWeightGradients] = []
    for block_cache in reversed(cache.blocks):
        gradient_hidden, block_gradients = transformer_block_backward(
            gradient_hidden,
            block_cache,
        )
        reversed_block_gradients.append(block_gradients)
    block_gradients = tuple(reversed(reversed_block_gradients))

    gradient_input_embedding = embedding_backward(
        gradient_hidden,
        cache.embedding,
    )
    gradient_token_embedding = (
        gradient_input_embedding + gradient_output_weight.T
    )
    return LanguageModelWeightGradients(
        token_embedding=gradient_token_embedding,
        blocks=block_gradients,
        final_norm_scale=gradient_final_norm,
    )


def named_parameters(
    parameters: LanguageModelParameters,
) -> Iterator[tuple[str, FloatArray]]:
    """Yield every unique trainable array in a deterministic optimizer order."""

    yield "token_embedding", parameters.token_embedding
    for index, block in enumerate(parameters.blocks):
        prefix = f"blocks.{index}"
        yield f"{prefix}.attention_norm_scale", block.attention_norm_scale
        yield f"{prefix}.query_weight", block.query_weight
        yield f"{prefix}.key_weight", block.key_weight
        yield f"{prefix}.attention_value_weight", block.attention_value_weight
        yield f"{prefix}.attention_output_weight", block.attention_output_weight
        yield f"{prefix}.feed_forward_norm_scale", block.feed_forward_norm_scale
        yield f"{prefix}.gate_weight", block.gate_weight
        yield f"{prefix}.feed_forward_value_weight", block.feed_forward_value_weight
        yield f"{prefix}.feed_forward_output_weight", block.feed_forward_output_weight
    yield "final_norm_scale", parameters.final_norm_scale
