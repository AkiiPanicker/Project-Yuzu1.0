"""One pre-normalized residual decoder transformer block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .feed_forward import (
    FeedForwardCache,
    FeedForwardWeightGradients,
    feed_forward_backward,
    feed_forward_forward,
)
from .multi_head_attention import (
    MultiHeadAttentionCache,
    MultiHeadAttentionWeightGradients,
    multi_head_attention_backward,
    multi_head_attention_forward,
)
from .normalization import (
    RMSNormCache,
    rms_norm_backward,
    rms_norm_forward,
)


FloatArray: TypeAlias = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class TransformerBlockWeightGradients:
    """Gradients for both norms, attention, and feed-forward parameters."""

    attention_norm: FloatArray
    attention: MultiHeadAttentionWeightGradients
    feed_forward_norm: FloatArray
    feed_forward: FeedForwardWeightGradients


@dataclass(frozen=True, slots=True)
class TransformerBlockCache:
    """Composition caches needed for a complete transformer-block backward pass."""

    attention_norm: RMSNormCache
    attention: MultiHeadAttentionCache
    feed_forward_norm: RMSNormCache
    feed_forward: FeedForwardCache


def transformer_block_forward(
    inputs: FloatArray,
    attention_norm_scale: FloatArray,
    query_weight: FloatArray,
    key_weight: FloatArray,
    attention_value_weight: FloatArray,
    attention_output_weight: FloatArray,
    feed_forward_norm_scale: FloatArray,
    gate_weight: FloatArray,
    feed_forward_value_weight: FloatArray,
    feed_forward_output_weight: FloatArray,
    *,
    n_heads: int,
    epsilon: float = 1e-5,
    rope_base: float = 10_000.0,
    position_offset: int = 0,
) -> tuple[FloatArray, TransformerBlockCache]:
    """Apply both pre-normalized sublayers and their residual connections."""

    normalized_attention, attention_norm = rms_norm_forward(
        inputs,
        attention_norm_scale,
        epsilon=epsilon,
    )
    attention_output, attention = multi_head_attention_forward(
        normalized_attention,
        query_weight,
        key_weight,
        attention_value_weight,
        attention_output_weight,
        n_heads=n_heads,
        rope_base=rope_base,
        position_offset=position_offset,
    )
    post_attention = np.asarray(inputs) + attention_output

    normalized_feed_forward, feed_forward_norm = rms_norm_forward(
        post_attention,
        feed_forward_norm_scale,
        epsilon=epsilon,
    )
    feed_forward_output, feed_forward = feed_forward_forward(
        normalized_feed_forward,
        gate_weight,
        feed_forward_value_weight,
        feed_forward_output_weight,
    )
    output = post_attention + feed_forward_output
    cache = TransformerBlockCache(
        attention_norm=attention_norm,
        attention=attention,
        feed_forward_norm=feed_forward_norm,
        feed_forward=feed_forward,
    )
    return output, cache


def transformer_block_backward(
    gradient_output: FloatArray,
    cache: TransformerBlockCache,
) -> tuple[FloatArray, TransformerBlockWeightGradients]:
    """Backpropagate through both sublayers and both residual branches."""

    gradient_normalized_feed_forward, feed_forward_gradients = (
        feed_forward_backward(gradient_output, cache.feed_forward)
    )
    gradient_post_attention_from_norm, gradient_feed_forward_norm = (
        rms_norm_backward(
            gradient_normalized_feed_forward,
            cache.feed_forward_norm,
        )
    )
    gradient_post_attention = (
        np.asarray(gradient_output) + gradient_post_attention_from_norm
    )

    gradient_normalized_attention, attention_gradients = (
        multi_head_attention_backward(
            gradient_post_attention,
            cache.attention,
        )
    )
    gradient_inputs_from_norm, gradient_attention_norm = rms_norm_backward(
        gradient_normalized_attention,
        cache.attention_norm,
    )
    gradient_inputs = gradient_post_attention + gradient_inputs_from_norm
    gradients = TransformerBlockWeightGradients(
        attention_norm=gradient_attention_norm,
        attention=attention_gradients,
        feed_forward_norm=gradient_feed_forward_norm,
        feed_forward=feed_forward_gradients,
    )
    return gradient_inputs, gradients
