"""NumPy GPT package."""

from .activations import (
    SiLUCache,
    SwiGLUCache,
    silu_backward,
    silu_forward,
    stable_sigmoid,
    swiglu_backward,
    swiglu_forward,
)
from .attention import (
    CausalAttentionCache,
    causal_attention_backward,
    causal_attention_forward,
)
from .config import ModelConfig
from .feed_forward import (
    FeedForwardCache,
    FeedForwardWeightGradients,
    feed_forward_backward,
    feed_forward_forward,
)
from .gradcheck import GradientCheckResult, check_gradient, finite_difference_gradient
from .layers import (
    EmbeddingCache,
    LinearCache,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
)
from .multi_head_attention import (
    MultiHeadAttentionCache,
    MultiHeadAttentionWeightGradients,
    multi_head_attention_backward,
    multi_head_attention_forward,
)
from .normalization import (
    LayerNormCache,
    RMSNormCache,
    layer_norm_backward,
    layer_norm_forward,
    rms_norm_backward,
    rms_norm_forward,
)
from .numerics import cross_entropy_with_logits, logsumexp, softmax
from .position import RoPECache, rope_backward, rope_forward

__all__ = [
    "CausalAttentionCache",
    "EmbeddingCache",
    "FeedForwardCache",
    "FeedForwardWeightGradients",
    "GradientCheckResult",
    "LayerNormCache",
    "LinearCache",
    "ModelConfig",
    "MultiHeadAttentionCache",
    "MultiHeadAttentionWeightGradients",
    "RMSNormCache",
    "RoPECache",
    "SiLUCache",
    "SwiGLUCache",
    "check_gradient",
    "causal_attention_backward",
    "causal_attention_forward",
    "cross_entropy_with_logits",
    "embedding_backward",
    "embedding_forward",
    "feed_forward_backward",
    "feed_forward_forward",
    "finite_difference_gradient",
    "layer_norm_backward",
    "layer_norm_forward",
    "linear_backward",
    "linear_forward",
    "logsumexp",
    "multi_head_attention_backward",
    "multi_head_attention_forward",
    "rms_norm_backward",
    "rms_norm_forward",
    "rope_backward",
    "rope_forward",
    "silu_backward",
    "silu_forward",
    "softmax",
    "stable_sigmoid",
    "swiglu_backward",
    "swiglu_forward",
]
