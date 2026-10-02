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
from .byte_data import (
    ByteBatch,
    ByteCorpusSplit,
    decode_utf8,
    encode_utf8,
    sample_next_token_batch,
    split_byte_tokens,
)
from .checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from .config import ModelConfig
from .feed_forward import (
    FeedForwardCache,
    FeedForwardWeightGradients,
    feed_forward_backward,
    feed_forward_forward,
)
from .gradcheck import GradientCheckResult, check_gradient, finite_difference_gradient
from .initialization import initialize_parameters
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
from .model import (
    LanguageModelCache,
    LanguageModelParameters,
    LanguageModelWeightGradients,
    TransformerBlockParameters,
    language_model_backward,
    language_model_forward,
    named_gradients,
    named_parameters,
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
from .optimizer import (
    AdamWState,
    AdamWStepStats,
    GradientClippingResult,
    adamw_step,
    clip_gradients_by_global_norm,
    global_gradient_norm,
    initialize_adamw_state,
)
from .position import RoPECache, rope_backward, rope_forward
from .sampling import GenerationResult, generate_tokens, select_next_token
from .transformer_block import (
    TransformerBlockCache,
    TransformerBlockWeightGradients,
    transformer_block_backward,
    transformer_block_forward,
)
from .training import (
    EvaluationMetrics,
    TrainingStepStats,
    evaluate_batch,
    train_step,
)
from .trainer import (
    TrainingLogRecord,
    TrainingLoopConfig,
    TrainingRunResult,
    run_training,
)

__all__ = [
    "AdamWState",
    "AdamWStepStats",
    "ByteBatch",
    "ByteCorpusSplit",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "CausalAttentionCache",
    "EmbeddingCache",
    "EvaluationMetrics",
    "FeedForwardCache",
    "FeedForwardWeightGradients",
    "GradientCheckResult",
    "GradientClippingResult",
    "GenerationResult",
    "LayerNormCache",
    "LanguageModelCache",
    "LanguageModelParameters",
    "LanguageModelWeightGradients",
    "LinearCache",
    "LoadedCheckpoint",
    "ModelConfig",
    "MultiHeadAttentionCache",
    "MultiHeadAttentionWeightGradients",
    "RMSNormCache",
    "RoPECache",
    "SiLUCache",
    "SwiGLUCache",
    "TransformerBlockCache",
    "TransformerBlockParameters",
    "TransformerBlockWeightGradients",
    "TrainingLogRecord",
    "TrainingLoopConfig",
    "TrainingRunResult",
    "TrainingStepStats",
    "adamw_step",
    "check_gradient",
    "causal_attention_backward",
    "causal_attention_forward",
    "clip_gradients_by_global_norm",
    "cross_entropy_with_logits",
    "decode_utf8",
    "embedding_backward",
    "embedding_forward",
    "evaluate_batch",
    "encode_utf8",
    "feed_forward_backward",
    "feed_forward_forward",
    "finite_difference_gradient",
    "global_gradient_norm",
    "generate_tokens",
    "initialize_adamw_state",
    "initialize_parameters",
    "layer_norm_backward",
    "layer_norm_forward",
    "linear_backward",
    "linear_forward",
    "language_model_backward",
    "language_model_forward",
    "load_checkpoint",
    "logsumexp",
    "multi_head_attention_backward",
    "multi_head_attention_forward",
    "named_gradients",
    "named_parameters",
    "rms_norm_backward",
    "rms_norm_forward",
    "rope_backward",
    "rope_forward",
    "sample_next_token_batch",
    "save_checkpoint",
    "select_next_token",
    "silu_backward",
    "silu_forward",
    "softmax",
    "split_byte_tokens",
    "stable_sigmoid",
    "swiglu_backward",
    "swiglu_forward",
    "transformer_block_backward",
    "transformer_block_forward",
    "run_training",
    "train_step",
]
