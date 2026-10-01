"""NumPy GPT package."""

from .config import ModelConfig
from .gradcheck import GradientCheckResult, check_gradient, finite_difference_gradient
from .layers import (
    EmbeddingCache,
    LinearCache,
    embedding_backward,
    embedding_forward,
    linear_backward,
    linear_forward,
)
from .numerics import cross_entropy_with_logits, logsumexp, softmax

__all__ = [
    "EmbeddingCache",
    "GradientCheckResult",
    "LinearCache",
    "ModelConfig",
    "check_gradient",
    "cross_entropy_with_logits",
    "embedding_backward",
    "embedding_forward",
    "finite_difference_gradient",
    "linear_backward",
    "linear_forward",
    "logsumexp",
    "softmax",
]

