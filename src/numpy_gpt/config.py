"""Validated model configuration and parameter-budget calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the first dense decoder-only model."""

    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 704
    rope_base: float = 10_000.0
    norm_epsilon: float = 1e-5
    tie_embeddings: bool = True
    use_bias: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameter_count(self) -> int:
        """Return the exact count for the planned bias-free, tied architecture."""

        embeddings = self.vocab_size * self.d_model
        attention_per_layer = 4 * self.d_model * self.d_model
        swiglu_per_layer = 3 * self.d_model * self.d_ff
        norms_per_layer = 2 * self.d_model
        biases_per_layer = 0
        if self.use_bias:
            biases_per_layer = 4 * self.d_model + 2 * self.d_ff + self.d_model
        output_head = 0 if self.tie_embeddings else embeddings
        final_norm = self.d_model
        return (
            embeddings
            + self.n_layers
            * (
                attention_per_layer
                + swiglu_per_layer
                + norms_per_layer
                + biases_per_layer
            )
            + final_norm
            + output_head
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls(**values)

