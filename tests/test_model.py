"""Composition and gradient tests for the stacked byte language model."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from typing import Callable
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    LanguageModelParameters,
    ModelConfig,
    TransformerBlockParameters,
    check_gradient,
    embedding_backward,
    embedding_forward,
    language_model_backward,
    language_model_forward,
    linear_backward,
    linear_forward,
    rms_norm_backward,
    rms_norm_forward,
    transformer_block_backward,
    transformer_block_forward,
)


ParameterReplacer = Callable[[np.ndarray], LanguageModelParameters]


def _tiny_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 7,
        "context_length": 5,
        "d_model": 4,
        "n_layers": 4,
        "n_heads": 2,
        "d_ff": 5,
        "seed": 163,
    }
    values.update(overrides)
    return ModelConfig(**values)


def _random_block(
    rng: np.random.Generator,
    config: ModelConfig,
    *,
    scale: float = 0.15,
) -> TransformerBlockParameters:
    width = config.d_model
    hidden = config.d_ff
    return TransformerBlockParameters(
        attention_norm_scale=1.0 + rng.normal(scale=scale, size=width),
        query_weight=rng.normal(scale=scale, size=(width, width)),
        key_weight=rng.normal(scale=scale, size=(width, width)),
        attention_value_weight=rng.normal(scale=scale, size=(width, width)),
        attention_output_weight=rng.normal(scale=scale, size=(width, width)),
        feed_forward_norm_scale=1.0 + rng.normal(scale=scale, size=width),
        gate_weight=rng.normal(scale=scale, size=(width, hidden)),
        feed_forward_value_weight=rng.normal(scale=scale, size=(width, hidden)),
        feed_forward_output_weight=rng.normal(scale=scale, size=(hidden, width)),
    )


def _random_parameters(
    rng: np.random.Generator,
    config: ModelConfig,
    *,
    scale: float = 0.15,
) -> LanguageModelParameters:
    return LanguageModelParameters(
        token_embedding=rng.normal(
            scale=scale,
            size=(config.vocab_size, config.d_model),
        ),
        blocks=tuple(
            _random_block(rng, config, scale=scale)
            for _ in range(config.n_layers)
        ),
        final_norm_scale=1.0 + rng.normal(scale=scale, size=config.d_model),
    )


class LanguageModelTests(unittest.TestCase):
    def test_forward_matches_explicit_primitive_composition(self) -> None:
        rng = np.random.default_rng(163)
        config = _tiny_config()
        parameters = _random_parameters(rng, config)
        token_ids = np.array([[0, 3, 1], [2, 6, 4]], dtype=np.int64)
        logits, _ = language_model_forward(token_ids, parameters, config)

        hidden, _ = embedding_forward(token_ids, parameters.token_embedding)
        for block in parameters.blocks:
            hidden, _ = transformer_block_forward(
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
            )
        normalized, _ = rms_norm_forward(
            hidden,
            parameters.final_norm_scale,
            epsilon=config.norm_epsilon,
        )
        expected, _ = linear_forward(normalized, parameters.token_embedding.T)
        np.testing.assert_allclose(logits, expected)

    def test_four_block_cache_and_logit_shapes_match_configuration(self) -> None:
        rng = np.random.default_rng(167)
        config = _tiny_config()
        parameters = _random_parameters(rng, config)
        token_ids = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
        logits, cache = language_model_forward(token_ids, parameters, config)
        self.assertEqual(logits.shape, (2, 4, config.vocab_size))
        self.assertEqual(len(cache.blocks), 4)
        actual_parameter_count = (
            parameters.token_embedding.size
            + parameters.final_norm_scale.size
            + sum(
                sum(getattr(block, field).size for field in block.__slots__)
                for block in parameters.blocks
            )
        )
        self.assertEqual(actual_parameter_count, config.parameter_count())
        for block_cache in cache.blocks:
            self.assertEqual(
                block_cache.attention.attention.probabilities.shape,
                (2, config.n_heads, 4, 4),
            )

    def test_future_tokens_do_not_affect_earlier_logits(self) -> None:
        rng = np.random.default_rng(173)
        config = _tiny_config()
        parameters = _random_parameters(rng, config)
        original_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        changed_ids = np.array([[0, 1, 2, 6, 5]], dtype=np.int64)
        original, _ = language_model_forward(original_ids, parameters, config)
        changed, _ = language_model_forward(changed_ids, parameters, config)
        np.testing.assert_allclose(changed[:, :3, :], original[:, :3, :])

    def test_tied_embedding_gradient_adds_input_and_output_uses(self) -> None:
        rng = np.random.default_rng(179)
        config = _tiny_config()
        parameters = _random_parameters(rng, config)
        token_ids = np.array([[1, 1, 4]], dtype=np.int64)
        logits, cache = language_model_forward(token_ids, parameters, config)
        gradient_logits = rng.normal(size=logits.shape)
        gradients = language_model_backward(gradient_logits, cache)

        gradient_hidden, gradient_output_weight, _ = linear_backward(
            gradient_logits,
            cache.output_linear,
        )
        gradient_hidden, _ = rms_norm_backward(
            gradient_hidden,
            cache.final_norm,
        )
        for block_cache in reversed(cache.blocks):
            gradient_hidden, _ = transformer_block_backward(
                gradient_hidden,
                block_cache,
            )
        gradient_input_embedding = embedding_backward(
            gradient_hidden,
            cache.embedding,
        )
        expected = gradient_input_embedding + gradient_output_weight.T
        np.testing.assert_allclose(gradients.token_embedding, expected)

    def test_all_parameter_gradients_match_finite_differences(self) -> None:
        rng = np.random.default_rng(181)
        config = _tiny_config(vocab_size=5, context_length=2, d_ff=3)
        parameters = _random_parameters(rng, config, scale=0.1)
        token_ids = np.array([[0, 3]], dtype=np.int64)
        logits, cache = language_model_forward(token_ids, parameters, config)
        gradient_logits = rng.normal(scale=0.3, size=logits.shape)
        gradients = language_model_backward(gradient_logits, cache)

        def objective(candidate: LanguageModelParameters) -> float:
            candidate_logits, _ = language_model_forward(
                token_ids,
                candidate,
                config,
            )
            return float(np.sum(candidate_logits * gradient_logits))

        checks: list[
            tuple[str, np.ndarray, np.ndarray, ParameterReplacer]
        ] = [
            (
                "token_embedding",
                parameters.token_embedding,
                gradients.token_embedding,
                lambda candidate: replace(
                    parameters,
                    token_embedding=candidate,
                ),
            ),
            (
                "final_norm_scale",
                parameters.final_norm_scale,
                gradients.final_norm_scale,
                lambda candidate: replace(
                    parameters,
                    final_norm_scale=candidate,
                ),
            ),
        ]
        gradient_fields = {
            "attention_norm_scale": lambda gradient: gradient.attention_norm,
            "query_weight": lambda gradient: gradient.attention.query,
            "key_weight": lambda gradient: gradient.attention.key,
            "attention_value_weight": lambda gradient: gradient.attention.value,
            "attention_output_weight": lambda gradient: gradient.attention.output,
            "feed_forward_norm_scale": (
                lambda gradient: gradient.feed_forward_norm
            ),
            "gate_weight": lambda gradient: gradient.feed_forward.gate,
            "feed_forward_value_weight": (
                lambda gradient: gradient.feed_forward.value
            ),
            "feed_forward_output_weight": (
                lambda gradient: gradient.feed_forward.output
            ),
        }
        for block_index, (block, block_gradient) in enumerate(
            zip(parameters.blocks, gradients.blocks, strict=True)
        ):
            for field_name, gradient_getter in gradient_fields.items():
                values = getattr(block, field_name)
                analytical = gradient_getter(block_gradient)

                def replace_block_parameter(
                    candidate: np.ndarray,
                    index: int = block_index,
                    name: str = field_name,
                ) -> LanguageModelParameters:
                    blocks = list(parameters.blocks)
                    blocks[index] = replace(blocks[index], **{name: candidate})
                    return replace(parameters, blocks=tuple(blocks))

                checks.append(
                    (
                        f"blocks[{block_index}].{field_name}",
                        values,
                        analytical,
                        replace_block_parameter,
                    )
                )

        for name, values, analytical, replace_parameter in checks:
            result = check_gradient(
                lambda candidate, replacer=replace_parameter: objective(
                    replacer(candidate)
                ),
                values,
                analytical,
                absolute_tolerance=5e-7,
                relative_tolerance=2e-5,
            )
            with self.subTest(name=name):
                self.assertTrue(result.passed, msg=str(result))

    def test_invalid_model_contracts_are_rejected(self) -> None:
        rng = np.random.default_rng(191)
        config = _tiny_config()
        parameters = _random_parameters(rng, config)
        token_ids = np.array([[0, 1]], dtype=np.int64)
        with self.subTest(case="block count"):
            with self.assertRaisesRegex(ValueError, "expected 4 transformer blocks"):
                language_model_forward(
                    token_ids,
                    replace(parameters, blocks=parameters.blocks[:-1]),
                    config,
                )
        with self.subTest(case="context length"):
            with self.assertRaisesRegex(ValueError, "exceeds context length"):
                language_model_forward(
                    np.zeros((1, 6), dtype=np.int64),
                    parameters,
                    config,
                )
        with self.subTest(case="position offset"):
            with self.assertRaisesRegex(ValueError, "exceeds context length"):
                language_model_forward(
                    token_ids,
                    parameters,
                    config,
                    position_offset=4,
                )
        with self.subTest(case="embedding shape"):
            with self.assertRaisesRegex(ValueError, "token_embedding"):
                language_model_forward(
                    token_ids,
                    replace(
                        parameters,
                        token_embedding=parameters.token_embedding[:, :3],
                    ),
                    config,
                )
        with self.subTest(case="untied"):
            with self.assertRaisesRegex(ValueError, "tied embeddings"):
                language_model_forward(
                    token_ids,
                    parameters,
                    replace(config, tie_embeddings=False),
                )
        with self.subTest(case="bias"):
            with self.assertRaisesRegex(ValueError, "bias-free"):
                language_model_forward(
                    token_ids,
                    parameters,
                    replace(config, use_bias=True),
                )


if __name__ == "__main__":
    unittest.main()
