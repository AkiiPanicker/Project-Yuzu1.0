"""Read-only checkpoint probes for teacher-forced continuation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .byte_data import decode_utf8, encode_utf8
from .checkpoint import load_checkpoint
from .model import language_model_forward
from .numerics import logsumexp, softmax


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    """One ranked next-byte candidate."""

    token_id: int
    text: str
    probability: float


@dataclass(frozen=True, slots=True)
class ContinuationProbeStep:
    """Teacher-forced diagnostics for one expected continuation byte."""

    offset: int
    prefix_token_count: int
    expected_token_id: int
    expected_text: str
    predicted_token_id: int
    predicted_text: str
    expected_probability: float
    expected_rank: int
    negative_log_likelihood: float
    candidates: tuple[TokenCandidate, ...]

    @property
    def correct(self) -> bool:
        return self.predicted_token_id == self.expected_token_id


@dataclass(frozen=True, slots=True)
class CheckpointContinuationProbeResult:
    """Checkpoint metadata and aggregate teacher-forced continuation metrics."""

    checkpoint_path: Path
    checkpoint_step: int
    context_length: int
    prompt_text: str
    expected_text: str
    prompt_token_count: int
    expected_token_count: int
    steps: tuple[ContinuationProbeStep, ...]
    mean_loss: float
    perplexity: float
    top1_accuracy: float
    exact_match: bool


def _validated_top_k(top_k: int, vocabulary_size: int) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError(f"top_k must be in [1, {vocabulary_size}]")
    return top_k


def _decoded_byte(token_id: int) -> str:
    return decode_utf8(np.asarray([token_id], dtype=np.int64))


def probe_checkpoint_continuation(
    checkpoint_path: str | Path,
    prompt: str,
    expected_text: str,
    *,
    top_k: int = 5,
) -> CheckpointContinuationProbeResult:
    """Score each expected byte while always feeding the gold preceding bytes."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(expected_text, str):
        raise TypeError("expected_text must be a string")
    prompt_tokens = encode_utf8(prompt).astype(np.int64, copy=False)
    expected_tokens = encode_utf8(expected_text).astype(np.int64, copy=False)
    if prompt_tokens.size == 0:
        raise ValueError("prompt must contain at least one UTF-8 byte")
    if expected_tokens.size == 0:
        raise ValueError("expected_text must contain at least one UTF-8 byte")

    source = Path(checkpoint_path)
    loaded = load_checkpoint(source)
    resolved_top_k = _validated_top_k(top_k, loaded.config.vocab_size)
    maximum_prefix_count = int(prompt_tokens.size + expected_tokens.size - 1)
    if maximum_prefix_count > loaded.config.context_length:
        raise ValueError(
            "teacher-forced prefixes require up to "
            f"{maximum_prefix_count} bytes, exceeding context length "
            f"{loaded.config.context_length}"
        )

    vocabulary_ids = np.arange(loaded.config.vocab_size, dtype=np.int64)
    steps: list[ContinuationProbeStep] = []
    losses: list[float] = []
    correct_count = 0
    for offset, expected_value in enumerate(expected_tokens):
        visible_tokens = np.concatenate(
            (prompt_tokens, expected_tokens[:offset]),
        ).astype(np.int64, copy=False)
        logits, _ = language_model_forward(
            visible_tokens[None, :],
            loaded.parameters,
            loaded.config,
        )
        scores = logits[0, -1].astype(np.float64, copy=False)
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("model produced nonfinite next-token logits")
        probabilities = softmax(scores)
        ranked_ids = np.lexsort((vocabulary_ids, -scores))
        expected_token_id = int(expected_value)
        predicted_token_id = int(ranked_ids[0])
        expected_rank = int(np.flatnonzero(ranked_ids == expected_token_id)[0]) + 1
        expected_probability = float(probabilities[expected_token_id])
        negative_log_likelihood = float(
            logsumexp(scores) - scores[expected_token_id]
        )
        candidates = tuple(
            TokenCandidate(
                token_id=int(token_id),
                text=_decoded_byte(int(token_id)),
                probability=float(probabilities[token_id]),
            )
            for token_id in ranked_ids[:resolved_top_k]
        )
        step = ContinuationProbeStep(
            offset=offset,
            prefix_token_count=int(visible_tokens.size),
            expected_token_id=expected_token_id,
            expected_text=_decoded_byte(expected_token_id),
            predicted_token_id=predicted_token_id,
            predicted_text=_decoded_byte(predicted_token_id),
            expected_probability=expected_probability,
            expected_rank=expected_rank,
            negative_log_likelihood=negative_log_likelihood,
            candidates=candidates,
        )
        steps.append(step)
        losses.append(negative_log_likelihood)
        correct_count += int(step.correct)

    mean_loss = float(np.mean(losses, dtype=np.float64))
    with np.errstate(over="ignore"):
        perplexity = float(np.exp(mean_loss))
    token_count = int(expected_tokens.size)
    top1_accuracy = correct_count / token_count
    return CheckpointContinuationProbeResult(
        checkpoint_path=source,
        checkpoint_step=loaded.optimizer_state.step,
        context_length=loaded.config.context_length,
        prompt_text=prompt,
        expected_text=expected_text,
        prompt_token_count=int(prompt_tokens.size),
        expected_token_count=token_count,
        steps=tuple(steps),
        mean_loss=mean_loss,
        perplexity=perplexity,
        top1_accuracy=top1_accuracy,
        exact_match=correct_count == token_count,
    )


def format_continuation_probe(result: CheckpointContinuationProbeResult) -> str:
    """Render a probe without emitting raw model-controlled terminal bytes."""

    if not isinstance(result, CheckpointContinuationProbeResult):
        raise TypeError("result must be a CheckpointContinuationProbeResult")
    correct_count = sum(step.correct for step in result.steps)
    lines = [
        f"checkpoint={result.checkpoint_path} step={result.checkpoint_step:06d} "
        f"context={result.context_length}",
        f"prompt={result.prompt_text!r} prompt_tokens={result.prompt_token_count}",
        f"expected_text={result.expected_text!r} "
        f"expected_tokens={result.expected_token_count}",
        f"teacher_forced_top1={correct_count}/{result.expected_token_count} "
        f"accuracy={result.top1_accuracy:.3f} exact_match={result.exact_match} "
        f"mean_loss={result.mean_loss:.6f} perplexity={result.perplexity:.3f}",
    ]
    for step in result.steps:
        candidates = ", ".join(
            f"{candidate.text!r}(id={candidate.token_id},p={candidate.probability:.6f})"
            for candidate in step.candidates
        )
        lines.append(
            f"offset={step.offset:02d} prefix_tokens={step.prefix_token_count} "
            f"expected={step.expected_text!r}(id={step.expected_token_id}) "
            f"predicted={step.predicted_text!r}(id={step.predicted_token_id}) "
            f"p_expected={step.expected_probability:.6f} "
            f"rank={step.expected_rank} nll={step.negative_log_likelihood:.6f} "
            f"top=[{candidates}]"
        )
    return "\n".join(lines)
