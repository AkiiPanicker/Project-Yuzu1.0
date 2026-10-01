# GPT from scratch

An educational decoder-only language model implemented with the Python standard
library and NumPy. The first milestone is correctness and understanding; later
milestones optimize quality and move the verified design to a CUDA-capable
backend.

## Non-negotiable constraints

- The model implementation uses only the Python standard library and NumPy.
- Every trainable operation receives a tested backward pass.
- Tiny tests must pass before model size or dataset size increases.
- Source documents are never modified or silently copied into the repository.
- Personal and sensitive documents are excluded by default.
- Data must pass rights, privacy, duplication, and extraction-quality checks.
- Results are compared against explicit baselines rather than subjective demos.

## Initial model

- Byte vocabulary: 256 values
- Context length: 256 tokens
- Width: 256
- Layers: 4
- Attention heads: 4
- SwiGLU hidden width: 704
- Approximate trainable parameters: 3,279,104

The planned block is a pre-normalized decoder block:

```text
x = x + causal_attention(rms_norm(x))
x = x + swiglu_feed_forward(rms_norm(x))
```

The complete network adds byte embeddings, RoPE, a final RMSNorm, tied output
weights, and next-token cross-entropy.

## Project layout

```text
configs/              Versioned experiment configurations
data/manifests/       Metadata only; no book or paper contents
docs/                 Architecture and data decisions
scripts/              Reproducible corpus and training utilities
src/numpy_gpt/         Model implementation
tests/                 Standard-library unittest suite
```

## Current status

Phase 1 is active: numerical primitives are being implemented and gradient-checked.
No training data has been ingested and no model training has started.

## Commands

```powershell
python -m unittest discover -s tests -v
python scripts/inventory_corpus.py
```

## Roadmap

1. Corpus audit, exclusions, and text-extraction assessment.
2. Tensor-operation gradient checks and numerical stability tests.
3. Byte tokenizer, batching, deterministic random seeds, and bigram baseline.
4. Linear, RMSNorm, RoPE, causal attention, and SwiGLU modules.
5. Full forward/backward transformer and deliberate tiny-set overfitting.
6. Checkpointing, AdamW, evaluation, and sampling.
7. Train the 3.28M model in NumPy.
8. Port the verified architecture to a GPU backend without changing semantics.
9. Add instruction tuning, summarization, retrieval, voice, and tools as separate
   evaluated stages.
