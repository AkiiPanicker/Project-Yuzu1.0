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

Phase 1 is active. Numerical primitives and one complete pre-normalized residual
transformer block are verified by 58 deterministic and finite-difference tests.
The four-block language model with tied input/output embeddings is verified by the
64-test checkpoint. Deterministic depth-scaled parameter initialization and named
parameter traversal are verified by the 70-test checkpoint. Global-norm clipping
and AdamW are verified by the 76-test checkpoint. Versioned, non-pickle checkpoint
save/restore is verified by the 82-test checkpoint. UTF-8 byte encoding, leak-free
splitting, and deterministic next-token batches are verified by the 88-test
checkpoint. One complete mean-loss, backward, clipped-AdamW training step and a
read-only evaluation path are verified by the 94-test checkpoint. A resumable,
terminal-visible training controller, JSON-lines metrics, checkpoint events, and
a small synthetic smoke-training command are verified by the 100-test checkpoint.
The first 20-step synthetic smoke run reduced validation loss from 5.544793 to
3.974303 and validation perplexity from 255.902 to 53.213, while writing both
planned checkpoints. This proves the small model can learn a repetitive byte
pattern; it does not demonstrate language ability. Greedy and seeded stochastic
autoregressive sampling with temperature,
top-k filtering, and context-limit stopping are verified by the 106-test
checkpoint. Checkpoint-backed one-shot generation, safe terminal rendering, and
generation timing are verified by the 112-test checkpoint. Greedy decoding from
step 20 produced ten spaces after `Yuzu `, showing byte-frequency learning rather
than sequence memorization. Exact resume to step 200 then reduced validation loss
from 3.974303 to 0.867316 and perplexity from 53.213 to 2.381. The identical
step-200 greedy probe produced ` tte patte`: learned sentence fragments, but not
the target `learns byt`. A teacher-forced continuation diagnostic is verified by
the 118-test checkpoint. Its step-20/step-200 comparison improved top-1 byte
accuracy from 10% to 70% and perplexity from 51.579 to 2.779, but the first
expected `l` remains rank three: space is still 8.09 times likelier. The next gate
scored that byte after the exact 16-byte training-like context and still returned
rank three, with space 8.24 times likelier. This rejects context mismatch as the
immediate explanation. The next gate is a controlled unchanged resume from step
200 to step 300, followed by fixed probes at 250 and 300. That run completed
stably: step 250 reached 9/10 teacher-forced accuracy and step 300 reached 10/10
with exact match, loss 0.230220, and perplexity 1.259. The next gate is unchanged
greedy generation from step 300. No book or paper data has been ingested.

## Commands

```powershell
python -m unittest discover -s tests -v
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000020.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000200.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python -c "import sys; sys.path.insert(0,'src'); from numpy_gpt import probe_checkpoint_continuation,format_continuation_probe; r=probe_checkpoint_continuation(r'runs\smoke-20261002-140621\checkpoint-step-00000200.npz','at a time.\nYuzu ','l',top_k=5); print(format_continuation_probe(r))"
python scripts\train_smoke.py --resume runs\smoke-20261002-140621\checkpoint-step-00000200.npz --run-dir runs\smoke-20261002-140621 --steps 300 --validation-interval 10 --checkpoint-interval 50
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000250.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000300.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\generate_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000300.npz --prompt "Yuzu " --max-new-tokens 10 --greedy
python scripts/train_smoke.py --steps 20
python scripts/inventory_corpus.py
```

## Roadmap

1. Corpus audit, exclusions, and text-extraction assessment.
2. Tensor-operation gradient checks and numerical stability tests.
3. Byte tokenizer, batching, deterministic random seeds, and bigram baseline.
4. Linear, RMSNorm, RoPE, causal attention, and SwiGLU modules.
5. Full forward/backward transformer and deliberate tiny-set overfitting.
6. Checkpointing, deterministic training, evaluation, and sampling.
7. Train the 3.28M model in NumPy.
8. Port the verified architecture to a GPU backend without changing semantics.
9. Add instruction tuning, summarization, retrieval, voice, and tools as separate
   evaluated stages.
