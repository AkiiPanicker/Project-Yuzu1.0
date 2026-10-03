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
planned checkpoints. This demonstrates optimization on the repetitive synthetic
corpus; it does not demonstrate language ability. Greedy and seeded stochastic
autoregressive sampling with temperature, top-k filtering, and context-limit
stopping are verified by the 106-test checkpoint. Checkpoint-backed one-shot
generation, safe terminal rendering, and generation timing are verified by the
112-test checkpoint. A teacher-forced continuation diagnostic is verified by the
118-test checkpoint. Step 20 generated ten spaces after `Yuzu ` and step 200
generated ` tte patte`, so neither had memorized the fixed target `learns byt`.
An unchanged resume then reached 9/10 teacher-forced accuracy at step 250 and
10/10 exact match at step 300, with mean loss 0.230220 and perplexity 1.259. The
unchanged step-300 greedy probe generated the exact suffix `learns byt`, producing
`Yuzu learns byt`. This closes the tiny synthetic-pattern integration gate:
teacher-forced top-1 predictions and free-running checkpoint generation agree on
this memorized continuation. It does not demonstrate generalization or
conversational ability. A stateless interactive checkpoint diagnostic is now
implemented with independent prompts, no conversation history, terminal-safe
output, and local JSON-lines session logs. Its eight new tests are verified by the
user-reported 126-test checkpoint. A user-run step-300 session then reproduced
`Yuzu learns byt`, displayed the scope/help text, and exited through `/exit`; the
local log independently records one turn and `reason=command`. This closes the
stateless diagnostic runtime gate. A separate metadata-only corpus-admission gate
is now implemented with an inventory-bound decision manifest, immutable
sensitive/duplicate/OCR holds, explicit rights/privacy/extraction reviews, and
live source hash checks. Its eight new tests await user verification; the initial
empty decision list is designed to refuse all files. No book or paper data has
been ingested.

## Current verification gate

Run the full suite first:

```powershell
python -m unittest discover -s tests -v
```

Only after all 134 tests finish with `OK`, run the admission command separately:

```powershell
python scripts\admit_corpus.py
```

Its initial closed-gate result is expected to exit `1`, so do not join these two
commands with `&&`.

## Historical and reproducibility commands

```powershell
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000020.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000200.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python -c "import sys; sys.path.insert(0,'src'); from numpy_gpt import probe_checkpoint_continuation,format_continuation_probe; r=probe_checkpoint_continuation(r'runs\smoke-20261002-140621\checkpoint-step-00000200.npz','at a time.\nYuzu ','l',top_k=5); print(format_continuation_probe(r))"
python scripts\train_smoke.py --resume runs\smoke-20261002-140621\checkpoint-step-00000200.npz --run-dir runs\smoke-20261002-140621 --steps 300 --validation-interval 10 --checkpoint-interval 50
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000250.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\probe_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000300.npz --prompt "Yuzu " --expected "learns byt" --top-k 5
python scripts\generate_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000300.npz --prompt "Yuzu " --max-new-tokens 10 --greedy
python scripts\interactive_checkpoint.py runs\smoke-20261002-140621\checkpoint-step-00000300.npz --greedy --max-new-tokens 10
python scripts/train_smoke.py --steps 20
python scripts/inventory_corpus.py
```

Before increasing model size, two architecture gates also remain open: compare
generation with a fixed-seed bigram baseline and record peak process memory along
with the existing tokens-per-second measurements.

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
