# Architecture decision record

## Objective

Build the strongest model that can be understood, tested, and trained under the
available compute. Similarity to frontier assistants is a direction, not an
evaluation metric. Each stage must have measurable loss, task quality, speed,
and memory baselines.

## Baseline: dense decoder-only transformer

The first implementation deliberately avoids mixture-of-experts, multi-head
latent attention, reinforcement learning, and distributed training. Those
features multiply failure modes before the basic gradients are trusted.

### Selected components

- Pre-normalized residual blocks for stable optimization.
- RMSNorm, with conventional LayerNorm retained as a comparison implementation.
- Rotary position embeddings (RoPE).
- Bias-free Q, K, V, output, and feed-forward projections.
- Standard causal multi-head attention for the first correct implementation.
- SwiGLU feed-forward network with hidden width rounded to a hardware-friendly
  multiple even though NumPy training starts on CPU.
- Tied input embeddings and language-model output weights.
- Stable log-sum-exp cross-entropy; no explicit probability tensor is needed for
  training loss.

### Deferred components

- Grouped-query attention: valuable for key/value-cache efficiency, but less
  useful before autoregressive inference becomes a bottleneck.
- Mixture-of-experts: adds routing, load balancing, communication, and sparse
  kernel complexity that is unjustified at this scale.
- Multi-head latent attention: promising for large key/value caches, but not the
  shortest route to a verified first model.
- Long-context modifications: a 256-token context keeps gradient debugging and
  attention memory manageable.

## Initial configuration

| Field | Value |
|---|---:|
| Vocabulary | 256 bytes |
| Context | 256 tokens |
| Layers | 4 |
| Model width | 256 |
| Heads | 4 |
| Head width | 64 |
| SwiGLU hidden width | 704 |
| Parameters | 3,279,104 |

The byte vocabulary avoids coupling the first model to a tokenizer-training
pipeline. A subword tokenizer becomes worthwhile for the GPU-scale model because
bytes consume context inefficiently.

## Execution backend strategy

NumPy remains the readable CPU specification and numerical oracle through the
first complete tiny-model overfit. A later native CUDA C++ backend will reproduce
the same operations behind a narrow Python interface. Transformer-specific
elementwise and reduction operations can use custom kernels, while cuBLAS is the
initial performance baseline for matrix multiplication.

The local machine reported an NVIDIA GeForce RTX 3050 with 4,096 MiB VRAM, WDDM
driver/KMD 616.92, and CUDA UMD compatibility 13.4. Neither `nvcc` nor Microsoft's
`cl` compiler was discoverable from the current shell, so the driver can run CUDA
applications but the native CUDA development toolchain is not yet available for
this project. The other laptops and cloud targets remain uninventoried.

## Scaling gates

The model is not enlarged until all conditions at the current scale pass:

1. Every primitive passes finite-difference gradient checks in float64.
2. The assembled model deliberately overfits a tiny sequence set.
3. Checkpoint restore reproduces the next update exactly.
4. Training and validation loss are reported separately.
5. Generation is compared with a bigram baseline under fixed seeds.
6. Memory and tokens-per-second are recorded.

## Primary research references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [OLMo 2 Furious](https://arxiv.org/abs/2501.00656)

These references are design inputs, not instructions to copy every frontier
feature. OLMo is especially useful because its artifacts and training decisions
are unusually transparent; Llama 3 informs dense-model practice; DeepSeek-V3 is
an advanced branch for later efficiency experiments.

## Implementation status - 2026-10-02

Verified with deterministic and finite-difference tests:

- Stable log-sum-exp, softmax, and cross-entropy.
- Linear and Embedding forward/backward operations.
- RMSNorm and conventional LayerNorm forward/backward operations.
- Overflow-safe sigmoid, SiLU, and the elementwise SwiGLU core.
- RoPE forward and inverse-rotation backward operations, including offsets.
- Causal scaled-dot-product attention forward/backward operations.
- Full bias-free multi-head self-attention composition.
- Complete bias-free three-projection SwiGLU feed-forward composition.
- Separate gate/value input projections, SwiGLU activation, output projection,
  and summed gate/value input-gradient branches.
- One complete pre-normalized residual transformer block.
- Explicit backward composition through both residual branches, both RMSNorms,
  multi-head attention, and the SwiGLU feed-forward network.
- Complete four-block byte language-model composition.
- Token embedding, four residual blocks, final RMSNorm, tied output projection,
  and reverse-order backward through the entire stack.
- Addition of the embedding-table gradients from input lookup and tied output use.
- Deterministic full-model parameter initialization from one configured seed.
- Unit-valued RMSNorm scales, base Gaussian projection scale, and depth-scaled
  attention/FFN residual-output projections.
- Unique named parameter traversal in a deterministic order.
- Overflow-resistant global L2 gradient norm and joint norm clipping.
- Named gradient traversal matching the parameter names, order, and shapes.
- Bias-corrected Adam first and second moments with decoupled weight decay.
- Matrix parameters receive weight decay; one-dimensional RMSNorm scales do not.
- Atomic validation before parameter mutation and optimizer statistics suitable
  for terminal-visible training logs.
- Versioned NumPy checkpoint archives containing configuration, parameters, AdamW
  moments, optimizer step, and optional random-generator state.
- Non-pickle loading with strict archive fields, names, shapes, dtypes, finite
  values, and format-version validation.
- Same-directory temporary writes, file synchronization, and atomic replacement
  so invalid state cannot partially overwrite a valid checkpoint.
- Exact-restoration testing against the next AdamW update.
- UTF-8 text-to-byte encoding and replacement-safe decoding for generated bytes.
- Contiguous train/validation splitting with enough independent tokens for a full
  context-plus-target window on each side.
- Seeded random next-token windows whose targets are inputs shifted by one byte.
- RNG-state restoration tests requiring the identical next sampled batch.
- Mean next-token loss composed with full-model backward propagation, global-norm
  clipping, and one atomic AdamW parameter update.
- A separate evaluation path that reports loss, perplexity, and token count
  without backward propagation or parameter mutation.
- Flat per-step diagnostics for step, loss, perplexity, token count, learning
  rate, gradient norm, and clipping coefficient.
- Absolute-step training control with periodic fixed-batch validation, terminal
  metrics, JSON-lines logs, and final/periodic atomic checkpoints.
- An atomic run manifest records model/training configuration plus SHA-256
  fingerprints of both token splits and rejects incompatible resume attempts.
- Training-RNG checkpoint restoration that must reproduce an uninterrupted run's
  next updates exactly.
- A deliberately small synthetic-data smoke configuration and terminal command;
  it does not read or approve any audited book or paper.
- Greedy next-token selection with deterministic first-index tie breaking.
- Seeded categorical sampling with positive temperature and deterministic top-k
  ranking, including stable normalization after subtracting the candidate maximum.
- Autoregressive generation that recomputes the full visible context, copies its
  prompt, and stops before exceeding the configured context length.
- Explicit stop reasons distinguishing a fulfilled token request from a context
  limit.
- Loading an exact saved checkpoint for bounded greedy or seeded stochastic
  generation without mutating the loaded state.
- UTF-8 prompt encoding, replacement-safe byte decoding, prompt/generated/full
  text reporting, checkpoint step, stop reason, elapsed time, and throughput.
- Terminal-safe `repr` rendering so generated control bytes are escaped rather
  than interpreted by the console.
- A one-shot generation command for inspecting a checkpoint before an interactive
  chat loop exists.

Demonstrated by the first synthetic smoke run:

- Twenty updates completed with 20 metric events, checkpoints at steps 10 and 20,
  and a final completion record.
- Validation loss fell from 5.544793 to 3.974303; byte-level validation perplexity
  fell from 255.902 to 53.213, a reduction by a factor of about 4.81.
- Training loss fell from 5.546263 to 4.147532 despite expected batch-to-batch
  fluctuation.
- Raw gradient norms stayed finite from 1.054 to 1.520. Every step exceeded the
  configured norm limit of 1.0 and was clipped, but the coefficient improved from
  as low as 0.658 to 0.949 while losses continued downward.
- The run used 173 generated training bytes, 43 generated validation bytes, a
  one-block 7,280-parameter model, and no audited documents.

This 0.14-second run is an optimization smoke test, not a throughput benchmark or
evidence of generalization. The validation split is a suffix of the repeated
synthetic sentence and excludes the `Yuzu ` -> `l` transition, so its loss
reduction demonstrates correct repeated updates on closely related bytes rather
than the named prompt or out-of-sample generalization.

Observed second checkpoint generation:

- Step-200 checkpoint, the identical five-byte prompt `Yuzu `, and identical
  greedy settings produced ` tte patte` rather than `learns byt`.
- The output contains fragments from `byte patterns`, showing more conditional
  structure than the step-20 spaces, but it still fails exact memorization.
- Generation completed structurally in 0.0235 seconds at 424.7 tokens/second.

Observed first checkpoint generation:

- Step-20 checkpoint, context length 16, prompt `Yuzu ` (five bytes), greedy mode.
- Ten requested bytes were generated in 0.0206 seconds at 484.6 tokens/second.
- The structural path succeeded with `stop_reason=max_new_tokens`, but all ten
  generated bytes were spaces: `Yuzu           `.
- This is not learned continuation. The current model favors a frequent byte and
  has not yet memorized that the prompt should continue with `learns`.

Demonstrated by exact resume from step 20 to step 200:

- Training resumed at step 21 with the saved parameter, AdamW, and RNG state and
  completed at the absolute target step 200.
- Validation loss fell from 3.974303 to 0.867316, about 78.2 percent; validation
  perplexity fell from 53.213 to 2.381, a reduction by a factor of about 22.3.
- Step-200 training loss was 0.920321. All reported losses remained finite while
  gradient clipping continued intermittently.
- The run directory now contains 200 metric records and checkpoint events at
  steps 10, 20, 50, 100, 150, and 200.
- These results satisfy the optimization/overfit gate, but the observed step-200
  fixed probe still fails the generation gate.

Why the validation metric did not predict this failure:

- The 216-byte synthetic corpus splits at byte 173. The 43-byte validation tail is
  exactly ` byte patterns one careful step at a time.\n`; it contains no `Yuzu `.
- Validation loss is measured on one fixed sampled batch of four 16-byte windows,
  or 64 possibly overlapping targets, rather than every validation transition.
- Only three of 157 legal training-window starts place `Yuzu ` at the beginning
  of a window. Most examples containing that transition include preceding text.
- Therefore validation loss 0.867316 and generation failure are compatible; the
  metric never directly certified the bare-prompt transition.

Verified by the user-reported 118-test suite:

- A read-only teacher-forced continuation probe that always feeds the expected
  preceding bytes after each scored position.
- Per-byte expected probability, deterministic rank, negative log-likelihood,
  greedy prediction, and top-five alternatives.
- Aggregate top-1 byte accuracy, exact-match status, mean loss, and perplexity.
- Terminal-safe formatting, UTF-8 byte accounting, full-context single-token
  scoring, and proof that repeated probes do not alter the checkpoint.

Observed teacher-forced comparison:

- Step 20 scored one of ten expected bytes at rank one, with mean loss 3.943107
  and perplexity 51.579. Its greedy prediction was a space at every offset.
- Step 200 scored seven of ten bytes at rank one, with mean loss 1.022074 and
  perplexity 2.779. This is an 18.56-fold perplexity reduction.
- The remaining misses are `l` after bare `Yuzu ` (rank three), `a` after `le`
  (rank three), and `b` after `learns ` (rank two).
- The first miss dominates free-running generation: space has probability
  0.514896 versus 0.063652 for `l`, so changing sampling settings would conceal
  rather than fix the learned distribution.

Controlled undertraining experiment:

- The exact 16-byte training-like context `at a time.\nYuzu ` still predicts
  space at step 200. Expected `l` remains rank three with probability 0.059196;
  space has probability 0.488026 and is 8.24 times likelier.
- Because bare and exact training-like contexts fail almost identically, test
  undertraining with only 100 additional updates while preserving the model,
  optimizer, learning rate, data, RNG trajectory, and decoding settings.
- Save steps 250 and 300, then compare their fixed teacher-forced probes against
  the recorded step-200 baseline before deciding whether to continue uniform
  training or implement prefix-balanced batches.

Observed controlled-resume result:

- The unchanged run resumed at step 201, completed step 300 in 0.67 seconds, and
  wrote the planned checkpoints at steps 250 and 300.
- From steps 200 to 300, sampled training loss fell from 0.920321 to 0.283254;
  fixed-batch validation loss fell from 0.867316 to 0.437441. All reported values
  remained finite.
- Step 250 reached 9/10 teacher-forced top-1 accuracy. Its only miss was already
  close: expected `l` had probability 0.355314 at rank two versus 0.389764 for
  space.
- Step 300 reached 10/10 top-1 accuracy and exact match, with mean loss 0.230220
  and perplexity 1.259. Expected `l` became rank one at probability 0.705429,
  versus 0.096345 for space.
- This passes the teacher-forced tiny-pattern memorization gate. It does not show
  language ability, generalization, or conversational behavior.

Next runtime evidence gate:

- Run the unchanged greedy `Yuzu ` probe against checkpoint step 300. Exact output
  must be `learns byt`; otherwise checkpoint-backed generation is inconsistent
  with the verified teacher-forced ranking.
- Do not train further before this integration check.

Not yet implemented:

- Interactive inference and conversation-history management.
- Key/value caching; full-context recomputation remains the correctness oracle.
