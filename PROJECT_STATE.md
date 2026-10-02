# Project state

Last updated: 2026-10-02 (Asia/Calcutta)
State version: 19

## Recovery instruction

Read this file before making project changes. Update it after every user prompt
that causes analysis, decisions, code changes, data work, tests, or execution.
Record what changed, what was verified, unresolved risks, and the exact next
action. This file is the continuity source if chat context is unavailable.

## Project identity

- Root: `C:\Users\AKSHAT\Desktop\numpy-gpt-from-scratch`
- Git branch: `main`
- Git state: transformer block, full model, initialization, and their tests are
  committed at `ea80058`; current state/documentation synchronization is
  uncommitted
- Runtime constraint: model code uses the Python standard library and NumPy
- Installed NumPy observed during setup: 2.3.5
- Long-term direction: build the strongest feasible GPT-style system from first
  principles, then add conversation, summarization, tools, browsing, and voice as
  separately tested subsystems
- Planned model name: **Yuvika**; the only allowed alias is **Yuzu**

## Collaboration rules supplied by the user

- Lead with gaps, risks, or uncomfortable facts rather than agreement.
- Prefix claims with confidence tags: `[Certain]`, `[Likely]`, or `[Guessing]`.
- Give structured disagreement with reasons, alternatives, and risks.
- Do not yield to unsupported pushback; revise only when new evidence warrants it.
- Consult primary research on modern architectures where useful.
- Keep this state file current after every user prompt execution.
- Make training and chatbot behavior observable in a terminal.
- The user will personally run project tests and training when given an exact
  command; do not execute them unless the user later asks.

## Locked architecture decisions

- Decoder-only autoregressive transformer.
- First full model target: 3,279,104 trainable parameters.
- Initial vocabulary: all 256 byte values.
- Initial context length: 256 byte tokens.
- Four transformer blocks, model width 256, four attention heads.
- Pre-normalized residual blocks.
- RMSNorm as the default; conventional LayerNorm retained for comparison.
- Rotary position embeddings (RoPE).
- Bias-free projections.
- Standard causal multi-head attention for the first implementation.
- SwiGLU feed-forward network with hidden width 704.
- Tied token-embedding and output-projection weights.
- Stable log-sum-exp cross-entropy.
- Grouped-query attention is deferred until inference optimization.
- Mixture-of-experts and multi-head latent attention are research branches, not
  baseline requirements.

## Recommended execution-backend direction (not yet implemented)

- Do not replace NumPy with scalar Python or `math` loops; that would remain on
  the CPU and discard efficient array kernels without enabling CUDA.
- Keep the verified NumPy implementation as the readable executable
  specification, numerical test oracle, and CPU fallback.
- Complete the tiny NumPy model and prove that it can overfit one small batch
  before starting a native GPU port.
- For low-level GPU control without PyTorch, JAX, CuPy, or Numba, add a separate
  CUDA C++ backend compiled with NVIDIA's CUDA toolchain and loaded from Python
  through a narrow standard-library `ctypes` interface.
- Initially use custom CUDA kernels for elementwise operations, reductions,
  normalization, RoPE, masking, and optimizer updates, while using cuBLAS for
  matrix multiplication. A from-scratch GEMM may be an educational experiment,
  but it should not be the training-performance baseline.
- Require every CUDA forward and backward operation to match the NumPy oracle
  within dtype-appropriate tolerances before it is accepted.
- Defer multi-machine synchronous training until one-GPU training is correct and
  profiled; separate gaming laptops are more useful initially for independent
  experiments and evaluation jobs.

Observed local GPU environment on 2026-10-02:

- NVIDIA GeForce RTX 3050 with 4,096 MiB VRAM and a reported 60 W power cap.
- Windows WDDM driver/KMD 616.92 and CUDA UMD compatibility 13.4.
- `nvcc` was not recognized and `where.exe cl` found no compiler in the current
  shell. This does not prove the toolchains are absent from disk, but neither is
  currently usable from that shell.
- Other gaming laptops and cloud GPU targets remain uninventoried.

## Future identity decisions

- Primary assistant identity: **Yuvika**.
- Only accepted alias: **Yuzu**.
- Wake-name handling should be deterministic at the interface layer rather than
  dependent on probabilistic model behavior.
- The eventual personality may be inspired by a person important to the user,
  but Yuvika remains an AI and must not claim to be that person.
- No private messages, recordings, personal history, or voice samples may be used
  until provenance and informed permission are explicitly recorded.
- Personality traits and prohibited behaviors are not yet specified.

## Completed work

1. Created the project folder on the Windows Desktop.
2. Initialized a local Git repository on branch `main`.
3. Created the initial directory structure and project documentation.
4. Added a validated `ModelConfig` dataclass and exact parameter-count method.
5. Added `configs/tiny.json` for the 3.28M-parameter model.
6. Added architecture and corpus-policy documents.
7. Added a standard-library corpus inventory script using Poppler commands when
   available; the script retains metadata but not extracted document text.
8. Audited the two user-provided source folders.
9. Added and ran three configuration tests successfully.
10. Compiled the current Python sources successfully.
11. Added `AGENTS.md` so repository work must preserve this recovery protocol.
12. Added numerically stable `logsumexp`, `softmax`, and
    `cross_entropy_with_logits` implementations.
13. Added a reusable centered finite-difference gradient checker with absolute
    and relative error reporting.
14. Added tests for extreme logits, shift invariance, loss correctness, invalid
    inputs, and analytical-versus-numerical cross-entropy gradients.
15. Started a live PowerShell terminal rooted at the project and used it to run
    the test suite so execution is visible beside the chat.
16. Implemented Linear forward/backward for arbitrary leading dimensions,
    including optional bias gradients.
17. Implemented Embedding lookup/backward with correct repeated-token gradient
    accumulation through `numpy.add.at`.
18. Added six deterministic tests covering Linear forward, every Linear gradient,
    Embedding lookup, repeated-token accumulation, Embedding weight gradients,
    and invalid token IDs.
19. Updated repository instructions to reserve tests and training for the user.
20. Added `docs\identity-and-personality.md` to preserve the planned Yuvika/Yuzu
    identity, implementation boundary, consent requirements, and deferred choices.
21. Added `docs\learning-path.md`, an exact reading order with current line ranges,
    tensor shapes, equations, tests, comprehension questions, and separate model,
    corpus, identity, and repository tracks.
22. Received the user's successful 16-test terminal result, verifying Linear and
    Embedding forward/backward implementations and their numerical gradients.
23. Implemented RMSNorm and conventional LayerNorm forward/backward operations.
24. Added six normalization tests covering definitions, all gradients, constant
    inputs, parameter shapes, and epsilon validation.
25. Extended the learning path with normalization equations and source ranges.
26. Received the user's successful 22-test terminal result, verifying RMSNorm and
    LayerNorm forward/backward implementations and their numerical gradients.
27. Implemented overflow-safe sigmoid and SiLU forward/backward operations.
28. Implemented the elementwise SwiGLU core and explicit gate/value gradients.
29. Added six activation tests and extended the learning path with SiLU/SwiGLU
    equations, numerical-stability rationale, and current source ranges.
30. Received the user's successful 28-test terminal result, verifying sigmoid,
    SiLU, and SwiGLU forward/backward behavior and numerical gradients.
31. Implemented RoPE forward and inverse-rotation backward operations with support
    for arbitrary leading dimensions and nonnegative position offsets.
32. Added six RoPE tests covering position-zero identity, explicit rotation,
    pair-norm preservation, gradients, chunk offsets, and invalid settings.
33. Synchronized the README, architecture record, learning path, and state with
    the current Phase 1 implementation status.
34. Received the user's explicit confirmation that the 34-test run ended with
    final status `OK`, verifying RoPE forward/backward behavior.
35. Implemented causal scaled-dot-product self-attention forward/backward with a
    strict future-token mask and arbitrary leading batch/head dimensions.
36. Added six causal-attention tests covering mask probabilities, prefix means,
    future isolation, Q/K/V gradients, one-token behavior, and invalid settings.
37. Updated the architecture record and learning path with causal-attention
    equations, source ranges, test ranges, and the next composition boundary.
38. Received the user's successful 40-test result, verifying causal attention
    structure, causality, edge cases, and Q/K/V numerical gradients.
39. Implemented full bias-free multi-head self-attention composition with Q/K/V
    projections, head splitting, RoPE, causal attention, merging, and output
    projection.
40. Added six full-attention tests covering primitive equivalence, multi-head
    shapes, future isolation, input and four weight gradients, zero values, and
    invalid configurations.
41. Updated the architecture record and learning path with the complete
    multi-head forward/backward composition and current source ranges.
42. Received the user's confirmation that the expanded 46-test suite passed,
    verifying full multi-head attention composition and all weight gradients.
43. Implemented the complete bias-free three-projection SwiGLU feed-forward
    module and its explicit backward composition.
44. Added six feed-forward tests covering primitive equivalence, arbitrary leading
    dimensions, zero values, input and three weight gradients, branch addition,
    and invalid weight shapes.
45. Updated the architecture record and learning path with feed-forward shapes,
    equations, source ranges, test ranges, and attention-versus-FFN roles.
46. Received the user's confirmation that the expanded 52-test suite passed,
    verifying the complete three-projection SwiGLU feed-forward composition and
    its input and weight gradients.
47. Received the local GPU inventory showing an RTX 3050 with 4 GiB VRAM and a
    working display driver, but no discoverable `nvcc` or `cl` command.
48. Confirmed that the verified 52-test milestone was committed at `b8ded9e`.
49. Implemented one complete pre-normalized residual transformer block with two
    RMSNorms, causal multi-head attention, SwiGLU feed-forward, and two residual
    connections.
50. Added explicit backward composition through both residual paths and grouped
    gradients for both norm scales and all seven projection matrices.
51. Added six transformer-block tests covering primitive equivalence, arbitrary
    leading dimensions, future isolation, ten numerical gradient groups,
    residual identity, and delegated configuration validation.
52. Updated the architecture record and learning path with backend inventory,
    block equations, source ranges, test ranges, and residual-gradient flow.
53. Received the user's confirmation that the expanded 58-test suite passed,
    verifying the complete pre-normalized residual transformer block, both
    residual-gradient paths, causality, and all ten tested gradient groups.
54. Added immutable parameter, gradient, and backward-cache records for the full
    stacked language model.
55. Implemented byte embedding lookup, four-block forward composition, final
    RMSNorm, tied output projection, and configuration/shape validation.
56. Implemented full backward composition through the blocks in reverse order,
    including addition of the input-lookup and tied-output embedding gradients.
57. Added six language-model tests covering explicit composition, four-block
    cache/logit shapes, exact parameter count, causality, tied-gradient addition,
    all parameter gradients, context offsets, and invalid model contracts.
58. Updated package exports, README, architecture status, and the learning path
    with full-model flow, equations, source ranges, and test ranges.
59. Received the user's confirmation that the expanded 64-test suite passed,
    verifying the four-block model, tied embedding gradient, causality, parameter
    count, configuration contracts, and all model-parameter gradients.
60. Added `initializer_std` to the validated configuration and versioned JSON.
61. Implemented reproducible full-model parameter initialization in float32 by
    default, with dtype and seed overrides.
62. Initialized RMSNorm scales to one, ordinary weights at the configured base
    Gaussian scale, and residual-output projections at
    `initializer_std / sqrt(2 * n_layers)`.
63. Added deterministic unique parameter naming for future optimizer and
    checkpoint traversal.
64. Added six initialization tests covering reproducibility, seed separation,
    exact names/shapes/count, dtype and finiteness, empirical initialization
    scales, and invalid settings.
65. Updated README, architecture status, and learning documentation with the
    initialization rationale, equations, source ranges, and test ranges.
66. Observed commit `ea80058`, which now contains the transformer block, complete
    model, initialization implementation, and their tests; only the latest
    state/documentation synchronization remains uncommitted.
67. Added a repository-level `.gitattributes` policy that keeps text files at LF
    on every platform, excludes binary project assets from text conversion, and
    renormalized the Git index to remove Windows `core.autocrlf` ambiguity.

## Corpus state

Source folders:

- `C:\Users\AKSHAT\Desktop\Quant Books`
- `C:\Users\AKSHAT\Desktop\imp books and papers`

Audit results:

- 58 files: 57 PDFs and one DOCX.
- 622,419,462 total bytes.
- 22,811 PDF pages before exclusions.
- 55 unique file hashes.
- Three exact duplicate groups excluded from repeat ingestion.
- `Last Will and Testament.docx` excluded as sensitive personal legal content.
- Seven PDFs quarantined for OCR and rights review.
- Forty-seven unique text-bearing PDFs await rights review.
- Sample-based projection: about 20.8M extracted characters, roughly 4-7M
  conventional subword tokens or about 20-25M byte tokens.
- No source document content has been copied into the project or approved for
  training.

## Verification status

- User-reported `python -m unittest discover -s tests -v` on 2026-10-02:
  the expanded 64-test suite passed with final status `OK`.
- Cross-entropy backward pass matched centered finite differences with maximum
  absolute error below `1e-9`.
- `python -m py_compile scripts\inventory_corpus.py src\numpy_gpt\config.py`:
  passed.
- Corpus inventory completed successfully and wrote
  `data\manifests\corpus_inventory.json`.
- Linear and Embedding forward/backward operations are verified by deterministic
  and finite-difference tests.
- RMSNorm and LayerNorm forward/backward operations are verified by deterministic
  and finite-difference tests.
- Sigmoid, SiLU, and SwiGLU forward/backward operations are verified by
  deterministic and finite-difference tests.
- RoPE forward/backward behavior, pair-norm preservation, and position offsets
  are verified by deterministic and finite-difference tests.
- Causal scaled-dot-product attention, masking, future isolation, and Q/K/V
  gradients are verified by deterministic and finite-difference tests.
- Full multi-head attention composition, causality, input gradients, and all four
  projection-weight gradients are verified.
- Complete three-projection SwiGLU feed-forward composition, input gradients, and
  all three projection-weight gradients are verified.
- The complete pre-normalized residual transformer block, both residual-gradient
  paths, future isolation, norm-scale gradients, and all seven projection-weight
  gradients are verified.
- The complete four-block model, tied input/output embedding gradient, causality,
  parameter count, context contracts, and every model-parameter gradient are
  verified.
- `src\numpy_gpt\initialization.py` and `tests\test_initialization.py` are newly
  written and unverified. The full suite is expected to contain 70 tests.

## Important limitations and risks

- The current corpus is too narrow and small for a general conversational model.
- Training rights have not been recorded for any source document.
- Seven PDFs have no adequate text layer and require OCR assessment.
- NumPy does not provide CUDA execution; the verified design must eventually be
  ported to a GPU-capable backend for larger runs.
- The local RTX 3050 and driver are inventoried, but the CUDA compiler and native
  host compiler are not available from the current shell; the other laptops,
  cloud GPUs, and network links remain unknown.
- The complete four-block model is verified. Deterministic parameter
  initialization is implemented but awaits user verification; no optimizer,
  tokenizer, trainer, checkpoint system, or inference loop has been implemented.
- Similarity to frontier assistants is a direction, not a measurable milestone;
  every stage needs explicit baselines and evaluations.
- A real-person-inspired personality can create privacy, impersonation, emotional
  dependency, and voice-abuse risks; no personal data has been authorized or used.

## Remaining major milestones

1. Verify deterministic parameter initialization.
2. Implement AdamW, gradient clipping, checkpointing,
   byte batching, evaluation, sampling, and terminal-visible training.
3. Approve and normalize training data, then overfit a tiny corpus before any
   longer run.
4. Train and evaluate the NumPy model; only then port verified semantics to a GPU
   backend for larger experiments.
5. Add instruction tuning, terminal chat, Yuvika/Yuzu identity routing, retrieval,
   tools, browsing, and voice as later evaluated layers.

## Terminal-observability requirement

All long-running training and interactive inference commands must expose useful
terminal output. Training output should include at least step, epoch, train loss,
validation loss, learning rate, gradient norm, tokens per second, elapsed time,
and checkpoint events. Interactive inference must provide a terminal chat loop
with visible prompt, generated response, generation speed, and an explicit exit
command. Logs must also be written to a run directory for later comparison.

## Current phase

Phase 0 is complete enough to proceed: project scaffold, architecture decision,
and corpus audit exist. Phase 1 is active. Stable probability functions,
cross-entropy backward, and gradient-checking infrastructure are verified.
All primitives, full multi-head attention, and the complete three-projection
SwiGLU feed-forward module are verified.
One pre-normalized residual transformer block and its explicit backward pass are
verified by the expanded 58-test suite.
The complete four-block byte language model with a final RMSNorm and tied
input/output embeddings is verified by the expanded 64-test suite.
Deterministic, depth-scaled parameter initialization and named parameter traversal
are implemented but await user verification.
The user also wants to learn the implementation while building it; new stages
should extend the learning-path document after source line numbers stabilize.

## Exact next action

The user should run the expanded test suite from the project root:

`python -m unittest discover -s tests -v`

Expected success signal: 70 tests run and the final line is `OK`. Do not implement
AdamW or gradient clipping until the user reports this result. Keep NumPy as the
canonical implementation and keep `nvcc`/`cl` setup deferred.

Do not start corpus extraction or model training until these primitives and their
backward passes pass deterministic numerical checks.

## Update log

### 2026-10-01 - State version 1

- Created this continuity file at the user's request.
- Recorded all work completed before neural-network implementation.
- Added the requirement for observable terminal training and chatbot inference.
- Next action remains implementation of stable loss functions and gradient tests.

### 2026-10-01 - State version 2

- Added repository instructions enforcing state recovery and terminal visibility.
- Started a project-rooted PowerShell terminal for visible execution.
- Implemented stable softmax, log-sum-exp cross-entropy, and its backward pass.
- Implemented centered finite-difference gradient checking.
- Expanded the suite from 3 to 10 tests; all tests pass.
- Queued `PROJECT_STATE.md` and `src\numpy_gpt\numerics.py` in the Codex editor.
- Exact next action: implement and gradient-check Linear and Embedding operations.

### 2026-10-01 - State version 3

- Recorded that the user will run all requested tests and training personally.
- Implemented explicit Linear and Embedding forward/backward operations.
- Added six deterministic and finite-difference layer tests.
- Exported the new operations through `numpy_gpt.__init__`.
- Queued `src\numpy_gpt\layers.py` in the Codex editor.
- No project code was executed during this update; the new code remains unverified.
- Exact next action: user runs the complete unittest command and reports whether
  all 16 tests finish with `OK`.

### 2026-10-01 - State version 4

- Recorded **Yuvika** as the primary model name and **Yuzu** as its only alias.
- Planned deterministic wake-name and self-identification behavior for a later
  interface phase.
- Added a personality boundary: original AI identity inspired by chosen traits,
  not an undisclosed impersonation of a real person.
- Recorded that private messages, mannerisms, recordings, and voice require
  provenance and informed permission before any use.
- Added `docs\identity-and-personality.md`; no model code or personal data changed.
- Verification remains pending: the user still needs to run the 16-test suite.

### 2026-10-01 - State version 5

- Created a step-by-step learning path that mirrors forward data flow and backward
  gradient flow instead of alphabetical file order.
- Added exact current file/line ranges, equations, shapes, derivations, and study
  questions for all implemented model components.
- Separated transformer mathematics from corpus tooling and future identity work.
- Queued `docs\learning-path.md` in the Codex editor beside the chat.
- No project tests or training were executed; the 16-test verification gate is
  still pending and remains the exact next action.

### 2026-10-02 - State version 6

- Recorded the user's terminal evidence: all 16 existing tests passed in 0.138
  seconds with final status `OK`.
- Marked Linear and Embedding forward/backward operations as verified.
- Implemented RMSNorm and LayerNorm with explicit analytical backward passes.
- Added six normalization tests, bringing the expected suite total to 22.
- Extended `docs\learning-path.md` with both normalization derivations, shapes,
  source ranges, and test ranges.
- Queued `src\numpy_gpt\normalization.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 22
  tests finish with `OK`; SwiGLU remains blocked until then.

### 2026-10-02 - State version 7

- Recorded the user's terminal evidence: all 22 tests passed in 0.115 seconds
  with final status `OK`.
- Marked RMSNorm and LayerNorm forward/backward operations as verified.
- Implemented overflow-safe sigmoid, SiLU, and the elementwise SwiGLU core with
  explicit analytical backward passes.
- Added six activation tests, bringing the expected suite total to 28.
- Extended `docs\learning-path.md` with activation equations, stability reasoning,
  source ranges, and test ranges.
- Queued `src\numpy_gpt\activations.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 28
  tests finish with `OK`; RoPE remains blocked until then.

### 2026-10-02 - State version 8

- Recorded the user's terminal evidence: all 28 tests passed in 0.144 seconds
  with final status `OK`.
- Marked overflow-safe sigmoid, SiLU, and SwiGLU forward/backward as verified.
- Implemented adjacent-pair RoPE with inverse-rotation backward and offset support.
- Added six RoPE tests, bringing the expected suite total to 34.
- Updated README status from Phase 0 to Phase 1.
- Added a dated implementation-status section to `docs\architecture.md`.
- Extended `docs\learning-path.md` with RoPE equations, shapes, source ranges,
  test ranges, norm preservation, and cached-decoding offset rationale.
- Queued `src\numpy_gpt\position.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 34
  tests finish with `OK`; causal attention remains blocked until then.

### 2026-10-02 - State version 9

- Recorded output showing all 34 individual test cases as `ok` and completion in
  0.134 seconds.
- Did not mark RoPE verified because the pasted output omitted unittest's final
  `OK` summary line.
- Made no code or architecture changes; README, architecture record, and learning
  path correctly continue to describe RoPE as awaiting verification.
- Exact next action: user confirms the omitted final status line; causal attention
  remains blocked until the status is explicitly `OK`.

### 2026-10-02 - State version 10

- Recorded the user's explicit confirmation that the 34-test run ended in `OK`.
- Marked RoPE forward/backward and position-offset behavior as verified.
- Implemented causal scaled-dot-product attention with a triangular future mask,
  stable masked softmax, and analytical gradients for Q, K, and V.
- Added six attention tests, bringing the expected suite total to 40.
- Updated `docs\architecture.md` and `docs\learning-path.md` with attention status,
  equations, shapes, source ranges, test ranges, and composition boundaries.
- Added a remaining-major-milestones section to answer whether the project is done.
- Queued `src\numpy_gpt\attention.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 40
  tests finish with `OK`; full multi-head projection composition remains blocked.

### 2026-10-02 - State version 11

- Recorded attached terminal evidence: all 40 tests passed in 0.149 seconds with
  final status `OK`.
- Marked causal attention structure, mask, future isolation, and Q/K/V gradients
  as verified.
- Implemented full bias-free multi-head attention composition with four Linear
  projections, head splitting/merging, RoPE on Q/K, and causal attention.
- Added six composition tests, bringing the expected suite total to 46.
- Updated `docs\architecture.md` and `docs\learning-path.md` with status, tensor
  shapes, forward/backward flow, source ranges, and test ranges.
- Queued `src\numpy_gpt\multi_head_attention.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 46
  tests finish with `OK`; complete feed-forward composition remains blocked.

### 2026-10-02 - State version 12

- Recorded the user's confirmation that the expanded 46-test suite passed.
- Marked full multi-head attention composition, causality, input gradients, and
  all four projection-weight gradients as verified.
- Implemented the complete three-projection SwiGLU feed-forward forward/backward
  composition.
- Added six feed-forward tests, bringing the expected suite total to 52.
- Updated `docs\architecture.md` and `docs\learning-path.md` with equations,
  tensor shapes, branch-gradient addition, source ranges, and test ranges.
- Queued `src\numpy_gpt\feed_forward.py` in the Codex editor.
- No project tests or training were run by the assistant.
- Exact next action: the user runs the expanded suite and reports whether all 52
  tests finish with `OK`; the transformer block remains blocked until then.

### 2026-10-02 - State version 13

- Recorded the user's confirmation that the expanded 52-test suite passed with
  final status `OK`.
- Marked the complete three-projection SwiGLU feed-forward composition, input
  gradients, and all three projection-weight gradients as verified.
- Synchronized the architecture record and learning path with the verified
  checkpoint.
- Made no model-code changes and did not run project tests or training.
- Exact next action: pause while the user commits this milestone; after the user
  asks to proceed, implement and gradient-check one pre-norm residual transformer
  block.

### 2026-10-02 - State version 14

- Evaluated the user's proposal to replace NumPy with "pure math" to gain CUDA
  control; recorded that scalar Python math does not provide GPU execution.
- Recommended a dual-backend design: retain NumPy as the verified executable
  specification and later add native CUDA C++ kernels behind a narrow `ctypes`
  boundary.
- Recommended custom kernels for transformer-specific elementwise and reduction
  work, with cuBLAS as the initial matrix-multiplication baseline.
- Deferred CUDA implementation until the full tiny NumPy model overfits one batch
  and the available GPUs, VRAM, drivers, toolkits, compilers, and network links
  are inventoried.
- Made no model-code changes and did not run project tests or training.
- Exact next action: the user finishes the current commit and provides complete
  `nvidia-smi` output before any GPU backend is designed or implemented.

### 2026-10-02 - State version 15

- Recorded the screenshot evidence: local NVIDIA GeForce RTX 3050, 4,096 MiB
  VRAM, WDDM driver/KMD 616.92, CUDA UMD compatibility 13.4, and no `nvcc` or
  `cl` command discoverable from the current shell.
- Confirmed the prior verified milestone is committed at `b8ded9e`.
- Implemented one pre-normalized residual transformer block and its explicit
  backward pass through both residual branches.
- Added six deterministic and finite-difference tests, bringing the expected
  suite total to 58 and covering ten independent gradient groups.
- Exported the block API and synchronized the architecture record, learning
  guide, implementation status, and exact source ranges.
- Did not execute project tests or training.
- Exact next action: the user runs the expanded suite and reports whether all 58
  tests finish with final status `OK`; stacked-model assembly remains blocked
  until then.

### 2026-10-02 - State version 16

- Recorded the user's confirmation that all 58 tests passed.
- Marked the complete pre-normalized residual block, both identity-gradient
  routes, causality, two norm-scale gradients, and seven projection-weight
  gradients as verified.
- Synchronized `README.md`, the architecture status, learning checkpoint, current
  stopping point, verification record, remaining milestones, and project phase.
- Recorded the user's decision to postpone `nvcc` and `cl` setup; NumPy remains
  the canonical implementation until the full model and tiny-overfit gates pass.
- Made no model-code changes and did not execute project tests or training.
- Exact next action: the user may commit this verified milestone; after the user
  asks to proceed, assemble and gradient-check the four-block language model.

### 2026-10-02 - State version 17

- Implemented the complete four-block byte language-model forward and backward
  composition using the verified embedding, transformer block, and RMSNorm
  operations.
- Added nested immutable parameter, gradient, and cache structures plus strict
  block-count, shape, context-length, and tied/bias-free contract validation.
- Implemented tied input/output embeddings and explicitly added both gradient
  contributions into the shared embedding table.
- Added six tests, bringing the expected suite total to 64; the full-model
  numerical test covers the shared embedding, final norm, and every parameter in
  all four transformer blocks.
- Updated package exports, README, architecture status, learning order, equations,
  source ranges, and current stopping point.
- Kept `nvcc` and `cl` setup deferred and did not execute tests or training.
- Exact next action: the user runs the expanded suite and reports whether all 64
  tests finish with final status `OK`; initialization and optimization remain
  blocked until then.

### 2026-10-02 - State version 18

- Recorded the user's confirmation that all 64 tests passed and marked the full
  four-block language-model composition and gradients verified.
- Added a validated `initializer_std` configuration field and implemented
  deterministic full-model allocation from one seeded NumPy generator.
- Initialized norm scales to one, ordinary weights at the base Gaussian scale,
  and residual-output projections at the depth-scaled GPT-style standard
  deviation `initializer_std / sqrt(2 * n_layers)`.
- Added deterministic unique parameter names for future optimizer and checkpoint
  traversal.
- Added six tests, bringing the expected suite total to 70, and synchronized all
  status and learning documentation.
- Recorded that commit `ea80058` contains the source and test milestones through
  initialization; the latest state/documentation synchronization is uncommitted.
- Kept `nvcc` and `cl` setup deferred and did not execute tests or training.
- Exact next action: the user runs the expanded suite and reports whether all 70
  tests finish with final status `OK`; AdamW and gradient clipping remain blocked
  until then.

### 2026-10-02 - State version 19

- Diagnosed the `LF will be replaced by CRLF` messages as Git line-ending policy
  warnings caused by global `core.autocrlf=true` and the absence of a repository
  `.gitattributes` file; no project data was lost.
- Added `.gitattributes` to enforce LF for repository text while marking common
  binary project formats as binary.
- Renormalized the Git index so the already staged documentation follows the
  explicit repository policy and future `git add` operations remain consistent.
- Made no model-code changes and did not execute project tests or training.
- Exact next action remains: the user runs the expanded suite and reports whether
  all 70 tests finish with final status `OK`; AdamW and gradient clipping remain
  blocked until then.
