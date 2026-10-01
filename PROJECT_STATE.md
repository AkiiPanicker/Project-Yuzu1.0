# Project state

Last updated: 2026-10-02 (Asia/Calcutta)
State version: 13

## Recovery instruction

Read this file before making project changes. Update it after every user prompt
that causes analysis, decisions, code changes, data work, tests, or execution.
Record what changed, what was verified, unresolved risks, and the exact next
action. This file is the continuity source if chat context is unavailable.

## Project identity

- Root: `C:\Users\AKSHAT\Desktop\numpy-gpt-from-scratch`
- Git branch: `main`
- Git state: initialized; initial files are uncommitted
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
  the expanded 52-test suite passed with final status `OK`.
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

## Important limitations and risks

- The current corpus is too narrow and small for a general conversational model.
- Training rights have not been recorded for any source document.
- Seven PDFs have no adequate text layer and require OCR assessment.
- NumPy does not provide CUDA execution; the verified design must eventually be
  ported to a GPU-capable backend for larger runs.
- All primitives, full multi-head attention, and the complete feed-forward module
  are verified; no residual transformer block, optimizer, tokenizer, trainer, or
  inference loop has been implemented.
- Similarity to frontier assistants is a direction, not a measurable milestone;
  every stage needs explicit baselines and evaluations.
- A real-person-inspired personality can create privacy, impersonation, emotional
  dependency, and voice-abuse risks; no personal data has been authorized or used.

## Remaining major milestones

1. Assemble and verify a residual pre-norm transformer block and the full
   language model.
2. Implement parameter initialization, AdamW, gradient clipping, checkpointing,
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
The user also wants to learn the implementation while building it; new stages
should extend the learning-path document after source line numbers stabilize.

## Exact next action

The user is committing the verified 52-test milestone before implementation
continues. After the user confirms the commit is complete, implement and
gradient-check one pre-norm residual transformer block. Do not start that module
before the user asks to proceed.

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
