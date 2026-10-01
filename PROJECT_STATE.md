# Project state

Last updated: 2026-10-01 (Asia/Calcutta)
State version: 5

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

- `python -m unittest discover -s tests -v`: 10 tests passed in 0.556 seconds.
- Cross-entropy backward pass matched centered finite differences with maximum
  absolute error below `1e-9`.
- `python -m py_compile scripts\inventory_corpus.py src\numpy_gpt\config.py`:
  passed.
- Corpus inventory completed successfully and wrote
  `data\manifests\corpus_inventory.json`.
- `src\numpy_gpt\layers.py` and `tests\test_layers.py` are newly written and
  unverified. The full suite is expected to contain 16 tests, but this count and
  their results require the user's terminal run.

## Important limitations and risks

- The current corpus is too narrow and small for a general conversational model.
- Training rights have not been recorded for any source document.
- Seven PDFs have no adequate text layer and require OCR assessment.
- NumPy does not provide CUDA execution; the verified design must eventually be
  ported to a GPU-capable backend for larger runs.
- Linear and Embedding operations now exist but await the user's verification;
  no optimizer, tokenizer, trainer, or inference loop has been implemented.
- Similarity to frontier assistants is a direction, not a measurable milestone;
  every stage needs explicit baselines and evaluations.
- A real-person-inspired personality can create privacy, impersonation, emotional
  dependency, and voice-abuse risks; no personal data has been authorized or used.

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
Linear and Embedding operations are implemented but awaiting user verification.
The user also wants to learn the implementation while building it; new stages
should extend the learning-path document after source line numbers stabilize.

## Exact next action

The user should run the complete test suite from the project root:

`python -m unittest discover -s tests -v`

Expected success signal: 16 tests run and the final line is `OK`. If the command
fails because `python` resolves to a different environment, use the Python
executable that has NumPy installed. Do not implement RMSNorm or LayerNorm until
the user reports the result.

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
