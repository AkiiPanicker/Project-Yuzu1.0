# Repository instructions

## Continuity

Before changing this repository, read `PROJECT_STATE.md` completely. After every
user prompt that results in project analysis, decisions, edits, tests, or command
execution, update `PROJECT_STATE.md` with:

- the work performed;
- decisions made and their rationale;
- verification commands and results;
- new or resolved risks;
- the exact next action;
- a dated update-log entry and incremented state version.

Treat `PROJECT_STATE.md` as the authoritative recovery record when chat context is
missing or incomplete.

## Runtime constraint

Model and training code may use only the Python standard library and NumPy unless
the user explicitly changes this constraint. External command-line tools may be
used for corpus preprocessing only when documented in the state file.

## Observability

Training and interactive generation must be runnable in a terminal. Training must
report step, epoch, train loss, validation loss, learning rate, gradient norm,
tokens per second, elapsed time, and checkpoints. Chat inference must provide an
interactive terminal loop with generation timing and an explicit exit command.

The user has asked to run project tests and training personally. After edits,
provide the exact project-relative file or command to run and the expected success
signal. Do not execute project tests or training unless the user later asks for it.

## Quality gates

Do not scale a component until its deterministic tests and numerical gradient
checks pass. Do not ingest a document until privacy, rights, extraction quality,
deduplication, and split-integrity decisions are recorded.
