"""Run a stateless interactive diagnostic against one NumPy GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import run_checkpoint_diagnostic  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect independent prompts using one loaded checkpoint. "
            "This diagnostic is not a chatbot."
        ),
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true")
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument("--log-file", type=Path)
    logging.add_argument("--no-log", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    if options.no_log:
        log_path = None
    elif options.log_file is not None:
        log_path = options.log_file
    else:
        log_path = options.checkpoint.parent / "interactive.jsonl"
    result = run_checkpoint_diagnostic(
        options.checkpoint,
        max_new_tokens=options.max_new_tokens,
        greedy=options.greedy,
        temperature=options.temperature,
        top_k=options.top_k,
        seed=options.seed,
        log_path=log_path,
    )
    return 130 if result.exit_reason == "interrupt" else 0


if __name__ == "__main__":
    raise SystemExit(main())
