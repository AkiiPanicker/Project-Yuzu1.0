"""Generate a terminal-safe continuation from a NumPy GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import format_generation_result, generate_from_checkpoint  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one bounded continuation from a saved checkpoint.",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Yuzu ")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    mode = "greedy" if options.greedy else "stochastic"
    print(
        f"mode={mode} temperature={options.temperature} "
        f"top_k={options.top_k} seed={options.seed}"
    )
    result = generate_from_checkpoint(
        options.checkpoint,
        options.prompt,
        max_new_tokens=options.max_new_tokens,
        greedy=options.greedy,
        temperature=options.temperature,
        top_k=options.top_k,
        seed=options.seed,
    )
    print(format_generation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
