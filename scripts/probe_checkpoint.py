"""Report teacher-forced byte predictions from a NumPy GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    format_continuation_probe,
    probe_checkpoint_continuation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a known continuation one expected byte at a time.",
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Yuzu ")
    parser.add_argument("--expected", default="learns byt")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = probe_checkpoint_continuation(
        options.checkpoint,
        options.prompt,
        options.expected,
        top_k=options.top_k,
    )
    print(format_continuation_probe(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
