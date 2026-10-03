"""Evaluate fail-closed metadata decisions without extracting corpus text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from numpy_gpt import (  # noqa: E402
    evaluate_corpus_admission,
    format_corpus_admission_summary,
    write_corpus_admission_report,
)


DEFAULT_INVENTORY = PROJECT_ROOT / "data" / "manifests" / "corpus_inventory.json"
DEFAULT_DECISIONS = PROJECT_ROOT / "data" / "manifests" / "corpus_decisions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "manifests" / "corpus_admission.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    try:
        report = evaluate_corpus_admission(options.inventory, options.decisions)
        output_path = write_corpus_admission_report(report, options.output)
    except (OSError, ValueError) as error:
        existing_report = str(options.output.exists()).lower()
        print(
            "admission_gate=ERROR "
            f"existing_report_may_be_stale={existing_report} error={error}",
            file=sys.stderr,
        )
        return 2
    print(format_corpus_admission_summary(report))
    print(f"report={output_path.resolve()}")
    return 0 if report.admitted_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
