"""Create a metadata-only audit of the proposed local training corpus.

This script uses only the Python standard library. If Poppler's ``pdfinfo`` and
``pdftotext`` commands are present, it samples the first, middle, and final pages
to estimate whether each PDF has an extractable text layer. It never stores the
sampled document text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable


DEFAULT_ROOTS = (
    Path(r"C:\Users\AKSHAT\Desktop\Quant Books"),
    Path(r"C:\Users\AKSHAT\Desktop\imp books and papers"),
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "manifests"
    / "corpus_inventory.json"
)
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}
SENSITIVE_NAME_MARKERS = ("last will", "testament")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run_text_command(arguments: list[str], timeout: int = 60) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(message)
    return completed.stdout


def pdf_page_count(path: Path, pdfinfo: str) -> int:
    output = run_text_command([pdfinfo, str(path)])
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError("pdfinfo did not report a page count")
    return int(match.group(1))


def pdf_text_sample(path: Path, pages: int, pdftotext: str) -> dict[str, Any]:
    selected_pages = sorted({1, max(1, (pages + 1) // 2), pages})
    characters = 0
    wordlike = 0
    for page in selected_pages:
        text = run_text_command(
            [
                pdftotext,
                "-f",
                str(page),
                "-l",
                str(page),
                "-nopgbrk",
                str(path),
                "-",
            ]
        )
        characters += len(text.strip())
        wordlike += len(re.findall(r"\w+", text, flags=re.UNICODE))
    return {
        "sampled_pages": selected_pages,
        "sample_characters": characters,
        "sample_wordlike_tokens": wordlike,
        "mean_characters_per_sampled_page": round(characters / len(selected_pages), 1),
    }


def iter_source_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
            if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES:
                yield path


def is_sensitive_name(path: Path) -> bool:
    lowered = path.name.casefold()
    return any(marker in lowered for marker in SENSITIVE_NAME_MARKERS)


def build_inventory(roots: list[Path]) -> dict[str, Any]:
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    records: list[dict[str, Any]] = []

    for path in iter_source_files(roots):
        stat = path.stat()
        record: dict[str, Any] = {
            "source_path": str(path.resolve()),
            "source_root": str(next(root for root in roots if path.is_relative_to(root))),
            "name": path.name,
            "suffix": path.suffix.casefold(),
            "bytes": stat.st_size,
            "sha256": sha256(path),
            "status": "needs_rights_review",
            "status_reasons": ["training rights have not been recorded"],
        }

        if is_sensitive_name(path):
            record["status"] = "excluded_sensitive"
            record["status_reasons"] = ["filename indicates personal legal content"]
            records.append(record)
            continue

        if path.suffix.casefold() == ".pdf":
            if pdfinfo is None or pdftotext is None:
                record["extraction"] = {"status": "poppler_unavailable"}
                record["status_reasons"].append("PDF text extraction was not assessed")
            else:
                try:
                    pages = pdf_page_count(path, pdfinfo)
                    sample = pdf_text_sample(path, pages, pdftotext)
                    record["pages"] = pages
                    record["extraction"] = {"status": "sampled", **sample}
                    if sample["mean_characters_per_sampled_page"] < 100:
                        record["status"] = "needs_ocr_and_rights_review"
                        record["status_reasons"].append(
                            "sampled pages have little or no extractable text"
                        )
                except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                    record["extraction"] = {
                        "status": "error",
                        "error": str(error)[:500],
                    }
                    record["status"] = "extraction_error"
                    record["status_reasons"].append("PDF inspection failed")
        elif path.suffix.casefold() == ".docx":
            record["status_reasons"].append("DOCX extraction is not implemented")

        records.append(record)

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    for matches in by_hash.values():
        if len(matches) < 2:
            continue
        canonical = matches[0]["source_path"]
        for duplicate in matches[1:]:
            if duplicate["status"] == "excluded_sensitive":
                continue
            duplicate["status"] = "excluded_exact_duplicate"
            duplicate["status_reasons"] = ["byte-for-byte duplicate"]
            duplicate["duplicate_of"] = canonical

    status_counts = Counter(record["status"] for record in records)
    suffix_counts = Counter(record["suffix"] for record in records)
    duplicate_groups = sum(1 for matches in by_hash.values() if len(matches) > 1)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": [str(root.resolve()) for root in roots],
        "tools": {
            "pdfinfo": pdfinfo,
            "pdftotext": pdftotext,
        },
        "summary": {
            "files": len(records),
            "unique_hashes": len(by_hash),
            "bytes": sum(record["bytes"] for record in records),
            "duplicate_groups": duplicate_groups,
            "by_suffix": dict(sorted(suffix_counts.items())),
            "by_status": dict(sorted(status_counts.items())),
        },
        "files": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help="Source folders to inspect",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON manifest destination",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [root for root in args.roots if not root.is_dir()]
    if missing:
        for root in missing:
            print(f"missing source folder: {root}")
        return 2

    inventory = build_inventory(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(inventory, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(json.dumps(inventory["summary"], indent=2))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

