"""Tests for fail-closed, metadata-only corpus admission."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from admit_corpus import main as admission_main  # noqa: E402
import numpy_gpt.corpus_admission as corpus_admission_module  # noqa: E402
from numpy_gpt import (  # noqa: E402
    evaluate_corpus_admission,
    format_corpus_admission_summary,
    sha256_file,
    write_corpus_admission_report,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _record(
    source: Path,
    *,
    status: str = "needs_rights_review",
    digest: str | None = None,
    size: int | None = None,
) -> dict[str, object]:
    resolved = source.resolve()
    return {
        "source_path": str(resolved),
        "name": source.name,
        "suffix": source.suffix or ".txt",
        "bytes": source.stat().st_size if size is None else size,
        "sha256": sha256_file(source) if digest is None else digest,
        "status": status,
        "extraction": {"status": "sampled"},
    }


def _decision(
    record: dict[str, object],
    *,
    rights: str = "approved",
    privacy: str = "approved",
    extraction: str = "approved",
) -> dict[str, object]:
    rights_basis = "public_domain" if rights == "approved" else "unknown"
    rights_evidence = "catalog record" if rights != "pending" else ""
    privacy_evidence = "manual metadata review" if privacy != "pending" else ""
    extraction_evidence = "sample text reviewed" if extraction != "pending" else ""
    return {
        "source_path": record["source_path"],
        "expected_sha256": record["sha256"],
        "rights": {
            "status": rights,
            "basis": rights_basis,
            "evidence": rights_evidence,
        },
        "privacy": {"status": privacy, "evidence": privacy_evidence},
        "extraction": {
            "status": extraction,
            "evidence": extraction_evidence,
        },
        "reviewed_by": "test reviewer",
        "reviewed_at_utc": "2026-10-03T00:00:00+00:00",
        "notes": "fixture",
    }


def _manifests(
    directory: str,
    records: list[dict[str, object]],
    decisions: list[dict[str, object]],
) -> tuple[Path, Path]:
    root = Path(directory)
    inventory_path = root / "inventory.json"
    decisions_path = root / "decisions.json"
    _write_json(
        inventory_path,
        {
            "schema_version": 1,
            "roots": [str(root.resolve())],
            "files": records,
        },
    )
    _write_json(
        decisions_path,
        {
            "schema_version": 1,
            "inventory_sha256": sha256_file(inventory_path),
            "decisions": decisions,
        },
    )
    return inventory_path, decisions_path


class CorpusAdmissionTests(unittest.TestCase):
    def test_approved_unchanged_source_is_admitted_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "public.txt"
            source.write_bytes(b"public domain training text")
            record = _record(source)
            inventory, decisions = _manifests(
                directory,
                [record],
                [_decision(record)],
            )
            original = source.read_bytes()

            original_read_bytes = Path.read_bytes
            manifest_reads: list[Path] = []

            def tracked_read_bytes(path: Path) -> bytes:
                if path in {inventory, decisions}:
                    manifest_reads.append(path)
                return original_read_bytes(path)

            with mock.patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=tracked_read_bytes,
            ):
                report = evaluate_corpus_admission(inventory, decisions)

            self.assertEqual(report.admitted_count, 1)
            self.assertEqual(report.refused_count, 0)
            self.assertEqual(report.records[0].admission_status, "admitted")
            self.assertEqual(report.records[0].live_sha256, record["sha256"])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(manifest_reads.count(inventory), 1)
            self.assertEqual(manifest_reads.count(decisions), 1)
            self.assertIn("admission_gate=OPEN", format_corpus_admission_summary(report))

            if os.name == "nt":
                inventory_value = json.loads(inventory.read_text(encoding="utf-8"))
                decision_value = json.loads(decisions.read_text(encoding="utf-8"))
                current_drive = Path(directory).drive.casefold()
                other_drive = "Z:" if current_drive != "z:" else "Y:"
                inventory_value["roots"].insert(0, f"{other_drive}\\unrelated")
                _write_json(inventory, inventory_value)
                decision_value["inventory_sha256"] = sha256_file(inventory)
                _write_json(decisions, decision_value)

                multi_drive_report = evaluate_corpus_admission(inventory, decisions)

                self.assertEqual(multi_drive_report.admitted_count, 1)

    def test_missing_pending_and_denied_rights_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records: list[dict[str, object]] = []
            for name in ("missing.txt", "pending.txt", "denied.txt"):
                source = Path(directory) / name
                source.write_text(name, encoding="utf-8")
                records.append(_record(source))
            decisions = [
                _decision(records[1], rights="pending"),
                _decision(records[2], rights="denied"),
            ]
            inventory, decision_path = _manifests(directory, records, decisions)

            report = evaluate_corpus_admission(inventory, decision_path)

        by_name = {record.name: record for record in report.records}
        self.assertIn("decision_missing", by_name["missing.txt"].reason_codes)
        self.assertIn("rights_pending", by_name["pending.txt"].reason_codes)
        self.assertIn("rights_denied", by_name["denied.txt"].reason_codes)
        self.assertEqual(report.admitted_count, 0)

    def test_privacy_and_extraction_reviews_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = (
                ("privacy-pending.txt", "pending", "approved", "privacy_pending"),
                ("privacy-excluded.txt", "excluded", "approved", "privacy_excluded"),
                ("extraction-pending.txt", "approved", "pending", "extraction_pending"),
                ("extraction-rejected.txt", "approved", "rejected", "extraction_rejected"),
            )
            records: list[dict[str, object]] = []
            decisions: list[dict[str, object]] = []
            expected: dict[str, str] = {}
            for name, privacy, extraction, reason in settings:
                source = Path(directory) / name
                source.write_text(name, encoding="utf-8")
                record = _record(source)
                records.append(record)
                decisions.append(
                    _decision(record, privacy=privacy, extraction=extraction)
                )
                expected[name] = reason
            inventory, decision_path = _manifests(directory, records, decisions)

            report = evaluate_corpus_admission(inventory, decision_path)

        by_name = {record.name: record for record in report.records}
        for name, reason in expected.items():
            self.assertIn(reason, by_name[name].reason_codes)
        self.assertEqual(report.admitted_count, 0)

    def test_immutable_inventory_holds_override_forged_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            statuses = {
                "sensitive.pdf": (
                    "excluded_sensitive",
                    "inventory_sensitive",
                    "0" * 64,
                ),
                "duplicate.pdf": (
                    "excluded_exact_duplicate",
                    "inventory_exact_duplicate",
                    "1" * 64,
                ),
                "scan.pdf": (
                    "needs_ocr_and_rights_review",
                    "inventory_ocr_quarantine",
                    "2" * 64,
                ),
            }
            canonical = _record(
                Path(directory) / "canonical.pdf",
                digest="1" * 64,
                size=123,
            )
            records: list[dict[str, object]] = [canonical]
            for name, (status, _, digest) in statuses.items():
                source = Path(directory) / name
                record = _record(
                    source,
                    status=status,
                    digest=digest,
                    size=123,
                )
                records.append(record)
            decisions = [_decision(record) for record in records]
            inventory, decision_path = _manifests(directory, records, decisions)

            report = evaluate_corpus_admission(inventory, decision_path)

        by_name = {record.name: record for record in report.records}
        for name, (_, reason, _) in statuses.items():
            self.assertIn(reason, by_name[name].reason_codes)
            self.assertNotIn("source_missing", by_name[name].reason_codes)
            self.assertIsNone(by_name[name].live_sha256)

    def test_live_source_integrity_failures_are_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.txt"
            missing.write_bytes(b"missing")
            missing_record = _record(missing)
            missing.unlink()

            not_file = root / "directory.txt"
            not_file.mkdir()
            not_file_record = _record(
                not_file,
                digest="1" * 64,
                size=0,
            )

            resized = root / "resized.txt"
            resized.write_bytes(b"a")
            resized_record = _record(resized)
            resized.write_bytes(b"longer")

            rehashed = root / "rehashed.txt"
            rehashed.write_bytes(b"abc")
            rehashed_record = _record(rehashed)
            rehashed.write_bytes(b"xyz")

            records = [
                missing_record,
                not_file_record,
                resized_record,
                rehashed_record,
            ]
            inventory, decision_path = _manifests(
                directory,
                records,
                [_decision(record) for record in records],
            )

            report = evaluate_corpus_admission(inventory, decision_path)

        by_name = {record.name: record for record in report.records}
        self.assertIn("source_missing", by_name["missing.txt"].reason_codes)
        self.assertIn("source_not_file", by_name["directory.txt"].reason_codes)
        self.assertIn("source_size_changed", by_name["resized.txt"].reason_codes)
        self.assertIn("source_hash_changed", by_name["rehashed.txt"].reason_codes)

    def test_stale_malformed_duplicate_and_unknown_decisions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_bytes(b"text")
            record = _record(source)
            inventory, decision_path = _manifests(
                directory,
                [record],
                [_decision(record)],
            )
            valid = json.loads(decision_path.read_text(encoding="utf-8"))

            cases: list[dict[str, object]] = []
            stale = copy.deepcopy(valid)
            stale["inventory_sha256"] = "f" * 64
            cases.append(stale)

            malformed = copy.deepcopy(valid)
            malformed["decisions"][0]["rights"]["status"] = "maybe"
            cases.append(malformed)

            boolean_version = copy.deepcopy(valid)
            boolean_version["schema_version"] = True
            cases.append(boolean_version)

            unhashable_rights = copy.deepcopy(valid)
            unhashable_rights["decisions"][0]["rights"]["status"] = []
            cases.append(unhashable_rights)

            unhashable_privacy = copy.deepcopy(valid)
            unhashable_privacy["decisions"][0]["privacy"]["status"] = {}
            cases.append(unhashable_privacy)

            unhashable_extraction = copy.deepcopy(valid)
            unhashable_extraction["decisions"][0]["extraction"]["status"] = []
            cases.append(unhashable_extraction)

            duplicate = copy.deepcopy(valid)
            duplicate["decisions"].append(copy.deepcopy(duplicate["decisions"][0]))
            cases.append(duplicate)

            unknown = copy.deepcopy(valid)
            unknown["decisions"][0]["source_path"] = str(
                (Path(directory) / "unknown.txt").resolve()
            )
            cases.append(unknown)

            for index, value in enumerate(cases):
                with self.subTest(index=index):
                    _write_json(decision_path, value)
                    with self.assertRaises(ValueError):
                        evaluate_corpus_admission(inventory, decision_path)

            valid_inventory = json.loads(inventory.read_text(encoding="utf-8"))
            invalid_inventories: list[dict[str, object]] = []

            boolean_inventory_version = copy.deepcopy(valid_inventory)
            boolean_inventory_version["schema_version"] = True
            invalid_inventories.append(boolean_inventory_version)

            missing_roots = copy.deepcopy(valid_inventory)
            missing_roots["roots"] = []
            invalid_inventories.append(missing_roots)

            wrong_name = copy.deepcopy(valid_inventory)
            wrong_name["files"][0]["name"] = "forged.txt"
            invalid_inventories.append(wrong_name)

            wrong_suffix = copy.deepcopy(valid_inventory)
            wrong_suffix["files"][0]["suffix"] = ".PDF"
            invalid_inventories.append(wrong_suffix)

            outside_root = copy.deepcopy(valid_inventory)
            outside_path = (Path(directory).parent / "outside.txt").resolve()
            outside_root["files"][0]["source_path"] = str(outside_path)
            outside_root["files"][0]["name"] = outside_path.name
            invalid_inventories.append(outside_root)

            unknown_status = copy.deepcopy(valid_inventory)
            unknown_status["files"][0]["status"] = "approved_by_metadata"
            invalid_inventories.append(unknown_status)

            inconsistent_duplicates = copy.deepcopy(valid_inventory)
            copied_record = copy.deepcopy(inconsistent_duplicates["files"][0])
            copied_path = (Path(directory) / "copy.txt").resolve()
            copied_record["source_path"] = str(copied_path)
            copied_record["name"] = copied_path.name
            inconsistent_duplicates["files"].append(copied_record)
            invalid_inventories.append(inconsistent_duplicates)

            for index, value in enumerate(invalid_inventories):
                with self.subTest(inventory_case=index):
                    _write_json(inventory, value)
                    _write_json(
                        decision_path,
                        {
                            "schema_version": 1,
                            "inventory_sha256": sha256_file(inventory),
                            "decisions": [],
                        },
                    )
                    with self.assertRaises(ValueError):
                        evaluate_corpus_admission(inventory, decision_path)

    def test_reason_results_are_deterministic_and_report_is_metadata_only_atomic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "private.txt"
            secret = b"SECRET DOCUMENT BODY"
            source.write_bytes(secret)
            record = _record(source)
            inventory, decisions = _manifests(directory, [record], [])

            first = evaluate_corpus_admission(inventory, decisions)
            second = evaluate_corpus_admission(inventory, decisions)
            output = Path(directory) / "report.json"
            write_corpus_admission_report(first, output)
            report_text = output.read_text(encoding="utf-8")

            self.assertEqual(first.records, second.records)
            self.assertEqual(first.reason_counts, second.reason_counts)
            self.assertNotIn(secret.decode("utf-8"), report_text)
            self.assertNotIn("sample_characters", report_text)
            self.assertEqual(source.read_bytes(), secret)
            serialized = json.loads(report_text)
            self.assertEqual(serialized["summary"]["admitted"], 0)
            self.assertEqual(
                serialized["authorization"]["status"],
                "informational_snapshot_only",
            )

            for protected in (inventory, decisions, source):
                with self.subTest(protected=protected.name):
                    before = protected.read_bytes()
                    with self.assertRaises(ValueError):
                        write_corpus_admission_report(first, protected)
                    self.assertEqual(protected.read_bytes(), before)

            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                relative_report = evaluate_corpus_admission(
                    Path("inventory.json"),
                    Path("decisions.json"),
                )
            finally:
                os.chdir(previous_cwd)
            with self.assertRaises(ValueError):
                write_corpus_admission_report(relative_report, inventory)

            hardlink = Path(directory) / "source-hardlink.json"
            try:
                os.link(source, hardlink)
            except OSError:
                hardlink = None
            if hardlink is not None:
                with self.assertRaises(ValueError):
                    write_corpus_admission_report(first, hardlink)

            previous = b"previous valid report"
            output.write_bytes(previous)
            with mock.patch.object(
                corpus_admission_module.os,
                "replace",
                side_effect=OSError("injected replacement failure"),
            ):
                with self.assertRaises(OSError):
                    write_corpus_admission_report(first, output)
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob(f".{output.name}.*.tmp")), [])

            with mock.patch.object(
                corpus_admission_module.os,
                "fdopen",
                side_effect=OSError("injected descriptor wrapping failure"),
            ):
                with self.assertRaises(OSError):
                    write_corpus_admission_report(first, output)
            self.assertEqual(output.read_bytes(), previous)
            self.assertEqual(list(Path(directory).glob(f".{output.name}.*.tmp")), [])

    def test_cli_exit_codes_and_error_does_not_overwrite_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.txt"
            source.write_bytes(b"public")
            record = _record(source)
            inventory, decisions = _manifests(
                directory,
                [record],
                [_decision(record)],
            )
            output = Path(directory) / "report.json"
            arguments = [
                "--inventory",
                str(inventory),
                "--decisions",
                str(decisions),
                "--output",
                str(output),
            ]

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(admission_main(arguments), 0)
            self.assertIn("admission_gate=OPEN", stdout.getvalue())

            _write_json(
                decisions,
                {
                    "schema_version": 1,
                    "inventory_sha256": sha256_file(inventory),
                    "decisions": [],
                },
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(admission_main(arguments), 1)

            sentinel = b"keep existing report"
            output.write_bytes(sentinel)
            _write_json(
                decisions,
                {
                    "schema_version": 1,
                    "inventory_sha256": "f" * 64,
                    "decisions": [],
                },
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(admission_main(arguments), 2)
            self.assertIn("existing_report_may_be_stale=true", stderr.getvalue())
            self.assertEqual(output.read_bytes(), sentinel)

            _write_json(
                decisions,
                {
                    "schema_version": 1,
                    "inventory_sha256": sha256_file(inventory),
                    "decisions": [_decision(record)],
                },
            )
            for protected in (inventory, decisions, source):
                with self.subTest(cli_protected=protected.name):
                    before = protected.read_bytes()
                    protected_arguments = [
                        "--inventory",
                        str(inventory),
                        "--decisions",
                        str(decisions),
                        "--output",
                        str(protected),
                    ]
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        self.assertEqual(admission_main(protected_arguments), 2)
                    self.assertIn("must not overwrite", stderr.getvalue())
                    self.assertEqual(protected.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
