"""Fail-closed, metadata-only admission checks for local corpus sources."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


CORPUS_ADMISSION_SCHEMA_VERSION = 1
RIGHTS_STATUSES = frozenset({"approved", "denied", "pending"})
APPROVED_RIGHTS_BASES = frozenset(
    {
        "copyright_owner",
        "explicit_permission",
        "license_allows_model_training",
        "public_domain",
    }
)
PRIVACY_STATUSES = frozenset({"approved", "excluded", "pending"})
EXTRACTION_STATUSES = frozenset({"approved", "rejected", "pending"})
HARD_HOLD_REASONS = {
    "excluded_sensitive": "inventory_sensitive",
    "excluded_exact_duplicate": "inventory_exact_duplicate",
    "needs_ocr_and_rights_review": "inventory_ocr_quarantine",
}
ALLOWED_SOURCE_SUFFIXES = frozenset({".docx", ".md", ".pdf", ".txt"})
INVENTORY_STATUSES = frozenset(
    {
        "excluded_exact_duplicate",
        "excluded_sensitive",
        "needs_ocr_and_rights_review",
        "needs_rights_review",
    }
)


@dataclass(frozen=True, slots=True)
class CorpusAdmissionRecord:
    """One source's metadata-only admission result."""

    source_path: str
    name: str
    suffix: str
    bytes: int
    inventory_sha256: str
    live_sha256: str | None
    inventory_status: str
    rights_status: str
    privacy_status: str
    extraction_status: str
    admission_status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusAdmissionReport:
    """Bound manifest digests, immutable decisions, and per-source results."""

    generated_at_utc: str
    inventory_path: Path
    decisions_path: Path
    inventory_sha256: str
    decisions_sha256: str
    records: tuple[CorpusAdmissionRecord, ...]

    @property
    def admitted_count(self) -> int:
        return sum(record.admission_status == "admitted" for record in self.records)

    @property
    def refused_count(self) -> int:
        return len(self.records) - self.admitted_count

    @property
    def reason_counts(self) -> dict[str, int]:
        counts = Counter(
            reason for record in self.records for reason in record.reason_codes
        )
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CORPUS_ADMISSION_SCHEMA_VERSION,
            "generated_at_utc": self.generated_at_utc,
            "authorization": {
                "status": "informational_snapshot_only",
                "requirement": (
                    "future extraction must rerun admission against the current "
                    "manifest bytes and live source files"
                ),
            },
            "inventory": {
                "path": str(self.inventory_path),
                "sha256": self.inventory_sha256,
            },
            "decisions": {
                "path": str(self.decisions_path),
                "sha256": self.decisions_sha256,
            },
            "summary": {
                "total": len(self.records),
                "admitted": self.admitted_count,
                "refused": self.refused_count,
                "by_reason": self.reason_counts,
            },
            "files": [
                {
                    "source_path": record.source_path,
                    "name": record.name,
                    "suffix": record.suffix,
                    "bytes": record.bytes,
                    "inventory_sha256": record.inventory_sha256,
                    "live_sha256": record.live_sha256,
                    "inventory_status": record.inventory_status,
                    "decision": {
                        "rights_status": record.rights_status,
                        "privacy_status": record.privacy_status,
                        "extraction_status": record.extraction_status,
                    },
                    "admission_status": record.admission_status,
                    "reason_codes": list(record.reason_codes),
                }
                for record in self.records
            ],
        }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without retaining file contents."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    """Read once, then hash and parse the same immutable byte buffer."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _validate_utc_timestamp(value: object, label: str) -> str:
    text = _require_nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must use UTC")
    return text


def _validated_inventory(inventory: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if (
        type(inventory.get("schema_version")) is not int
        or inventory["schema_version"] != 1
    ):
        raise ValueError("inventory schema_version must be 1")
    roots = inventory.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("inventory roots must be a nonempty list")
    canonical_roots: list[str] = []
    for index, root_value in enumerate(roots):
        root_text = _require_nonempty_string(root_value, f"inventory roots[{index}]")
        root_path = Path(root_text)
        if not root_path.is_absolute():
            raise ValueError(f"inventory roots[{index}] must be absolute")
        canonical_root = os.path.normcase(str(root_path.resolve(strict=False)))
        if canonical_root in canonical_roots:
            raise ValueError("inventory roots must be unique")
        canonical_roots.append(canonical_root)
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("inventory files must be a list")

    required = {"source_path", "name", "suffix", "bytes", "sha256", "status"}
    seen_paths: set[str] = set()
    by_digest: dict[str, list[Mapping[str, Any]]] = {}
    validated: list[Mapping[str, Any]] = []
    for index, record in enumerate(files):
        label = f"inventory files[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        if not required.issubset(record):
            raise ValueError(f"{label} is missing required fields")
        source_path = _require_nonempty_string(record["source_path"], f"{label}.source_path")
        source = Path(source_path)
        if not source.is_absolute():
            raise ValueError(f"{label}.source_path must be absolute")
        canonical_source = os.path.normcase(str(source.resolve(strict=False)))
        contained = False
        for root in canonical_roots:
            try:
                if os.path.commonpath((canonical_source, root)) == root:
                    contained = True
                    break
            except ValueError:
                continue
        if not contained:
            raise ValueError(f"{label}.source_path is outside inventory roots")
        if canonical_source in seen_paths:
            raise ValueError(f"inventory contains duplicate source_path: {source_path}")
        seen_paths.add(canonical_source)
        name = _require_nonempty_string(record["name"], f"{label}.name")
        if name != source.name:
            raise ValueError(f"{label}.name does not match source_path")
        suffix = _require_nonempty_string(record["suffix"], f"{label}.suffix")
        if suffix != source.suffix.casefold() or suffix not in ALLOWED_SOURCE_SUFFIXES:
            raise ValueError(f"{label}.suffix does not match an allowed source type")
        status = _require_nonempty_string(record["status"], f"{label}.status")
        if status not in INVENTORY_STATUSES:
            raise ValueError(f"{label}.status is unsupported")
        size = record["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"{label}.bytes must be a nonnegative integer")
        if not _is_sha256(record["sha256"]):
            raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        validated.append(record)
        by_digest.setdefault(record["sha256"], []).append(record)

    for digest, records in by_digest.items():
        duplicates = sum(
            record["status"] == "excluded_exact_duplicate" for record in records
        )
        if (len(records) == 1 and duplicates) or (
            len(records) > 1 and duplicates != len(records) - 1
        ):
            raise ValueError(
                "inventory duplicate statuses are inconsistent for sha256 "
                f"{digest}"
            )
    return tuple(validated)


def _validate_review(
    review: object,
    *,
    label: str,
    allowed_statuses: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(review, dict):
        raise ValueError(f"{label} must be an object")
    _require_exact_keys(review, {"status", "evidence"}, label)
    status = review["status"]
    if not isinstance(status, str) or status not in allowed_statuses:
        raise ValueError(f"{label}.status is unsupported")
    evidence = review["evidence"]
    if not isinstance(evidence, str):
        raise ValueError(f"{label}.evidence must be a string")
    if status != "pending" and not evidence.strip():
        raise ValueError(f"{label}.evidence is required for a completed review")
    return review


def _validated_decisions(
    decisions: Mapping[str, Any],
    *,
    inventory_digest: str,
    inventory_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    _require_exact_keys(
        decisions,
        {"schema_version", "inventory_sha256", "decisions"},
        "decision manifest",
    )
    if (
        type(decisions["schema_version"]) is not int
        or decisions["schema_version"] != CORPUS_ADMISSION_SCHEMA_VERSION
    ):
        raise ValueError("decision manifest schema_version is unsupported")
    if decisions["inventory_sha256"] != inventory_digest:
        raise ValueError("decision manifest is bound to a different inventory")
    entries = decisions["decisions"]
    if not isinstance(entries, list):
        raise ValueError("decision manifest decisions must be a list")

    expected_fields = {
        "source_path",
        "expected_sha256",
        "rights",
        "privacy",
        "extraction",
        "reviewed_by",
        "reviewed_at_utc",
        "notes",
    }
    by_path: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(entries):
        label = f"decisions[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be an object")
        _require_exact_keys(entry, expected_fields, label)
        source_path = _require_nonempty_string(entry["source_path"], f"{label}.source_path")
        if source_path not in inventory_by_path:
            raise ValueError(f"{label}.source_path is absent from the inventory")
        if source_path in by_path:
            raise ValueError(f"duplicate decision for source_path: {source_path}")
        if not _is_sha256(entry["expected_sha256"]):
            raise ValueError(f"{label}.expected_sha256 must be a SHA-256 digest")

        rights = entry["rights"]
        if not isinstance(rights, dict):
            raise ValueError(f"{label}.rights must be an object")
        _require_exact_keys(rights, {"status", "basis", "evidence"}, f"{label}.rights")
        rights_status = rights["status"]
        if not isinstance(rights_status, str) or rights_status not in RIGHTS_STATUSES:
            raise ValueError(f"{label}.rights.status is unsupported")
        basis = rights["basis"]
        evidence = rights["evidence"]
        if not isinstance(basis, str) or not isinstance(evidence, str):
            raise ValueError(f"{label}.rights basis and evidence must be strings")
        if rights_status == "approved":
            if basis not in APPROVED_RIGHTS_BASES:
                raise ValueError(f"{label}.rights.basis cannot approve training")
            if not evidence.strip():
                raise ValueError(f"{label}.rights.evidence is required")
        elif basis != "unknown":
            raise ValueError(f"{label}.rights.basis must be unknown unless approved")
        elif rights_status == "denied" and not evidence.strip():
            raise ValueError(f"{label}.rights.evidence is required when denied")

        _validate_review(
            entry["privacy"],
            label=f"{label}.privacy",
            allowed_statuses=PRIVACY_STATUSES,
        )
        _validate_review(
            entry["extraction"],
            label=f"{label}.extraction",
            allowed_statuses=EXTRACTION_STATUSES,
        )
        _require_nonempty_string(entry["reviewed_by"], f"{label}.reviewed_by")
        _validate_utc_timestamp(entry["reviewed_at_utc"], f"{label}.reviewed_at_utc")
        if not isinstance(entry["notes"], str):
            raise ValueError(f"{label}.notes must be a string")
        by_path[source_path] = entry
    return by_path


def _inventory_hold_reason(record: Mapping[str, Any]) -> str | None:
    status = record["status"]
    if status in HARD_HOLD_REASONS:
        return HARD_HOLD_REASONS[status]
    if status != "needs_rights_review":
        return "inventory_not_eligible"
    suffix = record["suffix"]
    if suffix == ".pdf":
        extraction = record.get("extraction")
        if not isinstance(extraction, dict) or extraction.get("status") != "sampled":
            return "inventory_not_eligible"
    elif suffix not in {".txt", ".md"}:
        return "inventory_not_eligible"
    return None


def _decision_reasons(
    record: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
) -> tuple[list[str], str, str, str]:
    if decision is None:
        return ["decision_missing"], "missing", "missing", "missing"

    reasons: list[str] = []
    if decision["expected_sha256"] != record["sha256"]:
        reasons.append("decision_hash_mismatch")
    rights_status = decision["rights"]["status"]
    privacy_status = decision["privacy"]["status"]
    extraction_status = decision["extraction"]["status"]
    if rights_status == "pending":
        reasons.append("rights_pending")
    elif rights_status == "denied":
        reasons.append("rights_denied")
    if privacy_status == "pending":
        reasons.append("privacy_pending")
    elif privacy_status == "excluded":
        reasons.append("privacy_excluded")
    if extraction_status == "pending":
        reasons.append("extraction_pending")
    elif extraction_status == "rejected":
        reasons.append("extraction_rejected")
    return reasons, rights_status, privacy_status, extraction_status


def _live_source_reasons(
    record: Mapping[str, Any],
) -> tuple[list[str], str | None]:
    path = Path(record["source_path"])
    try:
        if not path.exists():
            return ["source_missing"], None
        if not path.is_file():
            return ["source_not_file"], None
        if path.stat().st_size != record["bytes"]:
            return ["source_size_changed"], None
        live_digest = sha256_file(path)
    except OSError:
        return ["source_unreadable"], None
    if live_digest != record["sha256"]:
        return ["source_hash_changed"], live_digest
    return [], live_digest


def evaluate_corpus_admission(
    inventory_path: str | Path,
    decisions_path: str | Path,
) -> CorpusAdmissionReport:
    """Evaluate metadata decisions and live source integrity without extracting text."""

    inventory_source = Path(inventory_path).resolve(strict=False)
    decisions_source = Path(decisions_path).resolve(strict=False)
    inventory, inventory_digest = _read_json_object(inventory_source, "inventory")
    inventory_records = _validated_inventory(inventory)
    inventory_by_path = {record["source_path"]: record for record in inventory_records}
    decisions, decisions_digest = _read_json_object(
        decisions_source,
        "decision manifest",
    )
    decisions_by_path = _validated_decisions(
        decisions,
        inventory_digest=inventory_digest,
        inventory_by_path=inventory_by_path,
    )

    results: list[CorpusAdmissionRecord] = []
    for record in inventory_records:
        reasons: list[str] = []
        hold_reason = _inventory_hold_reason(record)
        if hold_reason is not None:
            reasons.append(hold_reason)
        decision_reasons, rights_status, privacy_status, extraction_status = (
            _decision_reasons(record, decisions_by_path.get(record["source_path"]))
        )
        reasons.extend(decision_reasons)
        live_digest: str | None = None
        if not reasons:
            live_reasons, live_digest = _live_source_reasons(record)
            reasons.extend(live_reasons)
        results.append(
            CorpusAdmissionRecord(
                source_path=record["source_path"],
                name=record["name"],
                suffix=record["suffix"],
                bytes=record["bytes"],
                inventory_sha256=record["sha256"],
                live_sha256=live_digest,
                inventory_status=record["status"],
                rights_status=rights_status,
                privacy_status=privacy_status,
                extraction_status=extraction_status,
                admission_status="admitted" if not reasons else "refused",
                reason_codes=tuple(reasons),
            )
        )

    return CorpusAdmissionReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        inventory_path=inventory_source,
        decisions_path=decisions_source,
        inventory_sha256=inventory_digest,
        decisions_sha256=decisions_digest,
        records=tuple(results),
    )


def write_corpus_admission_report(
    report: CorpusAdmissionReport,
    output_path: str | Path,
) -> Path:
    """Atomically write a metadata-only report without touching source files."""

    if not isinstance(report, CorpusAdmissionReport):
        raise TypeError("report must be a CorpusAdmissionReport")
    destination = Path(output_path)
    protected_paths = (
        report.inventory_path,
        report.decisions_path,
        *(Path(record.source_path) for record in report.records),
    )
    if any(_paths_alias(destination, protected) for protected in protected_paths):
        raise ValueError(
            "output_path must not overwrite the inventory, decisions, or a source file"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        try:
            stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        with stream:
            json.dump(report.to_dict(), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _paths_alias(first: Path, second: Path) -> bool:
    """Return whether paths resolve alike or existing entries are the same file."""

    first_resolved = os.path.normcase(str(first.resolve(strict=False)))
    second_resolved = os.path.normcase(str(second.resolve(strict=False)))
    if first_resolved == second_resolved:
        return True
    try:
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def format_corpus_admission_summary(report: CorpusAdmissionReport) -> str:
    """Return compact terminal output for the corpus-admission gate."""

    if not isinstance(report, CorpusAdmissionReport):
        raise TypeError("report must be a CorpusAdmissionReport")
    gate = "OPEN" if report.admitted_count else "CLOSED"
    lines = [
        f"admission_gate={gate} total={len(report.records)} "
        f"admitted={report.admitted_count} refused={report.refused_count}"
    ]
    lines.extend(
        f"reason={reason} count={count}"
        for reason, count in report.reason_counts.items()
    )
    return "\n".join(lines)
