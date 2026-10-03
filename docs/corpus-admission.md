# Corpus admission gate

This gate decides which inventoried source files may proceed to a future text
extraction stage. It reads metadata and hashes only. It never parses, copies, or
stores document text.

## Why the decision manifest is separate

`corpus_inventory.json` records observed file facts. It must not be edited to
claim permission. `corpus_decisions.json` records human review decisions and is
bound to the exact inventory file by SHA-256. Regenerating or editing the
inventory makes the decisions stale until they are reviewed again.

Possessing a PDF or buying a book is not itself evidence that model training is
permitted. Do not mark a source approved without a documented basis. This project
does not decide whether a license or law applies; the reviewer must supply a
specific evidence reference.

## Decision record

The initial decision list is empty, so the implementation is expected to refuse
all 58 inventoried files once verified. Add one record only after reviewing that
exact `source_path` and hash:

```json
{
  "source_path": "C:\\path\\to\\document.pdf",
  "expected_sha256": "64 lowercase hexadecimal characters",
  "rights": {
    "status": "approved",
    "basis": "public_domain",
    "evidence": "catalog, license, or permission reference"
  },
  "privacy": {
    "status": "approved",
    "evidence": "what was reviewed and by whom"
  },
  "extraction": {
    "status": "approved",
    "evidence": "representative pages and extraction quality reviewed"
  },
  "reviewed_by": "reviewer name",
  "reviewed_at_utc": "2026-10-03T00:00:00+00:00",
  "notes": "metadata-only notes; do not paste private content or license text"
}
```

Allowed rights statuses are `approved`, `denied`, and `pending`. An approved
record must use one of these bases:

- `copyright_owner`
- `explicit_permission`
- `license_allows_model_training`
- `public_domain`

Allowed privacy statuses are `approved`, `excluded`, and `pending`. Allowed
extraction statuses are `approved`, `rejected`, and `pending`. Completed reviews
require a nonempty evidence reference. Keep credentials, private document text,
and full license text out of this manifest.

## Immutable holds

No decision record can override these inventory states:

- `excluded_sensitive`
- `excluded_exact_duplicate`
- `needs_ocr_and_rights_review`
- extraction errors or unknown inventory states

For an otherwise eligible record, admission also requires the live source to
exist as a regular file with the same byte length and SHA-256 stored in both the
inventory and decision. Future extraction must rehash immediately before reading
content to close the time-of-check/time-of-use gap.

The evaluator reads each manifest once and hashes the exact bytes it parses. It
also rejects inconsistent inventory roots, paths, names, suffixes, states, and
duplicate groups. A report destination cannot alias either manifest or any
inventoried source, including through an existing hard link or symbolic path.

## Terminal command and exit codes

First, from the project root, require the complete verification gate:

```bat
python -m unittest discover -s tests -v
```

All 134 tests must finish with `OK`. Only then run:

```bat
python scripts\admit_corpus.py
```

Run the two commands separately; do not join them with `&&` because the expected
initial closed gate deliberately returns process exit `1`.

The command writes only `data\manifests\corpus_admission.json` and prints a
reason-count summary.

- Exit `0`: at least one source is admitted.
- Exit `1`: the manifests are valid, but zero sources are admitted.
- Exit `2`: a manifest is malformed, stale, unreadable, or cannot be written.

The initial expected state is exit `1`, `admission_gate=CLOSED`, 58 refused
files, and zero admitted files. That is a safety success, not an execution bug.

`corpus_admission.json` is an informational snapshot, not a reusable permission
token. It is intentionally ignored by Git while the inventory and human decision
manifest remain tracked. A later malformed or stale-manifest run returns exit `2`
and may leave an
older report on disk; the error line explicitly reports whether such an output
exists. Therefore future extraction must call the evaluator again
against the current manifest bytes and live source files; it must never trust a
saved report as authorization. An OCR-held source remains held even if a decision
claims approval. Clear that hold only by producing and auditing a derived OCR
artifact with provenance, regenerating its inventory record, and reviewing the
new exact hash; do not edit the old status by hand.
