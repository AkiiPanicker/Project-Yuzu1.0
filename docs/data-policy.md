# Local corpus policy

## Source folders

- `C:\Users\AKSHAT\Desktop\Quant Books`
- `C:\Users\AKSHAT\Desktop\imp books and papers`

The project stores only a metadata manifest until documents are approved. Source
files remain in place and are never changed.

Approval is recorded separately in `data/manifests/corpus_decisions.json`; local
possession alone is not a rights basis. The unverified `scripts/admit_corpus.py`
implementation is designed to bind that decision manifest to the exact inventory
bytes and, only for otherwise approved records, check live source hashes without
extracting document text. The schema and evidence rules are documented in
`docs/corpus-admission.md`.

## Admission gates

A document may enter a training split only when all applicable checks pass:

1. **Rights:** the owner, license, or applicable permission allows model training.
2. **Privacy:** personal, legal, medical, credential, and identifying information
   is excluded unless explicitly reviewed for a justified use.
3. **Extraction:** text is sufficiently accurate; scanned pages require OCR and
   a second quality check.
4. **Deduplication:** exact duplicates are removed, followed later by near-duplicate
   paragraph detection.
5. **Structure:** repeated headers, footers, page numbers, references, and broken
   hyphenation are normalized without destroying equations or code.
6. **Split integrity:** validation and test passages are separated before chunking.
7. **Provenance:** the manifest records the source, transformation, and decision.

## Default exclusions

- Wills, contracts, tax records, identity documents, and other personal legal files.
- Files with unknown rights until they receive an explicit decision.
- Password-protected or corrupt files.
- Image-only scans until OCR quality is measured.

## Expected limitation

The current collection is weighted toward finance, mathematics, engineering,
business, and a small amount of fiction. It is useful for domain experiments but
is not broad or conversational enough to pretrain a general assistant by itself.
