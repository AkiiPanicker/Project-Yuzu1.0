# Initial corpus audit

Generated from the two user-provided source folders on 2026-10-01. The detailed,
machine-readable record is in `data/manifests/corpus_inventory.json`.

## Inventory

| Measure | Result |
|---|---:|
| Files | 58 |
| PDFs | 57 |
| DOCX files | 1 |
| Total bytes | 622,419,462 |
| Total PDF pages, including duplicates and quarantined files | 22,811 |
| Unique file hashes | 55 |
| Exact duplicate groups | 3 |
| Sensitive files excluded | 1 |
| PDFs needing OCR and rights review | 7 |
| Text-bearing PDFs needing rights review | 47 |

## Mandatory exclusions and holds

- `Last Will and Testament.docx` is excluded as sensitive personal legal content.
- Three byte-for-byte duplicate PDFs are excluded from repeated ingestion.
- Seven PDFs have less than 100 extracted characters per sampled page and need OCR:
  - `A practical guide to quantative finanace interviews.pdf`
  - `Precalculus mathematics in a nutshell by George Finlay Simmons.pdf`
  - `Statistics by David Freedman.pdf`
  - `TechnicalAnalysisOfTheFinancialMarkets.pdf`
  - `17142797-Case-in-Point.pdf`
  - `Games People Play PDF.pdf`
  - `Steve-Nison-Japanese-Candlestick-Charting-Techniques-Prentice-Hall-Press-2001.pdf`

No document has been approved for training merely because its text can be
extracted. Rights review remains a separate gate.

## Estimated scale

Sampling the first, middle, and final pages of the 47 unique text-bearing PDFs
projects approximately 20.8 million extracted characters across 18,980 pages.
This is a rough estimate, not a full extraction count. It corresponds to roughly
4-7 million conventional subword tokens, depending on equations and formatting,
or approximately 20-25 million byte tokens.

This is useful for pipeline validation and domain adaptation experiments. It is
too small and too concentrated in finance, mathematics, engineering, business,
and a small amount of fiction to pretrain a broadly conversational assistant.

## Next data decisions

1. Run the full 134-test suite and require final status `OK`.
2. Run the initial metadata-only admission command and require a closed gate with
   58 refused and zero admitted files.
3. Use the hash-bound schema in `docs/corpus-admission.md` to record explicit
   rights, privacy, and extraction-quality decisions per intended source.
4. Visually inspect representative pages before selecting an OCR method.
5. Extract only currently admitted documents into one normalized record each.
6. Remove headers, footers, page numbers, broken hyphenation, and duplicate text.
7. Preserve equations and code using explicit boundary markers.
8. Create document-level train, validation, and test splits before chunking.
9. Measure the actual byte and subword token totals after normalization.
