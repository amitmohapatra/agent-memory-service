# ADR 0007: Docling parsing, natural chunking and the Document Context Graph

**Status:** accepted · **Date:** 2026-09-14

## Decision
- **Parsers behind a port.** `DoclingParser` (MIT) handles PDF/DOCX/PPTX/XLSX/HTML/images;
  Markdown/text/HTML also parse with the dependency-free `BuiltinParser`, which is also the
  fallback. Both emit parser-independent `Block`s, so hierarchy, chunking and graph building
  are written once. Docling's layout/OCR models (PDF, images) are fetched from Hugging Face on
  first use; DOCX/PPTX/XLSX/HTML need no models (verified: DOCX parsed offline in tests).
- **Hierarchy** document > section > subsection > paragraph/table/code with section paths,
  section numbers and page spans propagated to ancestors.
- **Natural chunking.** A unit that fits `max_chunk_tokens` (400) is one chunk; oversized
  prose splits on sentences with overlap; tables split by rows with the header repeated; code
  splits on blank lines. Token estimate is deterministic (chars/4 + lines) so it never needs a
  tokenizer download; the embedding adapter enforces the model's true limit at index time.
- **Contextual Retrieval.** `contextual_text` = `Document / Section / Page / Table / Entities`
  header + original text. Only `contextual_text` is embedded/BM25-indexed; `text` is kept for
  display and citations.
- **Document Context Graph** (deterministic, no LLM): PARENT/CHILD, PREVIOUS/NEXT, ON_PAGE,
  IN_TABLE, FOOTNOTE (`[^n]`, `^n`, "(note n)"), CROSS_REFERENCE ("Section 8", "Table 2",
  "Appendix B"), MENTIONS (`entity:<canonical>`), DEFINED_BY/DEFINES (glossary and prose
  definition patterns, parenthetical aliases). Stored in `context_edges`, queried by source or
  target id.
- **Durability.** Raw bytes are staged in PostgreSQL (`file_staging`) inside the accept
  transaction; the parse job runs from staged bytes, then the raw file is archived immutably
  to the file bucket, verified, and only then are staged bytes purged. Identical bytes within a
  tenant deduplicate by SHA-256.

## Evidence
`tests/fixtures/acme_fy26_annual_report.md` encodes the spec's cross-page case (definition p1,
value p11, footnote p20, restructuring p14). `test_context_graph_recovers_cross_page_links`
asserts the DEFINED_BY, FOOTNOTE and CROSS_REFERENCE edges exist end to end, and
`test_accept_parse_and_archive` asserts them from PostgreSQL after the parse job.
