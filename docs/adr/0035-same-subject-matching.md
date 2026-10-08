# ADR 0035: Same-subject matching: identifiers block, vocabulary allows, the encoder only asks

Date: 2026-10-08. Status: accepted. Amends ADR 0009 (consolidation's slot rules and the
conflict adjudicator's gate).

## Context

Consolidation can only replace a value or link two facts once it knows that two statements
are about the same thing. Until now it had two ways of knowing, both too narrow:

* **Subject equality.** A slot ("same subject + predicate") matched only when the stored
  subject strings were identical, so "FORKLIFT-4 uses 48V batteries" and "Forklift 4 uses 48V
  batteries" were two facts, and "PO-4471" was never "purchase order 4471".
* **Word overlap for the model.** The conflict adjudicator (`conflict_adjudication`, ADR 0009)
  was consulted only for pairs sharing at least half their words (Jaccard >= 0.5). That is both
  too strict and too loose: "The billing service runs on Cloud Run" and "Billing Service runs
  on Kubernetes in Frankfurt" (one subject, overlap 0.43) never reached it, while "The billing
  service runs on Cloud Run" and "The shipping service runs on Cloud Run" (two subjects,
  overlap 0.67) did - and a model that answers "update" there closes a true fact.

What decides identity is not similarity. "Warehouse 3" and "Warehouse 13", "SKU-1001" and
"SKU-1010", "5 kg" and "5 lb" are near-identical strings and near-identical vectors, and
different things; "DC 3" and "Distribution Centre 3" share almost nothing and are one thing.

## Decision

1. **A subject is parsed, then compared under rules that block or allow**
   (`domain/subjects.py`). Parsing splits a subject into **identifiers** - every number, code
   ("A12", "Q3", "SKU-1001"), number with its unit ("5 kg", "$5", "10%"), date and month name -
   **canonical words** (NFKC, case-folded, digits of any script read as ASCII, "#", "No.",
   "Nr.", "núm.", "رقم", "नंबर" before a number dropped, articles, connectives and titles
   dropped, a light plural fold, the Arabic article and spelling variants folded, every
   vocabulary alias replaced by its canonical phrase) and a **legal form** ("GmbH", "Inc").
   The verdict is one of:
   * **DIFFERENT** when both sides carry identifiers and they differ, or both carry legal
     forms that differ, or the two sides each have words the other lacks ("Acme Logistics" /
     "Acme Foods", "John Smith" / "John Miller"). Nothing lifts an identifier block - not a
     vector, not a model.
   * **SAME** when the canonical words and the identifiers are equal, or the letters are the
     same split differently ("Wal-Mart" / "Walmart").
   * **POSSIBLE** for what a reader would ask about: one side names more than the other
     ("Acme" / "Acme Logistics", "Forklift" / "Forklift 4"), one edit apart in words of five
     letters or more ("Jonathan" / "Jonathon"; "Jon" / "Joan" are two people), an initialism
     ("GFS" / "Global Freight Solutions"), or - with the encoder - a cosine of at least
     `DENSE_POSSIBLE` (0.80). A POSSIBLE pair is never merged by the service itself.
2. **The vocabulary is data** (`domain/vocabulary/generic.json`, `retail.json`), both packs
   always on, no option: number markers, connectives, titles and legal forms in English,
   German, Spanish, Arabic and Hindi; units and months in those languages; common
   abbreviations and spellings ("hazmat", "dept", "centre"); retail shorthand ("OOS", "PO",
   "DC", "SKU", "ASN", "3PL", "RTV"). The retail glossary's query expansion (`domain/glossary.py`)
   reads the same retail table - one table, two readers. An acronym that is also a word
   ("OH", "ST", "POS") counts only in capitals or directly before a number.
3. **A tenant's own vocabulary is learned from its own text, with nothing configured.** An
   abbreviation defined in a stored memory - "hazardous materials (hazmat)", "OOS (out of
   stock)", "WOS stands for weeks of supply", a compound's capitals "Zentrallager (ZL)" - is
   learned when it is checked to be one (`abbreviates`: the short form is made of the long
   form's word prefixes in order), so "Berlin (Germany)" teaches nothing. The names the stored
   memories mention are the tenant's known names: a short form two of them extend ("John"
   with "John Smith" and "John Miller") is DIFFERENT, not POSSIBLE. Both come from the rows
   consolidation has already loaded; nothing extra is read or stored.
4. **One service on the write path** (`modules/memory/subjects.py`, `SubjectMatcher`), shared
   by the provider and the pipeline:
   * *candidate generation* - the stored memories consolidation compares with are looked up by
     every spelling the matcher allows for the candidate's subject (`spellings`: identifier
     joins, alias forms), beside the normalized hash and the newest rows, on the existing
     `(tenant_id, subject, predicate)` index;
   * *slot rules* - "same subject" in the reinforce / single-valued / replacement-signal rules
     is the matcher's SAME, not string equality. A statement about a principal ("user:u1") is
     compared by identity, and then by slot: a single-valued predicate is the same subject;
     otherwise the two topics must share a third of their words;
   * *the adjudicator's gate* - with `conflict_adjudication` enabled, the one memory the model
     is asked about is the closest whose statement the matcher does not call DIFFERENT (SAME
     before POSSIBLE), instead of the closest by word overlap. Only when the words leave every
     pair undecided does the encoder score the subjects; its vectors are cached in process, so
     a recurring subject is encoded once. Without the model no vector is computed.
   Replacement semantics are unchanged: which predicates are single-valued, and when a value
   supersedes another, is ADR 0009's (and its successor's).
5. **No new dependency.** The typo rule is a one-edit check written here. rapidfuzz was
   measured against it on the labelled pairs (below) and not adopted.

## Consequences

* A respelled, abbreviated or aliased subject reinforces the memory it names instead of
  duplicating it; an identifier, unit or date that differs never merges, whatever the words.
* The adjudicator is consulted about the pairs it can decide - same subject, other words - and
  no longer about pairs that merely share words.
* The vocabulary grows by editing JSON; a pack entry that is ambiguous belongs under `cased`
  or not at all. A tenant's learned abbreviation is only as reachable as the memory that
  defines it: one defined long ago in another scope is not among the rows consolidation reads.
  A tenant-wide vocabulary store was not built for that.
* Cross-lingual subjects ("Forklift 4" / "Gabelstapler 4") are POSSIBLE only through the
  encoder, so only a write with the adjudicator can join them.

## Evidence

- `tests/eval/golden/subject_pairs.json`: 436 labelled pairs (en 245, es 55, de 54, ar 41,
  hi 41; retail, logistics, people, suppliers, software, finance, healthcare), 193 of them hard
  negatives (identifiers, units, dates, names and suppliers sharing words), halved into dev and
  test by a hash of the pair. → `benchmark/results/subject_gate.json` (words only, the eval
  gate) and `benchmark/results/subject_matching.json` (`benchmark/subjects.py`: the encoder,
  rapidfuzz, timings).
- `tests/unit/test_subjects.py`, `tests/integration/test_subject_matching.py`.
- The memory gate (`benchmark/results/memory_gate.json`) is unchanged: false-merge rate 0.00,
  dedup recall 1.00.

RESULTS_TABLE
