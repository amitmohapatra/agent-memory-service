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
   (`domain/subjects.py`). Parsing reads at most `MAX_SUBJECT_CHARS` (300) characters and
   splits a subject into:
   * **identifiers, in order** - every number (a decimal comma read as one: "1,5 kg" is 1.5;
     a minus kept: "Level -1"), code ("A12", "Q3", "SKU-1001"), number with its unit or
     currency before or after it ("5 kg", "$5", "5 €", "10%"), date, month name, and every
     **short code**: a word of at most three capitals ("LA", "IN", "DE") or at most two
     letters not followed by another word ("Block a", "Store la") - such a word is a code,
     never the connective it looks like. Each identifier remembers the word it labels
     ("Aisle 3 Bay 4" -> aisle 3, bay 4);
   * **words as written** - NFKC, case-folded, digits of any script read as ASCII, "#",
     "No.", "Nr.", "núm.", "رقم", "नंबर" before a number dropped, a leading article dropped,
     a connective between two words dropped, Arabic orthographic variants folded, every
     vocabulary alias replaced by its canonical phrase; titles and plurals are kept;
   * **loose words** - the same with plurals folded, titles and connectives dropped;
   * **titles** ("Mrs", "Herr", "السيد", "श्री") and a **legal form** ("GmbH", "Inc").

   The verdict is one of:
   * **DIFFERENT** when both sides carry identifiers and their sequences differ (value,
     order or count: "Dock 3 door 4" / "Dock 4 door 3", "Line 3 3" / "Line 3"), a word labels
     different identifiers, both carry different legal forms or different titles ("Mrs
     Patel" / "Mr Patel"), or each side has words the other lacks ("Acme Logistics" / "Acme
     Foods"). Nothing lifts such a block - not a vector, not a model.
   * **SAME** only when the words and identifiers are equal *in order*, or the letters are the
     same split differently ("Wal-Mart" / "Walmart").
   * **POSSIBLE** for what a reader would ask about: equal only once plurals are folded,
     titles dropped or the words reordered ("John Roberts" / "John Robert", "Dr. Priya
     Sharma" / "Priya Sharma", "Bank of China" / "China Bank"), one side naming more than
     the other ("Acme" / "Acme Logistics", "Forklift" / "Forklift 4"), one edit apart in
     words of five letters or more ("Jonathan" / "Jonathon"), an initialism ("GFS" /
     "Global Freight Solutions"), or - with the encoder - a cosine of at least
     `DENSE_POSSIBLE` (0.77). A POSSIBLE pair is never merged by the service itself.
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
   form's word prefixes in order), so "Berlin (Germany)" teaches nothing. A short form defined
   as two things ("Berlin Hub (BH)", "Bonn Hub (BH)") is not learned, and a learned form of
   two or three letters counts only in capitals or before a number, like a packed acronym.
   Definitions are looked for in the first `MAX_DEFINITION_CHARS` (2,000) characters of a
   text, with patterns anchored at word starts and bounded, so reading them is linear in the
   text (2,000,000 characters: a few milliseconds). The names the stored
   memories mention are the tenant's known names: a short form two of them extend ("John"
   with "John Smith" and "John Miller") is DIFFERENT, not POSSIBLE. Both come from the rows
   consolidation has already loaded; nothing extra is read or stored.
4. **One service on the write path** (`modules/memory/subjects.py`, `SubjectMatcher`), shared
   by the provider and the pipeline:
   * *candidate generation* - the stored memories consolidation compares with are also looked
     up by a few stored spellings of the candidate's subject (`spellings`: as written first -
     an identity such as "user:Alice" only as written, ids are case-sensitive - then
     lower-cased, the first identifier joined five ways, alias forms; at most 12), beside the
     normalized hash and the newest rows, on the existing `(tenant_id, subject, predicate)`
     index. It is a lookup aid, not a guarantee that every older row is found;
   * *slot rules* - "same subject" in the reinforce / single-valued / replacement-signal rules
     is the matcher's SAME, not string equality. The subject compared is the one written in
     the memory's entities when the stored one is its lower-cased copy ("Store LA", not "store
     la"). Two named subjects that are not SAME are never merged by the wording rules either
     (lexical and dense similarity), which used to join "Dock 3 door 4" and "Dock 4 door 3"
     because their word sets are equal. A statement about a principal ("user:u1") is
     compared by identity, and then by slot: a single-valued predicate is the same subject;
     otherwise the two topics must share a third of their words;
   * *the adjudicator's gate* - with `conflict_adjudication` enabled, the one memory the model
     is asked about is the closest whose statement the matcher does not call DIFFERENT (SAME
     before POSSIBLE), instead of the closest by word overlap. Only when the words leave every
     pair undecided does the encoder score the subjects of the askable memories; its vectors
     are cached in process (float32, keyed by encoder), so a recurring subject is encoded once.
     Without the model no vector is computed. A candidate or memory without a subject is
     asked about when its wording overlaps by half, as before. Measured on the ten LoCoMo
     conversations (5,882 turns, a counting stand-in for the model): 0.020 calls per turn
     against 0.004 before (120 against 23), so no further floor was added.
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

## Results

| matcher (SAME = merge) | precision | recall | F1 | false merges (all / hard negatives) | same / different pairs sent to the adjudicator |
|---|---|---|---|---|---|
| exact subject strings (before) | 1.00 | 0.07 | 0.13 | 0 / 0 | n/a |
| words only (every write) | 1.00 | 0.872 | 0.932 (dev 0.938, test 0.926) | 0 / 0 | 0.930 / 0.062 |
| words + encoder (adjudicator on) | 1.00 | 0.872 | 0.932 | 0 / 0 | 0.963 (dev 0.975, test 0.952) / 0.093 |

F1 by language (words only): en 0.95, ar 0.93, hi 0.93, de 0.89, es 0.89. The rules were refined
while reading errors on the whole set; only the encoder threshold was chosen on the dev half alone.

Cost: one comparison p50 0.04 ms cold, 0.002 ms warm; consolidating a statement against 20 stored
memories p50 0.19 ms / p95 0.33 ms (main: 0.067 / 0.10 ms). The encoder runs only on the 82 of
4,360 pairs the words leave undecided, p50 5.0 ms / p95 6.6 ms each, and only with the model on.
Every other eval gate (retrieval, grounding, graph, memory, tools) is identical to main's when
both run in the same environment.
