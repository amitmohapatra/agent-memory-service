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
     **short code**: a word of at most three capitals ("LA", "IN", "DE"), a code with signs
     written onto it ("C#", "C++", "A+", "A-"), a single Latin letter after a word ("block a
     north"), at most two letters not followed by another word ("Block a", "Store la"), and
     a two-letter legal form ("SE", "SA", "CO", "AG": "Sales SE", "Store CO") - such a word is
     a code, never the connective or company form it looks like. A comma before exactly three
     final digits ("1,250") is ambiguous and kept as written; U+2212 is a minus; an en dash
     between digits is a range ("3–5" as "3-5"). A month spelled out in full and followed by
     a capitalised word is a name ("April Jones"), not a date. Each identifier remembers the
     word it labels ("Aisle 3 Bay 4" -> aisle 3, bay 4);
   * **words as written** - NFKC, case-folded, digits of any script read as ASCII, "#",
     "No.", "Nr.", "núm.", "رقم", "नंबर" before a number dropped, a leading article dropped
     unless it is part of a capitalised name ("the warehouse", but "El Salvador"), Arabic
     orthographic variants folded, every vocabulary alias replaced by its canonical phrase;
     connectives, titles and plurals are kept;
   * **loose words** - the same with plurals folded, titles and connectives dropped -
     except a direction ("to" / "from", "zum" / "vom", "a" / "de", "إلى" / "من", "से" /
     "को"), which is never dropped;
   * **titles** ("Mrs", "Herr", "السيد", "श्री"), **directions** and a **legal form** of
     three letters or more ("GmbH", "Inc", "Ltd").

   The verdict is one of:
   * **DIFFERENT** when both sides carry identifiers and their sequences differ (value,
     order or count: "Dock 3 door 4" / "Dock 4 door 3", "Line 3 3" / "Line 3"), a word labels
     different identifiers, both carry different legal forms, titles ("Mrs Patel" / "Mr
     Patel") or directions ("Shipment to Berlin" / "Shipment from Berlin"), a subject that is
     only a code meets a name without one ("LA" / "Berlin Hub", "4471" / "Acme") - unless the
     code is the name's initialism ("BH"), which is POSSIBLE - or each side has words the
     other lacks ("Acme Logistics" / "Acme Foods"). Nothing lifts such a block - not a
     vector, not a model.
   * **SAME** only when the words and identifiers are equal *in order*, or the letters are the
     same split differently ("Wal-Mart" / "Walmart"). A legal form of three letters or more
     on one side only is set aside ("Acme Logistics GmbH" / "Acme Logistics").
   * **POSSIBLE** for what a reader would ask about: equal only once plurals are folded,
     titles or connectives dropped or the words reordered ("John Roberts" / "John Robert",
     "Dr. Priya Sharma" / "Priya Sharma", "Bank of China" / "China Bank", "ventas de la
     tienda 12" / "ventas tienda 12"), a code on one side only ("Sales SE" / "Sales"), one
     side naming more than
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
   as two things ("Berlin Hub (BH)", "Bonn Hub (BH)") is not learned, a learned form of two
   or three letters counts only in capitals or before a number, like a packed acronym, and
   the tenant's definition wins over a pack's ("Data Center (DC)": "DC 3" is then "Data Center
   3", not "Distribution Center 3").
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
     because their word sets are equal - but the adjudicator may still be asked about them
     (below). A statement about a principal ("user:u1") is
     compared by identity, and then by slot: a single-valued predicate is the same subject;
     otherwise the two topics must share a third of their words;
   * *the adjudicator's gate* - with `conflict_adjudication` enabled, the one memory the model
     is asked about is the closest whose statement the matcher does not call DIFFERENT (SAME
     before POSSIBLE), instead of the closest by word overlap. Only when the words leave every
     pair undecided does the encoder score the subjects of the askable memories; its vectors
     are cached in process (float32, keyed by encoder), so a recurring subject is encoded once.
     Without the model no vector is computed. A candidate or memory without a subject is
     asked about when its wording overlaps by half, as before. A subject that is only a code
     is never POSSIBLE against a name without one, so "LA" or "Q3" does not send every
     named memory to the model. Measured on the ten LoCoMo conversations (5,882 turns, a
     counting stand-in for the model): 0.022 calls per turn against 0.004 before (128
     against 23), so no further floor was added.
   * *caches* - only subjects of at most 64 characters are cached (4,096 of them): 20,000
     sentence-length texts leave a worker's memory where it was (+0 MB RSS, against +278 MB
     when sentences were cached too).
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

- `tests/eval/golden/subject_pairs.json`: 518 labelled pairs (en 275, de 71, es 70, ar 51,
  hi 51; retail, logistics, people, suppliers, software, finance, healthcare), 270 of them
  hard negatives - identifiers, units, dates, names and suppliers sharing words, and the
  classes a review found: short codes that look like connectives ("Store LA" / "Store AL"),
  identifier order ("Dock 3 door 4" / "Dock 4 door 3"), titles ("Mrs Patel" / "Mr Patel"),
  names a plural fold joins ("John Roberts" / "John Robert"), numbers as written ("1,5 kg" /
  "15 kg", "5 €" / "5 $", "Level -1" / "Level 1") and short forms defined twice - in every
  language they apply to; halved into dev and test by a hash of the pair.
  → `benchmark/results/subject_gate.json` (words only, the eval gate) and
  `benchmark/results/subject_matching.json` (`benchmark/subjects.py`: the encoder, rapidfuzz,
  timings), both from commit a932d43.
- `tests/unit/test_subjects.py`, `tests/integration/test_subject_matching.py` (including a
  person fact under a mixed-case id past the recency window).
- Every suite (unit and SDK, contract, integration, security, e2e, agent, failure, eval) passes
  with nothing skipped, and every other eval gate (grounding, graph, memory, retrieval,
  retrieval on PDFs, tools) has the same metrics as main's run in the same environment. The
  memory gate's false-merge rate is 0.00, dedup recall 1.00.

## Results

| matcher (SAME = merge) | precision | recall | F1 | false merges (all / 270 hard negatives) | same / different pairs sent to the adjudicator |
|---|---|---|---|---|---|
| exact subject strings (before) | 1.00 | 0.07 | 0.13 | 0 / 0 | n/a |
| words only (every write) | 1.00 | 0.746 | 0.855 (dev 0.863, test 0.847) | 0 / 0 | 0.936 / 0.078 |
| words + encoder (adjudicator on) | 1.00 | 0.746 | 0.855 | 0 / 0 | 0.968 (dev 0.975, test 0.961) / 0.100 |

No false merge in any class: identifiers 0 of 92, names 0 of 61, short codes 0 of 22, numbers
0 of 19, units 0 of 14, order 0 of 13, titles 0 of 11, plural names 0 of 9, general 0 of 9,
cross-lingual 0 of 8, learned 0 of 6, partial 0 of 6. F1 by language (words only): en 0.88,
hi 0.84, de 0.82, es 0.81, ar 0.79. Recall is lower than a looser matcher's on purpose: a
match that needs a plural folded, a title dropped or the words reordered is POSSIBLE, so the
service never merges it on its own; with the adjudicator on it is asked about. The rules were
refined while reading errors on the whole set; only the encoder threshold was chosen on the
dev half alone.

rapidfuzz against the one-edit rule, on the 90 pairs whose words each side alone has: the
one-edit rule routes 7 of 25 same-subject pairs and 1 of 65 different ones; ratio >= 85 routes
7 and 2, >= 90 routes 4 and 1. It never does better, so it is not a dependency.

Cost: one comparison p50 0.05 ms / p95 0.11 ms cold, 0.005 ms warm; consolidating a statement
against 20 stored memories p50 0.24 ms / p95 0.32-0.35 ms against main's 0.08 / 0.14-0.15 ms,
with identical decisions. The encoder runs only on the 80 of 5,180 pairs the words leave
undecided, p50 5.2 ms / p95 6.5 ms each, and only with the model on.
