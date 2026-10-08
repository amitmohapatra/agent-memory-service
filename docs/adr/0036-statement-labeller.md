# ADR 0036: What a user's statement does, labelled at write

Date: 2026-10-08. Status: accepted.

## Context

The memory pipeline extracted *facts* from a message but never recorded what the sentence
*did*. A standing instruction, a correction, the state of a forklift and the end of a
supplier contract were all just memories, so nothing downstream could keep a rule in every
answer or let "Forklift #4 is back on the floor" replace "Forklift #4 is out of service".
An audit against retail store-operations cases found three concrete failures in the one
place the extractor did look at what a sentence does, the standing-rule pattern
(`_STANDING_RULE` in `modules/memory/native.py`):

- "Whenever I ask for a stock audit, always format the response as a markdown table ..." was
  not recognised, and "When I ask ..." was read as a **question**: nothing was stored at all
  (case 8);
- a rule failed when any text preceded "Never" ("For weekly overviews, never include ...",
  case 11);
- the exception of a conditional rule ("... unless I specifically type 'include out of
  stock'") was not parsed.

The pattern was English-only and anchored at the start of the sentence.

## Decision

1. **A closed vocabulary, `StatementKind`** (`domain/enums.py`): `FACT`, `RULE`,
   `CONDITIONAL_RULE`, `STATUS`, `CORRECTION`, `LIFECYCLE`; a question, a greeting or a
   one-off request gets none. Precedence when two apply: CORRECTION > CONDITIONAL_RULE > RULE
   > LIFECYCLE > STATUS > FACT ("Actually, we terminated our contract with Uline" corrects).
   A request-scoped trigger ("whenever I ask for X") keeps a RULE; an exception, or a
   condition about the world ("if a delivery is late"), makes it CONDITIONAL.
2. **Stored with the memory, no migration.** `system_metadata["statement_kind"]`, and for a
   rule `rule_trigger` / `rule_exception` as said; `domain.memory.statement_kind_of` reads
   it. The kind is on every candidate a sentence produced; a verbatim turn carries the most
   telling kind of its sentences. Nothing on the read path reads it yet: retrieval and the
   bundle are unchanged.
3. **Three tiers, cheapest first** (`modules/memory/statements.py`):
   - **Lexicon**, every language, about 0.1 ms a sentence. Cue words are **data**:
     `modules/memory/lexicon/generic.json` (domain-free: always, unless, actually, from now
     on, in English, German, Spanish, Arabic and Hindi) and `retail.json` (out of stock,
     recalled, back on the floor, delisted, ...), both loaded by default
     (`StatementLabellerSettings.packs`). The code holds only structure: a standing word
     counts where it opens a clause, after a filler ("please", "important:") or a standing
     phrase set off by a comma ("from now on, ..."); a condition or exception clause is cut
     out to the end of its clause; "when"/"if" + a subject opens a condition, not a question.
     A sentence is read with its own language's cues and English's (`domain.language`), so a
     short cue of one language is not misread in another. Cues are **word families**, not
     phrases: Arabic words may carry their clitics and enclitics (wa-/al- before, -ni "me",
     the -i of a feminine imperative after), German and Spanish verbs their inflections
     (`discontinu*`, `eingestellt`, `cerrad[oa]s?`), a recipient is a role ("the store
     manager", Spanish personal *a*), and a condition counts as any occurrence when it is
     indefinite ("if a delivery is late", "if there's any change"). Sentence **shape** decides
     what words cannot: a consequent ("then", *dann*, *to*) and a vocative ("AI, ...") end a
     clause; a condition trailing a clause no instruction word opens makes the sentence a
     description ("Life is better when we're together"); a standing word after a copula or
     before a third-person verb describes a habit ("isn't always easy", "Nature always
     cheers me up").
   - **NLI**, the head the grounding cascade already loads (no new model), only where the
     lexicon is unsure: a standing word or a condition before a clause that may or may not be
     an instruction, or a word that only suggests a kind ("stopped", "started", "actually").
     The head scores the **one** hypothesis of the kind the lexicon suspects - one pair a
     sentence, every open sentence of an observation in one batch; a sentence no cue marks
     costs nothing. English hypotheses for every language (the head is cross-lingual):
     "This is an instruction." for both rule kinds (the lexicon has already found the
     condition), "Something is broken, not working, or unavailable.", "Something has ended,
     been terminated, or newly started." and "Something said earlier was wrong.", at
     thresholds 0.7 (rules) and 0.9 (the rest). They were chosen on the generalisation dev
     set (below) from cached head scores: among the choices within one dev item of the best,
     the one that fires least on chat. "The speaker is correcting an earlier mistake." was
     dropped because it holds for 47% of LoCoMo's chat sentences. A stand-in head
     (`LexicalNLI`) is never read as a classifier.
   - **LLM**, when the tenant's policy allows `contextual_extraction` - no new use, no new
     option - and only for sentences the head was unsure of (an entailment between the
     confirmation and the decision thresholds). The model proposes a kind as JSON; the
     proposal is taken only when the head confirms the sentence entails that kind.
4. **Unsure keeps the reading without the rule.** When neither the words nor the head settle
   it ("We always look forward to our camping trip" / "We never ship hazardous goods on
   Fridays"), the sentence is what it is without the suspected kind: a FACT, or - for a
   request ("Send alerts if the location deviates") - no kind at all. A durable rule nobody
   gave pollutes every later answer; a rule missed is still kept as the turn it was said in.
5. **The audit bugs are fixed where rules are made.** The extractor's rule branch reads the
   label instead of `_STANDING_RULE`: a RULE or CONDITIONAL_RULE in any language becomes the
   lasting PREFERENCE `rule` it was in English, with its trigger and exception; `is_question`
   (shared by the extractor and the labeller) no longer reads "When I ask ..." as a question.

## Evidence

Three labelled sets, and one unlabelled one (the labelled ones in `tests/eval/golden/`):

- `statement_kinds.json`: 333 dev and 154 held-out sentences in en/de/es/ar/hi, written by
  the packs' author (the held-out half before the packs). The labeller scores 0.986 / 1.000
  macro-F1 on them. Being by the same hand, they measure coverage of phrasing the author
  anticipated, not generalisation - and the first draft of this ADR, which scored them alone,
  overstated it.
- `statement_kinds_blind.json`: sentences a local language model wrote to order - four per
  request, one request per (language, kind, two sampled domains) - labelled with the kind
  asked for, then **cleaned against the written guidelines only**: garbled, truncated or
  ambiguous items dropped and mislabelled ones relabelled, every change listed with its
  reason. *Blind set 1* (retail, warehouse, software, clinic, hotel, finance, personal; 276
  generated, 235 kept, label noise 22.8%) was read and the packs and thresholds were tuned
  against it after a first score of 0.602 on its 276 raw sentences: it is the **generalisation dev set**. *Blind set
  2* (seed changed; logistics, pharmacy, e-commerce, manufacturing, school, restaurant, car
  repair, bank; 278 generated, 222 kept, label noise 24.5%) was generated afterwards, cleaned
  before any prediction on it existed, and scored once - the numbers below. (A later fix,
  made against LoCoMo chat only, was re-scored: blind 2 labels did not change.)
- LoCoMo's 16,758 dialogue sentences (human chat, no labels): rules, corrections, statuses and
  lifecycle changes are rare there, so every one the labeller finds is an upper bound on its
  false positives.

"main" is what the extractor did before this ADR: a question or acknowledgement stores
nothing (NONE), its English standing-rule pattern stores a rule, anything else a fact
(`benchmark.evaluation.statement_kinds.baseline_kind`). Macro-F1 is over the six kinds, NONE
counted in the confusions.

**Blind set 2 (held out, scored once)**, 222 sentences:

| | macro-F1 | FACT | RULE | COND_RULE | STATUS | CORRECTION | LIFECYCLE | NONE |
|---|---|---|---|---|---|---|---|---|
| main | 0.095 | 0.29 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 | 0.92 |
| B2 lexicon | 0.787 | 0.56 | 0.86 | 0.62 | 0.82 | 0.98 | 0.89 | 0.91 |
| B2 lexicon + NLI | 0.829 | 0.57 | 0.89 | 0.83 | 0.84 | 0.95 | 0.89 | 0.97 |

| macro-F1 by language | en (n=54) | de (n=49) | es (n=52) | ar (n=26) | hi (n=41) |
|---|---|---|---|---|---|
| main | 0.172 | 0.036 | 0.062 | 0.045 | 0.041 |
| B2 lexicon | 0.864 | 0.828 | 0.732 | 0.581 | 0.723 |
| B2 lexicon + NLI | 0.961 | 0.870 | 0.710 | 0.631 | 0.790 |

B2 is above main in every kind and every language, and below it in no (language, kind)
cell. Without the head - the lexicon alone, as a deployment with only the stand-in head
runs - NONE is 0.91 against main's 0.92. Spanish
is the one language where the head lowers the lexicon's score (0.732 -> 0.710); the weak
spots are Spanish and Arabic conditional rules and statements the packs have no word for
(FACT 0.57: most of its errors are conditional rules, statuses and lifecycle changes read as
plain facts - misses, which stay retrievable as the turn they were said in, rather than
false rules).

**Blind set 1 (dev)**, 235 sentences:

| | macro-F1 | FACT | RULE | COND_RULE | STATUS | CORRECTION | LIFECYCLE | NONE |
|---|---|---|---|---|---|---|---|---|
| main | 0.130 | 0.28 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 |
| B2 lexicon | 0.886 | 0.80 | 0.82 | 0.81 | 0.94 | 0.98 | 0.97 | 0.87 |
| B2 lexicon + NLI | 0.904 | 0.81 | 0.87 | 0.87 | 0.94 | 0.98 | 0.97 | 0.91 |

| macro-F1 by language | en (n=61) | de (n=51) | es (n=53) | ar (n=35) | hi (n=35) |
|---|---|---|---|---|---|
| main | 0.187 | 0.050 | 0.061 | 0.062 | 0.000 |
| B2 lexicon | 0.943 | 0.818 | 0.931 | 0.682 | 0.900 |
| B2 lexicon + NLI | 0.976 | 0.866 | 0.889 | 0.758 | 0.922 |

Here B2 is above main in every kind and language; in two (language, kind) cells it is not -
German NONE 0.80 against 0.86 and Arabic NONE 0.95 against 1.00 (requests the packs read as
statements).

**LoCoMo** (no labels): with the head, 9 sentences are labelled RULE, 65 CONDITIONAL_RULE, 25
LIFECYCLE, 15 STATUS and 7 CORRECTION of 16,758. The rule kinds are mostly "Let me know if
you need anything" (an offer phrased as a notify-me-when) and habits with a plural subject
("Rock concerts always have such an electrifying atmosphere"); no question is stored as a
rule in any set.

Cost on CPU (4 vCPU shared with other services, load average about 2; the head at 2
threads, the per-worker share it was frozen with): the lexicon is 0.1-0.2 ms a sentence
(p95 0.3-0.7). One NLI pair is 49 ms at the median (37 ms at 4 threads). A sentence costs a
pair only when the lexicon is unsure: 3.0% of LoCoMo's sentences (8.1% of its turns), 5.5%
of blind set 1 and 9.0% of blind set 2. Per statement, labelled as a message of its own:
p50 0.14 / 0.18 ms, p95 46.8 / **52.0** ms, mean 3.2 / 5.2 ms on blind sets 1 / 2 - on
instruction-dense text the p95 is the cost of one pair, and on blind set 2 it is 2 ms over a
50 ms budget at 2 threads. On LoCoMo, labelled turn by turn as the write path labels them:
p50 0.19 ms, p95 15.5 ms, mean 1.9 ms a statement; 0.5 / 54.9 / 5.4 ms (p50 / p95 / mean) a
turn. The tenant's model is called only for the head's unsure band.
Measured by `benchmark/statement_labeller.py` (`benchmark/results/statement_labeller.json`,
`statement_labeller_lexicon.json`) and the gate (`statement_kinds_gate.json`), which report
the blind sets beside the baseline.

## Consequences

- Downstream stages (rule assembly at read, replacement on status, lifecycle and correction)
  have a stored kind to act on; this ADR changes none of them.
- A new domain is a pack: a JSON file of cue words beside `retail.json`, named in
  `StatementLabellerSettings.packs`. A new language is a block in each pack.
- Write-path cost: the lexicon for every sentence; one NLI pair only for a sentence of a
  user-authored message the lexicon is unsure of (agent chatter keeps to the lexicon), at
  most `nli_max_sentences` per observation; the model only for the head's unsure band. On
  instruction-dense text the p95 per statement is the cost of one pair, which depends on the
  head's thread share: about 50 ms at 2 threads on the measured box.
- The blind sets are kept beside the golden set and reported by the gate against the
  baseline; blind set 2 is no longer unseen once this is read - a further claim of
  generalisation needs a new one.
