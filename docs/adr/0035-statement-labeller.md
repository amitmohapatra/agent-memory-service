# ADR 0035: What a user's statement does, labelled at write

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
     short cue of one language is not misread in another.
   - **NLI**, the mDeBERTa XNLI head the grounding cascade already loads (no new model), for
     what the lexicon leaves open: a standing word or a condition before a clause that may or
     may not be an instruction, and (outside English) a statement no cue word decided. English
     hypotheses for every language - the head is cross-lingual - one batch per observation,
     with per-kind thresholds calibrated on the dev half of the labelled set. A stand-in head
     (`LexicalNLI`) is never read as a classifier.
   - **LLM**, when the tenant's policy allows `contextual_extraction` - no new use, no new
     option - and only for sentences the head was unsure of (an entailment between the
     confirmation and the decision thresholds). The model proposes a kind as JSON; the
     proposal is taken only when the head confirms the sentence entails that kind.
4. **Unsure means FACT.** When the words cannot settle it ("We always look forward to our
   camping trip" / "We never ship hazardous goods on Fridays"; "If SF is your thing, check
   out The Expanse" / "If the forklift is down, route pallets to Dock 3"), the lexicon answers
   FACT and names the rule the model should check. A durable rule nobody gave pollutes every
   later answer; a rule missed is still kept as the turn it was said in.
5. **The audit bugs are fixed where rules are made.** The extractor's rule branch reads the
   label instead of `_STANDING_RULE`: a RULE or CONDITIONAL_RULE in any language becomes the
   lasting PREFERENCE `rule` it was in English, with its trigger and exception; `is_question`
   (shared by the extractor and the labeller) no longer reads "When I ask ..." as a question.

## Evidence

`tests/eval/golden/statement_kinds.json`: 333 dev and 154 held-out sentences in en/de/es/ar/hi
across retail, logistics, software, finance, healthcare, hospitality, travel and personal
domains, including the audit's cases. Both halves were written by the author of the packs,
the held-out half before the packs, and never used for tuning - but by the same hand, so
they measure coverage of phrasing the author anticipated, not generalisation. Precision on
text nobody wrote for the packs is measured on all 16,758 sentences of the LoCoMo dialogues
(human chat, where rules and corrections are rare: every label there is an upper bound on
false positives). Numbers: `benchmark/results/statement_kinds_gate.json` (gate),
`benchmark/results/statement_labeller.json` and `statement_labeller_lexicon.json`
(`python -m benchmark.statement_labeller`).

Macro-F1 over the six kinds (NONE counted in the confusions), per language; dev / held-out:

| | en | de | es | ar | hi | all |
|---|---|---|---|---|---|---|
| lexicon alone (measured) | 0.985 / 0.983 | 0.749 / 0.875 | 0.816 / 0.828 | 0.816 / 0.828 | 0.974 / 1.000 | 0.909 / 0.932 |
| lexicon + NLI head (cached real-head scores, offline) | 0.993 / 1.000 | 0.974 / 1.000 | 1.000 / 1.000 | 0.951 / 1.000 | 0.976 / 1.000 | 0.983 / 1.000 |

The second row replays the frozen head's entailment scores for every item and hypothesis
(17 hypotheses scored once, 487 items) through the shipped decision rules and thresholds; the
end-to-end frozen-head run is `tests/eval/test_statement_kinds_gate.py::test_frozen_head_statement_kinds`.
On these sets the head earns its place on maybe-rules (German, Spanish and Arabic rules after
a verb the packs cannot list): it lifts their macro-F1 by 0.13-0.22. The change screen for
open statements added nothing here - the packs already hold this vocabulary - and is kept,
at a high threshold, for statements in other languages the packs do not cover.

No question, greeting or one-off request is stored as a rule in either half. On LoCoMo the
lexicon labels 3 of 16,758 sentences RULE, 6 CORRECTION, 6 LIFECYCLE and 12 STATUS (the first
draft of the packs: 136, 50, 68 and 44 - the precision fixes were made against this text,
never against the held-out half).

Cost on CPU (4 vCPU, the head at 2 threads): the lexicon 0.10 ms a sentence (p95 0.21). One
NLI pair is 38 ms. A maybe-rule costs one pair; an open statement outside English one pair,
and two more only when the screen fires. 18% of the labelled sentences and 11.5% of LoCoMo's
reach the head at all, so the mean added write-path cost is about 7 ms and 5 ms a sentence;
a sentence that reaches the head pays 38 ms (114 ms in the rare screened-and-classified case).
The tenant's model is called only for the head's unsure band.

## Consequences

- Downstream stages (rule assembly at read, replacement on status, lifecycle and correction)
  have a stored kind to act on; this ADR changes none of them.
- A new domain is a pack: a JSON file of cue words beside `retail.json`, named in
  `StatementLabellerSettings.packs`. A new language is a block in each pack.
- Write-path cost: the lexicon for every sentence; the NLI head only for open sentences of
  user-authored messages (agent chatter keeps to the lexicon), at most
  `nli_max_sentences` per observation; the model only for the head's unsure band.
