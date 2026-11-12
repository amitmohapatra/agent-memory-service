# ADR 0045: With a model, one gap-fill call reads what the rules missed in a message

Date: 2026-11-09. Status: accepted. Supersedes the narrative units and the non-English typed
facts of `contextual_extraction`, and ADR 0037's model tier of the statement labeller.

## Context

With a model allowed, `contextual_extraction` did three things, each its own call: English
messages with two or more sentences no rule read got source spans (`narrative.py`), any other
language got typed facts (`source_facts.py`), and the labeller asked the model about sentences
its NLI head was unsure of. All of them read one sentence or one message alone: a fact spread
over two turns ("I adopted a dog." / "Her name is Biscuit."), a pronoun, an offer the user then
accepted, a vague date, an update of something said a month ago were out of reach. The
capability eval scored the same with and without the model (0.96 / 0.995 / 0.99): it did not
ask for any of this.

Mem0 extracts candidate facts from a message with the recent turns, then decides ADD / UPDATE /
DELETE / NOOP against the top similar memories (Chhikara et al. 2025, arXiv:2504.19413);
Hindsight's retain step writes narrative facts with their dates and entities
(Latimer et al. 2025, arXiv:2512.12818). Closed PRs #92 (the adjudicator sees the closest
neighbours) and #93 (model-first extraction: facts appended to the turn's key) were never
measured.

## Decision

The rules run first, unchanged. Then, for a person's own message, one call
(`modules/memory/gapfill.py`) gets the message, the four turns before it, what the rules read
and the speaker's eight closest memories, and returns operations: `add`, `update n`,
`contradict n`, each with an exact quote. Code checks every op: the quote occurs in the message
(or in the turn a short reply confirms); no number the quote does not state (a checked day or
a near year aside); the fact keeps its quote's language and script; a relative date is
`temporal.dated_mentions`' day, the model's only for a phrase the library cannot read and near
the day said; an `add` the rules read, a memory holds (normalized text, a cosine of 0.92, the
NLI head's entailment both ways) or a rule reading said again with the speaker's name is
dropped. An update or a contradiction goes through the cross-conversation replacement path
(`CrossConversationReplacement.settle`: the encoder relates, the NLI head scores, the
adjudicator settles the middle; unsure keeps both, the old one as an earlier value). Every
message a person writes is read, whatever its shape: what it states is the model's to judge, and a
question often carries a fact ("... with a cat that sheds a lot?"). Calls in flight are
bounded per tenant. The memories are `provider: llm`, classified like the rules' readings, and
never reinforce or replace what was said.

The narrative-unit and typed-fact paths are removed, and with them the labeller's model tier:
the sentences the head is unsure of are in the same call. A Hindsight server, when one is
configured, still reads a person's English messages instead (`NativeMemoryIntelligence.server_reads`).
No setting, use name, route or SDK call changed; the no-model path is the rules' path.

## Measurements

Haiku for the model, the answerer and the judge; four arms. The extraction eval
(`benchmark/extraction.py`, 80 cases: spread facts, pronouns, implied facts, offers accepted,
vague dates, updates, restatements, contradictions, opinions, multi-hop and temporal recall,
German / Spanish / Arabic / Hindi, chit-chat) reads what was stored. LoCoMo conversations 0-2
(385 questions of categories 1-4) is answered from the context and graded with the Mem0
paper's J prompt, verbatim.

| arm | recall | false facts | duplicates | wrong updates | LoCoMo J | $ / 1k messages | messages/s |
|---|---|---|---|---|---|---|---|
| A rules only (main) | 0.23 | 0 | 0.13 | 0 | 0.896, 0.873 (two runs) | 0 | 4.2 |
| B rules + gap-fill | **0.94-0.95** (two runs) | 0 | 0.10-0.20 | 0-0.05 | 0.862, 0.883 (two runs) | 0.37 | 0.53 |
| C model-first (#93) | 0.27 | 0 | 0.12 | 0 | 0.870 | 0.23 | 0.52 |
| B + #92 neighbours | 0.94 | 0 | 0.12 | 0.05 | 0.860 | 0.37 | 0.53 |

B is the winner on what is stored: four times the facts the rules keep, no false fact, as few
duplicates as the rules' own restatements make, and a wrong update in one run of two (a phone
number the model added instead of updating, whose two values the replacement path keeps apart).
LoCoMo does not separate the arms: two runs of A differ by 2.3 points and two of B by 2.1
(28 questions flip), and the means (A 0.884, B 0.873) are within that; the verbatim turns already
carry LoCoMo's answers.
C stores nothing new (its facts ride on the turn's key), and #92 added a wrong update and no
recall; both stay closed. Throughput is the model's latency on the in-process benchmark's
sequential queue (one message at a time); the worker runs messages concurrently, up to four
calls per tenant.

## Consequences

- A tenant with a model gets readable, dated facts from what was said across turns; the same
  call costs about $0.0004 a message with Haiku (input about 1.5k tokens, output about 400).
- The model's facts group with their turn at retrieval (same source and instant,
  `memory_ranking.statements`), so they add a way to find a turn rather than another line to
  read.
- Model errors that pass the checks remain possible: a pronoun bound to the wrong person
  (Arabic `ar-coref`: the speaker, not the sister, moved), an implied fact the model leaves out.
