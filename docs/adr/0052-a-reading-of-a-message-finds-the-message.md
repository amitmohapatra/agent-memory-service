# ADR 0052: What a model read from a message finds the message; the message as said is what is shown

Date: 2026-11-28. Status: accepted; measured on LoCoMo conversations 0-1 only so far (see
Measurements).

## Context

With a model on, a message is stored twice over: the turn kept as said, and the statements
the model read from it ("Caroline hopes to give a home to children who have not had one.",
marked "model-extracted, unverified"). Both are searched.

Ranking already lets the turn stand for the statement when the search finds both
(`memory_ranking.statements`, `_standing_for`: the turn holds every reading of it whole).
Two places did otherwise:

- **A reading found without its turn** stood alone, in its paraphrase, without what was said
  around it. In a LoCoMo context with the model on, 32 of 136 lines were such readings - 19%
  of the characters - and three of them said one thing ("give a home to children") from two
  messages.
- **`said_once`** dropped the turn when a reading of the same message was shown: the
  opposite of ranking's rule.

LongMemEval (Wu et al., 2024) measured the choice directly: the round as said, with the facts
read from it added to its *key*, retrieved and answered better than the facts as the value.
The facts help find the round; the round is what the reader should read.

## Decision

- **A reading found without its turn comes as the turn** (`RetrievalEngine._said_for_readings`):
  in the reading's place and at its score, with the reading among the turn's `duplicates`, so
  an expansion that reaches the reading by id does not bring it back. One indexed read
  (`MemoryRepository.turns_of`) and one batched fetch; only turns the reader may read, and
  only current ones. A reading with any source that is not a message, a memory derived from
  memories, and an exact lookup by id keep what they are.
- **`said_once` keeps the turn**: what a model read from a message, and a sentence word
  patterns cut out of it, go when the message is shown as said. A statement the profile lists
  still takes its message with it.

## Consequences

- With a model on, the context carries the messages that answer and none of their
  paraphrases; the model's readings still decide what is found.
- A reading whose message was not kept as said (the model judged it not worth keeping) is
  still shown as the reading.
- Without a model nothing changes: there are no readings to swap.

## Measurements

LoCoMo conversations 0-1 (304 questions), Haiku 5.5 answering and judging under the strict
rules, the model's store, memory attached, read-first and decline-first notes: 88.4% of real
questions (83.6% overall) with this decision, against 89.7% (86.2%) before it. A repeated
arm differs by 1-2 points, so this is within noise on real questions; the overall gap is the
questions about the wrong person (74.7% -> 67.6%, five questions). The decision stands on
what it removes (19% of a context in paraphrases, and a rule that contradicted ranking's);
the arms that search memory on demand, where a found reading now arrives as its turn with
its speaker, are not measured yet.
