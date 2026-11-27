# ADR 0050: What was stored before a model read messages is read again, and what was never the person's statement is forgotten

Date: 2026-11-27. Status: accepted; the model's verdicts on stored messages are not measured
yet (see Measurements).

## Context

ADR 0047 has the model read a person's message before anything else does: it says whose
words a sentence is, whether an instruction is for every time, and whether the message is
kept beyond its conversation (a statement; not a request or shared material). That holds
for messages that arrive from then on. What was stored before is still stored, and still
served:

- a skill's text pasted into a chat ("save this skill: ... Never create, update, or delete
  allocations.") is the standing rule of the person who pasted it, a line of their `user`
  block, and by being the later "never ..." about allocations it replaced the rule they
  gave themselves ("Never create an allocation when ATA is 0.");
- "ok for now skip the consent stage" is a memory of the person in every later
  conversation about a plan;
- a request ("show the top 20 as a chart") is a thing the user said, ranked beside what
  they told the assistant about their work.

Measured on the storage benchmark with no model (103 cases, 192 messages): 77% of pasted or
other people's text was stored as the user's own, and every request and "for now"
instruction was kept beyond its conversation. The context check (ADR 0048) removes such an
item from one context when a model is reachable on the read; it does not unmake the profile
built from them, restore the rule a pasted sentence replaced, or help a read with no model.

## Decision

A maintenance tool reads the stored messages again with the model and forgets what was made
from the ones it says are no statement (`tools/clean_up.py`).

- **The same reading as ingest.** `ObservationPipeline.reading` asks the model about a
  stored message exactly as `run` does for a new one - the same call, bound to the
  message's author (their key pays, their tenant's policy decides) - and writes nothing.
- **Read as it was the first time.** The model is shown the speaker's closest memories and
  adds nothing one of them holds. The memories made from the message itself are therefore
  left out of what it is shown: with them in, it would read nothing from a message whose
  rule is already stored, and the rule would be taken for unread.
- **Only what the patterns made, and only of a message that is no statement.** Candidates
  are the current memories a word pattern made from a person's message: the turn as said, and
  the rules, instructions, preferences and attributes read from it. A model's own reading
  (`provider: llm`) is never a candidate. Where the model says the message is a statement,
  or the message is by its shape not asked about, everything made from it stays.
- **What the model reads from the same words stays.** For a request or material, a
  pattern's reading is kept when the model's reading quotes its words: a rule, an
  instruction or a preference when that reading is itself the person's rule or preference;
  a fact about the person by any reading. "Show the top 20. From now on always answer as a
  table." loses the request and keeps the rule. The turn as said is never kept: the message
  is no statement.
- **Forgotten the way a person's "forget this" is.** `MemoryService.forget`, as the
  tenant's administrator: the index entry goes, the profile is rebuilt, and what the
  forgotten statement had replaced holds again.
- **It only removes.** Nothing is added or reworded: the model's reading decides what
  stays, and is not itself stored.
- **A dry run by default.** Without `--apply` nothing changes; the report counts what was
  read, kept, confirmed and would be forgotten, with examples. `--user`, `--before` and
  `--limit` bound a run.
- **Never a guess.** A message the model may not read (no model allowed for its author) or
  does not answer for is left as it is and counted (`unread`); a later run reads it.

## What it does not change

- A tenant with no model: nothing is read and nothing is forgotten.
- A statement that holds something the patterns misread (a quoted "never ..." inside the
  person's own account) keeps everything made from it; the context check is what keeps
  such a line out of a context it does not help.
- What a model read (ADR 0047), documents, events, imports and feedback notes.
- The transcript: a forgotten turn is still a message of its conversation, read by that
  conversation's window and by `memory_search` over messages.

## Measurements

None with a real model yet. `tests/integration/test_clean_up.py` scripts the model: it
proves what a verdict does - a dry run changes nothing, material and a request are
forgotten with what was made from them, the person's own rule holds again, a rule inside a
request stays, an unanswered message is left, one user's messages alone are read - not that
a model's verdict is right. The storage benchmark's `material` and `request` cases, read
before and after a run, are where a paid run shows it. One model call per stored message
that has a pattern-made memory.

## Consequences

- A deployment that stored months of messages without a model can be brought to what it
  would hold had the model read them, without clearing anything and while serving.
- A wrong verdict forgets something the person did state. It is a forget, not a delete:
  the message stays in its conversation, and saying it again stores it again. The dry
  run's examples are there to be read before `--apply`.
- It is the third part of one change: ADR 0047 keeps what is no statement from being
  stored, ADR 0048 keeps what does not help from being sent, this removes what was stored
  before either.
