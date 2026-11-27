# ADR 0049: What a person says for a period is kept until its last day, and no longer

Date: 2026-11-27. Status: accepted; the model's reading of a period is not measured yet (see
Measurements).

## Context

"For today only, use the outlet channel." "This week answer in German." A person gives an
instruction for a period. Memory had two readings of it, both wrong:

- kept for good. Without a model the word patterns read no end, so the instruction stood
  for every later conversation; reproduced with "for today only" returned days later.
- not kept at all. With a model reading the message (ADR 0047) an instruction "for now, for
  this task or for this conversation" is not memory - right for "for now skip the check",
  wrong for "for today only ...", which the person expects to hold in their other
  conversations today.

The parts needed were there: a memory has a valid time (`valid_to`), `build_memory` makes
it the memory's expiry, and the hourly expiry retires an expired memory from the index and
from the `user` block. Nothing read a period from what was said.

## Decision

- **The model names the last day.** Each operation of the model's reading (ADR 0047) has
  `until`: for a rule, preference or plan the speaker limits to a period, the last ISO day
  it holds, resolved against the day it was said; empty when no period is named. An
  instruction for a period that outlasts the conversation is a rule with `until`; an
  instruction for now with no period named is still not memory.
- **Checked in code, as every field is.** `until` must be a day, on or after the day it was
  said and within the years a date may be from it. One that is not (a phrase, a day already
  past) rejects the operation: what was said for a period is kept to that period or not at
  all, never as if it had no end.
- **It lapses when the next day starts.** The memory's `valid_to` is the start of the day
  after `until` (UTC); that is its expiry, so the hourly job retires it. The standing
  sections read `valid_to` from the index (`PAYLOAD_FIELDS` now returns it) and leave a
  statement out from the moment it is over, without waiting for the job.
- **The line says it.** `(said 2026-11-27; until 2026-11-27)`: the reader knows the rule is
  not for every time.

## What it does not change

- A tenant with no model: the patterns read no period. "For today only" is kept as before.
- The day is UTC. A person far from UTC has it lapse up to a day early or late by their own
  clock; a tenant time zone is not part of a message's reading today.
- A period with no day in it ("while the sale runs") has no `until`: the model is asked for
  a day, not a condition.
- The ranked memories section may still carry a statement for up to an hour after it is
  over (until the expiry job runs); its line shows the day it was said.

## Measurements

None with a real model yet. The tests script the model (`tests/unit/test_gapfill.py`,
`tests/integration/test_model_reads_first.py`): they prove what a named day does, not that a
model names the right one. The extraction benchmark's `moment` cases and the context
check's `moment_valid` case are where a paid run shows it.

## Consequences

- A time-limited instruction reaches the person's other conversations and then stops, with
  no one having to take it back.
- One more field for the model to get right. A wrong day ends a rule early (it is said
  again) or late (it is then an ordinary rule until it is replaced).
