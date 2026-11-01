# ADR 0043: An event recorded for a user follows that user

Date: 2026-10-31. Status: accepted. Amends ADR 0036 (decision 1: `EVENT` observations kept their
default audience) and ADR 0038 (only a user's messages were compared across conversations).

## Context

ADR 0036 made a user's own words `USER`-visible and left a record the request carries (`EVENT`,
`IMPORT`, `FILE`) with the conversation's audience (`THREAD`). ADR 0038 noted the consequence
and left it open as "an audience decision, not a replacement one".

An `EVENT` (`POST /v1/messages` with `role: "EVENT"`) is something that happened in the user's
world: "Change scheduled: host app-07 will be rebooted for kernel patching on Saturday at
02:00." recorded in one chat, "The payroll service runs on host app-07." said weeks earlier in
another, then "Will payroll be down this weekend?" in a third. The statement reached the third
chat; the event did not, so the second hop was out of reach. On the capability eval's harness
sample this was the multi-hop failure in context and search on a new conversation
(`multihop-it-payroll-reboot`), and the EVENT-only records "a new conversation cannot see" were
the last named search failure of the replacement work (PR #68).

## Decision

1. **An event recorded for a user is that user's** (`native.follows_user`): an `EVENT`
   observation that names a user, recorded by the user or by an agent acting for them, marks
   every candidate made from it, and `default_visibility` makes it `USER`, exactly as a
   statement under ADR 0036: read in every conversation of that user and by every agent acting
   for them, never by another person. Its keys are `user:<t>/<u>` plus the recorder's
   `principal:` key, and an agent principal is bound to its user (`agent:<u>/<a>`), so the same
   agent acting for another user reads none of it. The anchor stays the conversation
   (`scope_for`, ADR 0036 decision 2).
2. **Unchanged:** an event with no user (a system event) keeps `THREAD` (else `TENANT`);
   explicit `THREAD` or `WORKSPACE` on an event (`POST /v1/observations` hints) keeps it;
   `IMPORT`, `FILE`, the assistant's replies, agent-written memories, `SHARED` types and the
   agent re-typing of `USER`/`PREFERENCE` (`authored_type`) are as before. `said_by_user`
   still means the user's own words; an event is not one.
3. **An event replaces and is replaced as a statement is** (ADR 0038). The pipeline runs
   cross-conversation replacement for a user's `EVENT` as well as their message: "We replaced
   Brightway with Corrimax ..." recorded as an event closes "Our main supplier of shipping
   crates is Brightway." said in another conversation, and a later statement closes an earlier
   event. Otherwise a value an event changed would now stay current beside its replacement in
   every conversation of the user, which `THREAD` used to hide. Only the recorder's own
   statements are compared (`statement_of`: same owner principal, `USER`), so an event an
   agent records for the user is compared with that agent's records for the user, as a
   harness-relayed turn already is. An event is kept sentence by sentence (no turn as said,
   `_verbatim` is for messages), so each of its sentences is a unit.
4. **No graph fact read from an earlier value is served as current** (ADR 0038 decision 8).
   "PO-88120 has shipped" replaced by "PO-88120 was delivered ... at DC-2" went to the
   history, but the relation read from it (`po-88120 has shipped`) was still among the graph
   facts beside the current value. Once events follow the user this leaked in every
   conversation (the capability eval's `temporal-retail-po-status`); the same held for a
   user's statements. The graph stage drops a fact whose source memory, or a twin collapsed
   into it, is an earlier value among the candidates
   (`graph.retrieval._without_replaced_facts`, no extra read). The memory itself is still
   served in the history section.

No new setting, no new memory type, no new table, no API change.

## Existing data

Rows keep the audience they were written with, for ADR 0036's reason: a stored `THREAD` row
does not record whether the audience was the default or an explicit choice. A deployment that
wants earlier events to follow their user re-sends them (messages are idempotent on
`source_system` + `source_message_id`).

## Consequences

* Measured on the capability eval (private stack, run date pinned, nollm and Haiku answers):
  the numbers are in the pull request that carries this ADR.
* A user's events now also reach the agents acting for them in every conversation, and a
  cached bundle in another of their conversations is rebuilt on the write (the user's revision
  moves, ADR 0036 decision 2).
* One NLI batch per user event that relates to an earlier statement, as for a message.
