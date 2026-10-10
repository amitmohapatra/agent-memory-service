# ADR 0036: What a user says follows that user, in the words they said it

Date: 2026-10-08. Status: accepted. Amends ADR 0005 (default audiences) and ADR 0013 (an agent's
working notes).

## Context

"The master lock code for the hazardous materials cage in Warehouse 3 is 8492." said in one
chat, then "I need to get into the hazmat cage in Warehouse 3. What's the code?" in a new one,
found nothing. Three things stood between them:

* **The statement was the conversation's.** With no `visibility`, every memory made from a
  message in a thread was `THREAD` (`default_visibility`), readable only in that thread. A
  user's facts did not reach their next conversation, which is the point of a memory service;
  integrators worked around it by passing `visibility="USER"` on every write.
* **The sentence could be dropped.** A turn is kept verbatim beside what the rules read from
  it, except when a rule read all of it: then the reading replaced the turn. A reading is not
  the turn. "All seasonal holiday merchandise must be routed to Overflow Storage Facility B"
  was read as a seven-day `TASK` and nothing else, so the rule was gone in a week; a fact can be
  merged into older wording or fade unused after about seventy idle days, and only the turn is
  restated, keyed with the question it answers and kept from forgetting.
* **Long names were not facts.** The fact rule's subject stopped at five words, so "master lock
  code for the hazardous materials cage in Warehouse 3" was no subject at all.

A turn that an agent harness relays for the user carries the agent's lineage, and the agent rule
made such a user's `TASK` or `EPISODIC` statement the agent's `PRIVATE` note.

## Decision

1. **A user's own words default to `USER`.** `said_by_user` (an observation that is a message,
   a decision or feedback, written by its user and not by an agent: `Observation.agent_authored`)
   marks every candidate made from it, and `default_visibility(..., user_statement=True)` makes
   it `USER`: read in every conversation of that user and by every agent acting for them, never
   by another person, wherever it was said. `remember` by a user (no agent on the request) is
   the same. Agent-written memories, the assistant's replies, `EVENT`/`IMPORT`/`FILE`
   observations and `SHARED` types keep their defaults; explicit `THREAD` keeps a statement to
   its conversation and `WORKSPACE` shares it with a team, exactly as before. The conversation
   window, the thread summary, RUN and PRIVATE audiences are unchanged.
2. **The anchor stays where it was said.** Only the audience changes: the memory's scope is
   still the thread (`scope_for`), so consolidation, `/v1/memories?thread_id=` and the thread's
   revisions behave as before. Its keys are `user:<t>/<u>` plus the author's `principal:` key
   (`readable_by`), and the write moves the user's revision, so a bundle cached in another chat
   of theirs is rebuilt.
3. **The turn is kept beside any reading that is not lasting.** `domain.memory.lasting` (moved
   from the forgetting policy, which now reads it) names what is kept for good: a turn, a
   standing rule, a `LONG_TERM` `USER` or `PREFERENCE` memory. Only beside such a reading in the
   same words is the turn not stored again (it would outlive the reading's supersession as a
   stale copy). A reading and its turn never reinforce or merge into each other, and retrieval
   gives equal words from the same writer, instant and source turn one slot whatever subject
   each names (`engine._twin_context`: a topic subject counts as the writer, and an agent
   bound to a user, `agent:<u>/<h>`, as that user `user:<u>`, so a harness-relayed turn and
   its fact collapse; a different person's `user:`/`agent:` subject still keeps equal words
   apart). Graph and derived-source expansion add neither a collapsed twin (`engine.in_hand`)
   nor the other copy of a statement already in hand (`engine.not_in_hand`); exact, graph
   and derived candidates carry `owner_principal` so they collapse too.
   The twins are one statement to every write that removes or replaces it: `forget` (every
   copy not yet deleted, archived or superseded ones included, so `restore` cannot bring the
   words back), `update` (supersede), an automatic supersession in the pipeline, and a
   feedback retraction or correction (the CURRENT copies) act on both
   (`MemoryRepository.twins`: same words, owner and source, an indexed lookup).
4. **Long turns are kept in pieces, up to a bound.** A turn over `verbatim_max_chars` is stored
   as consecutive verbatim pieces cut at sentence ends (`verbatim_windows`), not truncated
   after the first; the restatement (ADR 0027) is carried by the first piece. At most
   `VERBATIM_MAX_PIECES` (6, about twelve thousand characters) are kept: a pasted corpus is
   not a statement, and a 2.1-million-character turn would otherwise have made a thousand
   lasting memories in one job. The rest stays in the message, the thread's archive and, for
   files, the document path. A constant, not a setting.
5. **A fact's subject may be a long noun phrase.** Up to twelve words; past five it may not
   contain a pronoun, a subordinating word, `and`/`but`, a possessive after its first word, or
   an event word (`_EVENT_HINT`: "The kids and I went...", "The guy we hired...", "Deployed
   the payment service to prod and the dashboard is green", "Yesterday John Smith sent the
   report and..."), and a sentence refused there falls through to the rules after it (the
   event rule, for those). The fact rule is English only, so there is no other language's
   counterpart to guard.

No new setting, no new memory type, no new table.

## Existing data

Rows keep the audience they were written with. A stored `THREAD` row does not record whether
`THREAD` was the default or the writer's explicit choice (only the observation's hints do), and
widening an explicit `THREAD` would undo the isolation integrators asked for, so no migration
rewrites keys and no read-side rule guesses. A deployment that wants an earlier conversation to
follow its user re-sends it (messages are idempotent on `source_system` + `source_message_id`)
or states the facts again.

## Consequences

* Single-fact recall works across a user's conversations, in any language: the verbatim turn
  carries the code where no rule reads the sentence (German, Hindi: integration tests).
* A participant of a shared thread no longer reads another user's default memories from it
  unless they were shared (`THREAD`, `WORKSPACE`); threads are single-owner through the API,
  and a workspace admin who reaches a member's thread reads only what was shared.
* A key restricted with `may_act_as=["user:alice", "agent:x"]` reads alice's statements from
  all of her threads when it acts for alice (or for `agent:x` on her behalf), not only from the
  thread named in the request: `USER` is alice's audience wherever she spoke.
* One more row per single-sentence user turn whose reading is not lasting (a fact, a task, a
  procedure, an event, a decision). Ranking sees one of the twins.
* Facts said in different threads are consolidated per thread as before: a value corrected in a
  later conversation does not supersede the earlier one; both are current and dated. Cross-thread
  supersession of a user's facts is a separate decision.
