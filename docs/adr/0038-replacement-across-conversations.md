# ADR 0038: A user's later statement replaces an earlier one, in any of their conversations

Date: 2026-10-11. Status: accepted. Follows up ADR 0036 (what a user says follows that user),
which left this open ("Cross-thread supersession of a user's facts is a separate decision").

## Context

ADR 0036 made a user's own words `USER`-visible: what they said in one conversation is read in
all of them. Consolidation still compared a new statement only with the memories of its own
conversation (`scope_key` is the thread), and a turn kept verbatim is never superseded by
consolidation at all - only its twin in the same words is. So a value the user replaced in a
later conversation stayed current beside its replacement, in every conversation:

* "Our primary supplier for heavy-duty pallets is Uline." in March, "Actually, we terminated
  our contract with Uline ... Our new exclusive pallet supplier is PackagingCorp." in May: both
  current, the old supplier still offered. In German, Spanish, Arabic and Hindi no rule reads
  either sentence, so not even the English slot rules could have closed it.
* Measured on the capability eval's core set (isolated store, main 420bbf3): same-session
  update/contradiction fell from 0.91 (before ADR 0036) to 0.04, temporal recency from 0.95
  to 0.33, every failure a `stale_leak` - the earlier conversation's statement (turn, fact or
  event; `USER`, `CURRENT`, never superseded). The values before ADR 0036 came from thread
  isolation (a new conversation saw nothing: new-session recall was 0.02), not from any
  replacement.

ADR 0036's claim that "a value replaced later in the conversation acts on both" held only
inside one conversation and only for a turn whose reading has the same words.

## Decision

1. **A user's new statement is compared with their current statements in all their
   conversations** (`modules/memory/replacement.py`, run by the observation pipeline in the
   observation's unit of work). Only a user's own words in a message (`USER`-visible, owned by
   them, not derived, not a standing rule or preference) on either side; an explicit `THREAD`
   or `WORKSPACE` statement keeps its audience and is never compared; another person's
   statement never. The 64 most recent of the user's statements are read
   (`MemoryRepository.statements_of`, a new partial index) and only what was said before the
   new statement.
2. **About the same thing, by its words first.** The two share a name ("Uline", "Maria Lopez";
   capitalised Latin runs, or any Latin run in another script; a single capitalised word is
   not a name in German) or an identifier ("MK-220", "Forklift #4", "Tor 14"), or the
   multilingual encoder puts them at a cosine of at least 0.5. Two statements that label the
   same kind of thing with different identifiers and share no other name are two things
   ("Forklift #3" / "Forklift #4", "SKU FT-1180" / "SKU PT-2207") and are never compared.
3. **Says the old one no longer holds: the NLI head.** The grounding cascade's head scores the
   contradiction both ways (the new turn and each of its sentences against the old statement,
   the old one against the new turn) in one batch. Thresholds, chosen on the capability set's
   dev split of 468 pairs of one user's statements (80 replacements), five languages and
   code-switched text:
   * >= 0.9: **superseded** - closed exactly as consolidation closes (`SUPERSEDED`,
     `superseded_by`, `valid_to`), kept for the temporal view. 100% precise on dev (22/22) and
     held-out (32/32).
   * >= 0.85, or >= 0.4 when the new statement says something changed (the labeller's
     `CORRECTION`, `LIFECYCLE` or `STATUS` - stored, or suspected by the lexicon - or a number
     that changed on the same named thing) and the two share a name or an identifier:
     **an earlier value** - still current and retrievable, ranked after every current item
     (`engine.current_first`), and labelled wherever it is served (`superseded: true` on a
     full context's memory and a search result, "(earlier value, since replaced)" in the
     rendered context).
   Held out: 32 of 35 replacements found; none of the 388 non-replacement pairs (dev and
   held-out) marked.
4. **The tenant's model, when its policy allows `conflict_adjudication`,** settles at most
   three related pairs per message that the head leaves in its unsure band (a contradiction of
   at least 0.5, below the marking thresholds), as consolidation's adjudicator does; an
   `update` or `contradict` verdict marks an earlier value, never a supersession. It is not
   asked about pairs the head reads as no contradiction, nor about pairs only the encoder
   relates below 0.5: with Granite 3.3 2B through the gateway, asking about those marked four
   statements on the llm_core set and every one was wrong (a multi-hop bridge, "Employees in
   the Berlin office follow the German calendar", marked by "Lena Vogel joined the Berlin
   office"). Without the model none of this runs, and nothing else depends on it.
5. **Every form of the statement goes with it:** its twins (`MemoryRepository.twins`) and, for
   a turn, what the rules read from it (a fact or an event of the same message inside the
   turn's words). A reading of one sentence never takes its whole turn with it.
6. **Reversible.** The replaced memory carries `system_metadata["replaced_by"]` (the newer
   memory, the grade, the score, the reason). When the newer statement is forgotten or
   retracted (an owner's `REJECT`), what it replaced is reinstated (`replacement.reinstate`,
   `MemoryRepository.replaced_by_any`, a second partial index).
7. **A failing model never costs the message.** A head that refuses (its queue full) or fails,
   or a model call that fails, leaves the statement stored and nothing replaced
   (`memory_replacement_skipped_total`); what was decided is counted by grade
   (`memory_replacements_total`).

No new setting, no new memory type, no new table. Migration 0027 adds the two indexes online.

## Consequences

* Measured on the core set (isolated store; build runs against the recorded main runs; the
  numbers and their files are in the commit message and the eval results).
* Not replaced, with or without the model: "I live in Seattle" / "I moved to Austin" share no
  name and sit at a cosine (0.32-0.44) where the head reads unrelated sentences as
  contradictions too ("the BBQ promo has ended" / "the Back to School promo is live"); and the
  state changes the head reads as no contradiction in Arabic and Hindi ("is back from leave",
  "finished the book"). The small model could not be trusted with either (decision 4).
* A standing rule is never replaced here, including a statement the labeller read as one
  ("Service van #7 is assigned to Ana Silva" in English, Spanish and Arabic).
* An earlier value is still served, last and labelled: a wrong mark costs a rank, not the
  statement. Only the stricter grade closes a statement, and that is undone with the newer
  one.
* One NLI batch per user message that has related earlier statements (none for a message
  that relates to nothing), and one encoder batch for the unanchored ones.
