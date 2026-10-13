# ADR 0038: A user's later statement replaces an earlier one, in any of their conversations

Date: 2026-10-11. Status: accepted. Follows up ADR 0036 (what a user says follows that user),
which left this open ("Cross-thread supersession of a user's facts is a separate decision").

## Context

ADR 0036 made a user's own words `USER`-visible: what they said in one conversation is read in
all of them. Consolidation still compared a new statement only with the memories of its own
conversation (`scope_key` is the thread), and a turn kept verbatim is never superseded by
consolidation at all - only its twin in the same words is. So a value the user replaced in a
later conversation stayed current beside its replacement, in every conversation:

* "Our main supplier of shipping crates is Brightway." in March, "We ended the Brightway
  agreement last week ... Shipping crates now come from Corrimax only." in May: both current,
  the old supplier still offered. In German, Spanish, Arabic and Hindi no rule reads either
  sentence, so not even the English slot rules could have closed it.
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
2. **About the same thing, by its words first.** The two share a name ("Brightway", "Ana
   Petrova"; capitalised Latin runs, or any Latin run in another script; a single capitalised
   word is not a name in German) or an identifier ("QX-410", "bike #4", "Halle 6"), or the
   rules (or, in llm mode, the tenant's extraction) read one single-valued slot of one subject
   from both with two values (`user lives_in Leeds` / `user lives_in Bristol`, including a
   reading consolidation has just closed), or the multilingual encoder puts them at a cosine of
   at least 0.5. Two statements that label the same kind of thing with different identifiers and
   share no other name are two things ("bike #3" / "bike #4", "SKU KV-3301" / "SKU LM-8812")
   and are never compared.
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
     that changed on the same named thing, or another value of a shared slot) and the two share
     a name, an identifier or a slot: **an earlier value** - still current and retrievable, but
     never among the current evidence (decision 8).
   The change cues are the lexicon's weak `LIFECYCLE`/`STATUS` terms; this ADR adds a handover
   ("handed ... over", "took over", "übergeben", "traspasó", "تولى", "सौंप") and a return
   ("back from", "zurück", "ha vuelto", "عادت", "लौट") in the five languages, as weak cues:
   the lexicon suspects, the head decides. A change cue never relates two statements by
   itself: a first version compared a stated change with nothing in common from cosine 0.3 at
   a contradiction of 0.95 ("I moved to Bristol" / "I live in Leeds"), was precise on the dev
   split, and in the first paired gate run (held-out split) cost a whole scenario of current
   statements, so it was removed.
   Held out: 32 of 35 replacements found; none of the 388 non-replacement pairs (dev and
   held-out) marked.
4. **The tenant's model, when its policy allows `conflict_adjudication`,** settles at most
   three related pairs per message that the head leaves in its unsure band (a contradiction of
   at least 0.5, below the marking thresholds), as consolidation's adjudicator does; an
   `update` or `contradict` verdict marks an earlier value, never a supersession. It is not
   asked about pairs the head reads as no contradiction, nor about pairs only the encoder
   relates below 0.5: with Granite 3.3 2B through the gateway, asking about those marked four
   statements on the llm_core set and every one was wrong (a bridge between two facts, "staff
   of the Lyon office follow the French calendar", marked by "Ines Moreau joined the Lyon
   office"). Without the model none of this runs, and nothing else depends on it.
5. **Every form of the statement goes with it:** its twins (`MemoryRepository.twins`) and, for
   a turn, what the rules read from it (a fact or an event of the same message inside the
   turn's words). A reading of one sentence never takes its whole turn with it.
6. **Reversible.** The replaced memory carries `system_metadata["replaced_by"]` (the newer
   memory, the grade, the score, the reason). When the newer statement is forgotten or
   retracted (an owner's `REJECT`), what it replaced is reinstated (`replacement.reinstate`,
   `MemoryRepository.replaced_by_any`, a second partial index).
7. **A fact misread as a standing rule is a fact here.** A statement stored as a rule
   (`category: rule`) that no sentence of which reads as a rule by the lexicon alone - a model
   confirmed it ("Our main supplier of shipping crates is Brightway") - is compared and replaced
   like a fact, but only about a shared name, identifier or slot, never on the encoder alone.
   A rule the lexicon reads as one ("Always order crates from Brightway") is still left to the
   rule that revises it.
8. **Current and history apart on the read path.** An earlier value is never among the current
   evidence: the context builder puts it in its own `history` section (at most four; handles
   `h1`..; a full context's `history`, each `superseded: true`; "## Earlier values (since
   replaced)" in the rendered context, after the current memories), and it is no evidence for
   `/v1/verify`. A search that found an earlier value brings what replaced it
   (`engine._follow_replacements`: the indexed `replaced_by`, a chain followed to its current
   end, at most two batched reads of four, the reader's visibility checked), so the old value
   never comes alone.
9. **A failing model never costs the message.** A head that refuses (its queue full) or fails,
   or a model call that fails, leaves the statement stored and nothing replaced
   (`memory_replacement_skipped_total`); what was decided is counted by grade
   (`memory_replacements_total`).

No new setting, no new memory type, no new table. Migration 0028 adds the two indexes online.

## Consequences

* Measured on the capability eval's core set, isolated store, real models, each build run
  paired with a recorded main 420bbf3 run by pinning that run's id and date (the eval names its
  users after the run id, and the id moves cosines near the relevance floor): the cells, their
  noise and the held-out split are in the follow-up "measured" commit and the eval results.
  The thresholds and cue classes were chosen on the eval's dev split only.
* Still not replaced, by design: a state change the head reads as no contradiction ("is
  back from sabbatical", "finished the book", a handover) in a language where it scores below
  0.4 - Arabic, Hindi and German cases on the dev split - and anything only the encoder
  relates below 0.5 with no slot read from both: "I live in Leeds" / "I moved to Bristol" is
  replaced in English (the rules read `lives_in`) and with the tenant's extraction, not in the
  other four languages without a model. The tenant's model (Granite 3.3 2B) was measured on the dev split's 188
  cross-conversation pairs: it called 97 of 143 unrelated pairs an update or contradiction,
  so it is never asked about pairs the head has not already found related and contradictory.
* A record carried by a request (`EVENT`) keeps ADR 0036's audience (the conversation): a new
  conversation does not see it, so its replacement is not seen there either. Changing that is
  an audience decision, not a replacement one.
* An earlier value is still served, apart and labelled: a wrong mark costs its place among
  the current evidence, not the statement. Only the stricter grade closes a statement, and
  that is undone with the newer one.
* One NLI batch per user message that has related earlier statements (none for a message
  that relates to nothing), one encoder batch for the unanchored ones, and on the read path at
  most two batched reads when an earlier value is served without its replacement.
