# ADR 0041: The read path under memory that is never cleared

Date: 2026-10-19. Status: accepted. Follows up [ADR 0036](0036-a-users-words-follow-the-user.md),
[ADR 0038](0038-replacement-across-conversations.md) and
[ADR 0040](0040-answer-coverage-and-finished-states.md).

## Context

Production memory is never cleared, and since ADR 0036 a user's words reach every one of
their conversations. Testers of the deployed service found five failures, all on the read path:

* a value restated over many chats and days (a store's buyer, a reorder point, "my regional
  merchandise manager") was answered with a stale one: every statement stayed current unless
  a later one was tied to it at write (ADR 0038);
* a two-hop question went unanswered: "Who is the buyer for store 4779?" found "Store 4779
  belongs to Region X29" and never asked about Region X29;
* standing rules ("never create when ATA is 0", "always answer stockout audits as a markdown
  table with columns ..., no prose") were not applied in a new chat: they reached it only
  through the `user` profile block, one bounded block of everything, newest first, so a rule
  said weeks ago fell off its end;
* a stated preference ("I care about high-ROS styles with short WOS") was
  not applied - the extraction rules did not read it as one;
* answers came from loosely related memory, and the no-LLM retrieval benchmarks sat at
  LoCoMo source recall@10 0.807 (main 70198fa) with a turn and the facts read from it ranked as
  separate, competing candidates.

The research report on updates, recency and abstention ranked the remedies: one document per
statement (LongMemEval's authors measured fact-augmented keys at Recall@5 0.644 against 0.478
for the same facts ranked separately), intent-gated latest-wins per slot, rule-based time
expressions, and an abstention cascade that checks coverage before relevance.

## Decision

All of it is read-time and model-free; nothing stored changes except what two extraction rules
now read.

1. **One statement, one document** (`memory_ranking.statements`, `GROUP_STATEMENTS`). Every
   form of one statement - the turn kept verbatim and what the rules read from it, by the same
   speaker, at the same instant, from the same source turns, current or earlier value alike -
   is one document in every arm: its rank in an arm is its best form's, the rules (speaker,
   time, period) read all of its forms, and the form shown is the one that ranked best on its
   own; the others ride along as its `duplicates`.
2. **Time-aware questions** (`periods`). A relative period in a question that names a day is
   read from that day ("last weekend before April 10, 2023"); a season of a named year is its
   three months; month names, dates written the language's way, years and the commonest
   relative periods in German, Spanish, Arabic and Hindi (an Arabic month name only beside a
   number or after في/شهر, since it doubles as a verb). The time rule reads wann/cuándo/متى/कब
   questions and those languages' time words.
3. **Intent.** A question about what was (`router._HISTORY`, every pack language: "used to
   be", "previously", "früher", "antes", "سابقا", "पुराना") keeps earlier values where they
   rank and the bundle carries them among the evidence. Any other question ranks earlier
   values after current ones, as before, and first applies the latest-value rules below. No
   recency decay anywhere.
4. **The current value of a slot** (`slots.latest_values`). The question says what the slot
   is: the things it names (identifiers, names) and the words it asks with. Of the current
   statements that name every one of those things, share an asked word (one beside what the
   question names; most of them for a slot that names nothing, "Who is my regional
   merchandise manager?") and hold a value of their own, the newest is current and every older
   one whose values differ is an earlier value of it: history section, never the answer. Not
   for a past-tense question unless it asks for the latest ("What was the last PO I
   raised?"), and, for a slot naming nothing, only for a question about one thing. Beside it
   the exact-slot rule (`engine.latest_per_slot`: same speaker, subject and predicate, a
   different object).
5. **A second hop** (`engine._second_hop`). When one of the three best memories names what the
   question names but not what it asks, and names another thing ("Store 4779 belongs to
   Region X29"), one keyword search (BM25, no encoder, k=8) for that thing and the asked words
   adds up to two memories that name it, after the memory that bridged to them. One hop.
6. **Standing rules and preferences per request** (`context.standing`). Each request reads
   the user's rules and instructions, and their preferences, that apply to it: a dense search
   in the primary space and a keyword search over those categories only, with the query's own
   vectors, beside the similarity scoring. Rules are the rule and instruction categories, any
   memory the labeller read as a rule or correction, and any statement giving an instruction
   it says is for every time that a no-model labeller left undecided (`standing_instruction`).
   A statement applies at cosine 0.4 or one shared content word or code; a later rule on the same subject (half of the shorter one's words)
   supersedes an earlier one; what was said in the same turn as an applying preference comes
   with it; an earlier value never applies. They come back as two profile sections,
   `standing_rules` and `preferences` (at most 6 and 4 lines), in any conversation, and are
   kept when the whole profile does not fit the pinned share.
7. **What the extraction reads.** "I care about / focus on ...", "permanently prefer ..." and
   their German, Spanish, Arabic and Hindi forms are lasting preferences. What a user manages
   or simply prefers ("I manage store X", "Prefiero las alertas por X") is not: it is one value
   of a slot that changes, a lasting preference is never replaced (ADR 0038 compares
   statements), and read as one it kept every earlier store and channel current
   (stress_accumulated `preference` fell 8 points below the candidate); the turn keeps it; "From now on, always answer ..." (a standing word opening the clause after
   another one) is a rule when what follows is an instruction.
8. **Upgrading a store an earlier build wrote** (`tools.backfill`, `make backfill`, documented
   in `docs/deploy/database.md`). For every user, their statements message by message in event
   time through the write path's replacement, then their memories indexed again; live-safe,
   idempotent, resumable (`--state`), nothing deleted. The read-path rules above need no
   backfill to apply to old memories; the backfill adds the write-time marks and the payload
   the rules section filters on.
9. **Abstention.** The overlap rule compares Snowball stems in the question's language (and
   English for Latin words) besides surface words, function words excluded; coverage reads
   disarming an alarm, logging in and the German verb to mail as asking for a code, a password
   or an address.

## What was measured and left out

All on the capability eval's dev split, replaying the bundles a no-LLM run recorded (b7 =
96420ef, the candidate's point 7), never the held-out half:

* **An NLI entailment check** (mDeBERTa, premise = the best memories, hypothesis = "The answer to
  the question "…" is stated here."): AUC 0.715 between unanswerable and answerable questions
  the gate did not refuse (0.60 Arabic to 0.79 Spanish); at its loosest useful threshold it
  refuses 10.7% of answerable questions to catch 22% of unanswerable ones, and costs 65 ms per
  pair on this host. Not adopted: it would raise false refusals above main's and the context
  p95 by more than the noise.
* **A slot-coverage gate** (abstain when no item names what the question names and shares an
  asked word): 30 more unanswerable questions refused, 616 more answerable ones. Not adopted.
* **An identifier / "who is my <role>" gate**: 2 more unanswerable, 281 more answerable. Not
  adopted.
* **Judging relevance on extracted facts only**: not built. Without a model the fact rules read
  English only, so in German, Spanish, Arabic and Hindi a statement has no fact reading and
  every one of those questions would be refused.
* **The subject gate** (an INCOMPLETE "no memory is about <Name>" made INSUFFICIENT): 14 more
  unanswerable questions, 274 answerable (new conversation). Not adopted.
* **Ask words "call", "anmelden", "entrar" and the English "mail"**: 3 unanswerable questions
  for 13 answerable ones, or the word also means signing up, going in or the post. Left out.

## Constants (none is a score weight; all model-free)

| constant | value | where it came from |
|---|---|---|
| `standing.RULE_SIMILARITY`, `PREFERENCE_SIMILARITY` | 0.4 | capability dev split: primary-space cosines of rules/preferences to their own vs other scenarios' dev questions (own p50 0.29, others' p99 0.35) |
| `standing.SHARED_STEMS` | 1 | same dev data: one shared word applies a statement to 69.5% of its own questions, 14.9% of others'; two: 49% / 3.9% |
| `standing.SAME_SUBJECT` | 0.5 | half of the shorter rule's words: a structural choice, not fitted |
| `standing.RULES_MAX` / `PREFERENCES_MAX` / `SEARCH_K` | 6 / 4 / 24 | bounds on the section's size and the round trip, not fitted |
| `slots` anchorless share | 0.6 of the asked words (at least 2) | structural ("most of them"), not fitted |
| `slots._shares` affix length | 4 characters | structural (a stem inside a longer word), not fitted |
| `engine.SECOND_HOP_BRIDGES` / `_K` / `_ADDED` | 5 / 8 / 2 | bounds (one hop, a small k), not fitted |

## Consequences

MEASURED_SECTION
