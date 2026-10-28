# Changelog

What changed in the service (`trellis-memory-service`) and its SDK (`trellis-memory`), newest
first. The service and the SDK carry separate versions ([versioning.md](docs/versioning.md));
each entry says which one moved. Decisions behind each change are in the
[ADRs](docs/adr/README.md).

## Unreleased

### Added
- The prompt form of the context names the memories it shows: `POST /v1/context`
  (`format="prompt"`) returns `memory_ids`, the ids of the memories `rendered` shows, in bundle
  order (absent when none). A person's verdict on the answer cites them (`evidence_refs`), so it
  moves their confidence when no grounding verdict has cited the answer's memories; the harness
  keeps them with the run. SDK `trellis-memory` 0.5.0: `PromptContext.memory_ids`. Additive:
  an SDK 0.4.0 client reads it as an extra field.
- The SDK's public API is held by CI: `sdk/python/public_api.json` is a snapshot of every name
  `trellis.memory` exports (signatures, model fields, enum members), and
  `sdk/python/tests/test_public_api.py` fails on any change to it until the snapshot is
  regenerated (`python scripts/sdk_api.py --update`), which it is only under a new SDK version
  that this CHANGELOG names. On a pull request, `scripts/sdk_api.py --check-base` holds the
  snapshot to the base branch's the same way. SDK and service unchanged.
- The conversation window as the conversation itself: `POST /v1/context` with
  `window_format="messages"` (SDK `ctx.context(..., window_format="messages")`) returns the
  thread's recent messages verbatim in `conversation.messages` — oldest first, tool calls with
  their results, never parted — and leaves them out of `rendered`, for a framework that replays
  them as its own messages, as its checkpoint would. Read from the store, so a turn written
  before is in the next window with the worker down. Messages carry their tool calls:
  `tool_calls` (`{id, name, arguments}`) on an `ASSISTANT` message and `tool_call_id` on an
  `INTERNAL` `TOOL` message, stored as written (migration `0029_message_tool_calls`, two
  nullable columns) and returned by the message reads. Service and SDK; the default
  (`rendered`) is unchanged.

### Fixed
- A learned skill says which standing rule forbids its step with the tool hints off too. With
  `tools.hints=false` (no `ToolHints`), "Never create a stock rule at store S40" in the
  context's `standing_rules` and a task at S40, the skill line read only
  "create-a-stock-rule...: create_stock_rule (worked 100% of 3 runs)", and the model called it:
  the note came only from the hints (`ToolHints.forbidden`). Each skill's steps are now checked
  against the section's rules with or without hints (`hints.ruled`, the same `standing.forbids`
  test the hints apply, on the task's values; with hints, their verdict for a step they hinted
  stands), carried as `SkillView.forbidden` and rendered under the skill. A rule about another
  store or an "always" rule adds no note. Service only; the wire form of a skill is unchanged.
- A tool that changes something gets no argument values and no "missing" question when the task
  is a question. Asked "How many units of style ZZ999999 do we hold?", the Tools section listed
  `create_stock_rule` with `store='S12', style='ZZ999999'`, in another run `min_units = 1,
  max_units = 3`, and the model created a rule nobody asked for. Such a tool may still be listed
  as a candidate (it was already never the next step, `hints._may_lead`); a read tool is
  prefilled as before, and a write tool on a request to act is too. The question marks are the
  ones the hints already read (`?`, `？`, `؟`). Service only.
- An earlier request's values no longer fill another call's arguments. The `min_units = 1,
  max_units = 3` above came from the memory of the earlier request "Create a stock rule for
  ST-101 at S40, min 1 max 3.", whose words "min 1 max 3" the memory-text source
  (`hints._from_memory_text`) read as labelled values for any task. A memory's labelled value
  now fills an argument only when the memory is about what the task is about: it names an
  identifier or name the task names ("Acme Steel"), or nothing but the value itself ("their
  supplier id is SUP-40"). Service only.
- Of two values of a one-valued slot, the one said later stays current, whatever order they are
  read in. "I work at Globex now." (`occurred_at` 5 days ago) sent first, then "I work at ACME
  Corp." (45 days ago) in a separate call: consolidation closed Globex and kept ACME current, so
  "where do I work?" answered ACME. When each was said decides (`TemporalState.since`: the date
  the statement names, else its `occurred_at` - when it was stored if none was given), the order
  of reading only on a tie; a value said before the current one is stored as its history
  (`SUPERSEDED`, `superseded_by` the current one, `valid_to` where that one begins), the event-time
  rule ADR 0040 set for replacement across conversations, now on the consolidation path too. A
  replacement holds from when it was said, not from when it was read, on both paths, so a search
  as of six weeks ago (`as_of`) reads ACME and one as of yesterday Globex. And two conversations of
  one user read at the same time each compared themselves with a store the other had not
  committed to, and both values stayed current (2 of 6 users' batches under load): the queue runs
  one conversation's messages one at a time (#57), and now one user's statements are compared and
  written one message at a time too (a transaction-scoped advisory lock on the user, taken by the
  write of an observation only: reading the message - the model's work - still runs in parallel,
  and other users never wait). No new setting; service only.
- An agent run can no longer rewrite or forget what the user said. Asked "What is 12 times 7?",
  a run called `memory_update` on the user's "I raised PO-77812 for 2,000 corrugated boxes; it's
  late." and replaced it with "12 times 7 = 84"; the new version kept the user as its owner and
  `user:<u> said` as its subject and predicate, so every later context presented the agent's
  words as the user's. A memory a person said (a `user:` owner, or a `said` turn of a `user:`
  subject, which a harness may relay; `pipeline.said_by_a_person`) is refused to an agent by
  `memory_update`, `memory_forget`, `POST /v1/memories/{id}/supersede`, `DELETE
  /v1/memories/{id}` and restore, whatever its rights (403: "... is the user's own statement:
  an agent cannot supersede it; use memory_remember to add your note"). An agent still updates
  and forgets its own memories, and a version it writes is its own: its owner, a kind an agent
  states (`authored_type`), no person as subject. The user's and a tenant admin's updates are
  unchanged. Service only.
- Two values of a one-valued slot sent in one `history.add` batch ("I work at ACME Corp.", "I
  work at Globex now.") no longer both stay current. Each message's memories are formed by its
  own job, and the jobs of one batch ran side by side: each compared its statement with the
  store before the other had committed, so neither replaced the other, where two calls did. A
  conversation's messages are now read one at a time, in the order they were stored (the
  queue's `lock` on `memory.process_observation`, per thread; the post-commit dispatch defers
  them in outbox order); other conversations run beside them. Service only.
- A learned skill no longer hands over a call the user's standing rule forbids. With "Never
  create a stock rule at store S40" in the context's `standing_rules` section, a skill learned
  from three runs at S12 made `create_stock_rule` the hints' next step, prefilled at S40, and
  the model called it twice. A candidate a current "never" rule of the section forbids
  (`standing.forbids`: the rule says every word of the tool's name, as stems, and names no code
  or number the call does not have: the task's values and those found for its arguments) is
  no next step and gets no next-step bonus, nothing is prefilled or asked for it, and the
  Tools section and the learned skill's line say which rule forbids it
  (`ToolHints.forbidden`). A rule about another store, an "always" rule, and one replaced,
  superseded or forgotten (not in the section) forbid nothing. The "never" words are the
  standing words the section already reads (`standing.NEVER_WORDS`: English, German, Spanish,
  Arabic); a rule in another language matches a tool it names (`create_stock_rule`). The
  `/v1/tools/hints` endpoint and the `tool_search` memory tool read the stored profile, which
  has no such section, and are unchanged. Service only.
- A value of the wrong type is never prefilled. Asked "How many units of style ZZ999999 do
  we hold?", the hints prefilled `min_units = 'ST-101'` and `max_units = 'ST-101'` from the
  memory "S12 should never hold fewer than 2 or more than 6 units of ST-101": only the
  task's own values were held to the argument's declared type. Every source (a learned
  binding, the graph, the profile, a memory, the task) now goes through one check
  (`hints._as_declared`): a value that is no number for a `number` or `integer` argument, or
  no whole one for an `integer`, is no value of it, and the next source is read. Service
  only.
- An agent no longer writes the user's profile. A memory an agent states (`memory_remember`,
  `POST /v1/memories` with an agent on the request) as `USER` or `PREFERENCE` is kept as the
  agent's own `AGENT` memory, by the rule that already re-typed an agent-authored observation's
  readings (ADR 0013, now one function, `pipeline.authored_type`), so it is never a line of the
  pinned `user` block: a help-desk run that remembered an asset's owner ("omar (u-77)",
  `kind="USER"`) had every later context say the user is Omar. It keeps the audience it was
  given and is still retrieved. The user's own statements (no agent on the request, or their
  turn relayed by a harness) state their profile as before. Service only.
- A user's question no longer becomes a memory, and a statement said again no longer replaces
  itself. A question behind a heading with no question mark ("For employee e-401: how many
  trips this year ...") was read as a statement and kept as a lasting turn; the question word
  after a heading now opens a question (`statements._HEADING`). A turn whose every sentence
  asks still forms nothing, in any language and with `?`, `？` or `؟`, and a turn that also
  states something ("I moved to Pune. Which store is nearest?") keeps it. A user's turn in the
  same words in a later conversation was compared with the earlier copy as a replacement, and a
  request repeated in four runs had three copies served under "Earlier values (since
  replaced)": the same words replace nothing (ADR 0038). The copies stay one per conversation;
  said twice in one conversation it is one memory, as before. Service only.
- A learned skill reaches a task worded otherwise: a procedure was offered (the context's
  skills, the hints' plan) only when the task's pattern shared 60% of its words with the
  procedure's, so "MFA lockout again" or "Re-enrol two-factor for a user who is locked out"
  got nothing from a skill learned on "reset multi-factor authentication for a locked-out
  user" (the context abstained). A procedure the words miss now matches by meaning: the
  cosine of the task's vector to its pattern's, in the space every query is searched in (so
  every script), at the encoder's own `DenseModel.procedure_cosine` (0.45 for the
  multilingual encoder, 0.86 for the English one where it is the only space; 0, words alone,
  for the hash stand-in). Calibrated on the capability eval's MFA skill: 32 of 50 phrasings
  in five languages match, none of the other 450 procedural queries, nor "reset a store's
  POS terminal" (0.36). The word match stays the fast path and ranks first. The task is
  encoded once per request - the context hands the procedures the retrieval's own vectors,
  the tool hints share theirs with the tool search - and each pattern once per process (an
  in-process cache behind the embedding cache): about 1 ms per request for 100 visible
  procedures. One matcher (`modules/tools/matching.py`) serves the context and the hints.
  Service only; the SDK is unchanged.
- The grounding judge (`/v1/verify`) no longer rejects a run that called tools only because
  its claims are not in the context: the judge never sees the tool results, so every
  tool-built answer read as unsupported and a live test put a false reject in the review
  queue for 40 of 40 correct answers. Such a run gets no verdict unless a claim is
  contradicted; a fully supported answer still confirms, and a run without tools is judged
  as before.
- Concurrent forgets of related memories (twins, and what one replaced) no longer deadlock,
  refuse a memory a twin's forget had just removed as "not visible in this scope", or reopen
  a forgotten old value as CURRENT: a forget locks the memory, its twins and what they
  replaced in one statement, in id order, and decides under that lock; a memory forgotten
  meanwhile is a success (204), a memory outside the caller's scope is still refused.
- The capability eval's harness mode (`--mode harness`) wraps its agent with
  `h.wrap(..., run_timeout=)`: the harness had removed `timeout=`, so the mode failed with a
  `TypeError` before asking anything. Its grounding sampling is turned off with
  `without={"grounding"}` on that wrap: the `grounding_sample` it set on the harness's
  `Settings` is no field there and was silently ignored. Its runs no longer pass
  `without={"records"}` (the harness has no such name now: it refused the run); instead each
  question is asked as a user of its own, the scenario written for it, so what a run records
  stays in that question's scope. A scenario is written once per question. Service and SDK
  unchanged.
- What the service learns from, after a live learning test (memory, procedures, hints,
  approvals):
  - the grounding judge's verdict waits for a tenant admin's review and is never a person's
    vote; a reject cites only the memories behind contradicted claims, a confirm none, and a
    run that called tools moves no memory; a passage about something else no longer
    contradicts a claim (`CONTRADICTION_MIN_COVERAGE`, same script only);
  - an agent run's own words, its tools' errors and raw tool output form no memory;
  - an unlabelled run whose calls all worked is no evidence for a procedure, however old (the
    24-hour weak positive is gone), and neither is a `system` failure of a run whose calls all
    worked (a time limit, a model error);
  - a failed call and its retry are one procedure step; a value is a learned literal only when
    it is the same across runs of different tasks, and is never prefilled for a task naming
    another; Code Mode's meta-tools are no steps;
  - tool hints fill an argument only from the task's words that match it (its label, its
    spelling, its documented example's shape), read no number out of an identifier ("ST-101"),
    rank a general view (overview, summary, dashboard) below a specific tool, never make a tool
    that changes something the next step of a question, and offer no candidates when nothing is
    relevant and no plan is learned; a learned plan is spelled out once in the context;
  - a tool-call decision counts only on an interrupt of the calling run, naming its reviewer,
    once per interrupt, and an approval only once the run recorded that call; a learned rule
    never relaxes an administrator's `approve_when`;
  - `GET /v1/tools` says whether each entry is indexed (`indexed`), so a client can wait for
    tool search before its first hints; and the context leaves out its "Evidence status"
    section when a hinted tool answers the question.
- A learned "approve automatically" rule needs at least 2 different reviewers: one reviewer's
  approvals of a tool's calls, however many, no longer make it a suggestion (the reviewers are
  read from the counted decisions; no new table). And each call a run recorded backs one counted
  approval: the interrupt id is the client's, so naming a new one made one call back any number.
  `GET /v1/tools/approval-suggestions` may list fewer suggestions; accepting one that fewer
  than 2 reviewers support is a 409.
- The capability eval's learned-procedure scenarios learn again: the runner labels its recorded
  tool runs as a person (`source="human"`) through the tenant's administrator credential, so the
  label is applied as it arrives and taught from; it labelled them `system`, a run's own report of
  success, which teaches no procedure. Eval only; service and SDK unchanged.
- Tool search reads the arguments' descriptions: a catalog entry is embedded with what its input
  schema says each argument is (`properties.<arg>.description`, nested objects and array items
  included, three levels deep), each distinct text once and at most 4,000 characters, so a tool
  whose purpose only a parameter's description names is found. Each tool record carries the
  hash of the text it was embedded from, and `periodic.tools_refresh` re-indexes the records an
  earlier build wrote, so an existing deployment picks the new text up by itself. No new
  setting.
- `memory-reindex --prune` dropped the live tool-search collection: it counted only the
  knowledge and memories collections as live. The tools collection is live now.
- After a model change tool search stayed empty until each catalog entry changed: the reindex
  tool never rebuilt the tools collection. It now re-embeds every catalog entry (of the
  `--tenant` only, when given; `tools` in its report), and `periodic.tools_refresh` also
  indexes any entry that has no record.
- A tool first seen in a recorded call (a bare catalog entry `record` creates) was never
  indexed; `periodic.tools_refresh` indexes it, so tool search can offer it by name.
- Accumulated memory (ADR 0041, decision 10): statements that are links of a chain about one
  code ("Store 6474 belongs to Region G10" / "Region G10 is replenished from ...") no longer
  replace one another; a rule restated with a new code is stored, listed once and ranked
  newest; "Only list ..." and its translations are standing rules; a question about a code no
  evidence names, of a form the store holds others of, is INSUFFICIENT (a code the user
  supplies to a request is not); the backfill retires the graph facts of what it replaced.
- The backfill brings a store an earlier build wrote to what a fresh store holds: it marks
  the standing instructions an earlier build never marked (a rule restated with a new value
  was two current rules), keeps the rules the NLI head confirmed (it made them facts),
  turns a rule reading this build reads no rule in, left alone by its message, into the turn
  this build keeps, and composes each user's profile block again.

### Added
- Memory that is never cleared reads right (ADR 0041). Of a user's statements of the slot a
  question about now asks for, the newest is current and the older ones are history, in five
  languages ("Who is my regional manager?", "What's the reorder point for PX-90?", "What was
  the last order I raised?"); a question about what was keeps them. A bounded second hop
  follows the evidence to the thing it names ("Branch 512 is in District D7" -> District D7's
  dispatcher), keeping one name for two things apart by its qualifier ("this week" / "last
  week"). The user's standing rules, corrections and preferences that apply to a request come
  back as profile sections `standing_rules` and `preferences` in any conversation, a later rule
  superseding an earlier one, a format rule or a list whole. A turn and what the rules read
  from it rank as one document. Time words are read anchored to a named day, by season, and in
  German, Spanish, Arabic and Hindi. The abstention overlap reads stems; disarming an alarm,
  logging in and the German verb to mail ask for a value. "I care about ..." is a
  lasting preference ("I manage ..." stays a statement a later one replaces); "From now on, always ..." is a rule. No model anywhere.
- `make backfill` (`memory_service.tools.backfill`): brings a store an earlier build wrote up
  to this one in place while serving - replacement across each user's conversations in event
  time, then the index again; idempotent, resumable (`--state`), nothing deleted
  (docs/deploy/database.md). No new setting. Compose runs it on every `up` (the one-shot
  `memory-backfill` service, beside the API, its checkpoint on the `backfill` volume), so an
  upgrade is pull, then `docker compose up -d --build`; the checkpoint records
  `BACKFILL_REVISION`, and one from an earlier revision starts over. With no user left to
  bring up it exits 0 before loading a model.
- A question asking for a phone number, an e-mail address or a code or password that no
  evidence about what it names states - "the alarm code for Store 33" when only Store 31's is
  known - or for a month the evidence never names (an item naming no month is about the month
  it was said in, or its relative dates name), is INSUFFICIENT; the evidence stays in the
  bundle, a note says what is missing (`RetrievalSettings.coverage_check`). A memory-only
  bundle that shares no word with the question but carries a memory at a relevance of
  0.4 or more is no longer INSUFFICIENT (`related_evidence_relevance`). A user's "I
  finished Dune" replaces their "I'm reading Dune at the moment" as an earlier value in all
  five languages, whatever the NLI head reads. A message stored out of order (a model reads
  messages in parallel) is compared with what the user said after it too: event time
  decides which statement is newer (ADR 0040).
- A value a user replaces in a later conversation is no longer served as current. Since a
  user's words follow the user (ADR 0036) the old and the new statement were both current in
  every conversation, and the same-session update and recency cells fell to 0.04 and 0.33. A
  new statement is now compared with the user's own statements in all their conversations:
  same thing by a shared name or identifier (two things of one kind, "bike #3" / "bike #4",
  never), by one subject's single-valued slot read from both with two values (`lives_in`),
  or by the multilingual encoder; the NLI head decides. At 0.9 the old statement and every
  form of it (turn, fact, event, twins) are superseded; at 0.85, or 0.4 for a stated change
  (a correction, a lifecycle or status cue - now also a handover or a return, in five
  languages) about the same named thing or slot, it stays current as an **earlier value**. A
  fact a model misread as a standing rule is compared like a fact. With the tenant's
  `conflict_adjudication`, the model settles at most three related pairs per message that the
  head leaves in its unsure band. Forgetting or retracting the newer statement reinstates what
  it replaced. A failing head or model never costs the message
  (`memory_replacement_skipped_total`; decisions in `memory_replacements_total`). Migration
  0028 adds two partial indexes, online (ADR 0038).
- Current and history apart in the context: an earlier value is never among the current
  memories. It is served in a `history` section of its own (full context `history[]`, each
  `superseded: true`; "## Earlier values (since replaced)" in the rendered context; handles
  `h1`..), is no evidence for `/v1/verify`, and never comes without what replaced it: a search
  that finds an earlier value brings its replacement, before it, past the relevance floor. A
  sentence shown whole in its turn is not shown again beside it. New in the API and the SDK:
  `ContextBundle.history`, `ContextMemory.superseded` (ADR 0038).
- Replacement in the languages the NLI head reads weakly (ADR 0038): where the speaker lives,
  or that they moved, is read from the words in German, Spanish, Arabic and Hindi as English's
  rules read it, so "Ich wohne in Leeds" and "Ich bin nach Bristol gezogen" relate without a
  model; with the tenant's `conflict_adjudication`, a stated change whose first named thing the
  older statement names is put to the model whatever the head read ("Maria Lopez is back from
  leave" scores 0.02 against "is on leave" in Arabic). The lexicon reads first-person "finished"
  in Spanish ("terminé") and Arabic ("أنهيت") and German "zu Ende", and a Hindi cue matches with
  or without the nukta ("ख़त्म" / "खत्म").
  Without a model, a stated change that leads with a name or identifier the older statement
  carries, and that the encoder puts at 0.5 or closer to it, marks it as an earlier value
  whatever the head read (never a supersession).
  In German, a single capitalised word both statements use bare (no determiner, no number
  beside it) is a shared name ("Ich lese gerade Dune" / "Ich habe Dune zu Ende gelesen").
- Feedback review by target (ADR 0039, amends 0028): `GET /v1/feedback?review=pending&group=target`
  lists the queue one row per target with `votes`, `thumbs_up`, `thumbs_down`, distinct
  `voters` and `latest_at`; `POST /v1/feedback/approve` and `POST /v1/feedback/dismiss` decide
  in bulk by `feedback_ids` (1 to 500) or by target (every vote pending on it), with the admin
  key, idempotently (a vote no longer pending is returned in `unchanged`, never applied twice;
  an id the tenant does not have refuses the whole call with 404). `review=` also takes
  `approved` and `dismissed`, for reporting. SDK: `ctx.feedback.pending(group="target",
  review=...)`, `approve_many`, `dismiss_many`, `FeedbackTargetGroup`, `FeedbackBulkReview`.
  No migration.
- The capability eval suite (`benchmark/capabilities/`): eleven memory capabilities -
  single-fact recall, multi-hop, recency, update, abstention, forgetting, scale, procedures,
  corrections, implicit and explicit personalization - as 105 scenarios over five domains,
  1,110 phrasings in each of English, German, Spanish, Arabic and Hindi plus code-switched
  ones, with a deterministic dev/held-out split. It drives the running service through the SDK
  (`nollm`, `llm` and agent-harness modes), waits on the service's own readiness signals,
  scores the pushed context, the agent's search and the harness's answers, and records the
  baseline in `benchmark/results/capability_gate.json` as a ratchet: each cell's threshold is
  its best recorded accuracy, and a drop beyond the cell's own measured noise, or a context p95
  rise over 10 %, fails `make eval-capabilities-gate`. A PR is gated on the core set
  (`SET=core`, nollm) and the `llm_core` set (llm mode), each run on a store of its own that
  is dropped afterwards; scale and the full set run nightly. Every run forgets what it wrote when it ends, through the API
  (`--keep` to keep it) ([the capability eval suite](docs/eval/capabilities.md)). Measures;
  changes no behaviour.
- Statement kinds: every sentence a user says is labelled at write with what it does - a
  `FACT`, a standing `RULE`, a `CONDITIONAL_RULE` (an exception or a trigger about the world),
  a `STATUS` of a thing, a `CORRECTION`, or a `LIFECYCLE` change - and the kind is stored on
  the memories it produced (`system_metadata["statement_kind"]`, with a rule's
  `rule_trigger` and `rule_exception` as said). Cue words are data: a generic pack and a
  retail pack (`modules/memory/lexicon/`) in English, German, Spanish, Arabic and Hindi; what
  they leave open goes to the NLI head the service already loads (one pair for a sentence
  the packs are unsure of, none for one they settle), and to the tenant's model only when the
  head is unsure and only if the head confirms it. Retrieval is unchanged
  ([concepts](docs/guide/02-concepts.md#what-a-statement-does), ADR 0037). No migration.
  Only a rule that says it is for every time (a standing word, a recurring trigger, a
  universal obligation) is kept as a lasting rule; any other conditional instruction ("Tell me
  if the price drops") keeps its kind as metadata and is stored as before. A failing NLI head
  keeps the lexicon's labels (`memory_statement_labeller_fallback_total`). Measured on
  sentences a local language model wrote to order (`tests/eval/golden/statement_kinds_blind.json`,
  blind set 2): macro-F1 0.829 over the six kinds against 0.105 for the extractor before it,
  below it in no kind, language or cell, and no non-rule kept as a lasting rule.
- Learned skills: an agent learns from all of its users (its own tool records, whichever user it
  ran for, under one audience) and is offered what it learned in its context on its own -
  "Learned skills for this task", in full, matched to the task; nothing is published or
  approved. A learned skill reaches the agent's other users once two users produced it, and
  one whose runs opened the agent's own skill is shown as what it adds to it. Administrators
  list them (`GET /v1/skills?agent=`) and dismiss one (`POST /v1/skills/{id}/dismiss`)
  ([learned skills](docs/api/skills.md), ADR 0034, superseding 0033's drafts). Migration
  `0026_procedure_agent`. SDK: `ctx.advanced.skills.list()`, `.dismiss()`; `LearnedSkill`,
  `SkillView`.
- Search past conversations by message: `kinds=["message"]` (`POST /v1/recall`, the
  `memory_search` agent tool, `ctx.search`) reads this conversation's messages and then the
  user's earlier conversations', this conversation's first among equal matches; each item
  carries its `thread_id`. Another user's threads are never read; without a user, this thread
  only ([past conversations](docs/guide/04-retrieval.md#past-conversations)).

- Same-subject matching on the write path (ADR 0035, [concepts](docs/guide/02-concepts.md#when-two-statements-are-about-the-same-thing)):
  a subject respelled ("FORKLIFT-4", "forklift #4"), abbreviated ("PO-4471", "purchase order
  4471"; "DC 3", "Distribution Centre 3") or written in another script's digits ("المستودع
  رقم ٣") reinforces the memory it names instead of duplicating it. Candidates are also looked
  up by a few stored spellings of the subject (case, the first identifier's joins, pack
  aliases), so some older rows are found beyond the newest ones consolidation reads.
  Differing identifiers, short codes ("Store LA" / "Store AL", "C#" / "C++", "Sales SE" /
  "Sales"), their order ("Dock 3 door 4" / "Dock 4 door 3"), units, signs and decimal commas
  ("5 kg" / "5 lb", "1,5 kg" / "15 kg", "Level -1" / "Level 1"), dates, titles ("Mrs Patel" /
  "Mr Patel"), directions ("to" / "from Berlin") and legal forms ("Acme Inc" / "Acme Ltd")
  never merge, and a match that needs a plural folded, a title or connective dropped or the
  words reordered ("John Roberts" / "John Robert") is never merged by the service itself; with
  `conflict_adjudication` on, it is what the model is asked about. Abbreviations come from two
  vocabulary packs (data, `domain/vocabulary/`: generic and retail, always on) and from a
  tenant's own text where it defines one ("hazardous materials (hazmat)", "OOS stands for out
  of stock"), the tenant's definition winning over a pack's; nothing to configure.
  576 labelled pairs in five languages: no false merge (0 of 322 hard negatives, 0 in every
  class), merge F1 0.84 (exact subject strings before: 0.13).

### Fixed
- A document is `READY` only once it is findable. The parse job set `READY` before the index job
  it chains had embedded a chunk, so `wait_ready` returned while `search` and `context` still
  found nothing; with the worker busy (LLM-assisted enrichment on a slow model), the index job
  ran seconds later and the question asked at `READY` answered "Memory holds nothing". The
  document now stays `STAGED` through parsing and indexing, and the index job marks it `READY`
  after the embeddings are written, before graph enrichment
  ([documents](docs/api/documents.md)). When the index job gives up (its retries spent, or past
  the job timeout on a large file) the document is `FAILED` with `last_error` "index failed:
  ...", and `memory-reindex` indexes such documents, and `STAGED` ones with a parsed version,
  and makes them `READY`.
- The job worker's queue pool is sized for its four Procrastinate workers (two connections
  each), and their four LISTEN connections count against `connection_budget`; with four
  connections for all of them, pool checkouts timed out under load. Its request pool has a
  connection for every job it runs at once, and in `staging` and `prod` a `connection_budget`
  below what the worker needs stops it at start instead of being exceeded with a warning
  ([database](docs/deploy/database.md)). A worker starting no
  longer takes the jobs of one silent for 30 s (Procrastinate's default) but for 145 s
  (`TASKS.stalled_after_seconds`, derived from the heartbeat, the 90 s a worker rides a
  database outage out, its last try's pool timeout and the shutdown grace, plus a heartbeat of
  margin; a worker killed with -9 waits that long too),
  so a job no longer runs twice.
- `/v1/verify` records no verdict on a run whose context held no evidence. Against an empty
  context every claim is "unsupported" - one the run's own tools returned too - so the judge
  rejected such runs, their outcome became a failure, and an agent sampled for grounding
  (`TRELLIS_GROUNDING_SAMPLE`, 10 % by default) never learned a skill from them
  ([trust](docs/guide/06-trust.md)).
- The worker runs the `chat-fast` queue and the `embedding` queue - the jobs that make a
  memory, a document's chunks or a tool findable - on slots no parse, archive or graph job
  (nor, with a model, a summary; without one summaries are short and take them after
  indexing) can take, and the `graph` queue on slots of its own, in four pools whose computing
  slots come to about one per CPU (`tasks.worker_concurrency`): half are chat's (indexing's
  when no chat job waits), up to `tasks.indexing_concurrency` (2, a new setting; at most a
  quarter) indexing's, one the graph queue's and the rest every other queue's, chat and
  indexing first. Chat and indexing share their slots by priority, chat first, except that
  with a model indexing keeps its own (a chat job then waits on the model); the graph pool's
  slots wait on the model too, and it gets half the CPUs (at least 2) on top. With a slow model, observation extraction
  and graph enrichment held every slot, and a tool catalog published just before a question
  waited 14 s to be searchable; and a graph backlog - an import queues thousands of enrichment
  jobs, which outrank parsing, archiving, reconciliation and summaries - held every slot until
  it drained ([operations](docs/guide/11-operations.md#workers-and-periodic-jobs)).
- The job worker rides out a database that cannot do its queue statements for up to 90 s: no
  connection free in the queue pool, a full disk or no memory, a server shutting down or
  starting, a lost LISTEN connection (a connection lost mid-statement still ends the worker).
  Each is retried with a growing pause, logged (`worker.db_retry`) and counted
  (`memory_worker_db_retries_total`; `memory_worker_db_unavailable_seconds` says for how long).
  Under memory pressure one pool timeout in a heartbeat made Procrastinate stop the worker and
  the process exited 0, so a supervisor restarting on failure left it down; and a job-status
  write to a full disk stopped it too. Past the 90 s, or when a pool stops on its own, it now
  logs `worker.failed` and exits non-zero, and compose restarts it (`restart: on-failure`). A
  log line that cannot be written is lost and counted (`memory_log_lines_dropped_total`)
  instead of raising into the code that logged it
  ([operations](docs/guide/11-operations.md#observability)). The API is unchanged: its
  enqueue goes through the outbox, whose sweep dispatches what a failed enqueue missed.
- A write is findable within seconds while the graph is enriched by a slow model. Graph
  enrichment ran inside `memory.index` and `document.index`, and with an LLM enabled each such
  job held a worker slot for the model's calls (up to minutes on a small CPU model): with every
  slot taken, indexing, parsing and the other jobs waited up to six minutes, and a memory
  remembered just before a question was not in its context. With a model, enrichment is now a
  job of its own, `graph.enrich_memories` / `graph.enrich_document`, queued once the source is
  indexed, on the `graph` queue below indexing, in a pool of its own - never inside the index
  job - and never one at a time per tenant (so serialised, the capability eval's LLM run still
  had 552 enrichment jobs queued when its questions began, and they met a graph without their
  facts). Without a model the index job still enriches, once the source is findable: there it
  is tens of milliseconds, and as separate jobs the capability eval's writes settled 30 s later
  ([operations](docs/guide/11-operations.md#workers-and-periodic-jobs)).
- A Hindi statement that begins "आज से" ("from today": when a value took effect) was read as a
  standing rule, stored as a lasting preference and never compared with what it replaced. It
  is a fact now, as "from today", "ab heute", "desde hoy" and "اعتبارًا من اليوم" already were;
  "अब से" ("from now on") still makes a rule (ADR 0038).
- Same-subject lookup: an older statement past the window of recent rows consolidation reads
  was missed when its subject was spelled with different punctuation ("forklift-4" stored,
  "forklift 4" said): the spelling variants were derived only when the subject was not
  already lower-case, which the fact rule's subjects always are. Every word-and-identifier
  pair is now joined every way (a space, nothing, a hyphen, "#", "no."; "Nr.", "núm." and "nº"
  are read as joins too), for the lower-cased and the as-written forms, in any script
  (Devanagari and Arabic vowel signs included); another identifier is never produced, and
  "Forklift 3" and "Forklift 4" still never merge (ADR 0035). Both sides are compared
  case-folded (`lower(subject)`, index `ix_memories_scope_subject_lower`, migration 0027), so
  "Forklift 4" stored through the API is found by "forklift-4" said in a turn. Both sides are
  folded by PostgreSQL (Python's lower() and the database's differ, e.g. "ΟΔΟΣ"); an identity
  ("user:Alice") still matches only as written, and one in another case ("User:Alice") matches
  neither "user:alice" nor "user:Alice". The exact matches are read per kind like the
  recent window: a user's newest turns, all about "user:<id>", crowded their older facts out
  of one shared limit, so a person fact past the window was never replaced - the test that
  claimed it never left the window, and now does.
- "When I ask for a stock audit, always use a table" opened like a question, so nothing was
  stored; a "when"/"if" followed by a subject now opens a condition, and the rule is kept with
  its trigger. A standing rule is recognised wherever its "never"/"always" sits ("For weekly
  overviews, never include ..."), not only first in the sentence, and in German, Spanish,
  Arabic and Hindi as well; its exception clause ("unless I type 'include out of stock'") is
  kept with it (ADR 0037).
- Single-fact recall across conversations: "The master lock code for the hazardous materials
  cage in Warehouse 3 is 8492." said in one chat is answered in the user's next one. What a
  user says - their messages, decisions, feedback and their own `remember` - now defaults to
  `USER` (every conversation of theirs and every agent acting for them, never another person)
  instead of the thread it was said in; `visibility="THREAD"` keeps a statement to its
  conversation and `"WORKSPACE"` shares it with a team, as before. A user's turn relayed by an
  agent harness is the user's, no longer the agent's private note. Existing rows keep their
  audience ([concepts](docs/guide/02-concepts.md#scope-and-visibility-are-separate-on-purpose),
  ADR 0036).
- A user's sentence is no longer lost when a rule reads all of it: the turn is kept verbatim
  beside any reading that can lapse, merge or fade (a fact, a task, a procedure, an event, a
  decision) - "All seasonal merchandise must go to Facility B" survived only as a seven-day
  task - and the two take one slot in a bundle; forgetting, updating, retracting or
  correcting either one acts on both, as does a value replaced later in the conversation. A
  turn longer than 2000 characters is kept in sentence-cut pieces, up to six (about 12,000
  characters), instead of truncated after the first. The fact rule reads noun-phrase subjects
  of up to twelve words ("the master lock code for the hazardous materials cage in
  Warehouse 3"), not five; a longer clause or event is left to the event rule (ADR 0036).
- A key restricted with `may_act_as` to a user now reads that user's statements from all of
  their threads, and the other participants of a shared thread no longer read a user's
  default memories from it (ADR 0036).
- A document uploaded into a thread that did not exist yet reached `READY` but was never
  returned by `search` or `context`: its THREAD audience named a thread nobody had been
  granted. The upload now creates the thread for the uploader, as a first message does
  ([documents](docs/api/documents.md)); examples 04 and 06 no longer send a message first.
- A document chunk's text is now an exact slice of the parsed document. A long paragraph's
  sentences were re-joined with one space, which erased the source's own whitespace (two
  spaces after a full stop in PDFs), so chunk text no longer matched the document; the overlap
  between parts is a slice too. A long list is split between items first, so an item that
  fits is never broken across two chunks. Only a split table's parts, which repeat the header
  row, are not slices ([concepts](docs/guide/02-concepts.md#documents), ADR 0007).
- A footnote is indexed with the sentence that cites it (`Footnote to:` in the chunk's
  header): a note marked `*`, `†`, `‡`, `§`, `¶` (or doubled, superscript digits, `[^n]`)
  such as "§§ Butenafine, butoconazole, …" names neither its subject nor the analysis it
  belongs to, so no question about it found it. On the public PDF gate the CDC footnote moved
  from rank 36 to rank 4 for "Which drugs were included in the topical antifungal analysis?".
  Documents indexed before need a reindex to pick it up.
- A chunk's embedding was cached under the hash of its body, but the string embedded is the
  body with its header; the same passage under another title, section or situating context
  reused a vector of a different string. The key is now the hash of the embedded text.
- Context expansion and the graph's evidence chunks no longer add a passage that is already
  among the candidates under another record (a copy of the same document): ranking collapses
  such twins on `text_hash`, but these stages re-added them and spent their small budgets
  on them, crowding out companions the reader did not have. Expansion now also follows the
  best-ranked seed's edges first within an edge kind, and graph evidence keeps the facts'
  order, instead of the database's row order. Verification still fetches a twin when its
  own node is the required companion.

### Changed
- An end user's thumbs-down on a memory now waits for the administrator like any other vote,
  the owner's included (it retracted the memory on the spot before), and any reader of the
  memory may cast one (it was owner-only; a rewrite still is); an owner's `correct`/`edit`
  carrying new text is still applied at once, as are a run's own status, tool-call decisions,
  `/v1/verify` and the administrator's own word, and `DELETE /v1/memories/{id}`. One vote per
  voter per target: a voter's newer vote replaces their earlier pending one, which is kept with
  `review.state = "dismissed"`, `review.note = "replaced by a newer vote from the same user"`
  and `review.replaced_by` naming the newer vote, never applied, so repeated clicks count once
  (ADR 0039). No new review state: SDK 0.4.0 reads every record (it ignores `replaced_by`), and
  `author_record` and the grouped counts leave replaced clicks out. The rule holds across
  approvals: the same thumb as the voter's approved vote is stored dismissed with
  `replaced_by` naming the vote that counts, and an approved flip first undoes the vote that
  was actually applied (`projection.reversed`; an approved vote not yet applied is closed
  instead). Whether a record is a vote is decided by who sent it (`feedback.is_vote`), never
  by the `source` it claims.
  A `system` verdict on the caller's own run applies at once only when the credential may
  act as that agent (a bearer token naming a person waits), and an agent acting for a tenant
  admin edits only its own user's memories. Migration
  `0027_votes_replaced_by_subject` adds `feedback.review_replaced_by` (and the subject index under Fixed).
- The conflict adjudicator (`conflict_adjudication`) is asked about the closest stored memory
  on the same - or possibly the same - subject, instead of the closest sharing half its words:
  a fact restated in other words now reaches it, and a fact about another subject that merely
  shares words ("billing service" / "shipping service") no longer does. A pair the words leave
  undecided is scored by the multilingual encoder only when the model is enabled.
- The retail glossary's acronyms are read from the retail vocabulary pack, which adds `ASN`,
  `BOL`, `RTV`, `POG`, `BOPIS`, `WMS`, `3PL`, `FIFO`, `SOH`, `EDI`, `FC` and `LP` to its query
  expansion.
- The context's `procedures` (with `format=full`) is `skills` (`id`, `name`, `steps`,
  `with_skill`, `fixes`, `success_rate`, `runs`), and `tool_search`'s and `POST /v1/tools/hints`'
  `plan` has the same shape. Tool hints are computed for five or more tools (or the catalog);
  any toolbox gets the learned skills. SDK: `PromptContext.skills` replaces `procedures`.
- The service requires `bifrost-sdk>=0.3`: model calls send the gateway's deny-all MCP scope,
  which 0.3.0 introduced.
- The model-adapter contract tests that build or load a model carry the `models` marker and
  check for torch/sentence-transformers themselves, so CI (without the `models` extra)
  deselects them instead of skipping the whole module; the pure fingerprint test still runs.

### Removed
- Two learning features that never fed anything (migration `0030_drop_dead_learning`):
  - **Prefetch learning.** `/v1/context` pre-included memories an agent's earlier tool calls
    kept using for requests of the same pattern, from counts the `periodic.prefetch_learn` job
    folded every 5 minutes. On a live stack 0 of 60 runs pre-included anything and no counted
    item was ever eligible: a pull was keyed by the model's own query wording, which differs
    every run. Gone with it: the agent-tool pull log (a row per memory agent-tool call), which
    nothing else read - `POST /v1/verify` and a confirmed run's feedback only marked pulls
    used. Tables `agent_tool_pulls` and `prefetch_stats` are dropped. The context's
    `diagnostics.prefetched` key is no longer set.
  - **Procedure distillation.** With the `procedure_abstraction` model use the tool-learning
    job had the tenant's model write a procedure's title and strategy; no read path showed
    them. The use is removed from the model policy vocabulary (`LLMUse`): stored policies drop
    it, and its usage rows go with it. Columns `procedures.title`, `strategy`, `distilled` and
    `owner_principal` (whose key paid for it) are dropped. A learned skill is named from its
    task pattern, as it already was without a model. SDK `trellis-memory` 0.5.1: `LLMUse`
    loses `procedure_abstraction`.
- The `MEMORY__BLOB__PROVIDER` setting. The blob store follows from where the service runs:
  GCS when `MEMORY__SERVICE__ENVIRONMENT` is `staging` or `prod`, or when a GCS emulator is
  configured (`STORAGE_EMULATOR_HOST`); the filesystem otherwise. A leftover
  `MEMORY__BLOB__PROVIDER` is ignored ([configuration](docs/configuration.md#blob-storage-archives-documents-large-tool-outputs)).
- **SDK, breaking:** `ctx.advanced.memories.list()`. It took a cursor but dropped the next
  one, so it could not page: use `memories.iter()` for every memory, or `memories.page()` for
  one page and its cursor.
- **SDK, breaking:** `ctx.feedback.list_for()`, for the same reason: use
  `ctx.feedback.page_for(kind, id)` (`.items`, `.next_cursor`).
- **SDK, breaking:** `record_tool(output_summary=, sub_calls=)`. Nothing sent them: the
  service derives the summary from `output`. `POST /v1/tools/invocations` still accepts both.
- **SDK, breaking:** `MemoryClient(circuit_failure_threshold=, circuit_open_seconds=)`. The
  breaker opens after 5 failed calls in a row for 30 s
  (`trellis.memory.breaker.FAILURE_THRESHOLD`, `OPEN_SECONDS`).

### Documentation
- A Start-here README; the high-level design with the five Trellis repositories
  ([ARCHITECTURE.md](docs/ARCHITECTURE.md)); a sequence diagram per key flow
  ([flows.md](docs/flows.md)); a [configuration reference](docs/configuration.md) with every
  setting, kept in step with the code by a test; [troubleshooting](docs/troubleshooting.md),
  [versioning](docs/versioning.md) and this changelog.
- Numbered [examples](examples/README.md), simple to complex, that run offline against an
  in-process service: `make examples`, run in CI.
- `make docs-check`: every relative link and anchor resolves, and every snippet calls SDK
  methods and keywords that exist. Run in CI.
- Dated reports moved out of the user docs to `benchmark/reports/`, design records to
  `design/`; superseded handoffs, plans and logs deleted (git history keeps them). The model
  uses page merged into [chapter 8](docs/guide/08-models.md#the-eleven-uses).
- ADR headers carry their amendments (0004, 0009, 0021, 0022, 0023); ADR 0017 is closed as
  superseded by 0024-0026.
- `.env.example` lists `MEMORY__RETAIL_CALENDAR`, the one setting it was missing.

## 2026-10-04: SDK `trellis-memory` 0.4.0 (service unchanged at 0.3.0)

### Changed
- Lean context, tool-hint and recall responses (ADR 0029): the prompt form of `/v1/context`
  carries the rendered text and the tools that fit, `{name, confidence}`; the full form is
  structured data only; every number is in 0..1; empty keys are absent. Recall items no
  longer carry a citation string. SDK 0.4.0 models follow.
- The API surface (ADR 0030): database outages are a retryable `503`/`504` with
  `Retry-After`; every `413` is `PAYLOAD_TOO_LARGE`; every write honours `Idempotency-Key`;
  `201`/`202` carry `Location`; one cursor convention for every list; conditional GETs on the
  tool catalog and the agent tools; every schema field described.
- The runtime (ADR 0031): cache freshness by the audience a write changes; overload bounds and
  request deadlines; one connection budget per pod and PgBouncer support; readiness decided by
  PostgreSQL and the process; the job worker's lifecycle and metrics; Qdrant layout from
  settings; the tenant registry's invalidation channel.
- SDK resilience and dev defaults (ADR 0032): `Retry-After`, full-jitter backoff, retries of
  the read-only POSTs, status-classed errors, a circuit breaker, `MEMORY_URL` and
  `TRELLIS_API_KEY` defaults; `may_act_as` checks agents too; the admission gate behind a
  tenant switch, off by default; the forget cascade moves readers' revisions; a development
  tenant and a development envelope key in `dev` and `test`.
- A vote waits for the tenant administrator before it changes anything (ADR 0028).
- A conversation turn can be restated at ingest and searched with the turn, opt-in per tenant
  (ADR 0027); memories are ranked by a general fusion and four rules (ADR 0026); late
  interaction over two keys per memory (ADR 0025); a retailer's fiscal calendar
  (`MEMORY__RETAIL_CALENDAR`).

## 0.3.0: 2026-09-30 (service and SDK)

### Removed
- The one-release aliases 0.2.0 kept: `POST /v1/files` and the `X-Memory-*` header spellings
  (`POST /v1/tools/record` had gone before), and the SDK's `ctx.files`. One route per
  operation, one spelling per header.
- `query_decomposition` as a model use: a model call on the read path took 3.2-12.2 s against
  a 300 ms budget.

### Changed
- The SDK's surface: the per-turn verbs are on the bound context (`context`, `remember`,
  `update`, `forget`, `search`, `history`, `feedback`, `record_tool`, `tool_hints`, `verify`,
  ...), the conversation is `ctx.history` (`add`, `thread`, `update`, `delete`) where it was
  `chat.*`, and everything else is under `ctx.advanced`.
- The multilingual runtime (ADR 0024): every query and record in every language is encoded
  in a multilingual dense space beside the English one, with a multilingual grounding model.

## 0.2.0: 2026-09-28 (service and SDK)

### Changed
- Trellis names and API conventions (ADR 0022): the SDK is `trellis.memory` (was
  `universal_memory`); the scope headers are `X-Trellis-Tenant`, `-Workspace` and `-User`;
  errors are RFC 9457 problem documents; W3C `traceparent` is continued and `X-Trace-ID`
  returned; operation ids are `<tag>.<function>`. The old spellings were kept as aliases for
  one release.
- Tenants, API keys and workspaces are the service's own (ADR 0021); feedback, cursor
  pagination and agent model keys (ADR 0023).
