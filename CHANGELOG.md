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
  uses page merged into [chapter 8](docs/guide/08-models.md#the-twelve-uses).
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
