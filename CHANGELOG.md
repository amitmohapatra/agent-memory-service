# Changelog

What changed in the service (`trellis-memory-service`) and its SDK (`trellis-memory`), newest
first. The service and the SDK carry separate versions ([versioning.md](docs/versioning.md));
each entry says which one moved. Decisions behind each change are in the
[ADRs](docs/adr/README.md).

## Unreleased

### Added
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
  Differing identifiers, short codes, their order, units, signs, decimal commas, dates,
  titles and legal forms ("Forklift #3" / "#4", "Store LA" / "Store AL", "Dock 3 door 4" /
  "Dock 4 door 3", "5 kg" / "5 lb", "1,5 kg" / "15 kg", "Level -1" / "Level 1", "Mrs Patel" /
  "Mr Patel", "Acme Inc" / "Acme Ltd") never merge, and a match that needs a plural folded, a
  title dropped or the words reordered ("John Roberts" / "John Robert") is never merged by
  the service itself. Abbreviations come from two vocabulary packs (data,
  `domain/vocabulary/`: generic and retail, always on) and from a tenant's own text where it
  defines one ("hazardous materials (hazmat)", "OOS stands for out of stock"); nothing to
  configure. SUBJECT_RESULTS

### Fixed
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
