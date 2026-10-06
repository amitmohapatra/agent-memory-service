# Key flows

How the service handles each kind of call, inside the service: which module does what, what is
committed before the response, and what runs later in the job worker. One sequence diagram per
flow. The per-area pages in [`api/`](api/README.md) show the same flows from the caller's side,
with every route and SDK call; [ARCHITECTURE.md](ARCHITECTURE.md) is the structure the flows
run through. The runnable form of each flow is in [`examples/`](../examples/README.md).

| Flow | Entry point | Runs in | Example |
|---|---|---|---|
| [1. Ingest: history add](#1-ingest-history-add) | `POST /v1/messages` | request, then jobs | [02](../examples/02_conversation_history.py), [03](../examples/03_remember_search_forget.py) |
| [2. Context build](#2-context-build) | `POST /v1/context` | request | [01](../examples/01_quickstart_context.py), [04](../examples/04_documents.py) |
| [3. Search and recall](#3-search-and-recall) | `POST /v1/recall` | request | [03](../examples/03_remember_search_forget.py), [05](../examples/05_agents_runs_and_sharing.py) |
| [4. Feedback and the review queue](#4-feedback-and-the-review-queue) | `POST /v1/feedback`, `/approve`, `/dismiss` | request, then a job | [08](../examples/08_feedback_and_review.py) |
| [5. Verify and grounding](#5-verify-and-grounding) | `POST /v1/verify` | request | [06](../examples/06_verify_grounding.py) |
| [6. Tool catalog and hints](#6-tool-catalog-and-hints) | `PUT /v1/tools/catalog`, `POST /v1/tools/invocations`, `POST /v1/tools/hints` | request, then jobs | [07](../examples/07_tool_memory_and_hints.py) |
| [7. Compaction and background jobs](#7-compaction-and-background-jobs) | the outbox, the worker, the crons | job worker | [10](../examples/10_background_jobs_and_summary.py) |

Every flow starts the same way: the credential is authenticated, the trusted headers and the
body's `scope` become one `MemoryExecutionContext` (a body that contradicts a header is
refused), and a write carries an `Idempotency-Key` (the SDK derives one). Every read is
filtered by audience **in the store**, before anything is ranked.

---

## 1. Ingest: history add

A message is durable when the response says so, and becomes memory afterwards. The request does
the cheap, transactional part (`modules/conversation/service.py`); everything expensive is a
job, committed in the same transaction as the message (the transactional outbox, ADR 0004).

```mermaid
sequenceDiagram
  autonumber
  participant C as Client (SDK history.add)
  participant API as API: POST /v1/messages
  participant CS as ConversationService
  participant PG as PostgreSQL
  participant R as Outbox relay
  participant W as Job worker
  participant Q as Qdrant
  participant K as Cache
  C->>API: {scope, messages} + Idempotency-Key
  API->>API: idempotency check (a replay returns the first answer)
  API->>CS: may this caller write the thread? (asked before any lock)
  CS->>PG: BEGIN, thread lock, upsert thread / session / turn
  CS->>PG: message row, observation row
  CS->>PG: outbox: memory.process_observation, archive.stage_message (60 s, per thread)
  CS->>PG: outbox: summary.refresh on every 20th message, else episode.index
  CS->>PG: bump the THREAD revision, COMMIT
  API-->>C: 202 {messages: [{message_id, job_ids}]}
  API->>K: append to the hot thread window (after commit, best effort)
  R->>W: dispatch the committed outbox rows (fast path, swept every minute)
  W->>PG: memory.process_observation: extract, classify, admission (tenant switch), consolidate
  Note over W,PG: dedupe is lexical and dense. A newer fact supersedes, a repeat reinforces.
  W->>PG: memories with evidence refs and validity, plus a memory.index job, one transaction
  W->>Q: memory.index: embed, upsert into the memory collection
  W->>PG: graph enrichment (entities, relations), bump the audience's revisions
```

`role: "EVENT"` is stored as an internal message and learned from as an event. A message in a
language other than English is kept verbatim and indexed in the multilingual dense space; the
model reads it into typed facts only when the tenant's key and policy allow
([chapter 8](guide/08-models.md#the-twelve-uses)). `ctx.remember(...)` (`POST /v1/memories`) is
the direct path: one memory, stored verbatim, never gated.

## 2. Context build

`POST /v1/context` is the call an agent makes every turn (`modules/context/builder.py`). A
cached bundle is served only while every revision it depends on is unchanged, so a write by
anyone in the bundle's audience retires it ([caching](ARCHITECTURE.md#caching)).

```mermaid
sequenceDiagram
  autonumber
  participant C as Client (SDK context)
  participant API as API: POST /v1/context
  participant B as ContextBuilder
  participant PG as PostgreSQL
  participant K as Cache
  participant E as RetrievalEngine
  participant Q as Qdrant
  C->>API: {scope, query, token_budget?, window?, tools?, format?}
  API->>B: build_api (model calls only if the policy's read_assist allows)
  B->>PG: read every revision the bundle depends on (one statement)
  B->>K: bundle key = scope + revision fingerprint + query
  alt cache hit
    K-->>B: the stored bytes
  else miss
    B->>E: route the query, then search under the authorized scope
    par encoders and search
      E->>Q: exact lookup, dense (two spaces) and BM25, audience-filtered in the store
    and graph prefetch
      E->>PG: one budgeted graph statement (150 ms)
    end
    E-->>B: fused candidates (memories ranked by ADR 0026), graph facts
    B->>PG: side sections in parallel: profile, thread summary, procedures, prefetched memories, recent messages, tool hints
    B->>B: expand over the document graph, verify evidence groups, relevance floor
    B->>B: pack within the token budget, render with handles ([m1], [k1])
    B->>K: one pipelined write: the bundle record, the prompt form, the full form
  end
  B-->>API: bundle (format=prompt or full)
  API->>PG: read audit (who read what, under which scope)
  API-->>C: {bundle_id, rendered, token_estimate, evidence_status, tools?}
```

The `bundle_id` names the record `verify` later checks an answer against (kept 30 minutes).
`evidence_status` is `COMPLETE`, `INCOMPLETE` or `INSUFFICIENT`; nothing raises on
`INSUFFICIENT`, so check it and say you do not know. The pipeline in detail:
[chapter 4](guide/04-retrieval.md) and [api/context.md](api/context.md).

## 3. Search and recall

`POST /v1/recall` (SDK `search`) is retrieval without the bundle: ranked, scope-filtered items
of the kinds you ask for (`memory`, `chunk`, `summary`, `episode`, `message`). It shares the
retrieval engine with the context build and skips the packing, the side sections and the cache.

```mermaid
sequenceDiagram
  autonumber
  participant C as Client (SDK search)
  participant API as API: POST /v1/recall
  participant S as SearchService
  participant A as Authorization
  participant E as RetrievalEngine
  participant Q as Qdrant
  participant PG as PostgreSQL
  C->>API: {scope, query, kinds?, limit?, time_from?, time_to?, as_of?, known_at?}
  API->>S: search(ctx, query, kinds, limit)
  S->>A: the audience this principal may read (visibility keys, memberships)
  S->>E: route (English cues, any other language as a general question)
  E->>Q: hybrid search with the audience filter in the query
  E->>PG: graph prefetch, exact identifiers, the thread's messages when kind=message
  E-->>S: fused, deduplicated, temporal view applied (as_of, known_at)
  S-->>API: items (id, kind, text, observed_on, document and page for a passage)
  API->>PG: read audit
  API-->>C: {items}
```

A superseded or forgotten memory is never served as current; `as_of` and `known_at` read it as
it was. Example [05](../examples/05_agents_runs_and_sharing.py) shows the audience at work: a
`RUN` note is found by its run and the runs it spawned, and by nobody else.

## 4. Feedback and the review queue

A verdict is stored, then projected by a job (`modules/feedback/service.py`). A vote that
would change what was learned on one person's or one agent's word waits for the tenant's
administrator (ADR 0028).

```mermaid
sequenceDiagram
  autonumber
  participant U as Caller (user, agent, judge, run)
  participant API as API: POST /v1/feedback
  participant F as FeedbackService
  participant PG as PostgreSQL
  participant A as Tenant admin
  participant W as Job worker
  U->>API: {target_kind, target_id, verdict, correction?, source}
  API->>F: record(ctx, feedback)
  F->>PG: authorize the target (read it, or own it to retract or correct)
  alt needs review: a vote on a memory, a run or a procedure
    F->>PG: feedback row, review.state = pending, no job
    API-->>U: 201 {review: {state: pending}, projection: null}
    A->>API: GET /v1/feedback?review=pending (the admin key)
    A->>API: POST /v1/feedback/{id}/approve (or /dismiss)
    API->>PG: review recorded, and on approve the feedback.project job
  else applied as it arrives: the judge, a run's own status, an owner's correction, a tool call, an admin
    F->>PG: feedback row + feedback.project job, one transaction
    API-->>U: 201 {review: null, projection: null}
  end
  W->>PG: feedback.project
  alt memory
    W->>PG: confirm reinforces, reject retracts, correct supersedes with the correction
  else run
    W->>PG: the run's outcome (a person over the judge over the run), cited memories' confidence
  else tool call or procedure
    W->>PG: tool statistics and approval pattern, or the procedure retired
  end
  W->>PG: projection written, the memory re-indexed, its audience's revisions bumped
```

The projection is a later fact, which is why `GET /v1/feedback/{id}` is worth calling after a
`POST`: example [08](../examples/08_feedback_and_review.py) reads it back. Details:
[api/feedback.md](api/feedback.md).

## 5. Verify and grounding

`POST /v1/verify` checks an answer claim by claim against the bundle it was built from
(`modules/grounding/cascade.py`). Deterministic first, the model last and only for the
borderline band.

```mermaid
sequenceDiagram
  autonumber
  participant C as Client (SDK verify)
  participant API as API: POST /v1/verify
  participant BR as Bundle records
  participant G as GroundingCascade
  participant N as NLI model
  participant L as Gateway (optional)
  participant F as FeedbackService
  C->>API: {scope, answer, bundle_id, run_id?}
  API->>BR: load the bundle record for this scope (404 after 30 minutes or for another scope)
  API->>G: verify(answer, evidence, unused evidence)
  G->>G: split into claims, resolve citation markers to the evidence they name
  G->>N: entailment of each claim against its evidence
  opt borderline claims, grounding_judge allowed and a key can pay
    G->>L: judge the borderline claims
  end
  G->>G: contradiction scan over retrieved-but-unused evidence
  G-->>API: per claim: supported, unsupported, contradicted or borderline
  API->>API: the supported evidence counts as used (agent-tool pull statistics)
  opt a run in the scope or in the body
    API->>F: the report as the judge's RUN feedback (applied without review)
  end
  API-->>C: {claims, per_claim_hallucination_rate, feedback_id?}
```

Without a gateway a borderline claim stays `borderline`. Offline the NLI is the lexical
stand-in, so example [06](../examples/06_verify_grounding.py) shows the mechanism, not the
quality of a real entailment model.

## 6. Tool catalog and hints

The service never runs a tool. It keeps the catalog, records what a run called, learns from
runs labelled successful, and answers which tool, which plan and which arguments
(`modules/tools/*`, ADR 0018).

```mermaid
sequenceDiagram
  autonumber
  participant H as Agent or harness
  participant API as API
  participant T as ToolMemoryService
  participant PG as PostgreSQL
  participant W as Job worker
  participant Q as Qdrant
  H->>API: PUT /v1/tools/catalog [{name, description, input_schema, side_effects}]
  API->>T: upsert by name (idempotent)
  T->>PG: catalog rows + tools.index job
  W->>Q: tools.index: the tools search collection
  loop each tool call the run makes
    H->>API: POST /v1/tools/invocations {tool, args, output, status, task, step}
    API->>T: record (idempotent on run, step, tool, arguments)
    T->>PG: invocation (arguments redacted per the catalog), tool statistics
  end
  H->>API: POST /v1/feedback {target_kind: run, verdict: confirm, source: system}
  W->>PG: tools.learn (every 5 minutes): one procedure per audience and task pattern, procedural graph edges
  H->>API: POST /v1/tools/hints {task, available, k}
  API->>T: hints(ctx, task)
  T->>Q: tools that fit the task
  T->>PG: the procedure for the task pattern, statistics, this run's earlier outputs
  T->>T: the next step, its arguments bound from earlier outputs, the ones still missing
  API-->>H: {tools: [{name, confidence, success_rate, next, args, missing}], plan}
```

`POST /v1/context` answers the same hints inline when it is sent `tools`. A run nobody labelled
counts as a weak success a day later if none of its calls failed. Details:
[api/tools.md](api/tools.md).

An active procedure becomes a skill only when a person publishes it
([learned skills](api/tools.md#learned-skills)):

```mermaid
sequenceDiagram
  autonumber
  participant A as Tenant administrator
  participant API as API
  participant S as SkillDrafts
  participant PG as PostgreSQL
  participant D as SKILLS_DIR or the gateway's skills repository
  participant H as Harness run
  A->>API: GET /v1/tools/skill-drafts
  API->>S: list(tenant)
  S->>PG: active procedures not decided for their current steps
  API-->>A: [{id, state: new or changed, name, description, body, support, success_rate}]
  A->>API: POST /v1/tools/skill-drafts/{id}/publish {name?}
  API->>S: publish
  S->>D: SKILL.md as its next version (refused when another tenant or a person owns the name)
  S->>PG: the decision, with the steps it was about
  API-->>A: {state: published, name, version, destination}
  H->>D: load the skill by name at run start (pinned for the run)
```

## 7. Compaction and background jobs

What keeps memory small and current runs off the request path, in the job worker
(`modules/jobs/registry.py`). A job is committed with the write that needs it and dispatched
after the commit. The outbox sweep is the floor on how late a write can become a memory when
the fast path misses it.

```mermaid
sequenceDiagram
  autonumber
  participant PG as PostgreSQL (outbox, rows)
  participant R as Outbox relay and sweep
  participant W as Job worker
  participant B as Blob store
  participant Q as Qdrant
  R->>PG: rows committed with each write
  R->>W: dispatch now (fast path), and every minute (periodic.outbox_sweep)
  par every 20th message of a thread
    W->>PG: summary.refresh: roll the thread's durable summary forward
    W->>Q: episode.index: the summary as a searchable episode
  and 60 s after a thread's messages, coalesced
    W->>B: archive.stage_message: group, compress, checksum, immutable upload, verify
    W->>PG: manifest, ARCHIVED, the staged payload purged later (periodic.archive_purge)
  and user facts
    W->>PG: profile.refresh: the pinned profile block
  end
  loop crons
    W->>PG: periodic.reconcile (5 min): re-queue stuck jobs, repair archives and index drift
    W->>PG: periodic.memory_expire (hourly): CURRENT to EXPIRED past expires_at
    W->>PG: periodic.memory_forget (daily): archive idle, low-scoring memories
    W->>PG: periodic.retention (daily): forget past the tenant's retention_days
    W->>PG: periodic.outbox_purge, idempotency_purge, read_audit_purge (hourly)
  end
  opt a gateway configured
    W->>PG: periodic.memory_reflect and memory_connect (every 6 hours)
  end
```

The summary plus the 20-message window always covers the whole thread, so a long conversation
fits a context at any length; example [10](../examples/10_background_jobs_and_summary.py)
crosses the mark. Every job, its queue, its retries and its schedule:
[chapter 11](guide/11-operations.md#workers-and-periodic-jobs).
