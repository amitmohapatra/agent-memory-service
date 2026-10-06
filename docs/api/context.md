# Context: what goes into the prompt, and whether the answer followed from it

One call assembles everything relevant to this turn, ranked, deduplicated, inside a token budget,
with an evidence report saying what it could *not* find. That is the call an agent should be
making — `POST /v1/context`. The rest of this area exists for when you need a part of it: the
ranked items alone (`recall`), the conversation alone (`threads`/`messages`), a standing question
kept warm (a profile block with a `source_query`), or a check on the answer you produced (`verify`).

## One turn

```mermaid
sequenceDiagram
  participant A as Agent
  participant C as POST /v1/context
  participant R as Retrievers
  participant G as Grounding
  A->>C: {query, scope, token_budget?, window?, tools?, format?}
  C->>R: conversation window · memories · document chunks · graph facts · summaries
  Note over R: audience-filtered in the store#59; memories: weighted RRF of BM25, two dense spaces and ColBERT arms, then session/speaker/time/period rules (ADR 0026)
  R-->>C: candidates
  C->>C: dedupe · rank · pack to the budget · render with citations
  C->>C: evidence report: required / satisfied / missing groups
  C-->>A: format=prompt: {bundle_id, rendered, token_estimate, evidence_status, tools?} · format=full: the same content as data
  A->>A: prompt the model with rendered
  A->>G: POST /v1/verify {answer, bundle_id} — did each claim follow?
  G-->>A: per-claim verdicts: supported · unsupported · contradicted · borderline
```

The evidence status is the part people skip and then wish they had not. `evidence_status` is
`COMPLETE`, `INCOMPLETE` or `INSUFFICIENT`: the service is telling you whether it found what an
answer to *this* question needs, and `format="full"` adds `missing_evidence`, the companion
passages that are not there. Nothing raises on `INSUFFICIENT` — there is no `require_evidence`
flag — so check it yourself and answer that you do not know rather than answer from a context
that cannot support it.

## Routes

| Route | Purpose | SDK |
| --- | --- | --- |
| `POST /v1/context` | the context for this turn: rendered for a prompt (`format="prompt"`, the default) or the same content as structured data (`format="full"`) | `ctx.context(query, token_budget=…, tools=[names], window=…, document_ids=…, format=…, debug=…)` |
| `POST /v1/recall` | ranked, scope-filtered items, no bundle assembly | `ctx.search(query, limit=…, kinds=[…], time_from=…, time_to=…, as_of=…, known_at=…, document_ids=…, threads=…)`; kinds: `memory`, `chunk`, `summary`, `episode`, `message`. `threads="all"` makes `message` search every conversation this user owns ([past conversations](../guide/04-retrieval.md#past-conversations)) |
| `POST /v1/verify` | verify an answer claim by claim against the context it was given; with a run, recorded as the judge's RUN feedback | `ctx.verify(answer, bundle_id=…, run_id=…)` |
| `GET /v1/threads/{id}` | one thread, with its durable `summary` once it has one | `ctx.history.thread()` |
| `PATCH /v1/threads/{id}` | title and metadata (creates the thread when it does not exist yet) | `ctx.history.update(title=…, metadata=…)` |
| `DELETE /v1/threads/{id}` | soft-delete a thread | `ctx.history.delete()` |
| `POST /v1/messages` | append a batch of messages; `role: "EVENT"` is something that happened | `ctx.history.add([...])` |
| `GET /v1/threads/{id}/messages` | the window, newest page last | `ctx.history(limit=…, include_internal=…)` |
| `GET /v1/messages/{id}` | one message | `ctx.history.message(id)` |

## The two forms

Each form holds only what its reader uses, and nothing twice. Every number is in 0..1, a field
or list with nothing in it is absent, and ranking internals (the raw fusion score, which
retrievers found an item) are not sent: `debug=true` adds them as `diagnostics`.

**`format="prompt"`** (the default) is what an agent puts in front of its model:

```json
{
  "bundle_id": "6f1c0e2a9b",
  "rendered": "## Tools\n- erp-create_po (confidence 0.74, next step): amount = 700, cost_centre = 'CC-7'; missing supplier_id: erp-create_po needs 'supplier id': what should it be?\n\n## Recent conversation\nUSER: …\n\n## Memories\n- [m1] …",
  "token_estimate": 180,
  "evidence_status": "COMPLETE",
  "tools": [{"name": "erp-create_po", "confidence": 0.74}, {"name": "erp-get_budget", "confidence": 0.74}]
}
```

`tools` is there only when the request sent `tools`; a caller offers its model only those.

**`format="full"`** is the same content as data, for a caller that builds its own prompt. It
has no `rendered`: that would be the same content twice.

```json
{
  "bundle_id": "ee0d62d9ae",
  "evidence_status": "COMPLETE",
  "token_estimate": 277,
  "missing_evidence": ["PAGE11"],
  "conversation": {"thread_id": "thread-po", "messages": [{"id": "msg_…", "role": "USER", "text": "…"}]},
  "thread_summary": "Priya runs procurement for the Berlin office.",
  "profile": [{"block": "user", "text": "name: Priya"}],
  "procedures": [{"id": "prc_…", "title": "Order supplies", "steps": ["erp-search_supplier", "erp-create_po"], "success_rate": 1.0, "runs": 3}],
  "tools": [{"name": "erp-create_po", "confidence": 0.74, "success_rate": 1.0, "next": true, "args": {"amount": 700, "cost_centre": "CC-7"}, "missing": [{"arg": "supplier_id", "question": "erp-create_po needs 'supplier id': what should it be?"}]}],
  "memories": [{"id": "mem_…", "text": "…", "relevance": 0.32, "observed_at": "2026-10-04T11:17:47Z", "subject": "user:u-priya", "dates": [{"text": "Last week", "date": "2026-09-21..2026-09-27"}], "sources": ["msg_…"]}],
  "knowledge": [{"id": "chk_…", "text": "…", "relevance": 0.61, "document_id": "doc_…", "page": 11, "section": "Results > Revenue"}],
  "graph_facts": [{"id": "rel_…", "subject": "Priya", "predicate": "works_at", "object": "Acme", "relevance": 0.4, "observed_at": "…"}],
  "summaries": [{"id": "sum_…", "text": "…", "relevance": 0.5}]
}
```

```python
prompt = await ctx.context("how did revenue develop?", token_budget=4000, tools=tool_names)
prompt.rendered, prompt.bundle_id, prompt.evidence_status, prompt.token_estimate
prompt.tools, prompt.tool_names  # the tools that fit, with confidence (only with tools=[...])

bundle = await ctx.context("how did revenue develop?", tools=tool_names, format="full")
bundle.memories, bundle.knowledge, bundle.graph_facts, bundle.summaries
bundle.conversation, bundle.thread_summary, bundle.profile, bundle.procedures, bundle.tools
bundle.missing_evidence, bundle.insufficient  # evidence_status == "INSUFFICIENT"
```

**Nothing is shown twice.** A memory whose every source is a message the recent conversation
already shows, and a `mentions` fact whose object the shown text already names, are left out
of both forms. They stay in the bundle the service keeps under `bundle_id`, so `/v1/verify` and
the handles still see them.

The pinned sections — profile, thread summary, procedures, tool hints, in that order — open
`rendered` and may take at most half of `token_budget` (in that priority); ranked evidence fills
the rest - but only evidence that clears the encoder's relevance floor: every ranked memory,
chunk and summary carries its dense similarity to the question as `relevance` (0..1), and one
under the floor (0.20 for the shipped multilingual encoder) is not packed, so a question the
store cannot answer comes back nearly empty instead of full of whatever ranked next
(`diagnostics.below_relevance_floor` counts them). Exact identifier hits and expansion
companions are exempt. They are one indexed read each and run concurrently with retrieval; none of them calls
a model. Memories an agent's own pulls kept using for requests of the same pattern are
pre-included ([agent-tools.md](agent-tools.md)). `window=False` leaves the conversation window
out, for a framework that keeps its own history. The cache key includes the `tools` request.

`rendered` presents memory as evidence to weigh with ids to cite — not as instructions to follow.
That framing is deliberate: a retrieved passage is data, and an agent that treats it as a command
is one prompt-injection away from a problem.

## Reads call a model only when the tenant's policy says so

Whether a read consults a model is the tenant's model policy, `read_assist`
(`PUT /v1/model-key/policy`); a request cannot override it — there is no per-request
`use_llm`. Even then only the read uses the policy's `uses` name run (`query_expansion` for a
question no rule classified, `entity_resolution` on the graph route, `grounding_judge` for
borderline claims in `/v1/verify`), and only when a model key can pay
([tenancy.md](tenancy.md)); the response header `X-Trellis-LLM-Tokens` says what the request
spent. Query decomposition was removed in 0.3.0. The pinned sections never call a
model: summaries, profiles and procedure titles are written in the background. With no key,
the deterministic path answers.

## Verifying an answer

```python
report = await ctx.verify(answer_text, bundle_id=prompt.bundle_id)  # run_id= to record it on a run
print(report.supported, report.unsupported, report.contradicted, report.borderline)
print(report.per_claim_hallucination_rate, report.nli_provider, report.representative)
for verdict in report.claims:
    print(verdict.verdict, verdict.claim)  # supported | unsupported | contradicted | borderline
```

The cascade is deterministic first: citations are resolved, then an NLI head scores each claim
against its best premises, and a contradiction scan runs. Only the borderline band is a candidate
for a model, and only when the policy allows `grounding_judge` with `read_assist` on and a key
can pay. With a run (`run_id`, or the scope's agent run) the verdict is recorded as RUN feedback
from the service's own judge (`source="judge"`, applied without review —
[feedback.md](feedback.md)); `report.feedback_id` names it. This is the same endpoint the
harness's grounded judge uses before it is willing to spend anything on an LLM judge.

## Conversation

```python
thread = await ctx.history.update(title="Q3 planning")  # optional; messages also create one
await ctx.history.add(
    [
        ("USER", "Revenue was EUR 412 million in FY26."),
        ("ASSISTANT", "Noted — that's up 4% year on year."),
        {"role": "AGENT", "content": "plan: check the FY25 figure", "kind": "INTERNAL"},
    ]
)
for message in await ctx.history(limit=20):
    print(message.role, message.content[:60])
```

`history.*` needs a `thread_id` (without one, an agent run's messages go to the thread
named by its run id). `session_id` and `turn_id` are optional and are the application's
own ids when given: a `turn_id` belongs to the session that created it, and a session belongs
to a thread. Without them a message joins the thread's own session, a USER message opens the
thread's next turn and any other message joins its latest turn; the acknowledgement returns the
ids used. Without a turn the SDK sends a fresh idempotency key per call, so two identical
messages ("ok") are two messages. `remember`, `observe`, `recall` and `context` need none of
that — a tenant is enough.
Internal messages stay out of the window unless `include_internal=True`.

## A standing question, kept warm

```python
block = await ctx.profile.edit(
    "user.suppliers", source_query="Which suppliers does this team buy from, and on what terms?"
)
```

A profile block with a `source_query` is that question's answer. The `profile.query` job builds a
context for it in the scope that set it and writes the answer into the block (by the tenant's
model under the `summaries` use when a key is registered, the best evidence line by line
otherwise) — when the question is set and hourly after. Reading it is reading the profile: it is
pinned into every bundle and never generates text on the read path. `source_query=None` clears
the question and keeps the last answer.
