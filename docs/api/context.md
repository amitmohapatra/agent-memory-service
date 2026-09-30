# Context: what goes into the prompt, and whether the answer followed from it

One call assembles everything relevant to this turn, ranked, deduplicated, inside a token budget,
with an evidence report saying what it could *not* find. That is the call an agent should be
making — `POST /v1/context`. The rest of this area exists for when you need a part of it: the
ranked items alone (`recall`), the conversation alone (`threads`/`messages`), a standing question
kept warm (`briefs`), or a check on the answer you produced (`verify`).

## One turn

```mermaid
sequenceDiagram
  participant A as Agent
  participant C as POST /v1/context
  participant R as Retrievers
  participant G as Grounding
  A->>C: {query, scope, token_budget, require_evidence?}
  C->>R: conversation window · memories · document chunks · graph facts · summaries
  Note over R: dense + BM25, fused (RRF), audience-filtered in the store
  R-->>C: candidates
  C->>C: dedupe · rank · pack to the budget · render with citations
  C->>C: evidence report: required / satisfied / missing groups
  C-->>A: ContextBundle {rendered, parts, evidence, token_estimate, cache_hit}
  A->>A: prompt the model with bundle.rendered
  A->>G: POST /v1/verify {answer, bundle} — did each claim follow?
  G-->>A: per-claim verdicts: supported · unsupported · contradicted · borderline
```

The evidence report is the part people skip and then wish they had not. `status` is `COMPLETE`,
`INCOMPLETE` or `INSUFFICIENT`, with `required_groups`, `satisfied_groups` and `missing_groups`:
the service is telling you it could not find what an answer to *this* question needs. With
`require_evidence=True` an `INSUFFICIENT` report raises `InsufficientEvidence` instead of
handing you a bundle you might answer from anyway.

## Routes

| Route | Purpose | SDK |
| --- | --- | --- |
| `POST /v1/context` | build a `ContextBundle` for this turn | `ctx.context(query, token_budget=…, tools=…, since_revision=…, require_evidence=…)` |
| `POST /v1/recall` | ranked, scope-filtered items, no bundle assembly | `ctx.search(query, limit=…, kinds=["chunk", "memory", "summary"])` |
| `POST /v1/verify` | verify an answer claim by claim against evidence | `ctx.verify(answer, bundle=…)` |
| `POST /v1/threads` | create (or idempotently fetch) a thread | `ctx.chat.create(title=…)` |
| `GET /v1/threads/{id}` | one thread | `ctx.chat.thread()` |
| `DELETE /v1/threads/{id}` | soft-delete a thread | `ctx.chat.delete_thread()` |
| `POST /v1/messages` | append a message | `ctx.chat.user(...)`, `.assistant(...)`, `.internal(...)` |
| `GET /v1/threads/{id}/messages` | the window, newest page last | `ctx.history(limit=…, include_internal=…)` |
| `GET /v1/messages/{id}` | one message | `ctx.chat.message(id)` |
| `GET /v1/threads/{id}/summary` | the thread's durable summary | `ctx.summary()` |
| `POST/PUT/GET/DELETE /v1/briefs…` | standing questions and knowledge pages | `ctx.advanced.briefs.*` |

## The bundle

```python
bundle = await ctx.context(
    "how did revenue develop?", token_budget=4000, tools={"available": tool_names, "k": 8}
)

bundle.rendered  # prompt-ready text, with citation markers
bundle.profile  # pinned blocks of the user, agent and workspace   (profile.md)
bundle.thread_summary  # the thread's durable summary: text, covers_to_sequence, version
bundle.conversation  # ConversationWindow: the messages after the summary that fit
bundle.procedures  # procedures learned for this task: id, title, steps, success_rate
bundle.tools  # tool hints (only when asked): candidates, plan, next, prefill, missing
bundle.memories  # durable facts and preferences      (ContextItem)
bundle.knowledge  # document passages, with document, page and evidence
bundle.graph_facts  # entity relations
bundle.summaries  # document and section summaries
bundle.revision, bundle.delta  # the scope revision; pass it back as since_revision
bundle.evidence.status  # COMPLETE | INCOMPLETE | INSUFFICIENT
bundle.evidence.missing_groups
bundle.token_estimate, bundle.token_budget, bundle.cache_hit
bundle.insufficient  # the status, as a boolean
bundle.evidence_items()  # the packed evidence as /v1/verify items, in citation order
```

The pinned sections — profile, thread summary, procedures, tool hints, in that order — open
`rendered` and may take at most half of `token_budget` (in that priority); ranked evidence fills
the rest - but only evidence that clears the encoder's relevance floor: every ranked memory,
chunk and summary carries its dense similarity to the question as `relevance` (0..1), and one
under the floor (0.20 for the shipped multilingual encoder) is not packed, so a question the
store cannot answer comes back nearly empty instead of full of whatever ranked next
(`diagnostics.below_relevance_floor` counts them). Exact identifier hits and expansion
companions are exempt. They are one indexed read each and run concurrently with retrieval; none of them calls
a model. Memories an agent's own pulls kept using for requests of the same pattern are
pre-included ([agent-tools.md](agent-tools.md)). `revision` is the scope's revision: a request
with `since_revision` lists only the items new or changed since then (`delta: true`); the
pinned sections always come whole, and when the record of that revision has expired (an hour)
the whole bundle comes back with `delta: false`. The cache key includes the `tools` request.

`rendered` presents memory as evidence to weigh with ids to cite — not as instructions to follow.
That framing is deliberate: a retrieved passage is data, and an agent that treats it as a command
is one prompt-injection away from a problem.

## Reads call a model only when the policy or the request says so

Every read takes `use_llm`. Omitted, the resolved model policy's `read_assist` decides;
`true`/`false` override it for one request. Either way only the read helpers the operator and
the policy allow run (query expansion, question decomposition, a grounding judge in the
borderline band) and only when a model key can pay ([tenancy.md](tenancy.md)); the response
header `X-Trellis-LLM-Tokens` says what the request spent. The pinned sections never call a
model: summaries, profiles and procedure titles are written in the background. With no key,
the deterministic path answers.

## Verifying an answer

```python
report = await ctx.verify(answer_text, bundle=bundle)
print(report.supported, report.unsupported, report.contradicted, report.borderline)
print(report.per_claim_hallucination_rate, report.nli_provider, report.representative)
for verdict in report.claims:
    print(verdict.verdict, verdict.claim)  # supported | unsupported | contradicted | borderline
```

The cascade is deterministic first: citations are resolved, then an NLI head scores each claim
against its best premises, and a contradiction scan runs. Only the borderline band is a candidate
for a model, and only with `use_llm=True`. This is the same endpoint the harness's grounded judge
uses before it is willing to spend anything on an LLM judge.

## Conversation

```python
thread = await ctx.chat.create(title="Q3 planning")  # idempotent; messages also create one
await ctx.chat.user("Revenue was EUR 412 million in FY26.")
await ctx.chat.assistant("Noted — that's up 4% year on year.")
await ctx.chat.internal("plan: check the FY25 figure", role="AGENT")  # kind=INTERNAL
for message in await ctx.history(limit=20):
    print(message.role, message.content[:60])
```

`chat.*` needs `thread_id`. `session_id` and `turn_id` are optional and are the application's
own ids when given: a `turn_id` belongs to the session that created it, and a session belongs
to a thread. Without them a message joins the thread's own session, a USER message opens the
thread's next turn and any other message joins its latest turn; the acknowledgement returns the
ids used. Without a turn the SDK sends a fresh idempotency key per call, so two identical
messages ("ok") are two messages. `remember`, `observe`, `recall` and `context` need none of
that — a tenant is enough.
Internal messages stay out of the window unless `include_internal=True`.

## Briefs: a standing question, kept warm

```python
from trellis.memory import BriefSpec

brief = await ctx.advanced.briefs.create(
    BriefSpec(
        kind="mental_model",  # or "knowledge_page"
        title="Supply risk for SKU-1",
        question="What threatens SKU-1 availability this quarter?",
        refresh_seconds=3600,  # 60 … 86400
        use_llm=False,  # synthesis stays deterministic unless permitted
    )
)
fresh = await ctx.advanced.briefs.get(brief.brief_id)  # a read: never generates, may be stale
print(fresh.status, fresh.output.text if fresh.output else None)
```

A brief is a definition plus its stored output. **Reading one never generates text** — it returns
what is stored with `status` `pending`, `ready` or `stale`, and a refresh is queued by the write
path. That is what makes a brief cheap to read on every turn.
