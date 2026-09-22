# 1 · Why a memory service

> You have built an agent. It works beautifully for one turn, and then it forgets.
> This chapter is about why the obvious fixes do not work, and what this service does
> instead.

**Next:** [2 · Concepts](02-concepts.md) · **Up:** [Documentation](../README.md)

---

## The problem, three times

### It forgets

The first instinct is to put the transcript in the prompt. That fails on three axes at
once, and only one of them is the context window:

- **Cost.** Every turn re-sends everything that came before. A long conversation pays for
  its own history on every single request.
- **Relevance.** A model given forty turns of history and one question has to find the
  question. Precision falls as you add context, not rises.
- **Truth.** The transcript says the user lives in Berlin *and* that they moved to Munich.
  A prompt has no opinion about which is current.

So you need something that decides *what is worth keeping*, and returns only what bears on
the question being asked right now.

### It has no idea who may see what

The second instinct is a vector database. Embed everything, search by similarity, done.

A vector store returns the nearest chunks. It has no notion of a tenant, a user, an agent
that must not see another agent's working notes, or a document a contractor was never
cleared for. You end up writing that logic yourself, in the application, on every read
path — and the first time you get it wrong, one customer sees another customer's data.

This service treats "who may read this" as a property of the memory, enforced below the
retrieval layer. Chapter 7 is entirely about it.

### It cannot tell you why it believes something

The third problem shows up when an agent says something wrong and you have to find out
where it came from. A chunk in a vector store is just text. There is no link back to the
message, file or tool run it came from, no record of when it was learned, and no way to
express "this used to be true".

Every memory here carries at least one piece of evidence pointing back at its source
(`domain/memory.py`: `evidence` has `min_length=1` — a memory with no provenance cannot be
constructed). And a fact that stops being true is *superseded*, not deleted, so the history
survives. Chapter 3 covers that.

---

## What the service actually does

You report what happened. It decides what is worth remembering.

```
your agent                    memory service
    │
    │  POST /v1/observations   "the user said: my timezone is Europe/Berlin"
    ├─────────────────────────────►  accepted (202), queued
    │                                    │
    │                                    ├─ extract candidate facts
    │                                    ├─ deduplicate against what is known
    │                                    ├─ decide type, lifetime, visibility
    │                                    └─ index for retrieval
    │
    │  POST /v1/context        "draft a reply about the Q3 numbers"
    ├─────────────────────────────►
    │◄───────────────────────────── a bounded, ranked bundle with provenance
```

Two things in that picture are the whole design.

**You never write a memory.** You report an *observation* — "this happened, or this was
learned" — and the service decides asynchronously whether anything about it is worth
keeping. That is why the write returns `202 Accepted` and not `201 Created`: at the moment
you are told the write succeeded, no memory exists yet. Chapter 2 explains the contract;
chapter 10 explains why the work is queued in the same transaction that accepted it.

**What comes back is bounded, not complete.** A context bundle is "bounded, ranked,
provenance-carrying evidence" and by design "never contains everything"
(`domain/context_bundle.py`). You give it a token budget; it gives you the best use of that
budget. Chapter 4 is about how it chooses.

---

## Distillation, measured

The point of an observation becoming a memory is compression with provenance. From the
throughput benchmark (`benchmark/results/memory.json`):

| | |
|---|---|
| observations submitted | 200 |
| memory rows produced | 87 |
| of those, currently true | 66 |

The other 21 are superseded or contradicted versions, kept because "what did we believe last
Tuesday" is a real question. Nothing was thrown away; the 200 raw observations are still
there, and every one of the 87 memories points back at the ones it came from.

---

## Why not just…

| …a vector database? | It has no tenancy, no time, no provenance, and no opinion about what is worth storing. You would build all four on top. |
| …a summariser? | A summary is lossy in a way you cannot query. "What did the user say about pricing in March" is not answerable from a paragraph that mentions pricing. |
| …a bigger context window? | Cost scales with every turn, and precision falls as you add context. The window is the least interesting of the three constraints. |
| …fine-tuning? | Wrong tool: you need facts that change hourly and are scoped per user, not weights that change monthly and are shared by everyone. |

---

## What this service is not

Being clear about this early saves disappointment later.

- **It is not an LLM wrapper.** Every one of its eleven optional model uses has a
  deterministic fallback, and the service ships with the LLM **off**. A full install with no
  model gateway at all is a working install. Chapter 8 lists each use and its fallback.
- **It is not an agent framework.** It has no opinion about how your agent loops, which
  model you call, or what a tool is. It is a service your agent talks to.
- **It is not finished.** Several seams are built and deliberately not wired — the admission
  gate, thread observation, relation invalidation. Each is flagged where it appears, with
  what it would do and what turning it on would change.

---

## What to read next

- New to the vocabulary → [2 · Concepts](02-concepts.md)
- Here for the retrieval quality → [4 · Retrieval](04-retrieval.md)
- Evaluating it for a regulated environment → [7 · Authorization](07-authorization.md) and
  [6 · Trust](06-trust.md)
- Need to deploy it this week → [11 · Operations](11-operations.md)
