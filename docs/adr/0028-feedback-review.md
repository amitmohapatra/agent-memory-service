# ADR 0028: A vote waits for review before the platform learns from it

Date: 2026-10-04. Status: accepted.

## Context

Feedback (ADR 0022) was applied as it arrived. A confirm added 0.1 confidence and one
reinforcement to a memory; a verdict on a run set the run's outcome and moved each memory
its answer cited by 0.05; a rejection of a procedure stopped it being offered. Confidence
and reinforcement feed the ranking (`by_standing`, at most +/-15%) and forgetting.

A vote can be wrong, or cast on purpose: anyone who may read a memory could confirm it
again and again under fresh feedback ids, and a thumbs-down on a correct answer lowered every
memory it cited. Nothing counted one vote per person, weighed who voted, or let anyone look
before the vote took effect.

## Decision

1. **A vote is stored and waits** (`review.state = pending`). It changes nothing - no
   confidence, reinforcement, run outcome, procedure or index - until it is approved.
2. **The tenant's administrator reviews**, with the credential that administers the tenant
   (the admin key, as for keys, workspaces and the model policy):
   - `GET /v1/feedback/pending` - the queue, newest first; each verdict carries its author's
     record in review (`author_record`: pending, approved, dismissed).
   - `POST /v1/feedback/{id}/approve` - applied as if it had just arrived.
   - `POST /v1/feedback/{id}/dismiss` - kept for statistics, never applied.
   A verdict is decided once (409 after that). A run verdict's evidence and the judge's own
   verdict on the same run (`GET /v1/feedback?target_kind=run&target_id=...`) are what a
   reviewer weighs a thumbs-down against.
3. **Applied as they arrive** (no queue), because none of them is a vote:
   - the grounding judge's verdict, written by the service itself (`/v1/verify`); a client
     that sends `source=judge` is not the judge and waits like anyone else;
   - a run reporting its own final status (`source=system`, the target is the calling run,
     citing no memories): the lowest-ranked word on a run, which never overrides a person or
     the judge; one that cites memories would move their confidence, so it waits;
   - an owner's reject, correct or edit of their memory: an edit, already limited to the
     owner (or a tenant admin), like supersede and forget;
   - a decision on a tool call: what is learned from it is an approval suggestion, which
     only takes effect when it is accepted (`POST /v1/tools/approval-suggestions/{id}/accept`,
     by the agent whose decisions it was learned from);
   - what the tenant's administrator says, through the admin key or in person.
4. Every record keeps who voted, on what, when, the verdict and, once decided, who reviewed
   it, when and why: the statistics survive whether or not the vote was applied.

## Consequences

- Without a reviewer, votes accumulate and the platform learns nothing from them; a tenant
  that wants votes to count has someone work the queue.
- Repeated or hostile votes no longer move anything on their own, so no per-voter limit or
  reputation is needed for safety; `author_record` lets the reviewer see a voter who is
  usually dismissed.
- Three new operations; feedback records carry `review`. Existing records have none and
  were applied as before (migration 0022).
