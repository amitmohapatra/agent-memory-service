# ADR 0039: Only an end user's thumbs wait, once per voter, reviewed by target

Date: 2026-10-11. Status: accepted. Amends [ADR 0028](0028-feedback-review.md).

## Context

ADR 0028 made a vote wait for the tenant's administrator. Two gaps remained, both noise:

- An owner's thumbs-down on their own memory was treated as an edit and applied at once: the
  memory was retracted on one click, although the same click on a run waits. A 👎 is a vote
  whoever casts it; what is an edit is a correction that says what the memory should say.
- Nothing counted one vote per person. Clicking 👍 five times stored five pending votes, and a
  user who changed their mind left both their 👍 and their 👎 in the queue. The queue was one
  row per vote, so an administrator looking at a popular answer read the same click many times
  and decided each one separately.

## Decision

1. **Only an end user's thumbs wait.** A `confirm`, `approve` or `reject` of a run, a memory or
   a procedure by a person or an agent carrying one is stored `pending` and changes nothing -
   confidence, reinforcement and so ranking, the index and cached bundles, the run's outcome,
   the skills learned from runs - until the administrator approves it. That now includes the
   owner's thumbs-down on their own memory, and since a thumbs-down no longer retracts on its
   own, any reader of a memory may cast one (it was limited to the owner, the user an agent
   acts for, or an admin, as forgetting is). A correction or edit that rewrites the memory
   keeps that limit: a workspace member who can read a colleague's shared memory cannot
   rewrite it (403), an agent acting for a user edits only that user's memories, and a tenant
   admin's edit applies at once.
2. **What is automated, or an edit, stays immediate:** the run's own status (`source=system`
   on the calling run, citing nothing), tool-call decisions, the service's own `/v1/verify`
   verdict, the administrator's own word (the admin key, the platform key, or an admin user in
   person), and an owner's `correct`/`edit` whose correction carries new text (a string or
   `{content: ...}`; one that carries no text is a thumbs-down in other words, and waits).
   `DELETE /v1/memories/{id}` is erasure, not feedback, and applies at once.
3. **One vote per voter per target.** A voter's newer pending vote on a target replaces their
   earlier pending vote on it: the earlier is kept as `dismissed`, with the note "replaced by a
   newer vote from the same user", `replaced_by` naming the newer vote and a projection of
   `none`, and is never applied. Not a new review state: SDK 0.4.0, the oldest the API
   supports (docs/versioning.md), validates the state against pending, approved and
   dismissed and ignores fields it does not know, so it reads a replaced vote. The voter is
   the person (`user_id`) whichever agent carried the vote; an agent with no person behind it is its own
   voter. Votes already decided are not touched; a retry with the same `feedback_id` replaces
   nothing. Submissions by one voter on one target are serialised, so two clicks at once still
   leave one pending vote, and replacing takes the same lock a review takes, so a vote is never
   both approved and replaced.
4. **A grouped queue and bulk decisions** (the admin key, as for the rest of review):
   - `GET /v1/feedback?review=pending&group=target`: one row per target with `votes`,
     `thumbs_up` (confirm/approve), `thumbs_down` (reject/correct/edit), distinct `voters` and
     `latest_at`, the most recently voted target first, cursor paged.
   - `POST /v1/feedback/approve` and `POST /v1/feedback/dismiss` take `feedback_ids` (1 to 500)
     or `target_kind` + `target_id` (every vote pending on it, up to 500 a call, `more` says when
     to call again). Idempotent: a vote no longer pending is returned in `unchanged` with its
     state and left as it is, so nothing is applied twice. An id the tenant does not have
     (another tenant's included) refuses the whole call with 404 before anything is decided.
     The one-vote routes keep their 409 for a vote decided before.
5. **Once per voter, across approvals.** A vote with the same thumb as the voter's approved
   vote changes nothing: it is stored dismissed under the client's id, `replaced_by` naming
   the vote that counts, and the voter's pending change of mind is withdrawn. A flipped vote
   waits; approving it undoes the effect of the voter's vote that was actually applied (its
   projection ran): reinforcement, a retraction it made, the confidence it moved on a run's
   cited memories, a procedure it rejected; then it applies its own (`projection.reversed`).
   An approved vote whose job has not run yet is closed by the newer one, so it never counts
   as well. Projections of one voter's votes on one target are serialised. A step clipped at
   a bound is undone by the full step, and a run's `mark_used` is not taken back.
   What is a vote is decided by who sent it and stored (`feedback.is_vote`), never by the
   `source` it claims: anything on a run, a memory or a procedure except the administrator's
   or the service's word, an automated word (`system`, `judge`) from a credential vouched for
   the agent, or an edit of a memory. An automated record never replaces a thumb, nor the
   reverse.
6. **A run's status is only as good as its credential.** No credential here is bound to one
   run: `agent_run_id` is named by the caller. A `system` verdict on the caller's own run is
   applied at once only when the credential may act as that agent (an issued key whose
   `may_act_as` allows it, or the platform or development key); one sent with an identity
   provider's token, which names a person, waits like a vote. A run-bound credential (issued
   by agent-runs per run) would close this fully; until then it is as strong as the key.
7. **Kept for reporting.** Pending and dismissed votes, replaced ones included, stay stored
   and queryable:
   `GET /v1/feedback?review=<state>` (with `group=target` too) and the target's own feedback
   list. `author_record` and the grouped counts leave replaced votes out; a replaced vote is the
   author changing their mind, not a verdict on them.

## Consequences

- A user's thumbs-down on their own memory no longer retracts it on the spot; it waits like any
  vote. A user who wants the memory gone forgets it (`DELETE`), and one who wants it to say
  something else corrects it with the new text.
- The queue holds at most one pending vote per voter per target, so its size is bounded by
  people and targets rather than by clicks.
- No new review state on the wire; `review.replaced_by` is a new optional field older SDKs
  ignore. Two new operations, one new list mode (`group=target`), and the review filter takes
  every state. Migration `0027_votes_replaced_by_subject` adds `feedback.review_replaced_by`
  and two partial indexes for the review reports (`ix_feedback_review`,
  `ix_feedback_review_target`).
- SDK: `ctx.feedback.pending(group="target", review=...)`, `approve_many`, `dismiss_many`.

> **Amendment, 2026-10.** The run's `mark_used` that decision 5 leaves in place no longer
> exists: the agent-tool pull log it wrote to, and the prefetch learning that read it, were
> removed (migration `0030_drop_dead_learning`). A confirmed run now moves only its cited
> memories' confidence, which a flipped vote undoes in full.
