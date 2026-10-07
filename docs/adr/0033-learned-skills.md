# ADR 0033: A learned procedure becomes a skill only when a person publishes it

Date: 2026-10-06. Status: superseded by [ADR 0034](0034-learned-skills-in-context.md).

## Context

The learning job (ADR 0018) keeps, per task pattern, the steps that worked and their track
record, and offers an **active** procedure to agents as a plan in tool hints and context. That
reaches only agents that ask the memory service for hints. Teams also keep **Agent Skills**
(`SKILL.md`) in a folder or in the Bifrost gateway's skills repository, which every framework
loads natively and the harness pins per run. A procedure that keeps working is exactly what a
skill is, but nothing turned one into the other, and a skill written by a model with no one
looking at it is a prompt injected into every later run.

## Decision

1. **A draft, never an automatic skill.** An active procedure not yet decided for its current
   steps is listed to the tenant's administrator as a draft (`GET /v1/tools/skill-drafts`):
   a `SKILL.md` rendered from the procedure only (title, task pattern, strategy, steps, failure
   modes). No new job and no new model call: the draft is read on request.
2. **The administrator publishes or dismisses.** Publishing writes it where agents already load
   skills: `SKILLS_DIR` when set, else the gateway's repository (`BIFROST_URL`, with
   `BIFROST_ADMIN_TOKEN` when its management API needs one), through bifrost-sdk's
   `Admin.skills`. Nothing configured is `503`.
3. **Versions are the store's.** `1.0.0`, then the next minor. Rollback is the store's own
   (the gateway's `shift_version`; the folder's own history). The harness pins the version a run
   loaded.
4. **The decision is kept with the steps** (`procedures.skill`, migration `0025`). Only the
   decision endpoints write it; a re-mine never does. New steps bring the draft back (`changed`
   when it was published, under the same name).
5. **Ownership.** A published skill's metadata names its tenant and procedure. A name that
   exists without this tenant's mark - a person's skill, another tenant's - is refused (`409`).

## Consequences

* An agent gets a learned skill with no change to how it loads skills
  (`skills=["refund-order"]`, `skills_dir`, or the framework's own loader).
* The administrator reviews the text before any agent reads it.
* Multi-tenant gateways share one skills namespace; the ownership mark keeps one tenant from
  overwriting another's skill, and names can be chosen at publish time.
