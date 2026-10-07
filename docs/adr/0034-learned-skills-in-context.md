# ADR 0034: An agent learns skills from all its users and is offered them in its context

Date: 2026-10-07. Status: accepted. Supersedes ADR 0033.

## Context

ADR 0033 turned an active procedure into an Agent Skill only when the tenant's administrator
published it, to `SKILLS_DIR` or the gateway's skills repository. In use that was the wrong
default on three counts:

* **A person in the loop for every skill.** Nothing reached an agent until someone reviewed a
  draft, so what the runs had already proved sat unused, and a second endpoint, a store, two
  settings and a version scheme existed only to move the text somewhere agents read it.
* **Learned per (user, agent).** Tool records are the agent's own (`PRIVATE`), and an agent's
  principal is bound to the user it runs for (`agent:<user>/<agent>`: the agent id is not
  authenticated). So an agent with many users learned once per user, and none of them reached
  the support a skill needs.
* **Two copies of one thing.** The context already offered active procedures ("Procedures
  that worked for this task"); a published skill repeated them in another form, and an agent
  with fewer than five tools was offered neither, because the harness sent no tools to ask
  with.

## Decision

1. **An agent learns from all its users.** An agent's own records, whichever user it ran for,
   are learned under one audience, `agent:<tenant>/<agent>` (`learning.audience_of`). Records
   shared wider (a group, a workspace) keep their own audience. Migration `0026` re-learns the
   agent records learned per user.
2. **An active procedure is the agent's learned skill, offered in its context.** The context
   section is "Learned skills for this task": each one in full (steps, what fixed a failing
   step, track record), up to three, matched to the task. `tool_search` returns the best as its
   plan. Nothing is published, written to a store or approved; the bar stays the procedure's
   (two runs, 60% success), and a skill that stops working is retired by the learning job.
3. **Who is offered one.** A learned skill reaches the agent's other users only once two
   different users produced it; before that only the user who did (`users`, `sole_user`), so
   one person's wording of a task never reaches another. Never another agent or tenant.
4. **The agent's own skills stay the person's.** A learned skill whose every run opened one of
   the agent's written skills (`load_skill`, recorded like any call) is shown as what it adds
   to that skill (`with_skill`), never as a copy; nothing edits a written skill.
5. **The administrator sees and dismisses.** `GET /v1/skills` lists what each agent learned;
   `POST /v1/skills/{id}/dismiss` stops one (the procedure's `rejected` status, held until its
   steps change). The draft, publish and store code, the `procedures.skill` column, and the
   `SKILLS_DIR` and `BIFROST_ADMIN_TOKEN` settings are removed.
6. **Learned skills for any toolbox, hints from five tools.** The harness always sends the
   agent's tool names; the memory service offers the learned skills to any agent with tools
   and computes tool hints (read after retrieval) only for five or more, where they narrow
   anything (`ToolsRequest.hinted`). The threshold lives in one place.

## Consequences

* An agent with many users learns in days what it learned per user before, and every
  framework gets it with no loader: it is in the context.
* No setting, no store and no review queue to run. A team that wants a learned skill as a
  written one copies its text into its own skills.
* What is offered is decided by data (the agent, its users, the runs), so it cannot be
  misconfigured; the administrator's one lever is dismissal.
* `procedures` in `format=full` context is now `skills` (with `name`, `with_skill`, `fixes`),
  and `tool_search`'s plan has the same shape: one view of a learned skill everywhere.
