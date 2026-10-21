# ADR 0042: The service accepts learned approval rules on its own

Date: 2026-10-21. Status: accepted. Amends [ADR 0028](0028-feedback-review.md) (a tool-call
decision's suggestion "only takes effect when it is accepted").

## Context

People's decisions on tool calls (`approve`, `reject`, `edit`) are counted per (agent, tool,
argument shape), and a pattern of at least five decisions suggests a rule: `auto_approve` at a
95% approval rate, `always_ask` at 50% or less. Until now a suggestion changed nothing until
someone called `POST /v1/tools/approval-suggestions/{id}/accept`, and no client surfaced the
listing, so the rules were learned and never used: an agent kept pausing for the same call a
person had approved dozens of times.

The owner decided that learned rules apply on their own; what needs a tenant admin's approval
stays what ADR 0028 and 0039 say it is, people's feedback verdicts.

## Decision

* The feedback projector accepts a pattern's rule as soon as the decision it projects completes
  the support for it (`modules.tools.approvals.auto_accept`), through the same composition into
  the tool's `approve_when` that the accept route uses.
* **The strictest tier waits for a person.** A tool whose risk is `irreversible` is never
  changed automatically, in either direction: its suggestions are listed and accepted by a
  person as before (and `auto_approve` is still never offered for it). A tool the catalog does
  not hold is not changed either: its risk is unknown.
* **Once.** The pattern records the rule and when it was accepted (`approval_patterns.
  auto_accepted`, `auto_accepted_at`, migration 0029). A rule a person revokes through the
  existing catalog update (`PUT /v1/tools/catalog` with the clause taken out of `approve_when`,
  or `""` to remove it) is not accepted again by later decisions; a person may still accept it
  with the route.
* **Logged and journaled.** Each acceptance is a structured log line
  (`approval_rule.auto_accepted`: tenant, agent, tool, the value-free argument shape, the rule,
  its support), the decision's projection names the resulting `approve_when`
  (`projection.approval_rule`, kept with the feedback record), and
  `GET /v1/tools/approval-suggestions/auto-accepted` lists every rule the service accepted,
  newest first, with whether it is still `in_effect`.

## Consequences

* A pause a person keeps answering the same way stops after the fifth consistent decision; one
  they keep rejecting becomes a rule that always asks.
* No new configuration: the thresholds are the ones that already chose the suggestions.
* The tenant's revision moves with each acceptance, so harnesses re-read the catalog (their
  conditional `GET /v1/tools`) and apply the rule within their refresh interval.
