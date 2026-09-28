# ADR 0021: Tenants, API keys and workspaces are the service's own

**Status:** accepted · **Date:** 2026-09-28 · **Amends:** ADR 0005 (reinstates `WORKSPACE`)

## Context
Several teams in one company will run agents against one deployment, each for its own
customers. The service authenticated the *calling service* (a static development key or an
external JWT) and then trusted `X-Memory-Tenant`; a credential reached every tenant by
changing a header unless a JWT issuer stamped a tenant claim. Nothing onboarded a tenant,
nothing issued a key, and no grant was ever revoked (`PRODUCT_DECISIONS.md`, §4). The
OpenFGA model already had `workspace` and `group` types, withdrawn from the visibility
ladder only because nothing wrote their membership.

## Decision
- A third authentication mode, `api_key`: keys the service issues and verifies against a
  stored SHA-256 (`domain/tenancy.py`, `modules/auth/keys.py`). A key names its tenant and
  optionally a workspace; the tenant is derived from the key and a header may only agree with
  it. One configured secret, `authentication.bootstrap_admin_key`, is the platform operator:
  it onboards tenants (`POST /v1/admin/tenants`, which returns the tenant's first admin key
  once) and acts for no tenant. That secret is the whole configuration a shared deployment
  needs beyond store URLs.
- Roles are `admin` (manages one tenant's keys, workspaces and groups) and `service` (acts
  for that tenant's users and agents - what a harness holds). Role gates are claims on the
  credential, checked in `api/deps.py`; data access stays with OpenFGA and visibility keys.
- **`WORKSPACE`** returns to the visibility ladder as a team's shared audience
  (`workspace:<tenant>/<id>`), with a `WORKSPACE` scope level to anchor team knowledge.
  Membership (`user:`, `agent:`, `group:`) is written as rows and as tuples in one unit of
  work; `viewer` is the relation resolved into `AuthorizedScope.workspace_ids`. A request
  naming a workspace reads that team; one naming none reads every team the caller is in -
  the same rule threads follow.
- **Writing into a team takes membership.** A workspace id is a caller-supplied anchor, so
  every path that mints a WORKSPACE audience (an observation, a message hint, a tool
  record: `modules/tenancy/gate.py` is their one home), a document ingested under a
  workspace and a thread opened in one are refused unless the caller is a `member` (admins
  compute to it; viewers read only), and WORKSPACE memory needs the team to exist. A
  workspace with no team row is a bare anchor: it grants nothing, the `workspace` tuples are
  not written for it, and an id already in use as an anchor by a thread or a document cannot
  become a team (409), so nothing labelled before teams existed is ever adopted by one.
  Callers from before teams existed are untouched. The model's rewrites then mean what they
  say: members and viewers read the documents shared with the team (`visibility=WORKSPACE`;
  a document uploaded inside a team under a narrower audience keeps that audience), a
  workspace admin reads and writes the team's threads. Workspace `admin` is a user
  relation.
- **Keys and roles.** `admin` keys are tenant-wide and cannot be bound to a workspace (a
  binding on a key that creates workspaces and issues keys would be a name only); `service`
  keys may be. `platform` is never issued, and a JWT that says `role: platform` is not the
  platform: only the bootstrap secret is. The bootstrap secret onboards tenants and never
  reads or writes memory, but it administers every tenant - it can issue a tenant's keys -
  so it is root: at least 32 characters in deployed environments, and unset once onboarding
  is done. A tenant's own administrators are suspended with it; the platform, which resumes
  it, is not. Administration is an `api_key`-mode power: in `jwt` mode the issuer's token is
  the calling service of a per-customer deployment and administers nothing, whatever its
  `role` claim says.
- **Revocation exists.** Removing a member or a group user deletes the tuples and bumps the
  membership revision, so the next request is denied. Revoking a key or suspending a tenant
  replaces the verifier's cache entries with tombstones rather than deleting them: a reader
  that fetched the row a moment before the change cannot write the stale record back over
  the deletion (entries are written set-if-absent), so the revocation holds from the next
  request on every instance. Unknown key ids are cached as missing for a few seconds, and
  each instance reads the store for at most `UNKNOWN_IDS_PER_MINUTE` ids it does not
  recognise (the registry holds every live key id), so a flood of distinct well-formed
  garbage tokens costs the store a bounded amount; the price is that a key issued on
  another instance during such a flood may be refused (logged as
  `api_key.unknown_id_budget_refused`) until the registry's next refresh, a minute at most.
  A tenant holds at most `MAX_KEYS_PER_TENANT` live keys.
- **Administration is idempotent on request and never shows a secret twice.** Onboarding,
  key issuance, workspace and group creation honour `Idempotency-Key`; without it every call
  is its own resource (a generated id has no natural key, and two keys may share a name), and
  a named id that exists is a 409 even when two requests race - the primary key decides. The
  issued token goes out on the first response only: the idempotency record keeps
  `token: null`, so a retried request gets the same record with `Idempotent-Replayed: true`
  and not a second look at the secret.
- **Suspension bites on the next request.** The verifier caches the tenant's status with
  the key; `PATCH /v1/admin/tenants/{id}` with a status change invalidates every key of that
  tenant, and a suspended tenant's key answers 403 `tenant is suspended` (a genuine
  credential, a policy refusal) rather than 401. Resuming works the same way.
- **Identifiers are never reused.** A deleted workspace or group keeps its rows for the
  record; creating another with the same id is 409, so audit entries and memory anchors
  keep their meaning. Deleting a workspace also revokes every key bound to it; deleting a
  group removes it from every workspace it was admitted to.
- **An admin key names its tenant.** On administration routes a header naming another
  tenant is 403, never a silent redirect; only the platform key administers by header.
- Per-tenant `retention_days` drives a daily sweep that forgets canonical memories through
  the same soft delete a caller's `DELETE` uses, by creation age (retention is a promise
  about how long something is kept, not how recently it was useful), draining a backlog in
  bounded batches up to a per-run cap so one tenant never holds a long transaction or skips
  the next (conversation rows, documents and observations are not yet covered). Per-tenant
  `rate_limit_per_minute` and suspension live in an in-process **tenant registry** the
  limiter and the context builder read without a store round trip; the API process primes
  it at start, refreshes it every minute itself (the worker has no middleware), and the
  instance that made a change applies it at once. The registry also maps the keys of
  overriding tenants to their tenant, since a key-holding caller sends no tenant header, and
  a suspension therefore stops `jwt` and `trusted_dev` callers too, not only keys. A read
  audit (`memory_reads`) records the authenticated credential, the principal it acted for,
  which records were served and under which scope, written in batches off the request path
  with a stated one-second loss window and purged after `TASKS.read_audit_retention_days`.

## Consequences
- `tests/security/test_isolation.py` now exercises `WORKSPACE`; its oracle had stated the
  rule for a year without an implementation to check. `tests/agent/` drives the whole
  lifecycle through the SDK alone: onboard, key, workspace, share, revoke, read-after-revoke,
  and the edges (`test_platform_edge_cases.py`): replayed issuance, wrong role, wrong tenant,
  suspension, id reuse, bound keys outliving their workspace, groups outliving their
  workspaces, idempotent revocation, audit paging, no bootstrap secret.
- The environment surface grows by one field (`bootstrap_admin_key`).
- **Rollout.** Apply migration 0014 before the first new-code instance starts: the request
  path of existing endpoints now reads the new tables (the workspace gate, the read audit),
  so there is no safe mixed window; the compose graph enforces the order. The partial index
  on `memories` is built without `CONCURRENTLY`, like 0009 and 0010 before it, which holds
  writes to `memories` for the build. The OpenFGA model is rolled forward by the provider
  itself: when the store's latest model differs from this build's, the build's model is
  written and used (models are immutable and append-only; any model in the history with
  the same meaning is reused, so a fleet running two builds does not append one per
  restart), so an upgraded deployment gets the relations its code writes. A pinned
  `openfga_model_id` that is not this build's model stops the service at start rather than
  failing the first tuple the pinned model does not know.
- Authorization tuples are written inside the request's unit of work but are not part of
  the database transaction: a commit that fails after a grant can leave a tuple with no row
  behind it. Revocation walks rows, so such a phantom would keep its access; a repair sweep
  that reconciles tuples against rows is the planned follow-up, not something this ADR
  claims. Membership changes of one team, and key issuance of one tenant, are serialised
  with an advisory lock so two administrators acting at once cannot leave the union of
  their changes in the tuples or overshoot the key cap.
- Quotas are per credential of a tenant (the bucket is `tenant + key`), so a tenant with N
  keys has N times its quota; `rate_limit_per_minute = 0` disables the limiter for the
  tenant (to stop serving it, suspend it). The limiter runs before authentication and names
  the effective limit on every response, including refused ones.
- `trusted_dev` keys remain omnipotent on a laptop and remain refused in deployed
  environments; `jwt` deployments bind the tenant through `tenant_claim` as before and have
  no administration surface.
