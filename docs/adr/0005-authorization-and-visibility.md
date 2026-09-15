# ADR 0005: OpenFGA relationships + visibility keys for store-side filtering

**Status:** accepted · **Date:** 2026-09-14

## Context
Every retrieval must be scope-filtered *before* any model sees data, with zero cross-tenant,
cross-user or private-agent leakage. OpenFGA answers "may P read object X?" but a vector or
BM25 query returns thousands of candidates; checking each one after retrieval is both slow
and the exact "retrieve globally then filter in memory" anti-pattern the spec forbids.

## Decision
1. **OpenFGA** (`deploy/openfga/model.fga`) holds relationships for tenant, workspace,
   group, user, agent, thread, work, document and memory. Object ids are tenant-prefixed
   (`thread:acme/thr_1`) so tenancy is structural. Tuples are written when objects are
   created (`AuthorizationService.grant_*`).
2. An **in-memory ReBAC provider** evaluates the same model with Zanzibar semantics for
   tests and single-process dev. A unit test parses the `.fga` DSL and asserts the Python
   model is identical; a Docker-marked contract test runs the same golden checks against a
   real OpenFGA server.
3. **Visibility keys.** At write time each memory/chunk/message gets `visibility_keys`
   derived from its visibility + anchors (`user:acme/u1`, `thread:acme/thr1`,
   `principal:acme/agent:research`, `tenant:acme`, `global:acme`, ...). At read time the
   principal's `AuthorizedScope` (OpenFGA `list_objects`, bounded, cached by revision) is
   turned into the set of audience keys it may read. The store-side filter is
   `tenant_id == T AND visibility_keys ∩ allowed ≠ ∅` — one `must_any` clause in Qdrant and
   one array-overlap predicate in PostgreSQL. No candidate outside the filter is ever
   materialised.
4. When `list_objects` would exceed `max_listed_objects`, the scope is marked `truncated`
   and retrieval falls back to bounded per-object `batch_check` calls.
5. The **calling service** is authenticated (`trusted_dev` | `jwt` | `gcp_iam` | `mtls`);
   only then are the trusted context headers (`X-Memory-Tenant/-Workspace/-User/-Groups`)
   honored. Body-supplied security fields must match the headers exactly or the request is
   rejected; `custom_metadata` may not contain reserved keys.

## Consequences
- PRIVATE means exactly one principal: an agent acting for a user sees the user's
  USER-level memories (it inherits the user's OpenFGA access) but not the user's PRIVATE
  ones, and vice versa. Tested in `tests/security`.
- `tests/security/test_isolation.py` cross-checks the specification against an independent
  oracle with property-based generation; it is part of `make security-test` and the release gate.
- Group membership comes from OpenFGA and, when `trust_header_groups` is on, from the
  authenticated upstream's `X-Memory-Groups` header.
