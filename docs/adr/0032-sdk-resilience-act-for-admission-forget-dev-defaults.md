# ADR 0032: SDK resilience, act-for checks, admission, forget cascade, dev defaults

Date: 2026-10-04. Status: accepted. Amends 0021 (`may_act_as`), 0022 (SDK errors), 0009
(admission).

## Context

An audit of what an agent harness meets on its first day found seven gaps, each small, that
together made the service feel broken before it was used:

- **The SDK gave up inside a rate-limit window.** Backoff was 0.1-0.4 s plus 50 ms of jitter,
  so three retries were spent in well under a second of a 60-second window; `Retry-After` was
  ignored. `context`, `search`, `verify` and `tool_hints` are POSTs and were never retried at
  all, though they change nothing. A gateway's problem without a `code` raised a bare
  `MemoryError(code="INTERNAL")` whatever its status, so a 404 or a 503 behind a proxy was
  indistinguishable from a bug.
- **No circuit breaker.** During an outage every memory call of every agent turn paid the
  connect timeout and every retry before it could degrade.
- **Zero configuration was not.** `MemoryClient` needed its URL and key spelled out although the
  platform names them (`MEMORY_URL`, `TRELLIS_API_KEY`); a development key answered
  `tenant_id: null` from `GET /v1/keys/self`, so the harness refused to start without a tenant
  the operator had to know to pass, and every memory call without `X-Trellis-Tenant` was a 422.
- **Registering a model key failed on a laptop.** The harness registers `BIFROST_VIRTUAL_KEY` for
  each agent on its first run; with the `.env.example` defaults that was `503 Agent credential
  encryption is not configured`, which reads as an outage.
- **`may_act_as` checked half of itself.** Only `user:<id>` entries were compared with the
  request; `agent:<id>` entries were stored, reported by `GET /v1/keys/self`, and compared with
  nothing, so a key restricted to one agent acted as any other by naming it.
- **The admission gate was built and not wired** (`modules/memory/admission.py`), while
  `docs/api/memory.md` described it as running.
- **Forgetting left readers' caches stale.** Forgetting a memory already retracted what was
  derived from it (recursively, through `memory_dependencies`) and queued its de-indexing, but
  only the forgotten memory's own revisions moved: a dependent anchored elsewhere stayed in its
  readers' caches.

## Decision

1. **SDK retries what cannot duplicate anything, for as long as the service asks.** GETs, writes
   carrying an `Idempotency-Key`, and the read-only POSTs (`/v1/context`, `/v1/recall`,
   `/v1/verify`, `/v1/tools/hints`) retry on a retryable problem, a timeout or a dropped
   connection; a connection that never opened and a `429` (refused before any work) retry for
   every call. The wait is `Retry-After`, capped at 30 s, else full-jitter backoff
   (`uniform(0, min(8, 0.5 * 2^(n-1)))`). A write without a key that may have reached the
   service is still never resent, and no key is invented for it.
2. **A problem without a `code` is classed by its status**: 400/413/422 `ValidationError`, 401
   `AuthenticationError`, 403 `AuthorizationError`, 404 `NotFoundError`, 409 `ConflictError`,
   429 `RateLimitedError`, 502/503 `DependencyUnavailableError`, 504 `TimeoutError` - the last
   four retryable - and anything else the base class, not retryable. Every error carries
   `retry_after`. The service's own `code` and `retryable` still win.
3. **A per-client circuit breaker** (`trellis.memory.breaker`, modelled on bifrost-sdk's): after
   5 calls in a row fail for want of the service (no response, or a 5xx; counted per call, not
   per attempt) it raises `CircuitOpenError` - a retryable `DependencyUnavailableError` with
   `retry_after` - without sending, for 30 s; then one call probes while the rest keep failing
   fast, and the probe's outcome closes or reopens it. A 4xx or a 429 never counts: a 429 is the
   service asking for less, and counting it turns backpressure into an outage.
   `circuit_failure_threshold=0` disables it.
4. **`MemoryClient()` is the whole configuration** where the platform's environment is set:
   `base_url` defaults to `$MEMORY_URL` (else `http://localhost:8080`) and `api_key` to
   `$TRELLIS_API_KEY`. The pool keeps idle connections 30 s; connecting is bounded by 5 s of the
   timeout; `context()` and `search()` take a per-call `timeout`.
5. **A development key acts in the development tenant**, `authentication.trusted_dev_tenant`
   (`MEMORY__AUTHENTICATION__TRUSTED_DEV_TENANT`, default `default`, the harness's local
   tenant), when a request names none: on every route that builds a context, on tenant
   administration (its row is created the first time it is administered), and in
   `GET /v1/keys/self`. `X-Trellis-Tenant` still names another. The development stack also
   authenticates the keys it issued, so a laptop exercises the service keys a deployment uses
   (`POST /v1/keys` with the development key); `jwt` and `api_key` modes are unchanged.
6. **A restricted key acts for the principals it lists and no others**: the request's user
   against `user:<id>`, its `agent_id` against `agent:<id>`, each a `403` naming the field. A
   request naming neither acts as the key itself, the anonymous service principal, which holds
   no grant on any user's or agent's memories. `*` lifts the restriction.
7. **The admission gate is a tenant's switch, off by default** (`Tenant.admission_gate`,
   migration `0023_tenant_admission_gate`, `POST`/`PATCH /v1/admin/tenants`). The observation
   pipeline always carries the gate and consults it for tenants that turned it on: an admitted
   memory records the decision in `system_metadata.admission`, a rejected candidate is `IGNORE`
   with the gate's reasons, a deferred one waits in working memory until it is said again.
   `remember` is never gated. It is off because the retrieval gates were measured with every
   candidate kept - the verbatim turn alone lifts LoCoMo's retrieval ceiling from 0.098 to
   0.685 (`config/constants.py`, `MemoryIntelligenceSettings.keep_verbatim_turns`) - so a default-on
   gate would change them without a measurement saying it should.
8. **Forgetting retracts everything derived from the forgotten memory, and the dependents leave
   readers' caches in the same commit.** The unit of work bumps the revisions of every dependent
   retracted in the transaction when it commits (forget, supersede, expiry and archive alike).
   A derived memory with other live sources is retracted too: it still says what the forgotten
   memory said. Its other sources stay, and the next reflection pass writes an insight from what
   is left. Memories derived only from other memories are untouched.
9. **Dev and test derive an envelope key when none is configured.** `envelope_settings` returns
   the operator's keyring when one is set and, in `dev`/`test` only, otherwise a key derived at
   startup from a fixed label and the database URL - never stored, the same in every process
   (API workers, the background worker that decrypts keys for jobs, a restart) - with an
   `agent_credentials.development_key` warning. It protects nothing, which is why staging and
   prod never get one and keep refusing registration.

## Consequences

- A harness during a memory outage degrades in microseconds per call after five failures, and
  rides out a rate-limit window instead of failing inside it; `exc.retryable` is right on every
  class, which agent-contracts' `AgentError.of` reads.
- A laptop runs the harness with `MEMORY_URL` and `TRELLIS_API_KEY` (or nothing, against the
  local stack) and no tenant anywhere; agent model keys register on the first run.
- A key whose `may_act_as` names agents now refuses requests naming other agents. A restricted
  key used for an agent of a listed user needs that agent listed too.
- Turning the admission gate on for a tenant changes what its extracted memories are; measure
  that tenant's retrieval before and after.
- Derived memories can disappear on a forget even though some of their sources live on, until
  reflection re-derives them.
- Rejected alternatives: a random per-process development envelope key (each API worker and
  the background worker would encrypt with a different key, and a restart would orphan every
  registered key); keeping a derived memory that has other live sources (it would keep serving
  the forgotten content); a default-on admission gate (it changes the measured retrieval
  without a measurement).
