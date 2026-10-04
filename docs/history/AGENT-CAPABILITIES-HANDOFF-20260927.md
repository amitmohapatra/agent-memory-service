# Agent capabilities and validation handoff — September 27, 2026

## Checkout and isolation

Active implementation: `/Users/ricky/usage_data/ams-agent-capabilities`, branch
`codex/agent-capabilities-20260927`, base `f988c26`. It contains the retrieval/integration
changes copied from `ams-hindsight-integration` plus the credential and standing-brief stack.
Changes are uncommitted, not merged or deployed; the branch HEAD alone does not contain
this work. Review this checkout and its new files, not just the branch commit. The separate integration checkout remained frozen
while its real-encoder benchmark arms ran; its recorded source hashes identify that tree.

Migrations 0012 (encrypted agent credentials) and 0013 (standing briefs) are new and were
applied only to these isolated databases:

- `memory_agent_cap_20260927`
- `memory_agent_cap_failure_20260927`
- `memory_agent_cap_queue_20260927`

Do not run this checkout's tests or migrations against a shared benchmark database.
Migrations 0009–0011 are inherited from the write-path base and remain corpus-changing.
The new migrations do not transform the source-memory corpus.

## Implemented

1. **Agent-owned Bifrost keys.** GET/PUT/DELETE `/v1/agents/model-key` use the authenticated
   tenant and user-bound agent principal. AES-256-GCM encrypts each key with a random nonce
   and authenticated tenant/principal/key-version metadata. HTTP responses contain status
   only. Idempotency records contain a key digest, not plaintext or ciphertext.
2. **Rotation and revocation.** Each write atomically advances a revision. Revocation keeps
   a tombstone; it cannot silently fall back to the operator key. An absent policy is also
   checked as a state: creating the first key/tombstone cancels further operator-key retries.
   The HTTP request hook validates the revision before every SDK transport attempt; a second
   check rejects responses completed after an observed rotation. Already-sent requests cannot
   be recalled. This does not promise atomic revocation of an entire remote network operation.
3. **Background billing identity.** Observation extraction, reflection, document parsing,
   indexing and graph enrichment bind the recorded owner. Jobs carry identity, not secrets;
   they resolve the current credential at execution time. Concurrent calls do not mutate a
   shared client's default key. Context, recall, verify and graph query default to
   `use_llm=false`, independently of configured ingestion model uses.
4. **MCP exclusion.** Native model calls send explicit empty MCP client/tool selections,
   `tool_choice=none`, no tool definitions and disabled gateway content logging. Responses
   requesting tools are rejected. Fixtures verify the wire contract; a live gateway deployment
   still needs to enforce that contract. No paid gateway calls were made.
5. **Hindsight reuse boundary.** The pinned extraction-preview SDK remains in the integration.
   The server SDK cannot carry a request-specific model VK. All agent identities use
   the existing native Bifrost source-span extractor, selected before calling any model.
   This also covers agents whose first key is registered/revoked while a call is pending.
   They do not spend a Hindsight server operator key or mirror memories into external banks.
6. **Standing briefs.** Mental models and knowledge pages share `BriefService`: a persisted
   question/title, exact execution-scope ownership, a bounded background evidence refresh,
   and a model-free stored-content read. Native mode returns cited source excerpts; assisted
   mode produces explicitly marked generated text with validated citation identifiers.
   This is not semantic proof that every generated claim is true.
7. **Brief lifecycle.** Create/update use the transactional outbox. A periodic indexed,
   `SKIP LOCKED` scan claims at most 100 due definitions, with jobs on the summary queue.
   Output is capped at 20 sources of 800 characters each. Source and authorization revisions,
   source expiry and temporal validity prevent old content being served as ready. Definition
   generations prevent in-flight old refreshes overwriting updates; deletion cannot be undone
   by a finishing job. Unchanged synthesis input avoids another model call merely because
   evidence retrieval timestamps or scores changed. Model/prompt fingerprints trigger fresh
   synthesis on the next refresh after configuration changes. Lists fetch definitions, not
   output blobs.
8. **SDK and schema.** Python SDK exposes key management and `ctx.briefs` create/update/get/list/
   delete. GET/DELETE preserve session lineage too. OpenAPI is regenerated from the app.

The retained retrieval stack includes Unicode sparse indexing, conservative semantic context
caching, revision-aware cache replay, qualified graph deduplication, provenance boundaries,
source chronology, CPU executor scheduling and pinned model profiles. See the integration
handoff for those details and the multilingual evaluation report for measured model results.

## Configure and use

Configure `MEMORY__AGENT_CREDENTIALS__ACTIVE_KEY_ID` and
`MEMORY__AGENT_CREDENTIALS__ENCRYPTION_KEYS` through the deployment secret manager. Each
keyring value must encode 32 random bytes in base64. Keep old key versions while stored
credentials still use them. An empty keyring allows native service operation but rejects
credential registration. Settings snapshots redact key values.

Automatic mode now activates assistance for a registered key without per-agent model/use
configuration. The deployment still owns its gateway URL and credential-encryption keyring.
`enabled=false` remains a hard prohibition; `enabled=true` retains explicit use selection.
Auto discovers recognized non-Chinese text model families through authenticated `/models`,
with a 5-minute cache bound to credential owner/revision. Opaque aliases are not guessed.
The default auto uses are contextual extraction, reflection, briefs and query expansion;
read requests still require `use_llm=true`. Native briefs/reads require no model credit.
Existing deployments that explicitly set false remain model-free. Offline tests must exclude
`bifrost`, `models` and `docker` markers unless specifically running their isolated fixtures.

```python
from trellis.memory import BriefSpec, MemoryClient

async with MemoryClient(base_url, api_key=service_key) as memory:
    # Bind a durable agent identity without a run for a brief intended across its runs.
    # A run/thread/session-bound brief is readable only in that same scope.
    ctx = memory.bind(tenant_id=tenant_id, user_id=user_id, agent_id="research")
    await ctx.set_model_key(virtual_key, idempotency_key="agent-key-v1")
    brief = await ctx.briefs.create(
        BriefSpec(
            title="Project status",
            question="Which projects are active?",
            kind="knowledge_page",
            use_llm=False,
        ),
        idempotency_key="project-status-v1",
    )
    current = await ctx.briefs.get(brief.brief_id)
    # pending: refresh has not completed; stale: old content is withheld;
    # ready: current.output contains stored evidence or explicitly marked synthesis.
    await ctx.revoke_model_key()
```

Refresh is scheduled at the configured interval, from 60 seconds to 24 hours. A revision
change hides output immediately on the next read; it does not promise immediate background
refresh. Updating the definition queues a refresh immediately. Native mode is extractive,
not an LLM-equivalent reasoner. A page is a maintained titled brief, not a full wiki with
folders, revision history, filesystem projection or collaborative editing.

## Validation

No paid model calls. Credential and synthesis HTTP responses are fixtures.

- First brief/native lifecycle pass: 10 passed, including complexity ratchet.
- Expanded capability checks after review repairs: **43 passed**, covering credentials,
  brief source deletion/expiry, definition races, citation rejection, scope isolation,
  session round trips, mocked model-use counts, SDK and OpenAPI contracts.
- Existing credential/native tests before briefs: 22 passed after retry repair.
- Full regression: **1,412 passed, 19 skipped, 12 deselected**, plus one stale
  environment-field assertion (47 versus the new 49) in 851.42 seconds. That assertion
  was corrected; the focused final rerun passed **87 tests, one skip** in 68.87 seconds,
  including real isolated cache/search contracts, all credential/brief integration cases,
  architecture and settings. Five cipher/settings-redaction tests also passed.
- Pyright: **0 errors**, 21 missing-import warnings for optional cloud/model packages in
  this host environment. Ruff checks pass; OpenAPI matches current code (26 paths).
- Final cache-profile repair: **96 passed, one skip**. Assisted context caches now bind
  gateway, strong/fast models, routing, output budget and the operator credential through a
  startup-computed digest. Unused model configuration does not fragment native caches.
- Final brief model/prompt-profile repair: **44 passed, one skip**, including SDK/OpenAPI
  and architecture checks. The fingerprint is stored in the existing JSON output, so legacy
  output without it is refreshed rather than silently reused across model/prompt changes.
  Four complexity/length ratchets passed afterward without raising their limits.
- Native review completed three repair passes and found real revocation/queue races,
  session scope and temporal validity issues. They were repaired and regression tested.
  The review did **not** reach a fresh zero-edit final-snapshot verdict; do not label it
  converged. No paid external reviewer or live-provider certification was used.

Validation logs are under `.bench_data/`: `full-capability-regression.log`,
`isolated-final-repairs.log`, `pyright-capabilities-final.log`, and
`ruff-capability-final.log`. Generated gate outputs were preserved in
`.bench_data/preserved-generated-gates/`; historical checked-in gate artifacts were restored
so this work does not silently rewrite earlier benchmark provenance.
The full-suite count predates the two final profile repairs; their focused runs are in
`cache-model-profile.log` and `brief-profile-final.log`. Current source hashes and the
validation matrix are recorded in `benchmark/results/capability_validation_20260927.json`.

`benchmark/results/brief_reads_20260927.json` contains 500 sequential and 500 eight-way
concurrent reads of a small native standing brief, after warmup. Retrieval and synthesis
were disabled after setup; every response remained ready with sources. Observed p50/p95/p99
were **22.03/37.45/66.33 ms** sequentially and **126.17/211.40/336.96 ms** concurrently.
This used real PostgreSQL, ASGI transport and in-process authorization/cache, with rate
limiting disabled and other host work running. It is not production/network/RAG latency.
The harness accepts an explicit already-migrated database, creates a unique tenant and
never truncates tables.

Reproduce against the dedicated databases:

```bash
cd /Users/ricky/usage_data/ams-agent-capabilities
source .bench_data/test-env.sh
/Users/ricky/usage_data/agent-memory-service/.venv/bin/python -m pytest tests sdk/python/tests \
  -m 'not docker and not models and not bifrost' -o faulthandler_timeout=0 \
  --timeout-method=thread
```

The macOS test interpreter needs `faulthandler_timeout=0` because cancelling its fault-dump
thread previously hung at teardown. Per-test timeouts remain enabled. Optional model,
Docker and live gateway suites are separately reported, not silently counted as passed.
The standard verify script's shared compose/database defaults were not used.

## Remaining boundaries

- No new LoCoMo **answer accuracy** result: no fresh paid reader/judge predictions. Full source
  retrieval and multilingual paragraph retrieval are separate metrics.
- No proof of production p99 or concurrent deployment throughput; real CPU measurements have
  explicit host/load caveats. Neural reranking was measured and rejected for the normal path.
- This is not full Hindsight API parity: deep reflect agent loops, webhook subscriptions,
  bank-template catalog and complete wiki features are not implemented.
- The multilingual evaluation covers retrieval in 12 languages, not every language, NLI,
  extraction, or end-to-end answer quality.
- All-layer optimality is not a proved property. Existing architecture/complexity gates remain
  in force; their budgets were not raised. The legacy complex grammar tail still exists.
- A cross-model Claude CLI review was not run under the no-credit constraint. Native source
  review and regression tests are documented separately from live provider certification.

## Contract test isolation repair

The existing cache/search contract fixtures used their adapters' shared localhost defaults.
The first full pass therefore created/deleted random `contract_*` Qdrant collections and
used random `contract:*` Redis keys on shared services. It did not reset benchmark collections
or migrate shared databases. This was an isolation gap in the harness, not an intended target.
Fixtures now require explicit `MEMORY_TEST_QDRANT_URL`, `MEMORY_TEST_QDRANT_GRPC_PORT` and
`MEMORY_TEST_CACHE_URL`, otherwise they skip server tests. The dedicated rerun uses Qdrant
16333/16334 and `memory-agent-cap-cache-20260927` (Redis at 16379, no persistence).

The environment-field ratchet increased from 47 to 49 solely for the two envelope-key
configuration fields. The code-complexity/length ratchets were not raised.

Full retrieval results are in `LOCOMO-OFFLINE-COMPARISON-20260927.md`: E5 source recall@50
79.70%, multi-hop source recall@50 61.92%, observed context-builder p99 489.84 ms with
host/load caveats. These do not establish answer accuracy or a production latency SLO.

A stale brief is withheld by the API; its previous stored output is replaced on refresh.
That read-time guarantee is not immediate physical erasure of every copied source byte.
The dedicated Redis/Qdrant containers and isolated databases are retained for reproduction.


## Reviewable patch stacks

`.bench_data/review-patches/01-retrieval-integration.patch` contains the frozen integration
changes relative to `f988c26`. `02-agent-capabilities.patch` adds the agent credentials,
briefs, final cache-profile repairs and their tests/docs. Apply them in that order to a
clean checkout at the named base. Their manifest records a clean-apply and byte-comparison
check. These are review artifacts, not commits or deployment authorization. Benchmark result
JSON, ignored dependencies, weights, credentials and runtime state are excluded from patches;
retain the named result files/worktrees separately.

For a fresh isolated test checkout, the local test-env file contains these explicit targets:

```bash
export MEMORY_TEST_DATABASE_URL=postgresql+psycopg://memory:memory@localhost:5432/memory_agent_cap_20260927
export MEMORY_TEST_APP_DB=memory_agent_cap_failure_20260927
export MEMORY_TEST_QUEUE_DB=memory_agent_cap_queue_20260927
export MEMORY_TEST_ADMIN_URL=postgresql://memory:memory@localhost:5432/postgres
export MEMORY_TEST_QDRANT_URL=http://localhost:16333
export MEMORY_TEST_QDRANT_GRPC_PORT=16334
export MEMORY_TEST_CACHE_URL=redis://localhost:16379/0
export PYTHONPATH=src:.sdk-test-deps:sdk/python/src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The names above identify this session's isolated local services. Other independent arms must
use their own database names and collections; do not share or truncate these during a run.
Install the lockfile dependencies in a fresh checkout; `.sdk-test-deps` and the vendor link
are local test-environment conveniences, not shipped package contents.


## Final model and accuracy decision

All three LoCoMo source-retrieval arms, all twelve-language paragraph screens and both
full SciFact encoder/fusion arms completed. No benchmark process is left running.
The current Granite default's source recall@50 is **77.42%**, complete@50 **70.96%**,
with observed context-builder p99 **327.78 ms** under the documented host/stand-in limits.
The E5 challenger reached **79.70%** source recall@50, complete@50 **73.18%**, and observed
p99 **489.84 ms**; it is not the shipped default. These are not generated-answer scores.

Full SciFact (5,183 documents, 300 questions) prevents an indiscriminate model replacement:
Granite hybrid nDCG@10/recall@10 is **0.7409/0.8912**; E5 is **0.7055/0.8482**. Keep the
existing default and the neural reranker off. The small Ettin screen improved ranking but
cost seconds on the measured CPU, so it did not earn the normal read path. Do not describe
this as proof that every possible reranker is too slow.

E5 remains the leading measured multilingual/conversational challenger. A future model
replacement must preserve document quality, calibrate abstention/deduplication, reproduce
its quantized bootstrap and rebuild isolated indices. The present twelve-language result
is paragraph retrieval; English NLI and full multilingual answers remain unvalidated.
See `SCIFACT-CPU-EVALUATION-20260927.md`, `MULTILINGUAL-CPU-EVALUATION-20260927.md`, and
`LOCOMO-OFFLINE-COMPARISON-20260927.md` for the full tables, category regressions and hashes.
No 85% or 90% answer-accuracy claim is earned by this work.
