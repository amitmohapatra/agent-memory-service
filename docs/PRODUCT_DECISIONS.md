# What to keep, what to cut, and why

A decision document for an enterprise agent-memory harness. Opinionated on purpose: every
dimension kept is one that must be verified forever, so the list is short and the reasons
are stated.

## 1. Reframing the goal: 100% accurate is achievable, but not as recall

No retrieval system reaches 100% recall, and any claim of it is measuring its own fixtures —
which is exactly the trap this repository is in today, with every quality gate reading 1.0
against corpora we wrote ourselves. Published results are the honest reference point:
Anthropic's Contextual Retrieval reports ~49% fewer failed retrievals from contextual
embeddings plus BM25, and ~67% with reranking added. Large improvements. Not 100%.

**The reachable goal is 100% grounded-or-abstained.** An enterprise answer is acceptable when
it is supported by evidence the user can open, and *refused* when it is not. A system that
answers 85% of questions with citations and declines the rest is trustworthy. A system that
answers 100% of questions, 12% of them confidently wrong, is unusable — and worse, unfixable,
because nobody can tell which 12%.

This is how coding agents achieve their reliability. They do not rely on a memory index to be
right about a fact: they read the source, cite the location, and re-read rather than trusting
what they remembered. Memory finds the source; the source settles the question.

**Design consequence:** memory must never be the last word on a fact that matters. It is the
index that locates evidence, plus a verifier that decides whether the evidence supports the
claim. That is a different product from "a vector database with an LLM on top", and it is the
one worth building.

> **Evidence.** Measured results, and the numbers this harness got wrong before it got them
> right, are in [MEASUREMENTS.md](MEASUREMENTS.md). Nothing below is decided on a number that
> file does not contain.

## 2. Keep — this is the product

**The grounding path.** `evidence_verification`, `abstain_when_insufficient`,
`escalation_max_rounds`, the NLI cascade. Already on by default. This is the mechanism that
converts "usually right" into "never confidently wrong", and it is the single feature that
makes the enterprise claim defensible. Everything else exists to feed it.

**The document context graph.** `DEFINED_BY`, `FOOTNOTE`, `CROSS_REFERENCE`, `PARENT`,
`ON_PAGE` — evidence resolved to a page, a section, the footnote that qualifies a number and
the definition of the term it uses. This is the most differentiated thing in the codebase.
A competitor returns a chunk; this returns a chunk with the sentence that changes its
meaning. For audit, compliance and finance-style questions, that *is* the requirement.
`definition_expansion`, `neighbor_expansion` and `parent_expansion` are all on. Keep all of it.

**Evidence and provenance on every memory.** `EvidenceRef` with document, page, message and
run. Without it, no citation, no verification, no audit. Non-negotiable.

**Temporal truth: supersession and contradiction.** Enterprise facts change — a price, a
policy, an owner. `SUPERSEDES`/`superseded_by`, `valid_from`/`valid_to`, `contradicts`. Stale
memory stated confidently is the single largest source of wrong answers in agent systems, and
this is the defence. Also why the echo rule matters: an agent repeating what it was told must
not raise confidence.

**Hybrid retrieval: dense + BM25 + RRF + cross-encoder rerank.** Commodity, and necessary.
Keep exactly this. One measured observation from this repository: recall on an external
corpus stayed high with a *random* embedding, meaning the lexical path was carrying the
result — dense alone would not have.

**Contextual chunks.** The largest published win available for the cost, and deterministic
here (document title, section path, page prepended before indexing) with no model call.

**Authorization: visibility ladder + OpenFGA.** Multi-tenant isolation is the price of entry
for enterprise. Not optional, not simplifiable.

**Tool memory procedures.** Deterministic mining of what actually worked, with support counts
and decay. No model call, cheap to run, and genuinely differentiated — models choose tools
from their own context, but they cannot know which chain has worked here before.

## 3. Cut — surface that costs verification and returns nothing measured

Every item below is either off, unmeasured, or an alternative to something already chosen.
Together they are the reason only 27 of 44 provider values were ever exercised: the surface
outgrew the verification budget.

**Seven of the eight off-by-default retrieval flags — done, 2026-09-20.** `minicoil`,
`colbert`, `pageindex`, `raptor`, `graph_ppr`, `graphrag_global` and `late_chunking` are
removed: flags, wiring, adapters, the `strategies` module, the late-interaction vector
support in the search layer, and 854 MB of ColBERT weights. `splade` remained as the single
experiment slot until the Phase-1 freeze (2026-09) removed it too: the sparse leg is BM25 with
server-side IDF, and a learned sparse encoder is a benchmark challenger, never a product flag.

The argument was redundancy, not quality — each named a capability something already-on
provides — and `tests/eval/test_capability_coverage.py` demonstrates each capability
surviving without its flag. Those tests are the justification; they still run, and the
fixture now asserts none of the seven names has reappeared in `RetrievalSettings`.

**`retrieval.rerank` is also off by default**, but on measurement rather than redundancy:
significantly *worse* on BeIR/SciFact (p = 0.012) at 21x the latency. The flag is kept, since
the finding indicts one out-of-domain cross-encoder rather than reranking as a technique. See
[MEASUREMENTS.md](MEASUREMENTS.md) §3e.

**Three of four memory-intelligence providers.** `mem0`, `cognee`, `langmem` were alternatives
to `native`, which is what runs. Each was an import, a wiring branch, a contract test that
did not exist, and a licence surface. Removed; `native` is the only provider.

**Four of five graph-enrichment providers.** Same argument. `graphiti`, `docling_graph` and
`cognee` are removed; `native` and `disabled` remain.

**Twenty-five memory types reduced to the caller-facing set.** `SEMANTIC`, `PREFERENCE`,
`EPISODIC`, `DECISION`, `PROCEDURAL`, `TOOL`, plus the internal ones the pipeline writes.
The rest are a 25-value enum on an expert-only hint field, which is a question rather than an
API — a caller cannot choose correctly and a wrong choice silently makes a memory
unretrievable.

**Benchmark challenger models out of the default download.** 8.8 GB of weights where the
three defaults need a fraction. `--all` still fetches them for benchmark runs.

**The rule going forward:** a provider value that is not exercised by a contract test does not
ship. The port contracts now exist; the enum should shrink to what they cover.

## 4. Add — enterprise table stakes that are missing

**Authorization revocation.** No tuple is ever deleted. A removed user keeps live grants.
This is the most serious open defect in the system and it is not a cost question.

**Retention and deletion enforcement.** `observations`, `tool_invocations` and `messages` grow
without bound. Enterprise buyers require per-tenant retention and verifiable deletion; today
neither exists as policy or as job.

**A read audit trail.** Who retrieved which memory, when, under which scope. Required by every
regulated buyer, and impossible to add retroactively over data already served.

**Per-tenant quotas and backpressure.** One tenant ingesting a corpus currently starves every
other tenant's queries — observed directly in this repository when a benchmark run degraded
the API.

**The accuracy baseline itself.** On a corpus nobody here wrote, with a stated number and a
regression ratchet. Until it exists, no decision in §3 can be evaluated, `reranker.candidate_k`
cannot be tuned, and "100% accurate" cannot be claimed or disproven.

## 5. Cost and memory, in the order that matters

1. **`reranker.candidate_k` (20)** — one request costs 1 embedding and 20 cross-encoder pairs.
   The reranker is ~87% of per-request model cost. This is the tuning dial; it needs §4's
   baseline to tune against.
2. **A separate model tier** — built, then removed (ADR 0019 is superseded): on a single
   8 vCPU VM it only added an HTTP hop to every query.
3. **`on_disk_payload`** — vectors in RAM, payload on disk. The difference between a small and
   a large Qdrant node above ~10M chunks.
4. **Reclaim superseded collections** — 20 exist where 2 are used, each a full vector set.
5. **`context_edges` fan-out** — 12,670 rows for 1,303 chunks. Establish which edge kinds are
   actually followed before paying for all eleven.

## 6. What this buys

A system that answers with a citation to a page and a section, refuses when the evidence does
not support an answer, notices when a fact has been superseded or contradicted, enforces
tenant isolation, and remembers which tool sequences have actually worked — on CPU, in one
process.

That is a narrower product than the current configuration surface implies, and a much more
defensible one.
