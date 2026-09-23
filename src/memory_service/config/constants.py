"""What one package ships: the frozen model set and the tuning of every stage.

Nothing here is read from the environment. Changing a value is a code change, reviewed like
one, and a change to a model or its ONNX graph is also ``make reindex``: the embedding
fingerprint names the vector collections, so vectors from two encoders can never share one.

``Settings`` (``config/settings.py``) is the operator's surface - topology and credentials
only. The test suite's in-process stand-ins are ``application.container.Overrides``. This
module is the third thing, and the largest: the numbers that make the product what it is.

Why the tuning is frozen rather than configurable: three value sets and three mechanisms
(defaults, ``.env``, Makefile ``-e``) used to configure the same retriever, so the service
that was benchmarked and the service that shipped were never the same artefact. A retrieval
depth is not something an operator chooses; it is something this repository measures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

#: Where the weights live, in search order: the image bakes them under ``/models``; a host
#: checkout keeps them in ``./models`` (``make models``). When neither directory holds the
#: model it is loaded by its hub id, which ``HF_HUB_OFFLINE=1`` turns into a startup error
#: rather than a download.
MODEL_ROOTS: tuple[Path, ...] = (Path("/models"), Path("models"))


def local_model_path(local_dir: str) -> str | None:
    """The first root that holds ``local_dir``, or ``None`` when no root does."""
    for root in MODEL_ROOTS:
        candidate = root / local_dir
        if candidate.is_dir():
            return str(candidate)
    return None


class DenseModel(BaseModel):
    """The dense encoder, as loaded by sentence-transformers or by our own ONNX runner."""

    model_config = ConfigDict(frozen=True)

    id: str = "ibm-granite/granite-embedding-small-english-r2"
    #: directory name under a model root; ``model_path`` overrides the lookup entirely
    local_dir: str = "granite-embedding-small-english-r2"
    model_path: str | None = None
    revision: str | None = None
    #: Which runner loads it. ``torch``: sentence-transformers. ``onnx``: the tokenizer +
    #: ``onnxruntime`` session this repository owns (``adapters/models/embeddings.py``),
    #: because sentence-transformers' own ONNX backend reaches the graph through
    #: ``optimum.onnxruntime``, and ``optimum-onnx`` pins ``optimum~=2.1``, which cannot be
    #: installed beside sentence-transformers 6. Switching this is also ``make reindex``.
    runtime: Literal["torch", "onnx"] = "onnx"
    #: sentence-transformers backend (``runtime="torch"`` only)
    backend: Literal["torch", "onnx", "openvino"] = "torch"
    #: A specific ONNX graph under the model directory (``onnx/model_qint8.onnx`` vs
    #: ``onnx/model.onnx``). Part of the fingerprint: int8 and fp32 graphs of the same model
    #: must never silently share a collection.
    graph_file: str | None = None
    dimension: int = 384
    max_seq_length: int = 512
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"
    #: Intra-op threads the model may use: ``torch.set_num_threads`` for the torch runner,
    #: ORT ``intra_op_num_threads`` for the ONNX one. Two, because the deployment runs three
    #: uvicorn workers on eight vCPU and one encode fanning over every core leaves the other
    #: two workers nothing. That is arithmetic and a division of cores, not a measurement:
    #: every timing on this branch is single-query, one runner at a time, so "bounded beats
    #: the twelve-thread executor" is reasoning until the 8 vCPU VM times two callers.
    threads: int = 2

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


class SparseModel(BaseModel):
    """Client-side BM25 term frequencies with Qdrant's server-side IDF; no weights."""

    model_config = ConfigDict(frozen=True)

    name: Literal["bm25"] = "bm25"
    version: str = "v1"


class NLIModel(BaseModel):
    """The claim-support cross-encoder behind the grounding cascade."""

    model_config = ConfigDict(frozen=True)

    id: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    local_dir: str = "deberta-v3-base-mnli-fever-anli"
    model_path: str | None = None
    revision: str | None = None
    graph_file: str | None = None
    batch_size: int = 16
    max_length: int = 512
    #: as ``DenseModel.threads``; both runners call ``torch.set_num_threads``, which is
    #: process-wide, so the two counts are deliberately the same number
    threads: int = 2

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


class CrossEncoderModel(BaseModel):
    """A reranker. The shipped set has none (see ``FrozenModels.reranker``); the class exists
    for the benchmark challengers under ``benchmark/``."""

    model_config = ConfigDict(frozen=True)

    id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    local_dir: str = "ms-marco-MiniLM-L6-v2"
    model_path: str | None = None
    backend: Literal["torch", "onnx"] = "torch"
    batch_size: int = 16

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


class FrozenModels(BaseModel):
    model_config = ConfigDict(frozen=True)

    dense: DenseModel = DenseModel()
    sparse: SparseModel = SparseModel()
    nli: NLIModel = NLIModel()
    #: Off on measured evidence: SciFact-1000 nDCG@10 79.33 vs 84.51 without it (paired sign
    #: test p = 0.012) at 21x the latency. ``RetrievalSettings.rerank`` is the switch; with
    #: no model here it has nothing to load.
    reranker: CrossEncoderModel | None = None


FROZEN_MODELS = FrozenModels()

#: docling's layout and table models, fetched by ``download_models`` next to the weights
DOCLING_ARTIFACTS_DIR = "docling"

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

SERVICE_NAME = "memory-service"
API_VERSION = "v1"
HOST = "0.0.0.0"  # noqa: S104 - a container listens on every interface
#: never log raw source text (prompts, model output, message bodies)
LOG_SOURCE_TEXT = False
MAX_BODY_BYTES = 25 * 1024 * 1024
#: extra requests tolerated above ``service.rate_limit_per_minute``
RATE_LIMIT_BURST = 200


@dataclass(frozen=True)
class Headers:
    tenant: str = "X-Memory-Tenant"
    workspace: str = "X-Memory-Workspace"
    user: str = "X-Memory-User"
    groups: str = "X-Memory-Groups"
    api_key: str = "X-API-Key"


HEADERS = Headers()

# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseTuning:
    pool_timeout_seconds: float = 5.0
    statement_timeout_ms: int = 15_000
    #: A pooled connection is thrown away and reopened after this long. It replaces the
    #: pre-ping, which cost a round trip on every checkout - up to three per request against
    #: a remote database - to catch a connection closed by the server, a proxy or an idle
    #: timeout while the pool held it.
    #:
    #: It only makes that case rare if it is shorter than whatever closes connections on the
    #: other side, and the usual things are not long: pgbouncer's server_idle_timeout
    #: defaults to 600 s and managed PostgreSQL offerings idle out between 5 and 10 minutes.
    #: At 1800 s the window was wider than all of them and the "rare" in that sentence was
    #: not earned. Five minutes is inside every one of them; the cost is 24 connections
    #: (8 per process, three API workers) reopened every five idle minutes.
    pool_recycle_seconds: int = 300
    #: libpq gives up on opening a connection after this long. Without it libpq waits
    #: indefinitely, and "indefinitely" is reachable: a PostgreSQL container whose port is
    #: still published but whose server has stopped answering completes the TCP handshake
    #: and never replies to the startup packet. Bounded so /health/ready always answers.
    connect_timeout_seconds: int = 5


DATABASE = DatabaseTuning()


@dataclass(frozen=True)
class CacheTuning:
    hot_thread_ttl_seconds: int = 6 * 3600
    hot_thread_max_messages: int = 200
    working_memory_ttl_seconds: int = 1800
    embedding_ttl_seconds: int = 7 * 24 * 3600
    context_bundle_ttl_seconds: int = 300
    authz_ttl_seconds: int = 60
    #: short on purpose: a backend failure raises CacheUnavailable quickly and callers
    #: degrade to the canonical store instead of hanging
    connect_timeout_seconds: float = 0.5
    socket_timeout_seconds: float = 0.5


CACHE = CacheTuning()


@dataclass(frozen=True)
class BlobLifecycle:
    """GCS lifecycle for the archive buckets: autoclass, or explicit tiering by age."""

    policy: Literal["autoclass", "explicit"] = "autoclass"
    nearline_days: int = 60
    coldline_days: int = 180
    archive_days: int = 365


BLOB_LIFECYCLE = BlobLifecycle()


@dataclass(frozen=True)
class SearchTuning:
    collection_prefix: str = "mem"
    on_disk_payload: bool = True
    timeout_seconds: float = 5.0


SEARCH = SearchTuning()


@dataclass(frozen=True)
class TaskTuning:
    default_retries: int = 5
    job_timeout_seconds: int = 600
    periodic_reconcile_seconds: int = 300
    #: a job whose worker stopped heartbeating for this long is re-queued
    stalled_after_seconds: float = 120
    #: How long a dispatched outbox row is kept. It is a receipt once the queue owns the
    #: job, but keeping it briefly makes a relay crash debuggable; rows marked dead are
    #: never purged. Measured before this existed: 734 rows for 788 ingested turns, growing
    #: with every write forever.
    outbox_retention_seconds: int = 24 * 3600


TASKS = TaskTuning()


@dataclass(frozen=True)
class AuthorizationTuning:
    max_listed_objects: int = 2000
    decision_cache: bool = True


AUTHORIZATION = AuthorizationTuning()


@dataclass(frozen=True)
class LLMTransport:
    """Passed straight to the shared Bifrost client, which owns retries and the breaker."""

    retry_backoff_seconds: float = 0.5
    #: consecutive failures that open the circuit
    circuit_failure_threshold: int = 5
    circuit_open_seconds: float = 30.0


LLM_TRANSPORT = LLMTransport()

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class ArchiveSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    segment_target_bytes: int = Field(
        default=4 * 1024 * 1024, description="compressed; benchmark 1-8MB"
    )
    segment_max_messages: int = 5000
    zstd_level: int = 6
    purge_grace_seconds: int = 24 * 3600
    purge_min_payload_bytes: int = Field(
        default=4096, description="only payloads larger than this are purged from hot DB"
    )
    tenant_shards: int = 64


ARCHIVE = ArchiveSettings()


class GraphSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_visited: int = 200
    default_hops: int = 1
    #: How long a query may wait for the retrieval-time traversal before answering without
    #: graph facts.
    #:
    #: The traversal is started as soon as the scope is known, so on a slow encoder it costs
    #: nothing - it finishes underneath. That is exactly why it needs a ceiling: the moment
    #: the encoder gets faster (int8 ONNX), or the graph gets deep enough for a three-hop
    #: walk to outrun it, an unbounded traversal becomes the tail of every entity, temporal
    #: and multi-hop question. 150 ms is the band the roadmap derives for an 8 vCPU VM whose
    #: encode is ~40 ms and whose p99 target is 300 ms.
    #:
    #: Expiry drops facts; it never cancels the traversal. See ``GraphStage.__call__``.
    prefetch_budget_ms: int = Field(default=150, ge=1)
    #: How many expired traversals may be finishing at once before the next one is cancelled
    #: instead of parked.
    #:
    #: The budget bounds the wait, not the concurrency, and the condition that parks a
    #: traversal - a graph slower than the budget - is exactly the condition that parks the
    #: next one too. Each parked traversal holds a connection out of a pool of
    #: ``pool_size + max_overflow`` (8 + 8 per process, see ``DatabaseSettings``) that the
    #: read path checks out of, so an uncapped leak turns a latency problem into pool
    #: exhaustion, which is worse than the tail the budget exists to cut. Past this many,
    #: the aborted statement is the cheaper harm.
    #:
    #: This default is therefore **half the pool**, not the two fifths the figure here used
    #: to imply - it read 10 + 10, which the settings have never been. The two numbers are
    #: only safe as a pair, so moving either one means re-reading this.
    max_parked_traversals: int = Field(default=8, ge=1)


GRAPH = GraphSettings()


class NLISettings(BaseModel):
    """Thresholds of the grounding cascade (the model itself is ``FROZEN_MODELS.nli``)."""

    model_config = ConfigDict(frozen=True)

    supported_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="entailment (or contradiction) score to decide"
    )
    borderline_band: tuple[float, float] = Field(
        default=(0.3, 0.7), description="entailment band in which the LLM judge is consulted"
    )
    premises_per_claim: int = Field(
        default=5, ge=1, description="evidence items scored per claim (best lexical overlap)"
    )
    max_claims: int = Field(default=40, ge=1)


NLI = NLISettings()


class MemoryIntelligenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    dedup_lexical_threshold: float = 0.92
    dedup_dense_threshold: float = 0.90
    dedup_candidate_k: int = 20
    # admission gate (worthiness x novelty x confidence x expected utility -> admit/defer/reject)
    admission_worthiness_min: float = Field(default=0.35, ge=0.0, le=1.0)
    admission_confidence_min: float = Field(default=0.3, ge=0.0, le=1.0)
    admission_score_min: float = Field(default=0.4, ge=0.0, le=1.0)
    admission_defer_band: float = Field(
        default=0.08, ge=0.0, le=1.0, description="score band below the minimum that defers"
    )
    #: Keep the turn itself, not only what the rules could parse out of it.
    #:
    #: Extraction is rule-based and first-person ("I work at X"). A conversation *about*
    #: someone — third-person narrative, which is most real dialogue — matches nothing, and
    #: `_from_sentence` returns None. Measured on LoCoMo: 452 of 788 turns (57.4%) produced
    #: no candidate at all, so the text was never indexed and no retriever could reach it.
    #: The perfect-retrieval ceiling under that regime is 0.098; keeping the turn verbatim
    #: raises it to 0.685.
    #:
    #: The verbatim copy is an OBSERVATION, which is in DERIVED_MEMORY_TYPES, so it is
    #: excluded from supersession and reflection (landing.py:63, :77) and cannot disturb the
    #: fact machinery or the false-merge gate. It augments the rule output; it never
    #: replaces it. Applies to user-authored messages outside a thread only: a thread's
    #: turns are kept by the hot-thread cache and the archive, and an agent's messages are
    #: working chatter that must not inherit a shared visibility.
    keep_verbatim_turns: bool = True
    #: Longest turn kept verbatim. Beyond this the turn is truncated rather than dropped.
    verbatim_max_chars: int = Field(default=2000, ge=200)
    # landing reflection and derived memories
    landing_reflection_k: int = Field(default=8, ge=0, le=8)
    belief_min_support: int = Field(default=2, ge=2)
    entity_summary_min_facts: int = Field(default=2, ge=1)
    # forgetting: importance x recency x access decay
    forgetting_half_life_days: float = Field(default=30.0, gt=0.0)
    forgetting_archive_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    forgetting_min_idle_days: float = Field(default=30.0, ge=0.0)
    forgetting_batch: int = Field(default=500, ge=1)


MEMORY_INTELLIGENCE = MemoryIntelligenceSettings()


class DocumentSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: ``builtin`` is also the fallback when docling cannot be constructed in this image.
    parser: Literal["docling", "builtin"] = "docling"
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 40
    chunk_overlap_tokens: int = 40
    contextual_chunks: bool = True
    keep_tables_intact: bool = True
    keep_code_intact: bool = True
    max_file_bytes: int = 100 * 1024 * 1024


DOCUMENTS = DocumentSettings()


#: Shipped retrieval depth, and the only number that sets it.
#:
#: Prefetch and fusion depth exist to give reciprocal rank fusion something to reorder; they
#: are not a second opinion about how much evidence a caller wants. Written down separately
#: they drifted - the service shipped 100/100/50 while every judged benchmark ran 200/200/100
#: - so no artifact could say which ratio had been measured. One knob now, and a ratio.
FINAL_K = 50
#: Prefetch/fusion depth as a multiple of ``final_k``.
#:
#: 2.0 is not a proposal: it is the ratio every measurement this repository owns was produced
#: at - the shipped 100/100/50 and the judged 200/200/100 are the same ratio - so writing it
#: down changes no result on disk and makes the two configurations one derivation.
#:
#: The roadmap's Phase 2 step 4 wants it at 1.25 (63/63/50 shipped): RRF only reorders inside
#: the prefetch union, so a quarter again may well be all the headroom the reorder needs, and
#: the search and assemble stages scale roughly linearly with it. That is a retrieval-quality
#: change, not a refactor, and it is gated on a judged run reporting ``evidence_recall >=
#: 0.987`` plus ``tests/eval/test_retrieval_gate.py``. When the gate passes, this constant is
#: the only line that moves.
DEPTH_RATIO = 2.0


def derived_k(final_k: int) -> int:
    """Prefetch and fusion depth for a final depth: ``ceil(DEPTH_RATIO * final_k)``."""
    return math.ceil(final_k * DEPTH_RATIO)


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Longest query text that is retrieved on. Anything beyond this is cut.
    #:
    #: A query is not free: it is embedded, run through the sparse encoder, and then paired
    #: with every reranked candidate — and a cross-encoder pair is as expensive as its longest
    #: side. Measured on the degenerate-input benchmark, a 2,000-character wall of noise cost
    #: 20 seconds against 1 second for an ordinary question, on the same corpus. Nothing is
    #: lost by cutting: the embedding models truncate at 512 tokens regardless, so the text
    #: past this point never reached the model — it was only ever paid for.
    max_query_chars: int = Field(
        default=2048, ge=64, description="query text beyond this is truncated before retrieval"
    )
    exact: bool = True
    bm25: bool = True
    dense: bool = True
    graph: bool = True
    #: Reciprocal rank fusion of the dense and sparse prefetches, natively in Qdrant.
    rrf_k: int = 60
    #: Derived from ``final_k``; see ``derived_k``. Set explicitly only to pin a depth that
    #: is not the shipped one (``benchmark/env.py`` pins the judged 200/200/100).
    prefetch_k: int = Field(
        default=derived_k(FINAL_K), description="per-retriever candidates before fusion"
    )
    fused_k: int = Field(default=derived_k(FINAL_K), description="candidates after fusion")
    #: Cross-encoder reranking. **Off on measured evidence.**
    #:
    #: BeIR/SciFact, 1,000 documents, 70 paired queries, clean vector store, real models:
    #:
    #:              recall@10   nDCG@10   p50
    #:   rerank on     97.14%    79.33%   11,151 ms
    #:   rerank off    98.57%    84.51%      533 ms
    #:
    #: Paired, the reranker rescued *zero* queries the first stage missed and lost one. nDCG
    #: better on 4 queries, worse on 16, identical on 50 — exact sign test p = 0.012, mean
    #: delta -0.0518 with a 95% interval of [-0.0917, -0.0120] that excludes zero. It is
    #: significantly worse here, not merely not better. At 11.2 s per query against 0.53 s,
    #: 20 RPS needs ~161 cores with it and ~8 without. ``FROZEN_MODELS.reranker`` is None:
    #: turning this on loads nothing until a model is frozen there, on a new measurement.
    rerank: bool = False
    #: candidates handed to the reranker when one is wired (benchmark 15-25)
    rerank_k: int = 20
    #: The one depth knob: what a caller receives. ``prefetch_k`` and ``fused_k`` follow it.
    final_k: int = Field(default=FINAL_K, ge=1)
    parent_expansion: bool = True
    neighbor_expansion: bool = True
    definition_expansion: bool = True
    expansion_budget_items: int = 8
    evidence_verification: bool = True
    escalation_max_rounds: int = 2
    abstain_when_insufficient: bool = True
    #: On conversational (memory-only) bundles, require that some retrieved memory *about
    #: the person the question names* shares a content term with the rest of the question.
    #: The plain overlap rule above cannot see a wrong-person presupposition — "what was
    #: grandma's gift to Melanie?" when it was Caroline's grandma — because a two-person
    #: conversation shares terms with any question about either of them. Measured on LoCoMo:
    #: every one of 304 bundles, 71 of them unanswerable by construction, reported COMPLETE.
    subject_evidence_check: bool = True

    @model_validator(mode="before")
    @classmethod
    def _derive_depth(cls, data: Any) -> Any:
        """``final_k`` alone decides the depth; an explicit value still wins."""
        if not isinstance(data, dict) or "final_k" not in data:
            return data
        depth = derived_k(int(data["final_k"]))
        return {"prefetch_k": depth, "fused_k": depth, **data}

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """``model_copy(update={"final_k": n})`` derives the depth above it, like construction.

        Pydantic's ``model_copy`` assigns straight onto the copy and runs no validator, so the
        one-knob rule would have held at construction and silently not held here - and
        ``model_copy`` is how every benchmark ablation and every test builds a tuning. A depth
        passed explicitly alongside ``final_k`` still wins, exactly as it does in the
        constructor (``benchmark/env.py`` pins the judged 200/200/100 that way).
        """
        if update and "final_k" in update:
            depth = derived_k(int(update["final_k"]))
            update = {"prefetch_k": depth, "fused_k": depth, **update}
        return super().model_copy(update=dict(update) if update else None, deep=deep)


RETRIEVAL = RetrievalSettings()


class ContextSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_budget: int = 8000
    conversation_max_messages: int = 20
    conversation_token_budget: int = 2000
    #: Keep what the engine ranked, and let the token budget do the bounding.
    #:
    #: This was 12 against final_k = 20, so eight already-retrieved, already-ranked
    #: candidates were dropped for free — and the bundles that survived used 5.8% of the
    #: 6000-token budget (measured: median 1380 rendered chars, ~345 tokens). Two numbers
    #: that must agree were written down twice and drifted. token_budget stays the real
    #: constraint.
    memories_max: int = 50
    knowledge_max: int = 12
    graph_facts_max: int = 12
    summaries_max: int = 4
    #: How long served-memory ids may sit in the builder's buffer before one bulk bump, and
    #: how many ids force an early flush.
    #:
    #: The bump is an UPDATE plus a COMMIT - a WAL flush - on the same pool the reads use. One
    #: per served bundle is ~20 of them a second at the 20 rps target, each touching up to
    #: ``memories_max`` rows, for a counter whose only reader is the nightly forgetting pass.
    #: Buffering trades a 2 s delay in that counter, which nothing reads sooner, for a single
    #: statement per tenant per window. Ids repeated inside one window count once.
    access_flush_seconds: float = Field(default=2.0, gt=0.0)
    access_flush_max_ids: int = Field(default=200, ge=1)


CONTEXT = ContextSettings()
