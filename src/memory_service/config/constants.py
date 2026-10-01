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
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memory_service.ports.search import VectorName

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
    """A dense encoder, as loaded by sentence-transformers or by our own ONNX runner."""

    model_config = ConfigDict(frozen=True)

    id: str = "ibm-granite/granite-embedding-small-english-r2"
    #: directory name under a model root; ``model_path`` overrides the lookup entirely
    local_dir: str = "granite-embedding-small-english-r2"
    model_path: str | None = None
    revision: str | None = None
    #: the checkpoint's licence, reported with the provider
    license: str = "Apache-2.0"
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
    #: Retrieval-trained encoders such as E5 require asymmetric task prefixes.
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = Field(default=32, ge=1)
    #: The least cosine to the query a ranked item needs to be packed into a context, when
    #: this encoder's space is the one every query is searched in. A cosine is only
    #: comparable within one encoder, so the floor belongs to it; 0 is no floor. Not part of
    #: the fingerprint: it reads vectors, it does not make them.
    relevance_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Bound each non-preemptible indexing turn on the shared model runner. A full
    #: 32-passage forward pass blocked interactive queries for seconds on CPU.
    document_batch_size: int = Field(default=1, ge=1)
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
    version: str = "v3-snowball"


class NLIModel(BaseModel):
    """The claim-support cross-encoder behind the grounding cascade.

    Multilingual, and frozen on measured evidence: the FP32 graph scores the English golden
    grounding set 36/40 - the same as the English DeBERTa it replaces - and 78.67% on
    same-language XNLI across 15 languages
    (``benchmark/results/multilingual_nli_mdeberta_fp32.json``).
    Its int8 quantisation lost eleven points on English and is not shipped. One model, one
    runtime: the ONNX session this repository owns (``adapters/models/onnx_nli.py``).
    """

    model_config = ConfigDict(frozen=True)

    id: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    local_dir: str = "mdeberta-v3-base-xnli-multilingual-nli-2mil7"
    model_path: str | None = None
    revision: str | None = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    license: str = "MIT"
    runtime: Literal["onnx"] = "onnx"
    #: the FP32 graph; the quantised ``onnx/model_quantized.onnx`` failed quality
    graph_file: str | None = "onnx/model.onnx"
    batch_size: int = Field(default=8, ge=1, le=64)
    max_length: int = Field(default=512, ge=32, le=512)
    #: as ``DenseModel.threads``: ORT ``intra_op_num_threads``, one share of the cores
    threads: int = 2

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


class LateInteractionModel(BaseModel):
    """The late-interaction (ColBERT) encoder: one 64-wide vector per token, scored by MaxSim.

    mxbai-edge-colbert-v0-32m (mixedbread, Germany; Apache-2.0), the publisher's own ONNX
    export, FP32. The tokenisation is the checkpoint's own (``onnx_config.json``: the
    ``[Q]``/``[D]`` marker after ``[CLS]``, lower-casing, punctuation dropped from documents,
    no query expansion); reproduced here it scores within 0.011 of PyLate on every LoCoMo
    question of conversation 0 with an identical top 10. Its int8 graph changed the top 10 of
    every one of those questions and is not shipped (ADR 0025).
    """

    model_config = ConfigDict(frozen=True)

    id: str = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
    local_dir: str = "mxbai-edge-colbert-v0-32m"
    model_path: str | None = None
    revision: str | None = "bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66"
    license: str = "Apache-2.0"
    runtime: Literal["onnx"] = "onnx"
    graph_file: str = "model.onnx"
    dimension: int = 64
    batch_size: int = Field(default=8, ge=1)
    #: as ``DenseModel.threads``
    threads: int = 2

    @property
    def source(self) -> str:
        return self.model_path or local_model_path(self.local_dir) or self.id


class FrozenModels(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: the English specialist: named vector ``dense_en``, searched for Latin-script queries
    dense: DenseModel = DenseModel()
    #: The multilingual encoder: named vector ``dense_ml``, searched for every query.
    #:
    #: Bekko a8m: ModernBERT (Answer.AI/LightOn) + mmBERT (JHU) lineage, a Japanese
    #: maintainer, MIT. Measured (``docs/CPU-MULTILINGUAL-DECISION-20260928.md``): XQuAD
    #: paragraph R@10 0.9883 over 12 languages where the English encoder reads 0.6596; fused
    #: with it and BM25, SciFact 0.7557/0.8926 against 0.7409/0.8912 for English + BM25; 1,200
    #: encodes at 20 RPS with p99 108.84 ms. Its larger sibling (a25m) costs a p99 of 1,046 ms
    #: for no English gain and is not shipped.
    dense_ml: DenseModel = DenseModel(
        id="hotchpotch/bekko-embedding-v1-a8m",
        local_dir="bekko-embedding-v1-a8m",
        revision="c721113d59a1d91b447450324f51c4b3332c924a",
        license="MIT",
        batch_size=8,
        # Fusion scores only order, so without a floor a context fills its budget with
        # whatever ranked next: ten questions nothing in the corpus answers packed 30.8
        # memories each. Measured on a LoCoMo conversation (docs/MEASUREMENTS.md, section
        # 8): at 0.20 every evidence memory of 135 questions is still packed and those
        # off-topic questions pack 0.9; 0.25 already loses 1.5% of the evidence.
        relevance_floor=0.2,
    )
    sparse: SparseModel = SparseModel()
    nli: NLIModel = NLIModel()
    #: the late-interaction arm of every collection
    colbert: LateInteractionModel = LateInteractionModel()


FROZEN_MODELS = FrozenModels()

#: docling's layout and table models, fetched by ``download_models`` next to the weights
DOCLING_ARTIFACTS_DIR = "docling"

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

SERVICE_NAME = "trellis-memory"
API_VERSION = "v1"
HOST = "0.0.0.0"  # noqa: S104 - a container listens on every interface
#: never log raw source text (prompts, model output, message bodies)
LOG_SOURCE_TEXT = False
MAX_BODY_BYTES = 25 * 1024 * 1024
#: Requests per tenant per minute unless the tenant's own quota says otherwise
#: (``PATCH /v1/admin/tenants/{id}``), counted in the cache. 1200 is exactly the 20 rps the
#: service is measured at, so a load run against one tenant spent itself answering 429s: the
#: default is a guard against a runaway client, not the target rate.
RATE_LIMIT_PER_MINUTE = 6000
#: extra requests tolerated above the per-minute limit
RATE_LIMIT_BURST = 200


@dataclass(frozen=True)
class Headers:
    """The trusted context headers (ADR 0022). ``llm_tokens`` is a response header."""

    tenant: str = "X-Trellis-Tenant"
    workspace: str = "X-Trellis-Workspace"
    user: str = "X-Trellis-User"
    api_key: str = "X-API-Key"
    llm_tokens: str = "X-Trellis-LLM-Tokens"
    request_id: str = "X-Request-ID"
    traceparent: str = "traceparent"


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
    #: How long ``memory_reads`` keeps who-read-what. Long enough for an annual access
    #: review with a quarter's slack; the table grows with every recall otherwise.
    read_audit_retention_days: int = 400


TASKS = TaskTuning()


@dataclass(frozen=True)
class AuthorizationTuning:
    max_listed_objects: int = 2000
    decision_cache: bool = True
    #: One OpenFGA call's budget. It was 3 s, hardcoded in the client construction, and under
    #: load a ListObjects that queued behind others ran past it and surfaced as a 503: long
    #: enough to hold a worker, too short for a busy authorization server. A read path waits
    #: at most ``timeout_seconds * (retries + 1)`` plus the pauses.
    timeout_seconds: float = 8.0
    #: Retries of a call that failed transiently (a timeout, a refused or reset connection, a
    #: 5xx); a decision the server made is never retried.
    retries: int = 2
    #: Pause before a retry, doubled each time.
    retry_pause_seconds: float = 0.1


AUTHORIZATION = AuthorizationTuning()


@dataclass(frozen=True)
class LLMTransport:
    """Passed straight to the shared Bifrost client, which owns retries and the breaker."""

    retry_backoff_seconds: float = 0.5
    #: consecutive failures that open the circuit
    circuit_failure_threshold: int = 5
    circuit_open_seconds: float = 30.0


LLM_TRANSPORT = LLMTransport()


@dataclass(frozen=True)
class LLMTuning:
    """How the service calls the model. A tenant's policy may name the model for a use
    (``PUT /v1/model-key/policy``); ``auto`` discovers a recognised text model through the
    gateway's authenticated ``/models``."""

    model: str = "auto"
    #: the cheap model, for the classification-sized uses below
    fast_model: str = "auto"
    fast_uses: tuple[str, ...] = ("contextual_extraction", "query_expansion", "chunk_context")
    #: A reasoning model spends its output budget thinking before it emits anything, so too
    #: small a ceiling returns 200 OK with an empty string.
    max_tokens: int = 1024
    timeout_seconds: float = 30.0
    #: retries on 429/5xx/timeouts; the sleeps sit outside the request timeout
    max_retries: int = 2


LLM = LLMTuning()


@dataclass(frozen=True)
class HindsightTuning:
    """The Hindsight extraction preview (``[hindsight]`` extra, ``MEMORY__HINDSIGHT__*``)."""

    bank_id: str = "extraction-preview"
    timeout_seconds: float = 30.0
    max_concurrency: int = 1


HINDSIGHT = HindsightTuning()

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
    #: The retrieval-time traversal's time budget, enforced by PostgreSQL: the budgeted
    #: pool's connections carry it as their ``statement_timeout`` (see
    #: ``adapters/graph/postgres_store.py``), so the server stops a traversal past it and
    #: the query is answered without graph facts.
    #:
    #: The traversal starts as soon as the scope is known and is one statement over
    #: indexed, per-hop-limited CTEs, so on a healthy database it finishes underneath the
    #: encoder. The budget is for the graph that is not healthy - a star-shaped entity whose
    #: DISTINCT ON sorts every edge it has - and 150 ms is the band the roadmap derives for
    #: an 8 vCPU VM whose encode is ~40 ms and whose p99 target is 300 ms. It is server time,
    #: not a timer in the client: the client's timer measured its own scheduling as much as
    #: the graph and dropped graph facts on any busy box (MEASUREMENTS.md, section 8).
    prefetch_budget_ms: int = Field(default=150, ge=1)
    #: The budgeted traversal's own pool, per process: one traversal per graph-routed read,
    #: so this many run at once before one waits for a connection. Kept apart from the main
    #: pool because the timeout is a property of the connection.
    budgeted_pool_size: int = Field(default=4, ge=1)
    budgeted_pool_overflow: int = Field(default=4, ge=0)
    #: ``GET /v1/graph/entities``: the most entities one search returns.
    entity_search_max: int = Field(default=100, ge=1)
    #: ``GET /v1/graph/entities/{id}``: current relations and history rows in a profile.
    profile_relations_max: int = Field(default=50, ge=1)
    profile_history_max: int = Field(default=20, ge=1)
    #: Entity summaries, refreshed by the enrichment job for the entities it touched: the
    #: facts one summary is written from, how many entities one job refreshes (most
    #: mentioned first), and how many of those may be rewritten by the model per job - an
    #: entity named in every message would otherwise cost a model call per message.
    entity_summary_facts: int = Field(default=12, ge=1)
    entity_summaries_per_job: int = Field(default=32, ge=0)
    entity_summary_model_calls_per_job: int = Field(default=4, ge=0)


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
    #: excluded from supersession and reflection and cannot disturb the
    #: fact machinery or the false-merge gate. It augments the rule output; it never
    #: replaces it. Applies to user-authored messages, including messages inside threads:
    #: archival storage alone does not make older turns searchable. Agent working chatter
    #: is excluded so it cannot inherit a shared visibility.
    keep_verbatim_turns: bool = True
    #: Longest turn kept verbatim. Beyond this the turn is truncated rather than dropped.
    verbatim_max_chars: int = Field(default=2000, ge=200)
    #: Index a verbatim turn together with the message said just before it in the same
    #: conversation (same workspace and thread, within ``preceding_turn_window``). A reply
    #: rarely restates its question - "Yes, last weekend with my kids" answers "Did you go
    #: camping?" - and a bare question is never kept as a memory of its own, so without
    #: this the question's words are in no index at all. Index text only: the memory's
    #: content, payload text and rendering are unchanged. Language-agnostic.
    #:
    #: Offline LoCoMo A/B (turn-level BM25 + one 384-d dense encoder, weighted RRF; the
    #: service-faithful baseline reads 0.664): recall@10 +0.047 on its own, 183 questions
    #: better and 114 worse. Since ADR 0025 it is the memory's second key, beside its own
    #: text, and the turn it names is its neighbour in the learned fusion's lift.
    #: Changing it changes what is indexed: ``make reindex``.
    index_preceding_turn: bool = True
    preceding_turn_max_chars: int = Field(default=500, ge=0)
    preceding_turn_window: timedelta = timedelta(hours=6)
    # forgetting: importance x recency x access decay
    forgetting_half_life_days: float = Field(default=30.0, gt=0.0)
    forgetting_archive_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    forgetting_min_idle_days: float = Field(default=30.0, ge=0.0)
    forgetting_batch: int = Field(default=500, ge=1)
    #: Keep out of automatic forgetting what months-long recall depends on: the user's own
    #: lasting facts and preferences (their name, their dog's name, "never suggest
    #: cilantro") and the verbatim turns every other memory is evidence from. Scored by
    #: importance x recency x use, a never-recalled turn fell under the threshold in about
    #: forty idle days and a never-recalled user fact in about ninety. They can still be
    #: forgotten explicitly (DELETE /v1/memories/{id}) and by tenant retention.
    forgetting_protect_core: bool = True


MEMORY_INTELLIGENCE = MemoryIntelligenceSettings()

# Complete source text admitted into one background reflection prompt.
REFLECTION_SOURCE_CHARS = 2000


class DocumentSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: ``builtin`` is also the fallback when docling cannot be constructed in this image.
    parser: Literal["docling", "builtin"] = "docling"
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 40
    chunk_overlap_tokens: int = 40
    contextual_chunks: bool = True
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
    #: A query is not free: it is embedded and run through the sparse encoder. Measured on
    #: the degenerate-input benchmark, a 2,000-character wall of noise cost 20 seconds against
    #: 1 second for an ordinary question, on the same corpus. Nothing is lost by cutting: the
    #: embedding models truncate at 512 tokens regardless, so the text past this point never
    #: reached the model — it was only ever paid for.
    max_query_chars: int = Field(
        default=2048, ge=64, description="query text beyond this is truncated before retrieval"
    )
    exact: bool = True
    bm25: bool = True
    dense: bool = True
    graph: bool = True
    #: Unclassified questions can still name known graph entities in any language.
    #: Resolve names under the caller's scope, with the same bounded traversal budget.
    semantic_graph: bool = True
    #: Outer fusion of hybrid document candidates with optional strategy retrievers.
    rrf_k: int = 60
    # Dense/sparse fusion uses the same one-based rank convention as rrf_fuse. One
    # preserves Qdrant's historical default (zero-based k=2); tune explicitly, not silently.
    hybrid_rrf_k: int = Field(default=1, ge=0, le=1000)
    #: Weight of each hybrid arm (``dense_en``, ``dense_ml``, ``bm25``) in the store's RRF,
    #: fitted offline from per-arm rank dumps (``benchmark/fit_rrf_weights.py``) for the
    #: shipped encoders (``benchmark/results/phase9/rrf_weight_fit_ensemble.json``).
    #: ``None`` is equal weights, which every measurement before the fit was made at.
    #:
    #: Measured over LoCoMo's 1,986 questions against equal weights at the same depth
    #: (``docs/PHASE9-RESULTS-2026-09-29.md``, item 1): recall@10 +0.0143, multi-hop@10
    #: +0.0176, better on 55 questions and worse on 20; all-answerable recall@10/20/50/100
    #: 0.651/0.719/0.813/0.817 -> 0.666/0.730/0.819/0.821, for +7.8 ms p50. An arm missing
    #: from the mapping weighs 1.0, so a single-encoder deployment still fuses correctly.
    #:
    #: ``colbert`` is the late-interaction arm (ADR 0025): MaxSim over the union of the
    #: other arms' candidates, at 2.0. Offline over SciFact's 300 questions the fusion reads
    #: nDCG@10 0.746 -> 0.759 and recall@10 0.872 -> 0.883 with it. These weights serve the
    #: documents and the episodes; the memories are ranked by the learned fusion.
    hybrid_weights: dict[VectorName, float] | None = Field(
        default_factory=lambda: {
            VectorName.BM25: 2.0,
            VectorName.DENSE_EN: 0.5,
            VectorName.DENSE_ML: 2.0,
            VectorName.COLBERT: 2.0,
        }
    )
    #: The memories are ranked by ``modules/retrieval/learned_fusion.py`` (ADR 0025): each
    #: arm's own top this-many, read unfused in one round trip.
    memory_arm_depth: int = Field(default=100, ge=10, le=500)
    #: The top this-many of each of its two first stages are what the learned score orders
    #: (``learned_fusion``); the rest follow in first-stage order. The coefficients are
    #: fitted at this value, so it moves only with a refit.
    memory_pool_k: int = Field(default=30, ge=1, le=100)
    #: Derived from ``final_k``; see ``derived_k``. Set explicitly only to pin a depth that
    #: is not the shipped one (``benchmark/env.py`` pins the judged 200/200/100).
    prefetch_k: int = Field(
        default=derived_k(FINAL_K), description="per-retriever candidates before fusion"
    )
    fused_k: int = Field(default=derived_k(FINAL_K), description="candidates after fusion")
    #: The one depth knob: what a caller receives. ``prefetch_k`` and ``fused_k`` follow it.
    final_k: int = Field(default=FINAL_K, ge=1)
    # Wider recall for memory-only ranked pools. Full LoCoMo source recall improved from
    # 77.40% to 83.90%; the user accepted measured p99 550.6 ms on 2026-09-26.
    # Explicit caller limits and document/mixed pools retain final_k. ContextSettings
    # packs this depth within the unchanged token budget. Zero restores final_k behavior.
    memory_recall_k: int = Field(default=100, ge=0, le=200)
    #: Actor/topic search for multi-hop memory questions: the people a question names are
    #: searched again, each as their own subject, for the question's topic, and fused with
    #: the original ranking (weighted twice). Measured over LoCoMo's 1,986 questions on one
    #: corpus (docs/MEASUREMENTS.md, section 8.4): multi-hop complete coverage @50 +2.5 and
    #: @100 +2.8 points, all-question recall @10/@50/@100 +0.6/+0.6/+0.4, no depth worse.
    #: It fires only for English multi-hop cues over memory-only pools (250 of the 1,986) and
    #: its extra encode and searches stop at the timeout below. The cost, paired per question
    #: on the 2015 dev box: +122 ms median on those 250, all-question p95 503 -> 547 ms.
    memory_entity_search: bool = True
    memory_entity_search_timeout_ms: int = Field(default=200, ge=1, le=500)
    #: Model-assisted query expansion sits on the read path, in front of the search, and the
    #: model call behind it is bounded only by ``LLMTuning`` (30 s and two retries): one slow
    #: answer was the whole of a retrieval's latency budget. Past this deadline the query is
    #: searched as written. The written query is encoded while the model is asked, so an
    #: expansion that misses the deadline or adds no terms costs nothing beyond the wait.
    query_expansion_timeout_ms: int = Field(default=250, ge=1, le=5000)
    parent_expansion: bool = True
    #: Source memories fetched to validate the derived memories in one result pool.
    derived_source_k: int = Field(default=6, ge=0, le=16)
    # Soft diversity cap before the primary cut. Zero preserves score order. Overflow
    # fills spare slots; exact hits, selected single documents and companions are exempt.
    max_chunks_per_document: int = Field(default=0, ge=0, le=50)
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
    # Primary memories only: graph source companions have their own bounded retrieval
    # allowance and share the hard token budget, like required document companions.
    memories_max: int = RETRIEVAL.memory_recall_k
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
