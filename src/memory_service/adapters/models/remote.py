"""Model ports served over HTTP, so inference is a deployment rather than a library.

In-process inference welded three models into both the API and the worker: ~6 minutes of
startup, two resident copies of the same weights, and no way to scale ingestion separately
from query traffic — indexing a corpus starved the API because they compete for the same
cores. None of that is inherent; it is only what loading weights inside the request process
costs.

These adapters speak the HuggingFace *text-embeddings-inference* (TEI) API, which serves
embedding, reranking and sequence classification on CPU. The service keeps knowing a URL and
a model *name*; no inference library is imported here, which is what lets the model tier be
upgraded, batched, or moved to a GPU without touching this code base.

``fingerprint()`` returns the remote model identity, so the existing collection naming keeps
working unchanged: swapping the served model still creates a new vector space rather than
silently mixing two.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from memory_service.config.settings import EmbeddingSettings, NLISettings, RerankerSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.observability.logging import get_logger
from memory_service.observability.tracing import span
from memory_service.ports.models import NLIScore, ProviderInfo, RerankResult
from memory_service.ports.search import SparseVector

log = get_logger(__name__)

#: Statuses where the same request may succeed later; anything else is the caller's fault.
#:
#: Deliberately this module's own, not the gateway client's identical set: these are the
#: service's in-cluster model servers, and pulling an LLM-gateway dependency into the embed
#: path to share eight integers would be the wrong direction for a coupling.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


#: How a served model announces itself. TEI answers ``/info`` with a model_id; every
#: OpenAI-compatible gateway answers ``/v1/models``. Detecting is better than declaring: the
#: operator already has to supply the URL, and making them also name the dialect is a second
#: fact that can disagree with the first.
@dataclass(frozen=True)
class Dialect:
    """One inference server's wire format. Adding a server means adding an entry below.

    Three servers, three shapes: TEI answers ``/info`` and embeds at ``/embed``; Infinity
    answers ``/models`` and embeds at ``/embeddings``; an OpenAI-compatible gateway answers
    ``/v1/models`` and embeds at ``/v1/embeddings``. Encoding that as branches inside each
    method meant every new server touched three call sites — the switch statement this code
    base already decided it did not want.
    """

    name: str
    #: GET path that both identifies the server and names the model it serves
    probe: str
    identity: Callable[[Any], str]
    embed_path: str
    embed_body: Callable[[str, list[str]], dict[str, Any]]
    embed_parse: Callable[[Any, int], list[list[float]]]
    rerank_path: str
    rerank_body: Callable[[str, str, list[str], int], dict[str, Any]]
    rerank_parse: Callable[[Any], list[dict[str, Any]]]


def _openai_rows(payload: Any, expected: int) -> list[list[float]]:
    """``data`` re-sorted by ``index``: the specification permits any order, and a permuted
    batch misassigns every vector to the wrong chunk — degraded retrieval, never an error."""
    rows = sorted(payload["data"], key=lambda row: int(row.get("index", 0)))
    if len(rows) != expected:
        raise DependencyUnavailable(
            f"embedding server returned {len(rows)} vectors for {expected} inputs"
        )
    return [[float(x) for x in row["embedding"]] for row in rows]


DIALECTS: tuple[Dialect, ...] = (
    Dialect(
        name="tei",
        probe="/info",
        identity=lambda body: str(body.get("model_id") or body.get("model") or "unknown"),
        embed_path="/embed",
        embed_body=lambda model, texts: {"inputs": texts},
        embed_parse=lambda payload, expected: [[float(x) for x in row] for row in payload],
        rerank_path="/rerank",
        rerank_body=lambda model, query, docs, top_k: {
            "query": query,
            "texts": docs,
            "raw_scores": False,
        },
        rerank_parse=list,
    ),
    Dialect(
        name="infinity",
        probe="/models",
        identity=lambda body: str((body.get("data") or [{}])[0].get("id") or "unknown"),
        embed_path="/embeddings",
        embed_body=lambda model, texts: {"model": model, "input": texts},
        embed_parse=_openai_rows,
        rerank_path="/rerank",
        rerank_body=lambda model, query, docs, top_k: {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": top_k,
        },
        rerank_parse=lambda payload: list(payload.get("results", payload)),
    ),
    Dialect(
        name="openai",
        probe="/v1/models",
        identity=lambda body: str((body.get("data") or [{}])[0].get("id") or "unknown"),
        embed_path="/v1/embeddings",
        embed_body=lambda model, texts: {"model": model, "input": texts},
        embed_parse=_openai_rows,
        rerank_path="/v1/rerank",
        rerank_body=lambda model, query, docs, top_k: {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": top_k,
        },
        rerank_parse=lambda payload: list(payload.get("results", payload)),
    ),
)


async def identify(client: httpx.AsyncClient) -> tuple[Dialect, str]:
    """``(dialect, served_model_id)`` — what this server speaks and what it is serving.

    The served id matters as much as the dialect. A URL says where to send text, not which
    model answers: point ``EMBEDDING__URL`` at a server running a different encoder and
    nothing downstream notices, because the fingerprint naming the vector collection was
    built from the *declared* name. Vectors from the wrong model then land in the
    right-looking collection, which is silent corruption rather than an error.
    """
    for dialect in DIALECTS:
        try:
            response = await client.get(dialect.probe, timeout=5.0)
            if response.status_code < 400:
                return dialect, dialect.identity(response.json())
        except (httpx.HTTPError, ValueError, IndexError, KeyError):
            continue
    raise DependencyUnavailable(
        "could not identify the model server: it answered none of "
        + ", ".join(d.probe for d in DIALECTS)
    )


def _slug(name: str) -> str:
    """A fingerprint fragment safe to put in a collection name.

    Fingerprints end up in Qdrant collection names, which reject ``:`` and ``/``. The
    in-process adapters already spell theirs with hyphens and the last path segment
    (``st-granite-embedding-small-english-r2-torch-d384``); a served model id is a path or a
    vendor string, so it has to be reduced the same way.
    """
    tail = name.rstrip("/").rsplit("/", 1)[-1]
    return "".join(c if c.isalnum() or c == "-" else "-" for c in tail).strip("-").lower()


def _same_model(declared: str, served: str) -> bool:
    """Whether two spellings name the same model.

    A served id is often a path (``/models/granite-embedding-small-english-r2``) while the
    configured name is a repository (``ibm-granite/granite-embedding-small-english-r2``).
    Comparing the last path segment matches those without pretending that two genuinely
    different models are the same.
    """
    return declared.rstrip("/").rsplit("/", 1)[-1] == served.rstrip("/").rsplit("/", 1)[-1]


class _RemoteModel:
    """Shared plumbing: one bounded client, retries only where they can help."""

    def __init__(self, base_url: str, model: str, *, timeout: float, retries: int) -> None:
        if not base_url:
            raise ProviderNotConfigured("a served model needs url set to its inference server")
        self.declared_model = model
        self.retries = retries
        self._dialect: Dialect | None = None
        self._served_model: str | None = None
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
            headers={"Content-Type": "application/json"},
        )

    async def _post(self, path: str, body: dict[str, Any], *, use: str) -> Any:
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = await self._client.post(path, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error = DependencyUnavailable(f"{use} server unreachable ({type(exc).__name__})")
            else:
                if response.status_code < 400:
                    return response.json()
                error = DependencyUnavailable(
                    f"{use} server returned {response.status_code}: {response.text[:200]}"
                )
                if response.status_code not in RETRYABLE_STATUS:
                    raise error
            if attempt < self.retries:
                await asyncio.sleep(0.2 * (2**attempt))
        raise error or DependencyUnavailable(f"{use} call failed")

    async def identify(self) -> tuple[Dialect, str]:
        """Ask the server who it is, once. Cached for the life of the adapter."""
        if self._dialect is None:
            self._dialect, self._served_model = await identify(self._client)
            if (
                self.declared_model
                and self._served_model not in ("unknown", "")
                and not _same_model(self.declared_model, self._served_model)
            ):
                log.warning(
                    "model.served_name_differs",
                    declared=self.declared_model,
                    served=self._served_model,
                    detail="the served identity is what the fingerprint uses",
                )
        return self._dialect, self._served_model or self.declared_model

    @property
    def model(self) -> str:
        """The served model where known, the declared one before the handshake."""
        return self._served_model or self.declared_model

    async def ping(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 500

    async def close(self) -> None:
        await self._client.aclose()


class RemoteEmbedding(_RemoteModel):
    """``POST /embed`` — one vector per text, order preserved."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        super().__init__(
            settings.url or "",
            settings.model,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
        )
        self.dimension = settings.dimension
        self.batch_size = settings.batch_size
        self.info = ProviderInfo(
            name="remote-embedding",
            version=settings.model,
            license="see model card",
            origin="text-embeddings-inference",
            # "local" the way Qdrant and Dragonfly are local: a separate process inside the
            # operator's deployment, not a third-party API. `provider_policy.allow_remote_models`
            # exists to stop data leaving the deployment, and a self-hosted inference sidecar
            # does not — marking it "remote" would have made the policy block the very thing
            # it was written to permit. Point base_url at a hosted endpoint and that judgement
            # changes; that is a deployment decision, as it already is for the vector store.
            locality="local",
            data_residency="deployment",
        )

    def fingerprint(self) -> str:
        """Bound to the *served* model, so pointing at a different one creates a new vector
        space instead of writing foreign vectors into the existing collection."""
        return f"remote-{_slug(self.model)}-d{self.dimension}"

    async def verify(self) -> None:
        """Startup handshake: who is serving, and does it return the promised shape?

        The declared dimension creates the Qdrant collection. If the server actually returns
        a different width, every write fails later — or worse, if the widths happen to agree
        while the models differ, retrieval quietly degrades with no error anywhere. One probe
        embedding at startup turns both into a refusal a human can read.
        """
        dialect, served = await self.identify()
        probe = await self.embed_query("dimension probe")
        if len(probe) != self.dimension:
            raise ProviderNotConfigured(
                f"embedding server at this url serves {served!r} returning {len(probe)} "
                f"dimensions, but models.embedding.dimension is {self.dimension}. Set the "
                "dimension to match the served model (and reindex: the vector space changed)."
            )
        log.info(
            "model.verified",
            role="embedding",
            dialect=dialect,
            served=served,
            dimension=self.dimension,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        dialect, _ = await self.identify()
        vectors: list[list[float]] = []
        # the server batches internally, but an unbounded body is still a bad idea
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            with span("embedding.remote", count=len(chunk), dialect=dialect.name):
                payload = await self._post(
                    dialect.embed_path, dialect.embed_body(self.model, chunk), use="embedding"
                )
            vectors.extend(dialect.embed_parse(payload, len(chunk)))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class RemoteReranker(_RemoteModel):
    """``POST /rerank`` — cross-encoder scores for one query against many documents.

    This is the model that decides capacity: retrieval sends ``candidate_k`` pairs per
    request, so it is the first thing worth batching across requests or moving to its own
    tier. Served separately, that becomes a deployment decision instead of a code change.
    """

    def __init__(self, settings: RerankerSettings) -> None:
        super().__init__(
            settings.url or "",
            settings.model,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
        )
        self.info = ProviderInfo(
            name="remote-reranker",
            version=settings.model,
            license="see model card",
            origin="text-embeddings-inference",
            # "local" the way Qdrant and Dragonfly are local: a separate process inside the
            # operator's deployment, not a third-party API. `provider_policy.allow_remote_models`
            # exists to stop data leaving the deployment, and a self-hosted inference sidecar
            # does not — marking it "remote" would have made the policy block the very thing
            # it was written to permit. Point base_url at a hosted endpoint and that judgement
            # changes; that is a deployment decision, as it already is for the vector store.
            locality="local",
            data_residency="deployment",
        )

    def fingerprint(self) -> str:
        return f"remote-{_slug(self.model)}"

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int
    ) -> list[RerankResult]:
        if not documents:
            return []
        dialect, _ = await self.identify()
        with span("reranker.remote", count=len(documents), dialect=dialect.name):
            payload = await self._post(
                dialect.rerank_path,
                dialect.rerank_body(self.model, query, list(documents), top_k),
                use="reranker",
            )
        results = [
            RerankResult(
                index=int(row["index"]),
                score=float(row.get("score", row.get("relevance_score", 0.0))),
            )
            for row in dialect.rerank_parse(payload)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class RemoteNLI(_RemoteModel):
    """``POST /predict`` — sequence classification over (premise, hypothesis) pairs.

    Only the grounding cascade calls this, so it is the model most worth deploying on its own
    schedule: verification traffic has nothing to do with retrieval volume.
    """

    #: label spellings vary by checkpoint; every MNLI head means the same three things
    _LABELS = {
        "entailment": "entailment",
        "entail": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
        "contradict": "contradiction",
        "label_0": "entailment",
        "label_1": "neutral",
        "label_2": "contradiction",
    }

    def __init__(self, settings: NLISettings) -> None:
        super().__init__(
            settings.url or "",
            settings.model,
            timeout=settings.timeout_seconds,
            retries=settings.max_retries,
        )
        self.representative = True
        self.info = ProviderInfo(
            name="remote-nli",
            version=settings.model,
            license="see model card",
            origin="text-embeddings-inference",
            # "local" the way Qdrant and Dragonfly are local: a separate process inside the
            # operator's deployment, not a third-party API. `provider_policy.allow_remote_models`
            # exists to stop data leaving the deployment, and a self-hosted inference sidecar
            # does not — marking it "remote" would have made the policy block the very thing
            # it was written to permit. Point base_url at a hosted endpoint and that judgement
            # changes; that is a deployment decision, as it already is for the vector store.
            locality="local",
            data_residency="deployment",
        )

    def fingerprint(self) -> str:
        return f"remote-{_slug(self.model)}"

    async def entail(self, premises: Sequence[str], hypothesis: str) -> list[NLIScore]:
        if not premises:
            return []
        with span("nli.remote", count=len(premises)):
            predictions = await self._post(
                "/predict",
                {"inputs": [[premise, hypothesis] for premise in premises]},
                use="nli",
            )
        return [self._score(row) for row in predictions]

    def _score(self, row: Any) -> NLIScore:
        buckets = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        for item in row if isinstance(row, list) else [row]:
            label = self._LABELS.get(str(item.get("label", "")).strip().casefold())
            if label:
                buckets[label] = float(item.get("score", 0.0))
        total = sum(buckets.values())
        if total <= 0:  # a head we do not recognise: neutral is the honest answer
            return NLIScore(entailment=0.0, neutral=1.0, contradiction=0.0)
        return NLIScore(**{k: v / total for k, v in buckets.items()})


class RemoteSparse(_RemoteModel):
    """``POST /sparse`` — SPLADE term weights, served on their own tier.

    This was the one model with no remote adapter, which meant "run the models separately"
    was only ever three-quarters true: SPLADE is a BERT-sized model (~110M parameters) and it
    runs on *every chunk at ingest*, so keeping it in-process put the largest per-document
    cost in the same container as the API. It is also the model most worth scaling on its own
    schedule, because ingest volume and query volume have nothing to do with each other.

    The port is synchronous — the indexer calls ``encode_documents`` directly — so the async
    HTTP call is bridged here rather than pushed up into the pipeline. That is deliberate:
    the alternative is changing the ``SparseEncoder`` protocol and every implementation of it
    to suit one transport.
    """

    info = ProviderInfo(
        name="remote-sparse",
        license="see model card",
        origin="served",
        locality="remote",
    )

    def __init__(self, base_url: str, model: str, *, timeout: float = 30.0, retries: int = 2):
        super().__init__(base_url, model, timeout=timeout, retries=retries)

    async def _encode(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        payload = await self._post("/sparse", {"inputs": list(texts)}, use="sparse")
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        return [
            SparseVector(
                indices=[int(i) for i in row.get("indices", [])],
                values=[float(v) for v in row.get("values", [])],
            )
            for row in rows
        ]

    def _run(self, texts: Sequence[str]) -> list[SparseVector]:
        """Bridge to the synchronous port without assuming whether a loop is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._encode(texts))
        # Called from inside a running loop: hand the coroutine to a worker thread with its
        # own loop. Blocking the caller's loop here would deadlock, and the port gives us no
        # way to await.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._encode(texts)).result()

    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return self._run(texts)

    def encode_query(self, text: str) -> SparseVector:
        out = self._run([text])
        return out[0] if out else SparseVector(indices=[], values=[])

    def fingerprint(self) -> str:
        return _slug(f"remote-sparse-{self._served_model or self.declared_model}")
