"""The model server's wire contract, one role at a time.

The point of this server is that the service stops loading weights and speaks HTTP to a
separate process per model. That only holds if what each role puts on the wire is what the
matching ``Remote*`` adapter expects to read — a mismatch there is invisible until a
deployment switches to the served tier and every call 404s or silently returns nothing.

The models themselves are stubbed: this is about the contract, not the weights.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memory_service.ports.models import NLIScore, RerankResult
from memory_service.ports.search import SparseVector
from memory_service.tools import model_server


class _Embedding:
    async def embed_documents(self, texts):
        return [[float(len(t)), 0.5] for t in texts]


class _Reranker:
    async def rerank(self, query, documents, *, top_k):
        # deliberately returned out of order: the server is responsible for sorting
        return [RerankResult(index=i, score=float(i) / 10) for i in range(len(documents))][:top_k]


class _NLI:
    async def entail(self, premises, hypothesis):
        return [NLIScore(entailment=0.7, neutral=0.2, contradiction=0.1)]


class _Sparse:
    def encode_documents(self, texts):
        return [SparseVector(indices=[1, 7], values=[0.4, 0.6]) for _ in texts]


def test_every_role_is_rejected_unless_known(monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_MODEL_ROLE", "nonsense")
    with pytest.raises(SystemExit, match="MEMORY_MODEL_ROLE"):
        model_server._role()


@pytest.mark.parametrize("role", list(model_server.ROLES))
def test_info_identifies_the_role_and_model(role: str) -> None:
    """Dialect detection reads /info; a server answering for the wrong model must be visible."""
    client = TestClient(model_server.create_app(role))
    body = client.get("/info").json()
    assert body["role"] == role
    assert body["model_id"], "a served model must say which model it is serving"


@pytest.mark.parametrize(
    ("role", "path"),
    [("embed", "/embed"), ("rerank", "/rerank"), ("nli", "/predict"), ("sparse", "/sparse")],
)
def test_a_role_exposes_only_its_own_endpoint(role: str, path: str) -> None:
    """One process serves one model. A container that answers every endpoint has the memory
    profile of the monolith this split exists to avoid, and would hide a misconfigured role."""
    app = model_server.create_app(role)
    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert path in paths
    others = {"/embed", "/rerank", "/predict", "/sparse"} - {path}
    assert not (others & paths), f"{role} also exposes {others & paths}"


def test_rerank_returns_index_score_pairs_sorted_by_score(monkeypatch) -> None:
    app = model_server.create_app("rerank")
    monkeypatch.setattr(model_server, "_role", lambda: "rerank")
    client = TestClient(app)
    # inject the stub into the lazy cache by calling through a patched constructor
    import memory_service.adapters.models.rerankers as rr

    monkeypatch.setattr(rr, "CrossEncoderReranker", lambda settings: _Reranker())
    body = client.post("/rerank", json={"query": "q", "texts": ["a", "b", "c"]}).json()
    assert [row["index"] for row in body] == [2, 1, 0], "must be ordered by score, best first"
    assert all({"index", "score"} == set(row) for row in body)


def test_rerank_accepts_documents_as_well_as_texts(monkeypatch) -> None:
    import memory_service.adapters.models.rerankers as rr

    monkeypatch.setattr(rr, "CrossEncoderReranker", lambda settings: _Reranker())
    client = TestClient(model_server.create_app("rerank"))
    body = client.post("/rerank", json={"query": "q", "texts": [], "documents": ["a", "b"]}).json()
    assert len(body) == 2, "clients that spell the field `documents` must not get an empty list"


def test_nli_returns_all_three_labels(monkeypatch) -> None:
    """Sending only the winning label would discard the contradiction signal, which is the
    half the grounding cascade uses to detect conflict."""
    import memory_service.adapters.models.nli as nli

    monkeypatch.setattr(nli, "TransformersNLI", lambda settings: _NLI())
    client = TestClient(model_server.create_app("nli"))
    body = client.post("/predict", json={"inputs": [["a premise", "a hypothesis"]]}).json()
    labels = {row["label"] for row in body[0]}
    assert labels == {"entailment", "neutral", "contradiction"}


def test_sparse_returns_indices_and_values(monkeypatch) -> None:
    import memory_service.adapters.models.advanced as advanced

    monkeypatch.setattr(
        advanced, "FastEmbedSparseEncoder", lambda model, model_path=None: _Sparse()
    )
    client = TestClient(model_server.create_app("sparse"))
    body = client.post("/sparse", json={"inputs": ["one", "two"]}).json()
    assert len(body) == 2
    assert body[0] == {"indices": [1, 7], "values": [0.4, 0.6]}


def test_embed_accepts_a_bare_string(monkeypatch) -> None:
    import memory_service.adapters.models.embeddings as emb

    monkeypatch.setattr(emb, "SentenceTransformersEmbedding", lambda settings: _Embedding())
    client = TestClient(model_server.create_app("embed"))
    body = client.post("/embed", json={"inputs": "one text"}).json()
    assert len(body) == 1 and len(body[0]) == 2


def test_the_http_hint_schema_matches_the_domain_one() -> None:
    """Every hint the pipeline honours must be expressible over HTTP.

    ``ProcessingHints`` is splatted straight into the domain model
    (``ProcessingHints(**body.hints.model_dump())``), so a field present in one and absent in
    the other is silently unreachable rather than a type error. That is exactly how
    ``custom_type`` went missing: the domain gained it, the HTTP schema did not, and
    ``memory_type=CUSTOM`` stayed impossible to submit through the API while looking supported
    in the enum and documented in the field description.
    """
    from memory_service.api.schemas.conversation import ProcessingHintsIn
    from memory_service.domain.observation import ProcessingHints

    http = set(ProcessingHintsIn.model_fields)
    domain = set(ProcessingHints.model_fields)
    assert http == domain, (
        f"hint schema drift — only in domain: {sorted(domain - http)}; "
        f"only in HTTP: {sorted(http - domain)}"
    )
