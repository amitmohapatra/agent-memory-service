"""The remote model adapters, against a real HTTP server.

They must be indistinguishable from the in-process ones at the port boundary, because that
is the whole point: `models.*.provider` becomes a deployment choice rather than a code one.
The TEI wire format is pinned here so an upgrade that changes it fails loudly instead of
silently degrading retrieval.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from memory_service.adapters.models.remote import RemoteEmbedding, RemoteNLI, RemoteReranker
from memory_service.config.settings import EmbeddingSettings, NLISettings, RerankerSettings
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured

pytestmark = pytest.mark.contract


class FakeTEI:
    """A real socket speaking the text-embeddings-inference API."""

    def __init__(self, *, fail_times: int = 0, status: int = 200) -> None:
        self.requests: list[tuple[str, dict]] = []
        self._remaining_failures = fail_times
        self._status = status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - the stdlib's spelling
                status = outer._routes().get(self.path, 404)
                outer._send(self, status, {"model_id": "fake"} if status < 400 else {})

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"] or 0)) or b"{}"
                )
                outer.requests.append((self.path, body))
                if outer._remaining_failures > 0:
                    outer._remaining_failures -= 1
                    outer._send(self, 503, {"error": "warming up"})
                    return
                if outer._status >= 400:
                    outer._send(self, outer._status, {"error": "nope"})
                    return
                outer._send(self, 200, outer._answer(self.path, body))

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @staticmethod
    def _answer(path: str, body: dict) -> Any:
        if path == "/embed":
            return [[0.1, 0.2, 0.3] for _ in body["inputs"]]
        if path == "/rerank":
            # deliberately out of order: the adapter must sort
            return [
                {"index": i, "score": score}
                for i, score in enumerate([0.1, 0.9, 0.5][: len(body["texts"])])
            ]
        return [
            [
                {"label": "entailment", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "contradiction", "score": 0.1},
            ]
            for _ in body["inputs"]
        ]

    @staticmethod
    def _send(handler: BaseHTTPRequestHandler, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)

    def __enter__(self) -> FakeTEI:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _routes(self) -> dict[str, int]:
        """Which discovery endpoints this server answers. TEI has /info; a server must be
        explicit about every probe, or detection picks whichever it accidentally allows."""
        return {"/info": 200, "/models": 404, "/v1/models": 404}

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"


async def test_remote_embedding_returns_one_vector_per_text_in_order() -> None:
    with FakeTEI() as tei:
        provider = RemoteEmbedding(EmbeddingSettings(url=tei.url, dimension=3, batch_size=2))
        vectors = await provider.embed_documents(["a", "b", "c"])
        assert len(vectors) == 3
        assert await provider.embed_query("q") == [0.1, 0.2, 0.3]
        await provider.close()
    # batch_size=2 over three texts is two calls; an unbounded body is not sent
    assert [path for path, _ in tei.requests].count("/embed") == 3
    assert len(tei.requests[0][1]["inputs"]) == 2


async def test_the_fingerprint_names_the_remote_model() -> None:
    """Collection names embed this, so a served-model swap must create a new vector space."""
    with FakeTEI() as tei:
        first = RemoteEmbedding(EmbeddingSettings(url=tei.url, model="model-a", dimension=3))
        second = RemoteEmbedding(EmbeddingSettings(url=tei.url, model="model-b", dimension=3))
        assert first.fingerprint() != second.fingerprint()
        assert "model-a" in first.fingerprint()
        await first.close()
        await second.close()


async def test_remote_reranker_sorts_by_score_and_honours_top_k() -> None:
    with FakeTEI() as tei:
        provider = RemoteReranker(RerankerSettings(url=tei.url))
        results = await provider.rerank("q", ["x", "y", "z"], top_k=2)
        await provider.close()
    assert [r.index for r in results] == [1, 2], "best first, regardless of server order"
    assert len(results) == 2


async def test_remote_nli_normalises_the_label_head() -> None:
    with FakeTEI() as tei:
        provider = RemoteNLI(NLISettings(url=tei.url))
        scores = await provider.entail(["premise one", "premise two"], "a claim")
        await provider.close()
    assert len(scores) == 2
    assert scores[0].entailment == pytest.approx(0.7)
    assert sum([scores[0].entailment, scores[0].neutral, scores[0].contradiction]) == pytest.approx(
        1.0
    )


async def test_a_warming_up_server_is_retried_and_a_bad_request_is_not() -> None:
    """A model server that is still loading answers 503; that is worth waiting for. A 4xx is
    the caller's fault and retrying it only spends the request's deadline."""
    with FakeTEI(fail_times=2) as tei:
        provider = RemoteEmbedding(EmbeddingSettings(url=tei.url, dimension=3, max_retries=3))
        assert await provider.embed_query("q") == [0.1, 0.2, 0.3]
        await provider.close()
    assert len(tei.requests) == 3, "two 503s then success"

    with FakeTEI(status=400) as tei:
        provider = RemoteEmbedding(EmbeddingSettings(url=tei.url, dimension=3, max_retries=3))
        with pytest.raises(DependencyUnavailable, match="400"):
            await provider.embed_query("q")
        await provider.close()
    assert len(tei.requests) == 1, "a 400 must not be retried"


async def test_a_remote_provider_without_a_url_fails_at_construction() -> None:
    with pytest.raises(ProviderNotConfigured, match="url"):
        RemoteEmbedding(EmbeddingSettings())


class FakeGateway(FakeTEI):
    """An OpenAI-compatible gateway: identifies itself at /v1/models, not /info."""

    @staticmethod
    def _answer(path: str, body: dict) -> Any:
        if path == "/v1/embeddings":
            # deliberately shuffled: the spec permits any order, and a permuted batch
            # misassigns every vector to the wrong chunk
            rows = [
                {"index": i, "embedding": [float(i), 0.2, 0.3]} for i in range(len(body["input"]))
            ]
            return {"data": list(reversed(rows)), "model": body["model"]}
        return {
            "results": [
                {"index": i, "relevance_score": s}
                for i, s in enumerate([0.1, 0.9, 0.5][: len(body["documents"])])
            ]
        }

    def _routes(self) -> dict[str, int]:
        return {"/info": 404, "/models": 404, "/v1/models": 200}


async def test_the_same_adapter_speaks_whichever_dialect_the_server_announces() -> None:
    """One `url` setting, no second fact to get wrong.

    Asking an operator to supply both a URL *and* the dialect it speaks is asking them to
    keep two things in agreement when the server already knows the answer. TEI answers
    /info; an OpenAI-compatible gateway answers /v1/models.
    """
    with FakeGateway() as gateway:
        provider = RemoteEmbedding(EmbeddingSettings(url=gateway.url, dimension=3, batch_size=8))
        vectors = await provider.embed_documents(["a", "b", "c"])
        await provider.close()
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0], "vectors must follow the input order"
    assert any(path == "/v1/embeddings" for path, _ in gateway.requests)


async def test_a_gateway_that_drops_a_vector_is_an_error_not_a_silent_mismatch() -> None:
    class Short(FakeGateway):
        @staticmethod
        def _answer(path: str, body: dict) -> Any:
            return {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}

    with Short() as gateway:
        provider = RemoteEmbedding(EmbeddingSettings(url=gateway.url, dimension=3))
        with pytest.raises(DependencyUnavailable, match="1 vectors for 2 inputs"):
            await provider.embed_documents(["a", "b"])
        await provider.close()


async def test_the_reranker_reads_whichever_score_field_the_dialect_uses() -> None:
    with FakeGateway() as gateway:
        provider = RemoteReranker(RerankerSettings(url=gateway.url, model="rerank-v3"))
        results = await provider.rerank("q", ["x", "y", "z"], top_k=2)
        await provider.close()
    assert [r.index for r in results] == [1, 2], "relevance_score, best first"


async def test_a_server_that_announces_nothing_is_refused_with_a_reason() -> None:
    class Silent(FakeTEI):
        def _routes(self) -> dict[str, int]:
            return {"/info": 404, "/models": 404, "/v1/models": 404}

    with Silent() as server:
        provider = RemoteEmbedding(EmbeddingSettings(url=server.url, dimension=3))
        with pytest.raises(DependencyUnavailable, match="answered none of"):
            await provider.embed_query("q")
        await provider.close()


async def test_the_fingerprint_follows_the_served_model_not_the_declared_one() -> None:
    """The failure this prevents: a URL pointing at a different encoder than the config names.

    The fingerprint names the vector collection. Built from the declared name, vectors
    produced by a *different* served model land in the collection belonging to the declared
    one — no error, just retrieval that quietly stops working.
    """

    class ServesSomethingElse(FakeTEI):
        def _routes(self) -> dict[str, int]:
            return {"/info": 200, "/models": 404, "/v1/models": 404}

    with ServesSomethingElse() as server:
        provider = RemoteEmbedding(
            EmbeddingSettings(url=server.url, model="granite-embedding-small", dimension=3)
        )
        before = provider.fingerprint()
        await provider.identify()  # the server says it serves "fake"
        after = provider.fingerprint()
        await provider.close()
    assert "granite-embedding-small" in before
    assert "fake" in after, "after the handshake the fingerprint must name what is served"
    assert before != after


async def test_a_dimension_that_does_not_match_the_server_is_refused_at_startup() -> None:
    """The declared dimension creates the collection; the server decides the real width."""
    with FakeTEI() as server:
        provider = RemoteEmbedding(EmbeddingSettings(url=server.url, dimension=1024))
        with pytest.raises(ProviderNotConfigured, match="3 dimensions"):
            await provider.verify()
        await provider.close()


async def test_verify_accepts_a_server_that_matches() -> None:
    with FakeTEI() as server:
        provider = RemoteEmbedding(EmbeddingSettings(url=server.url, dimension=3))
        await provider.verify()
        await provider.close()


def test_a_fingerprint_is_always_a_legal_collection_name() -> None:
    """Fingerprints become Qdrant collection names, which reject ``:`` and ``/``.

    The first version of these adapters returned ``remote:models/granite-...:384``. Wiring
    succeeded, the service reported healthy, and every write then failed with a 422 naming
    the collection rather than the cause — visible only by running the stack.
    """
    import re

    from memory_service.adapters.models.remote import _slug

    for raw in (
        "models/granite-embedding-small-english-r2",
        "/models/ms-marco-MiniLM-L6-v2",
        "text-embedding-3-small",
        "vendor:model@v2.1",
    ):
        assert re.fullmatch(r"[a-z0-9-]+", _slug(raw)), f"{raw!r} -> {_slug(raw)!r}"
