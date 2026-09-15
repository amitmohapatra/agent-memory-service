"""Unit tests for the deterministic pieces of baseline retrieval (M6)."""

from __future__ import annotations

import math

import pytest

from memory_service.adapters.models.embeddings import HashEmbedding
from memory_service.adapters.models.rerankers import LexicalReranker
from memory_service.adapters.models.sparse import Bm25SparseEncoder, term_id, tokenize
from memory_service.domain.conversation import Message
from memory_service.domain.enums import MessageKind, MessageRole, QueryType
from memory_service.domain.ids import content_hash, new_id
from memory_service.modules.context.builder import render_window
from memory_service.modules.retrieval.engine import Candidate, _dedup, rrf_fuse
from memory_service.modules.retrieval.router import QueryRouter
from memory_service.ports.search import SearchHit

pytestmark = pytest.mark.unit


# --- router --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("show me chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH", QueryType.EXACT_IDENTIFIER),
        ("what is the status of ACME-1234?", QueryType.EXACT_IDENTIFIER),
        ("what did I say earlier in this thread about pricing?", QueryType.CONVERSATION_HISTORY),
        ("what is my timezone?", QueryType.USER_MEMORY),
        ("why did we decide to use Postgres?", QueryType.DECISION),
        ("give me a summary of the main risks", QueryType.GLOBAL_SUMMARY),
        ("why did Adjusted EBITDA increase despite lower revenue?", QueryType.DOCUMENT_MULTI_HOP),
        ("who owns the billing service?", QueryType.ENTITY_RELATION),
        ("what changed since last quarter?", QueryType.TEMPORAL),
        ("what does table 2 on page 11 say?", QueryType.DOCUMENT_LOCAL),
        ("restructuring savings", QueryType.GENERAL_SEMANTIC),
    ],
)
def test_router_types(query: str, expected: QueryType) -> None:
    routed = QueryRouter().route(query)
    assert routed.query_type is expected, routed.signals


def test_router_is_deterministic_and_sets_needs() -> None:
    r = QueryRouter()
    a = r.route("what is my favourite editor?", has_thread=True)
    b = r.route("what is my favourite editor?", has_thread=True)
    assert a == b
    assert a.needs_memories and not a.needs_knowledge and a.needs_conversation
    exact = r.route("open msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH", has_thread=False)
    assert exact.identifiers == ["msg_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"]
    assert not exact.needs_conversation and not exact.needs_memories
    multi = r.route("compare FY25 and FY26 margins")
    assert multi.needs_graph and multi.query_type is QueryType.DOCUMENT_MULTI_HOP


# --- fusion -----------------------------------------------------------------------------


def _hits(retriever: str, *ids: str) -> list[SearchHit]:
    return [
        SearchHit(record_id=i, score=1.0 - n / 10, retriever=retriever, payload={"n": n})  # type: ignore[arg-type]
        for n, i in enumerate(ids)
    ]


def test_rrf_rewards_agreement_and_is_stable() -> None:
    fused = rrf_fuse([_hits("dense", "a", "b", "c"), _hits("bm25", "b", "a", "d")], k=60)
    ids = [rid for rid, _, _, _ in fused]
    # a and b appear in both lists -> ahead of c and d
    assert set(ids[:2]) == {"a", "b"} and ids[2:] == ["c", "d"]
    a = next(f for f in fused if f[0] == "a")
    assert math.isclose(a[1], 1 / 61 + 1 / 62)
    assert a[2] == ["dense", "bm25"]
    # ties broken by id -> deterministic
    tie = rrf_fuse([_hits("dense", "z"), _hits("bm25", "y")])
    assert [t[0] for t in tie] == ["y", "z"]


def test_dedup_merges_retrievers_and_keeps_best_score() -> None:
    c1 = Candidate(record_id="x", kind="chunk", text="t", score=0.2, retrievers=["dense"])
    c2 = Candidate(record_id="x", kind="chunk", text="t", score=0.9, retrievers=["bm25"])
    c3 = Candidate(record_id="y", kind="chunk", text="u", score=0.5, retrievers=["bm25"])
    out = _dedup([c1, c2, c3])
    assert [c.record_id for c in out] == ["x", "y"]
    assert out[0].score == 0.9 and out[0].retrievers == ["bm25", "dense"]


# --- sparse / dense / rerank ------------------------------------------------------------


def test_tokenizer_stems_and_drops_stopwords() -> None:
    toks = tokenize("The restructuring charges were excluded from Adjusted EBITDA!")
    assert "the" not in toks and "were" not in toks and "from" not in toks
    assert "ebitda" in toks and "adjust" in toks and "charg" in toks
    assert toks == tokenize("the restructuring charges were excluded from adjusted ebitda")
    assert term_id("ebitda") == term_id("ebitda") and term_id("a") != term_id("b")


def test_bm25_encoder_saturates_and_is_stable() -> None:
    enc = Bm25SparseEncoder()
    one, many = enc.encode_documents(["ebitda", "ebitda " * 50])
    assert one.indices == many.indices
    # saturated term frequency: 50 repeats is worth < k1 + 1 times a single occurrence
    assert 1.0 < many.values[0] / one.values[0] < 2.2
    q = enc.encode_query("Adjusted EBITDA adjusted")
    assert set(q.values) == {1.0} and len(q.indices) == len(set(q.indices))
    assert enc.encode_query("x y") == enc.encode_query("x y")
    assert enc.fingerprint() == "bm25-v1-k1.2-b0.75"
    assert enc.encode_query("the of and").indices == []


async def test_hash_embedding_is_deterministic_unit_norm() -> None:
    emb = HashEmbedding(dimension=64)
    a, b = await emb.embed_documents(["revenue declined", "revenue declined"])
    assert a == b and len(a) == 64
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-6)
    q = await emb.embed_query("revenue declined")
    other = await emb.embed_query("completely unrelated words here")
    dot = sum(x * y for x, y in zip(a, q, strict=True))
    dot_other = sum(x * y for x, y in zip(a, other, strict=True))
    assert dot > dot_other
    assert emb.fingerprint() == "hash-v1-d64" and emb.dimension == 64
    assert emb.info.locality == "local"


async def test_lexical_reranker_orders_by_overlap() -> None:
    rr = LexicalReranker()
    docs = ["nothing relevant here", "Adjusted EBITDA increased despite lower revenue", "EBITDA"]
    out = await rr.rerank("why did adjusted ebitda increase", docs, top_k=2)
    assert [r.index for r in out] == [1, 2]
    assert out[0].score >= out[1].score
    assert rr.fingerprint().startswith("lexical")


# --- context window ----------------------------------------------------------------------


def _msg(content: str, *, kind: MessageKind = MessageKind.VISIBLE, seq: int = 1) -> Message:
    return Message(
        message_id=new_id("message"),
        tenant_id="t",
        thread_id="thr",
        session_id="ses",
        turn_id="trn",
        sequence=seq,
        role=MessageRole.USER,
        kind=kind,
        content=content,
        content_hash=content_hash(content),
        author_principal="user:u1",
    )


def test_render_window_keeps_recent_visible_messages_within_budget() -> None:
    msgs = [_msg("old " * 200, seq=1), _msg("hidden", kind=MessageKind.INTERNAL, seq=2)]
    msgs += [_msg(f"recent {i}", seq=3 + i) for i in range(3)]
    w = render_window("thr", msgs, token_budget=40)
    assert w.thread_id == "thr"
    assert "hidden" not in w.rendered and "old" not in w.rendered
    assert w.rendered.splitlines() == ["USER: recent 0", "USER: recent 1", "USER: recent 2"]
    assert w.message_ids == [m.message_id for m in msgs[2:]]
    assert 0 < w.token_estimate <= 40
    # a single oversized message is still kept (never an empty window when there is content)
    big = render_window("thr", [_msg("x" * 4000)], token_budget=10)
    assert len(big.message_ids) == 1
