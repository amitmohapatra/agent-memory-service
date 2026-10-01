"""Point-in-time memory search (``as_of``: valid time, ``known_at``: knowledge time) and the
searchable thread episode: superseded memories stay in the index as history that only a
point-in-time search reads, and a thread is one episode record."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from memory_service.modules.conversation.summary import (
    EPISODE_INDEX_DELAY_SECONDS,
    TASK_EPISODE_INDEX,
    enqueue_episode_index,
)
from memory_service.modules.rag.indexer import (
    MEMORIES,
    TIME_MAX,
    TIME_MIN,
    episode_id,
    episode_index_text,
    memory_time_payload,
)
from memory_service.modules.retrieval.engine import PointInTime, point_in_time_filter
from memory_service.ports.search import SearchFilter, SearchRecord, VectorName
from tests.unit import test_llm_retrieval as base

pytestmark = pytest.mark.unit

parts = base.parts

JAN = datetime(2026, 1, 10, tzinfo=UTC)
FEB = datetime(2026, 2, 10, tzinfo=UTC)
MAR = datetime(2026, 3, 10, tzinfo=UTC)
APR = datetime(2026, 4, 10, tzinfo=UTC)


def test_without_a_moment_the_filter_is_the_ordinary_current_one() -> None:
    flt = point_in_time_filter(SearchFilter(tenant_id="t"), PointInTime())
    assert flt.must["current"] is True
    assert flt.within == {}


def test_a_moment_lets_history_in_and_cuts_it_to_that_moment() -> None:
    flt = point_in_time_filter(SearchFilter(tenant_id="t"), PointInTime(as_of=FEB))
    assert "current" not in flt.must
    assert flt.within == {"valid_from": (None, FEB), "valid_to": (FEB, None)}


def test_known_at_intersects_an_observed_range_rather_than_replacing_it() -> None:
    observed = SearchFilter(tenant_id="t", within={"observed_at": (JAN, APR)})
    flt = point_in_time_filter(observed, PointInTime(known_at=FEB))
    assert flt.within["observed_at"] == (JAN, FEB)
    assert flt.within["known_to"] == (FEB, None)


def test_the_time_payload_closes_knowledge_only_for_a_superseded_memory() -> None:
    def memory(status: str, valid_from=None, valid_to=None) -> SimpleNamespace:
        temporal = SimpleNamespace(
            status=SimpleNamespace(value=status), valid_from=valid_from, valid_to=valid_to
        )
        return SimpleNamespace(temporal=temporal, updated_at=MAR)

    current = memory_time_payload(memory("CURRENT"))  # type: ignore[arg-type]
    assert current == {
        "valid_from": TIME_MIN.isoformat(),
        "valid_to": TIME_MAX.isoformat(),
        "known_to": TIME_MAX.isoformat(),
    }
    old = memory_time_payload(memory("SUPERSEDED", JAN, MAR))  # type: ignore[arg-type]
    assert old == {
        "valid_from": JAN.isoformat(),
        "valid_to": MAR.isoformat(),
        "known_to": MAR.isoformat(),
    }


async def test_the_store_answers_as_of_and_known_at(parts) -> None:
    """Paris (Jan-Mar, replaced in March) then Lisbon (from March, learned in March)."""
    indexer, embedding, store = parts
    engine = base._engine(parts)
    rows = {
        "mem_paris": (
            "the user lives in Paris",
            False,
            {"valid_from": JAN, "valid_to": MAR, "known_to": MAR, "observed_at": JAN},
        ),
        "mem_lisbon": (
            "the user lives in Lisbon",
            True,
            {"valid_from": MAR, "valid_to": TIME_MAX, "known_to": TIME_MAX, "observed_at": MAR},
        ),
    }
    texts = [text for text, _, _ in rows.values()]
    dense = await embedding.embed_documents(texts)
    sparse = indexer.sparse.encode_documents(texts)
    await store.upsert(
        [
            SearchRecord(
                record_id=rid,
                collection=indexer.collection(MEMORIES),
                tenant_id="t",
                dense={VectorName.DENSE_ML: dense[i]},
                sparse=sparse[i],
                payload={
                    "kind": "memory",
                    "text": text,
                    "visibility_keys": ["tenant:t"],
                    "current": current,
                    **{k: v.isoformat() for k, v in times.items()},
                },
            )
            for i, (rid, (text, current, times)) in enumerate(rows.items())
        ]
    )

    async def found(at: PointInTime | None = None) -> set[str]:
        hits = await engine._hybrid(
            "where does the user live",
            base.VISIBILITY,
            kind="memory",
            document_ids=None,
            at=at,
        )
        return {h.record_id for h in hits}

    assert await found() == {"mem_lisbon"}, "an ordinary search reads only what is current"
    assert await found(PointInTime(as_of=FEB)) == {"mem_paris"}
    assert await found(PointInTime(as_of=APR)) == {"mem_lisbon"}
    assert await found(PointInTime(known_at=FEB)) == {"mem_paris"}
    assert await found(PointInTime(known_at=APR)) == {"mem_lisbon"}


def test_an_episode_is_one_record_per_thread_and_leads_with_its_dates() -> None:
    assert episode_id("thr_1") == "epi_thr_1"
    text = episode_index_text("We chose vendor X.", "Vendor review", JAN, MAR)
    assert text == "[2026-01-10 to 2026-03-10] conversation Vendor review: We chose vendor X."
    assert episode_index_text("Hi.", None, JAN, JAN) == "[2026-01-10] conversation Hi."


async def test_every_message_in_a_window_shares_one_episode_job() -> None:
    enqueued = []
    uow: Any = SimpleNamespace(enqueue=lambda spec: _record(enqueued, spec))
    start = 1_000 * EPISODE_INDEX_DELAY_SECONDS
    await enqueue_episode_index(uow, "t", "thr_1", now=start + 1)
    await enqueue_episode_index(uow, "t", "thr_1", now=start + 2)
    await enqueue_episode_index(uow, "t", "thr_1", now=start + EPISODE_INDEX_DELAY_SECONDS)
    keys = [spec.idempotency_key for spec in enqueued]
    assert keys[0] == keys[1] != keys[2]
    assert {spec.task_name for spec in enqueued} == {TASK_EPISODE_INDEX}
    assert all(spec.schedule_in_seconds == EPISODE_INDEX_DELAY_SECONDS for spec in enqueued)


async def _record(into: list, spec) -> None:
    into.append(spec)
