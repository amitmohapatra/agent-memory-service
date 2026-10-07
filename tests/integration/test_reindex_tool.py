"""Rebuilding the search index from PostgreSQL (``tools.reindex``) against a real store.

PostgreSQL is the source of truth and the search index is derived: after the index is lost
the tool puts every READY document's chunks, every CURRENT memory and every thread's episode
back, a tenant-scoped rebuild never touches another tenant's points, and whatever fails is
reported rather than skipped. ``tests/unit/test_reindex_safety.py`` covers the drop rules
against a stand-in store; this runs the whole rebuild.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from typing import Any

import pytest

from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.conversation import Message, Session, Thread, Turn
from memory_service.domain.enums import Lifetime, MemoryType, MessageRole, Visibility
from memory_service.domain.ids import content_hash
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.rag.indexer import KNOWLEDGE, MEMORIES
from memory_service.ports.search import SearchFilter
from memory_service.tools import reindex
from memory_service.tools.reindex import ReindexReport, rebuild_search_index
from tests.integration.conftest import integration_overrides, integration_settings

pytestmark = pytest.mark.integration

ACME = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
GLOBEX = MemoryExecutionContext(tenant_id="globex", user_id="u9", workspace_id="ws1")
REPORT = b"# Annual report\n\nRevenue grew to EUR 98 million in FY26.\n\nEBITDA rose 4%.\n"


async def _remember(container, uow_factory, ctx, content: str) -> str:
    async with uow_factory() as uow:
        ack = await container.services["memory"].remember(
            uow,
            ctx,
            content=content,
            memory_type=MemoryType.SEMANTIC,
            lifetime=Lifetime.LONG_TERM,
            visibility=Visibility.USER,
        )
        await uow.commit()
    await container.tasks.drain()
    return ack.memory_id


async def _document(container, uow_factory, ctx) -> str:
    async with uow_factory() as uow:
        ack = await container.services["ingestion"].accept_file(
            uow, ctx, filename="report.md", media_type="text/markdown", data=REPORT, title="FY26"
        )
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()
    return ack.document_id


async def _thread(uow_factory, tenant_id: str, user_id: str) -> str:
    thread = Thread(tenant_id=tenant_id, owner_user_id=user_id, title="Planning")
    async with uow_factory() as uow:
        await uow.threads.add(thread)
        session = Session(thread_id=thread.thread_id, tenant_id=tenant_id, user_id=user_id)
        await uow.sessions.add(session)
        turn = Turn(
            session_id=session.session_id,
            thread_id=thread.thread_id,
            tenant_id=tenant_id,
            sequence=1,
        )
        await uow.turns.add(turn)
        content = "We agreed to move the offsite to Lisbon in June."
        await uow.messages.add(
            Message(
                thread_id=thread.thread_id,
                session_id=session.session_id,
                turn_id=turn.turn_id,
                tenant_id=tenant_id,
                role=MessageRole.USER,
                sequence=1,
                content=content,
                content_hash=content_hash(content),
                author_principal=f"user:{user_id}",
            )
        )
        await uow.commit()
    return thread.thread_id


async def _seed(container, uow_factory) -> dict[str, Any]:
    register_handlers(container)
    return {
        "document": await _document(container, uow_factory, ACME),
        "acme_memory": await _remember(
            container, uow_factory, ACME, "The offsite moved to Lisbon."
        ),
        "globex_memory": await _remember(
            container, uow_factory, GLOBEX, "Globex renewed the Initech contract."
        ),
        "thread": await _thread(uow_factory, "acme", "u1"),
    }


def _names(container) -> tuple[str, str]:
    indexer = container.services["indexer"]
    return indexer.collection(KNOWLEDGE), indexer.collection(MEMORIES)


async def _ids(container, collection: str, tenant_id: str) -> set[str]:
    return set(await container.search.record_ids(collection, SearchFilter(tenant_id=tenant_id)))


async def test_a_whole_store_rebuild_restores_every_record_after_a_drop(
    container, uow_factory
) -> None:
    seeded = await _seed(container, uow_factory)
    knowledge, memories = _names(container)
    acme_chunks = await _ids(container, knowledge, "acme")
    assert acme_chunks, "the document was indexed at ingest"

    report = await rebuild_search_index(container, drop=True)

    assert report.ok and report.failures == []
    assert report.dropped == [knowledge, memories]
    assert report.documents == 1 and report.chunks >= 1
    assert report.memories >= 2 and report.episodes == 1
    assert await _ids(container, knowledge, "acme") == acme_chunks
    assert seeded["acme_memory"] in await _ids(container, memories, "acme")
    assert seeded["globex_memory"] in await _ids(container, memories, "globex")
    as_dict = report.as_dict()
    assert set(as_dict) == {"documents", "chunks", "memories", "episodes", "dropped", "failures"}
    assert as_dict["documents"] == 1


async def test_a_lost_index_is_rebuilt_without_dropping_anything(container, uow_factory) -> None:
    seeded = await _seed(container, uow_factory)
    knowledge, memories = _names(container)
    await container.search.drop_collection(knowledge)
    await container.search.drop_collection(memories)

    report = await rebuild_search_index(container)

    assert report.dropped == []
    assert report.ok and report.documents == 1
    assert seeded["acme_memory"] in await _ids(container, memories, "acme")
    assert await _ids(container, knowledge, "acme")


async def test_a_collection_already_gone_is_not_reported_as_dropped(container, uow_factory) -> None:
    await _seed(container, uow_factory)
    knowledge, memories = _names(container)
    await container.search.drop_collection(knowledge)
    report = await rebuild_search_index(container, drop=True)
    assert report.dropped == [memories]
    assert report.ok


async def test_a_tenant_rebuild_removes_and_restores_only_that_tenant(
    container, uow_factory
) -> None:
    seeded = await _seed(container, uow_factory)
    knowledge, memories = _names(container)
    globex_before = await _ids(container, memories, "globex")
    acme_before = await _ids(container, memories, "acme")

    report = await rebuild_search_index(container, tenant_id="acme", drop=True)

    assert len(report.dropped) == 2
    assert report.dropped[0].startswith(f"{knowledge} (tenant acme: ")
    assert report.dropped[1] == f"{memories} (tenant acme: {len(acme_before)} points)"
    assert report.documents == 1 and report.episodes == 1
    assert await _ids(container, memories, "globex") == globex_before, "never another tenant's"
    assert seeded["acme_memory"] in await _ids(container, memories, "acme")


async def test_a_tenant_drop_after_the_index_was_lost_rebuilds_instead_of_failing(
    container, uow_factory
) -> None:
    """After an index loss or a model change the collections do not exist yet: a tenant's
    drop has nothing to remove, and the rebuild goes ahead."""
    seeded = await _seed(container, uow_factory)
    knowledge, memories = _names(container)
    await container.search.drop_collection(knowledge)
    await container.search.drop_collection(memories)

    report = await rebuild_search_index(container, tenant_id="acme", drop=True)

    assert report.ok
    assert report.dropped == [
        f"{knowledge} (tenant acme: 0 points)",
        f"{memories} (tenant acme: 0 points)",
    ]
    assert report.documents == 1 and report.episodes == 1
    assert seeded["acme_memory"] in await _ids(container, memories, "acme")


async def test_a_tenant_rebuild_counts_only_that_tenants_rows(container, uow_factory) -> None:
    await _seed(container, uow_factory)
    acme = await rebuild_search_index(container, tenant_id="acme")
    globex = await rebuild_search_index(container, tenant_id="globex")
    nobody = await rebuild_search_index(container, tenant_id="nobody")
    assert (acme.documents, acme.episodes) == (1, 1)
    assert (globex.documents, globex.episodes) == (0, 0) and globex.memories >= 1
    assert nobody == ReindexReport()


async def test_what_fails_is_reported_and_the_rest_is_still_rebuilt(
    container, uow_factory, monkeypatch
) -> None:
    seeded = await _seed(container, uow_factory)
    indexer = container.services["indexer"]
    real_index_memories = indexer.index_memories

    async def broken_document(tenant_id: str, document_id: str, **_: Any) -> int:
        raise ValueError("parser output is corrupt")

    async def broken_for_globex(tenant_id: str, ids: list[str]) -> int:
        if tenant_id == "globex":
            raise RuntimeError("embedding service down")
        return await real_index_memories(tenant_id, ids)

    async def broken_episode(tenant_id: str, thread_id: str) -> bool:
        raise KeyError(thread_id)

    monkeypatch.setattr(indexer, "index_document", broken_document)
    monkeypatch.setattr(indexer, "index_memories", broken_for_globex)
    monkeypatch.setattr(indexer, "index_episode", broken_episode)

    report = await rebuild_search_index(container)

    assert not report.ok
    assert report.documents == 0 and report.episodes == 0 and report.memories >= 1
    assert sorted(report.failures) == sorted(
        [
            f"document acme/{seeded['document']}: ValueError: parser output is corrupt",
            "memories globex [0:1]: embedding service down",
            f"episode acme/{seeded['thread']}: KeyError: '{seeded['thread']}'",
        ]
    )


# --------------------------------------------------------------------------- the command


def _command(monkeypatch, make_settings, tmp_path, *argv: str) -> list[Any]:
    built: list[Any] = []
    settings = integration_settings(make_settings, blob={"filesystem_root": str(tmp_path)})

    async def build(_settings: Any, version: str) -> Any:
        c = await build_container(settings, version, overrides=integration_overrides(blob=None))
        built.append(c)
        return c

    monkeypatch.setattr(reindex, "build_container", build)
    monkeypatch.setattr(sys, "argv", ["reindex", *argv])
    return built


def test_the_command_rebuilds_and_prints_its_report(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    built = _command(monkeypatch, make_settings, tmp_path, "--tenant", "nobody-cli")
    assert reindex.main() == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == (
        "{'documents': 0, 'chunks': 0, 'memories': 0, 'episodes': 0, 'dropped': [], 'failures': []}"
    )
    assert len(built) == 1


def test_a_dry_run_prune_reports_and_rebuilds_nothing(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    _command(monkeypatch, make_settings, tmp_path, "--prune-dry-run", "--tenant", "nobody-cli")

    async def must_not_run(*_: Any, **__: Any) -> ReindexReport:
        raise AssertionError("a dry run rebuilds nothing")

    monkeypatch.setattr(reindex, "rebuild_search_index", must_not_run)
    assert reindex.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "would drop 0 retired collection(s): []"


def test_a_prune_drops_then_rebuilds(monkeypatch, make_settings, tmp_path, capsys) -> None:
    _command(monkeypatch, make_settings, tmp_path, "--prune", "--tenant", "nobody-cli")
    assert reindex.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert "dropped 0 retired collection(s): []" in out
    assert out[-1].startswith("{'documents': 0")


def test_the_command_exits_one_when_anything_failed(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    _command(monkeypatch, make_settings, tmp_path)
    seen: dict[str, Any] = {}

    async def failed(container: Any, **kwargs: Any) -> ReindexReport:
        seen.update(kwargs)
        return ReindexReport(memories=3, failures=["memories acme [0:3]: down"])

    monkeypatch.setattr(reindex, "rebuild_search_index", failed)
    assert reindex.main() == 1
    assert seen == {"tenant_id": None, "drop": False}
    assert "'failures': ['memories acme [0:3]: down']" in capsys.readouterr().out


def test_the_module_runs_as_a_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["reindex", "--help"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy re-executing an imported module
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module("memory_service.tools.reindex", run_name="__main__")
    assert exit_.value.code == 0
    assert "Rebuild the search index from PostgreSQL" in capsys.readouterr().out
