"""Tool search: catalog entries embedded into the ``tools`` collection (name, description,
argument names and the arguments' schema descriptions: ``ToolDescriptor.index_text``),
searched with the same hybrid fusion as memories, per tenant and workspace.

Each record carries the ``text_hash`` of the text it was embedded from, prefixed by the
record's payload layout (``RECORD_FORMAT``). A changed catalog entry is re-indexed by its own
``tools.index`` job; ``refresh`` (a periodic job) indexes the rest: an entry with no record
(one a recorded call created, a job that gave up, a new collection after a model change) and
one whose text a release changed, so an existing deployment picks the new text up by itself.
``rebuild`` (``make reindex``) re-embeds every entry.

A record an earlier layout wrote for the entry's current text is moved to the current layout
in place, payload only (its vectors are of the same text): by ``refresh``, and by ``indexed``
for the entries it is asked about, so a harness waiting for its tools to be indexed is not
held until the next refresh tick. Layout 2 added ``tool_name``, the keyword payload index a
search's ``available`` filter reads (layout 1 filtered on ``text``, which is not indexed).
Until no release that writes layout 1 can be running, the filter also accepts the name in
``text`` (``search``), so a record an earlier worker wrote back is still found.

A tool record's audience is its catalog scope: ``tenant:<t>`` for a tenant-wide entry,
``workspace:<t>/<w>`` for a workspace's own; a search reads both, and a workspace's entry
shadows the tenant's of the same name.

A search's fusion score only orders: divided by the task's best, the first hit reads 1.0
whether or not any tool is about the task. So each hit also says whether it ``matches``: its
cosine to the task, in the space every query is searched in, is at least that encoder's
calibrated ``tool_cosine`` (one bounded read of the hits, as the memories' similarity is
read). An encoder with none (the stand-in) leaves every hit a match.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Final, NamedTuple

from memory_service.domain.ids import content_hash
from memory_service.domain.script import detect_script
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.rag.indexer import Indexer
from memory_service.modules.retrieval.engine import QueryVectors
from memory_service.observability.logging import get_logger
from memory_service.ports.search import CollectionSpec, SearchFilter, SearchRecord, SearchStore
from memory_service.ports.uow import UnitOfWorkFactory

log = get_logger(__name__)

TOOLS: Final = "tools"
#: Candidates one search asks the store for, among the caller's available tools when it names
#: them (narrowed in the store: a tenant's catalog may hold thousands, and the best 32 of all of
#: them may hold none of a run's).
SEARCH_LIMIT: Final = 32
_RRF_K: Final = 1
#: Catalog entries one ``refresh`` page compares with their records.
REFRESH_BATCH: Final = 256
#: How long a process trusts a ``refresh`` it already ran: a new release is a new process, so
#: the first tick after a deploy compares every record; later ticks only catch a record an
#: older process wrote during a rolling deploy.
REFRESH_EVERY_SECONDS: Final = 6 * 3600


class ToolMatch(NamedTuple):
    """One tool a search found."""

    name: str
    #: the fusion score over the task's best hit, 0..1: what orders the tools
    relevance: float
    #: whether the tool's indexed text is close to the task at all (``tool_cosine``)
    matches: bool = True


#: The tool record's payload layout, part of its ``text_hash``: a record an older layout wrote
#: reads as stale, and ``refresh`` / ``indexed`` move it to this one. 2: ``tool_name``.
RECORD_FORMAT: Final = 2


def record_hash(text: str) -> str:
    """The ``text_hash`` of a record embedded from ``text`` in the current layout."""
    return f"v{RECORD_FORMAT}:{content_hash(text)}"


def _hash(entry: ToolDescriptor) -> str:
    return record_hash(entry.index_text())


def _hash_v1(entry: ToolDescriptor) -> str:
    """What layout 1 (no ``tool_name``) stored for the same text: a bare content hash."""
    return content_hash(entry.index_text())


def _payload(entry: ToolDescriptor, text: str) -> dict[str, object]:
    return {
        "kind": "tool",
        # the name, as every layout wrote it: what a search reads a hit's name from
        "text": entry.name,
        "tool_name": entry.name,
        "text_hash": record_hash(text),
        "visibility_keys": tool_audience(entry.tenant_id, entry.workspace_id)[-1:],
    }


def tool_audience(tenant_id: str, workspace_id: str | None) -> list[str]:
    """The keys a catalog scope is searched by."""
    keys = [f"tenant:{tenant_id}"]
    if workspace_id:
        keys.append(f"workspace:{tenant_id}/{workspace_id}")
    return keys


class ToolIndex:
    def __init__(self, uow_factory: UnitOfWorkFactory, indexer: Indexer, store: SearchStore):
        self.uow_factory = uow_factory
        self.indexer = indexer
        self.store = store
        self._ensured: set[str] = set()
        #: collection -> when this process last compared every record with its entry
        self._refreshed: dict[str, float] = {}

    @property
    def collection(self) -> str:
        return self.indexer.collection(TOOLS)

    def forget_collections(self) -> None:
        """The collection was dropped outside this index (a test reset): ensure it again."""
        self._ensured.clear()
        self._refreshed.clear()

    async def ensure(self) -> None:
        if self.collection in self._ensured:
            return
        await self.store.ensure_collection(
            CollectionSpec(
                name=self.collection,
                dense=self.indexer.spaces.dimensions,
                sparse=True,
                sparse_idf=getattr(self.indexer.sparse, "server_side_idf", True),
                # a catalog is small and every tool search reads it
                on_disk_payload=False,
            )
        )
        self._ensured.add(self.collection)

    async def index(self, tenant_id: str, tool_ids: Sequence[str]) -> int:
        """Embed the named catalog entries (the ``tools.index`` job)."""
        async with self.uow_factory() as uow:
            entries = await uow.tools.catalog_by_ids(tenant_id, tool_ids)
        if not entries:
            return 0
        await self.ensure()
        await self.store.upsert(await self._records(entries))
        log.info("tools.indexed", tenant_id=tenant_id, count=len(entries))
        return len(entries)

    async def indexed(self, entries: Sequence[ToolDescriptor]) -> set[str]:
        """The ids of the entries tool search finds as they are now: their record holds their
        current text. One just published or changed is not, until its ``tools.index`` job
        ran: what a harness waits for before it first asks for hints."""
        if not entries:
            return set()
        await self.ensure()
        held = await self._held(entries)
        current = {e.tool_id for e in entries if held.get(e.tool_id) == _hash(e)}
        older = [e for e in entries if held.get(e.tool_id) == _hash_v1(e)]
        if older:
            # the window after a release: moved now, not at the next refresh tick
            try:
                await self._relayout(older)
            except Exception as exc:  # not indexed yet; the refresh moves them
                log.warning(
                    "tools.relayout_failed",
                    count=len(older),
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            else:
                current |= {e.tool_id for e in older}
        return current

    async def _held(self, entries: Sequence[ToolDescriptor]) -> dict[str, object]:
        """Each entry's record's ``text_hash``, by tool id (absent: no record)."""
        return {
            r.record_id: r.payload.get("text_hash")
            for r in await self.store.get(self.collection, [e.tool_id for e in entries])
        }

    async def _relayout(self, entries: Sequence[ToolDescriptor]) -> None:
        """Move records of the entries' current text to the current payload layout, in place:
        no embedding, the vectors stay."""
        await self.store.set_payloads(
            self.collection, {e.tool_id: _payload(e, e.index_text()) for e in entries}
        )

    async def refresh(self, *, force: bool = False) -> int:
        """Index every catalog entry that has no record, or whose record was embedded from
        other text than its ``index_text`` gives now (a release changed what tools are searched
        by, or the record predates ``text_hash``). Runs once per process per
        ``REFRESH_EVERY_SECONDS`` unless ``force``; returns how many records were written."""
        collection = self.collection
        started = time.monotonic()
        last = self._refreshed.get(collection)
        if not force and last is not None and started - last < REFRESH_EVERY_SECONDS:
            return 0
        written = await self._sweep(None, every=False)
        self._refreshed[collection] = started
        if written:
            log.info("tools.refreshed", count=written)
        return written

    async def rebuild(self, tenant_id: str | None = None) -> int:
        """Re-embed every catalog entry of ``tenant_id`` (every tenant's when None) from the
        catalog, whatever its record holds: the reindex tool's part for tool search."""
        written = await self._sweep(tenant_id, every=True)
        log.info("tools.rebuilt", tenant_id=tenant_id, count=written)
        return written

    async def _sweep(self, tenant_id: str | None, *, every: bool) -> int:
        """Page the catalog by ``tool_id`` and index the entries ``every`` or, without it, the
        ones whose record is missing or holds another text's hash (or another layout's: moved
        in place)."""
        await self.ensure()
        written, after = 0, ""
        while True:
            async with self.uow_factory() as uow:
                entries = await uow.tools.catalog_page(
                    tenant_id=tenant_id, after=after, limit=REFRESH_BATCH
                )
            if not entries:
                break
            after = entries[-1].tool_id
            todo = list(entries)
            if not every:
                held = await self._held(entries)
                todo = [e for e in entries if held.get(e.tool_id) != _hash(e)]
                older = [e for e in todo if held.get(e.tool_id) == _hash_v1(e)]
                if older:
                    await self._relayout(older)
                    written += len(older)
                    todo = [e for e in todo if held.get(e.tool_id) != _hash_v1(e)]
            if todo:
                await self.store.upsert(await self._records(todo))
                written += len(todo)
            if len(entries) < REFRESH_BATCH:
                break
        return written

    async def _records(self, entries: Sequence[ToolDescriptor]) -> list[SearchRecord]:
        texts = [e.index_text() for e in entries]
        dense = await self.indexer.embed_cached(texts, [content_hash(t) + ":tool1" for t in texts])
        sparse = self.indexer.sparse.encode_documents(texts)
        return [
            SearchRecord(
                record_id=entry.tool_id,
                collection=self.collection,
                tenant_id=entry.tenant_id,
                dense={space: vectors[i] for space, vectors in dense.items()},
                sparse=sparse[i],
                payload=_payload(entry, texts[i]),
            )
            for i, entry in enumerate(entries)
        ]

    async def encode(self, task: str) -> QueryVectors:
        """The task's vectors, as a search reads them: every dense space its script calls
        for, and the sparse one."""
        script = detect_script(task)
        return QueryVectors(
            dense=await self.indexer.spaces.embed_query(task, script=script),
            sparse=self.indexer.sparse.encode_query(task),
            script=script,
        )

    async def search(
        self,
        tenant_id: str,
        workspace_id: str | None,
        task: str,
        *,
        vectors: QueryVectors | None = None,
        limit: int = SEARCH_LIMIT,
        available: Sequence[str] | None = None,
    ) -> list[ToolMatch]:
        """The tools found, best first, one per name; among ``available`` alone when given
        (the store filters by name before it ranks). ``vectors`` are the task's, when a
        retrieval already encoded it (the pushed context): no second encode."""
        if available is not None and not available:
            return []
        await self.ensure()
        if vectors is None or not vectors.dense:
            vectors = await self.encode(task)
        dense, sparse = vectors.dense, vectors.sparse
        names = sorted(set(available)) if available is not None else None
        hits = await self.store.search_hybrid(
            self.collection,
            dense=dense,
            sparse=sparse,
            flt=SearchFilter(
                tenant_id=tenant_id,
                must={"kind": "tool"},
                must_any={"visibility_keys": tool_audience(tenant_id, workspace_id)},
                # ``tool_name`` is an indexed keyword field: a catalog of thousands is narrowed
                # in the index. Or ``text``, the name as layout 1 stored it: while releases are
                # mixed an earlier worker writes layout-1 records back on each refresh tick, and
                # a record waits for this release's next refresh or ``indexed`` to be moved;
                # filtered on ``tool_name`` alone, a run's hints were empty in the meantime.
                # Both alternatives sit inside the tenant, kind and audience above.
                should_any={"tool_name": names, "text": names} if names is not None else {},
            ),
            limit=limit,
            prefetch_limit=limit,
            rrf_k=_RRF_K,
        )
        best = max((h.score for h in hits), default=0.0) or 1.0
        close = await self._close(task, vectors, [h.record_id for h in hits])
        out: dict[str, ToolMatch] = {}
        for hit in hits:
            name = str(hit.payload.get("text", ""))
            if name and name not in out:
                matches = close is None or hit.record_id in close
                out[name] = ToolMatch(name, hit.score / best, matches)
        return list(out.values())

    async def _close(
        self, task: str, vectors: QueryVectors, record_ids: Sequence[str]
    ) -> set[str] | None:
        """The records whose cosine to the task is at least the encoder's ``tool_cosine``;
        ``None`` when the encoder has none (every record counts)."""
        spaces = self.indexer.spaces
        floor = spaces.primary.tool_cosine
        if floor <= 0 or not record_ids:
            return None
        space = spaces.primary_space.name
        dense = vectors.dense.get(space) or await spaces.embed_primary_query(task)
        cosines = await self.store.similarity(self.collection, space, dense, record_ids)
        return {rid for rid, cosine in cosines.items() if cosine >= floor}
