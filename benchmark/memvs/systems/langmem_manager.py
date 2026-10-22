"""LangMem: ``create_memory_store_manager`` - the model reads each session with the owner's
closest existing memories and writes, rewrites (and may delete) memories in a LangGraph store
(the same ``SqliteStore`` with the shared encoder as ``langgraph_store``). Searched by meaning,
as an agent's ``create_search_memory_tool`` searches it.

Provenance is observed: every item a session's manager call created or changed (a new key, or
a new ``updated_at``) is attributed to that session.
"""

from __future__ import annotations

import sqlite3

from langchain_openai import ChatOpenAI
from langgraph.store.sqlite import SqliteStore
from langmem import create_memory_store_manager

from benchmark.memvs.model import Hit, Query, Role, Session, SystemName, Unit
from benchmark.memvs.systems.base import (
    SEARCH_DEPTH,
    Attribution,
    MemorySystem,
    MemorySystemError,
    Retrieved,
    render,
    spoken,
)

DB_FILE = "store.sqlite"
ATTRIBUTION_FILE = "attribution.json"
#: the most memories one owner's namespace is listed for when diffing a write
LIST_LIMIT = 100_000
USER_KEY = "langgraph_user_id"


def owner(unit: Unit) -> str:
    return f"{unit.tenant}:{unit.user}"


def namespace(unit: Unit) -> tuple[str, ...]:
    return ("memories", owner(unit))


class LangMem(MemorySystem):
    name = SystemName.LANGMEM
    turn_level = False

    async def open(self) -> None:
        gateway = self.env.gateway
        if gateway is None:
            raise MemorySystemError("LangMem's manager needs the model (MEMVS_LLM_*)")
        self._conn = sqlite3.connect(
            self.env.store_dir / DB_FILE, check_same_thread=False, isolation_level=None
        )
        self.store = SqliteStore(
            self._conn,
            index={"dims": self.env.embedder.dimension, "embed": self.env.embedder.embed},
        )
        self.store.setup()
        model = ChatOpenAI(
            model=gateway.model,
            base_url=gateway.base_url,
            api_key=gateway.key,
            temperature=0,  # type: ignore[arg-type]
        )
        self.manager = create_memory_store_manager(
            model, namespace=("memories", "{" + USER_KEY + "}"), store=self.store
        )
        self.attribution = Attribution.load(self.env.store_dir / ATTRIBUTION_FILE)

    def _versions(self, unit: Unit) -> dict[str, str]:
        items = self.store.search(namespace(unit), limit=LIST_LIMIT)
        return {i.key: i.updated_at.isoformat() for i in items}

    async def write(self, unit: Unit, session: Session) -> None:
        before = self._versions(unit)
        messages = [
            {"role": "user" if t.role is Role.USER else "assistant", "content": spoken(t)}
            for t in session.turns
        ]
        if session.at:
            messages.insert(
                0, {"role": "system", "content": f"This conversation took place on {session.at}."}
            )
        await self.manager.ainvoke(
            {"messages": messages}, config={"configurable": {USER_KEY: owner(unit)}}
        )
        after = self._versions(unit)
        self.attribution.credit(
            sorted(k for k, v in after.items() if before.get(k) != v), session.session_id
        )
        self.attribution.save()

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        items = self.store.search(namespace(unit), query=query.text, limit=SEARCH_DEPTH)
        hits = [
            Hit(
                str((i.value.get("content") or {}).get("content", i.value.get("content", i.value))),
                (),
                self.attribution.of(i.key),
            )
            for i in items
        ]
        return Retrieved(hits, render(hits), insufficient=not hits)

    async def close(self) -> None:
        self.attribution.save()
        self._conn.close()
