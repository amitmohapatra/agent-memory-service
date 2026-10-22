"""LangGraph's long-term memory: a ``BaseStore`` with a vector index (``SqliteStore``, the
persistent store that needs no server), one item per turn under the owner's namespace,
searched by meaning - the store LangGraph's memory guides put behind ``create_react_agent``."""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.store.base import PutOp
from langgraph.store.sqlite import SqliteStore

from benchmark.memvs.model import Hit, Query, Session, SystemName, Unit
from benchmark.memvs.systems.base import SEARCH_DEPTH, MemorySystem, Retrieved, render, turn_text

DB_FILE = "store.sqlite"
FIELD = "text"


def namespace(unit: Unit) -> tuple[str, ...]:
    return ("memories", unit.tenant, unit.user)


class LangGraphStore(MemorySystem):
    name = SystemName.LANGGRAPH_STORE

    async def open(self) -> None:
        conn = sqlite3.connect(
            self.env.store_dir / DB_FILE, check_same_thread=False, isolation_level=None
        )
        self._conn = conn
        self.store = SqliteStore(
            conn,
            index={
                "dims": self.env.embedder.dimension,
                "embed": self.env.embedder.embed,
                "fields": [FIELD],
            },
        )
        self.store.setup()

    async def write(self, unit: Unit, session: Session) -> None:
        self.store.batch(
            [
                PutOp(
                    namespace(unit),
                    t.turn_id,
                    {
                        FIELD: turn_text(t),
                        "turn": t.turn_id,
                        "session": session.session_id,
                        "at": session.at,
                    },
                )
                for t in session.turns
            ]
        )

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        items = self.store.search(namespace(unit), query=query.text, limit=SEARCH_DEPTH)
        hits = [_hit(i.value) for i in items]
        return Retrieved(hits, render(hits), insufficient=not hits)

    async def close(self) -> None:
        self._conn.close()


def _hit(value: dict[str, Any]) -> Hit:
    return Hit(value[FIELD], (value["turn"],), (value["session"],), at=value.get("at"))
