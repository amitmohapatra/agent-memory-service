"""The OpenAI Agents SDK's memory: a ``SQLiteSession`` per owner, the conversation items it
replays into the next run (runs in agent-harness's environment, which has the SDK).

A session has no search: ``Runner.run(..., session=)`` prepends its items, newest last, and the
SDK's ``SessionSettings(limit=)`` keeps the latest ones. Read here as the SDK reads it -
``get_items(limit)`` - newest first for the recall ranking, and as many of the latest as the
memory budget holds, in order, for the model. Items carry no ids of their own, so the turn
each position holds is recorded beside the database when it is written.
"""

from __future__ import annotations

import json

from agents import SQLiteSession

from benchmark.memvs.llm import CHARS_PER_TOKEN, MEMORY_TOKEN_BUDGET
from benchmark.memvs.model import Hit, Query, Role, Session, SystemName, Unit, turn_session
from benchmark.memvs.systems.base import SEARCH_DEPTH, MemorySystem, Retrieved, spoken

DB_FILE = "sessions.sqlite"
POSITIONS_FILE = "positions.json"


def session_id(unit: Unit) -> str:
    return f"{unit.tenant}/{unit.user}"


class OpenAISession(MemorySystem):
    name = SystemName.OPENAI_SESSION

    async def open(self) -> None:
        self.db = self.env.store_dir / DB_FILE
        self.positions_path = self.env.store_dir / POSITIONS_FILE
        self.positions: dict[str, list[str]] = (
            json.loads(self.positions_path.read_text()) if self.positions_path.exists() else {}
        )
        self.sessions: dict[str, SQLiteSession] = {}

    def _session(self, unit: Unit) -> SQLiteSession:
        key = session_id(unit)
        if key not in self.sessions:
            self.sessions[key] = SQLiteSession(key, self.db)
        return self.sessions[key]

    async def write(self, unit: Unit, session: Session) -> None:
        items = [
            {"role": "user" if t.role is Role.USER else "assistant", "content": spoken(t)}
            for t in session.turns
        ]
        await self._session(unit).add_items(items)  # type: ignore[arg-type]
        self.positions.setdefault(session_id(unit), []).extend(t.turn_id for t in session.turns)
        self.positions_path.write_text(json.dumps(self.positions))

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        items = await self._session(unit).get_items(limit=SEARCH_DEPTH)
        turns = self.positions.get(session_id(unit), [])[-len(items) :] if items else []
        hits = [
            Hit(f"{i['role']}: {i['content']}", (t,), (turn_session(t),))  # type: ignore[index]
            for i, t in zip(reversed(items), reversed(turns), strict=True)
        ]
        # the latest items that fit the budget, oldest first: what the run would replay
        kept: list[str] = []
        used = 0
        for h in hits:
            cost = len(h.text) // CHARS_PER_TOKEN + 1
            if kept and used + cost > MEMORY_TOKEN_BUDGET:
                break
            kept.append(h.text)
            used += cost
        return Retrieved(hits, "\n".join(reversed(kept)), insufficient=not hits)

    async def close(self) -> None:
        for s in self.sessions.values():
            s.close()
