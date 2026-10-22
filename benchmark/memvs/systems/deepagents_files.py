"""Deep Agents' long-term memory: files under ``/memories/`` in a ``StoreBackend`` (here over a
persistent ``SqliteStore``), one file per session, one line per turn - the shape the agent's
conversation offloading writes. A deep agent reads memory with its file tools (``ls``,
``grep``, ``read_file``); it has no search by meaning.

The agent's own loop does not fit the benchmark's model (its system prompt and tools alone are
about 5,000 tokens and the framework head-to-head measured ``Model input still does not fit``
on an 8,192-token window), so the reads here are the backend's own ``grep`` - a literal search,
as the agent's tool calls it - with the question's words as the patterns: lines ranked by how
many of the words they hold, rarer words counting more (a word's weight falls with the number
of lines it matches). The ranking is the benchmark's; every match is the backend's.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict

from deepagents.backends import StoreBackend
from langgraph.store.sqlite import SqliteStore

from benchmark.memvs.model import Hit, Query, Session, SystemName, Unit
from benchmark.memvs.systems.base import SEARCH_DEPTH, MemorySystem, Retrieved, render, turn_text

DB_FILE = "store.sqlite"
ROOT = "/memories/"
#: a pattern shorter than this matches inside too many words to tell lines apart
MIN_PATTERN = 3
_WORD = re.compile(r"\w+", re.UNICODE)
#: a stored line: "[turn id] [date] speaker: text"
_LINE = re.compile(r"^\[(?P<turn>[^\]]+)\] (?:\[(?P<at>[^\]]+)\] )?(?P<text>.*)$")


def _patterns(question: str) -> list[str]:
    words = {w for w in _WORD.findall(question) if len(w) >= MIN_PATTERN}
    # a literal grep is case-sensitive: the word as asked and in lower case
    return sorted(words | {w.lower() for w in words})


class DeepAgentsFiles(MemorySystem):
    name = SystemName.DEEPAGENTS

    async def open(self) -> None:
        self._conn = sqlite3.connect(
            self.env.store_dir / DB_FILE, check_same_thread=False, isolation_level=None
        )
        self.store = SqliteStore(self._conn)
        self.store.setup()
        self.backends: dict[str, StoreBackend] = {}

    def _backend(self, unit: Unit) -> StoreBackend:
        key = f"{unit.tenant}/{unit.user}"
        if key not in self.backends:
            ns = ("filesystem", unit.tenant, unit.user)
            self.backends[key] = StoreBackend(namespace=lambda _runtime: ns, store=self.store)
        return self.backends[key]

    async def write(self, unit: Unit, session: Session) -> None:
        stamp = f"[{session.at}] " if session.at else ""
        body = "\n".join(f"[{t.turn_id}] {stamp}{turn_text(t)}" for t in session.turns)
        self._backend(unit).write(f"{ROOT}{session.session_id}.md", body + "\n")

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        backend = self._backend(unit)
        lines_of: dict[str, set[tuple[str, int]]] = {}
        text_of: dict[tuple[str, int], str] = {}
        for pattern in _patterns(query.text):
            found = backend.grep(pattern, path=ROOT)
            for m in found.matches or ():
                where = (m["path"], m["line"])
                lines_of.setdefault(pattern.lower(), set()).add(where)
                text_of[where] = m["text"]
        score: dict[tuple[str, int], float] = defaultdict(float)
        for lines in lines_of.values():
            weight = 1 / math.log2(1 + len(lines))
            for where in lines:
                score[where] += weight
        ranked = sorted(score, key=lambda w: (-score[w], w))[:SEARCH_DEPTH]
        hits = [h for h in (self._hit(text_of[w], w[0]) for w in ranked) if h]
        return Retrieved(hits, render(hits), insufficient=not hits)

    @staticmethod
    def _hit(line: str, path: str) -> Hit | None:
        m = _LINE.match(line.strip())
        if not m:
            return None
        session = path.removeprefix(ROOT).removesuffix(".md")
        return Hit(m["text"], (m["turn"],), (session,), at=m["at"])

    async def close(self) -> None:
        self._conn.close()
