"""Mem0 OSS (``mem0ai``): its ``Memory`` on an embedded, on-disk Qdrant and a SQLite history.

* ``Mem0``: ``add(..., infer=False)`` - every message kept as written, embedded, searched by
  meaning; no model at write time.
* ``Mem0LLM``: ``add(...)`` as shipped (``infer=True``) - one extraction call per session to the
  benchmark's model, which writes the facts it finds and updates earlier ones.

The owner is ``user_id`` (``tenant:user``); the session's date travels in each item's metadata.
Mem0 returns the ids each ``add`` created or changed: they are attributed to that write's turns
(verbatim) or session (extraction).
"""

from __future__ import annotations

from typing import Any

from mem0 import Memory

from benchmark.memvs.model import Hit, Query, Role, Session, SystemName, Unit, turn_session
from benchmark.memvs.systems.base import (
    SEARCH_DEPTH,
    Attribution,
    MemorySystem,
    MemorySystemError,
    Retrieved,
    render,
    spoken,
)

COLLECTION = "memvs"
ATTRIBUTION_FILE = "attribution.json"
#: the embedding server needs no key; the OpenAI client insists on one
UNUSED_KEY = "unused"
#: a model address nothing listens on: verbatim mode must never reach a model
NO_MODEL = "none"
NO_MODEL_URL = "http://127.0.0.1:9/v1"


def owner(unit: Unit) -> str:
    return f"{unit.tenant}:{unit.user}"


class Mem0(MemorySystem):
    name = SystemName.MEM0
    infer = False

    def config(self) -> dict[str, Any]:
        store = self.env.store_dir
        cfg: dict[str, Any] = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": COLLECTION,
                    "path": str(store / "qdrant"),
                    "embedding_model_dims": self.env.embedder.dimension,
                    "on_disk": True,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.env.embedder.model,
                    "openai_base_url": self.env.embedder.base_url,
                    "api_key": UNUSED_KEY,
                    "embedding_dims": self.env.embedder.dimension,
                },
            },
            "history_db_path": str(store / "history.db"),
        }
        gateway = self.env.gateway
        if gateway is None and self.infer:
            raise MemorySystemError("Mem0's extraction needs the model (MEMVS_LLM_*)")
        # Mem0 builds its LLM client even when nothing will call it (infer=False)
        cfg["llm"] = {
            "provider": "openai",
            "config": {
                "model": gateway.model if gateway else NO_MODEL,
                "openai_base_url": gateway.base_url if gateway else NO_MODEL_URL,
                "api_key": gateway.key if gateway else UNUSED_KEY,
                "temperature": 0,
            },
        }
        return cfg

    async def open(self) -> None:
        self.memory = Memory.from_config(self.config())
        self.attribution = Attribution.load(self.env.store_dir / ATTRIBUTION_FILE)

    async def write(self, unit: Unit, session: Session) -> None:
        messages = [
            {"role": "user" if t.role is Role.USER else "assistant", "content": spoken(t)}
            for t in session.turns
        ]
        result = self.memory.add(
            messages,
            user_id=owner(unit),
            metadata={"session": session.session_id, "at": session.at},
            infer=self.infer,
        )
        items = result.get("results", [])
        if self.infer:
            self.attribution.credit([r["id"] for r in items], session.session_id)
        else:
            # one item per message, in message order (system messages are never sent)
            for item, turn in zip(items, session.turns, strict=True):
                self.attribution.credit([item["id"]], turn.turn_id)
        self.attribution.save()

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        found = self.memory.search(query.text, top_k=SEARCH_DEPTH, filters={"user_id": owner(unit)})
        hits = [self._hit(r) for r in found.get("results", [])]
        return Retrieved(hits, render(hits), insufficient=not hits)

    def _hit(self, r: dict[str, Any]) -> Hit:
        sources = self.attribution.of(str(r["id"]))
        meta = r.get("metadata") or {}
        if self.infer:
            return Hit(r["memory"], (), sources, at=meta.get("at"))
        return Hit(
            r["memory"],
            sources,
            tuple(sorted({turn_session(s) for s in sources})),
            at=meta.get("at"),
        )

    async def close(self) -> None:
        self.attribution.save()
        client = getattr(self.memory.vector_store, "client", None)
        if client is not None:
            client.close()


class Mem0LLM(Mem0):
    name = SystemName.MEM0_LLM
    infer = True
    turn_level = False
