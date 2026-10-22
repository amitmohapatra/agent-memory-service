"""CrewAI's memory (``crewai.Memory``, the unified memory that replaced the short-term,
long-term and entity memories in 1.x) on LanceDB, with the shared encoder.

* ``CrewAIMemory``: the documented model-free mode - every field given to ``remember`` (scope,
  categories, importance), consolidation off, ``recall(depth="shallow")``.
* ``CrewAIMemoryLLM``: what a CrewAI agent does after a task (``crewai/agent/core.py``):
  ``extract_memories`` on the raw text, then ``remember_many``, every field inferred by the
  model; ``recall`` at its default depth.

The owner is the scope ``/tenant/user``. Records written by a session are found by listing the
owner's scope before and after it (CrewAI writes them in the background).
"""

from __future__ import annotations

from typing import Any

from crewai import LLM, Memory

from benchmark.memvs.model import Hit, Query, Session, SystemName, Unit, turn_session
from benchmark.memvs.systems.base import (
    SEARCH_DEPTH,
    Attribution,
    MemorySystem,
    MemorySystemError,
    Retrieved,
    render,
    turn_text,
)

ATTRIBUTION_FILE = "attribution.json"
CATEGORY = "conversation"
IMPORTANCE = 0.5
#: CrewAI's ceiling: only an exact duplicate (cosine >= 1.0) would ask a model to merge; in the
#: model-free arm no model is configured, so such a call fails loudly instead of passing
NO_CONSOLIDATION = 1.0
#: the most records one owner's scope is listed for
LIST_LIMIT = 100_000


def scope(unit: Unit) -> str:
    return f"/{unit.tenant}/{unit.user}"


class CrewAIMemory(MemorySystem):
    name = SystemName.CREWAI
    uses_model = False

    def _memory(self) -> Memory:
        kwargs: dict[str, Any] = {
            "storage": str(self.env.store_dir / "lancedb"),
            "embedder": self.env.embedder.embed,
        }
        gateway = self.env.gateway
        if self.uses_model:
            if gateway is None:
                raise MemorySystemError("CrewAI's analysis needs the model (MEMVS_LLM_*)")
            kwargs["llm"] = LLM(
                model=f"openai/{gateway.model}",
                base_url=gateway.base_url,
                api_key=gateway.key,
                temperature=0,
            )
        else:
            kwargs["consolidation_threshold"] = NO_CONSOLIDATION
        return Memory(**kwargs)

    async def open(self) -> None:
        self.memory = self._memory()
        self.attribution = Attribution.load(self.env.store_dir / ATTRIBUTION_FILE)

    async def write(self, unit: Unit, session: Session) -> None:
        for turn in session.turns:
            record = self.memory.remember(
                turn_text(turn),
                scope=scope(unit),
                categories=[CATEGORY],
                importance=IMPORTANCE,
                metadata={"turn": turn.turn_id, "session": session.session_id, "at": session.at},
            )
            if record is not None:
                self.attribution.credit([record.id], turn.turn_id)
        self.attribution.save()

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        matches = self.memory.recall(
            query.text,
            scope=scope(unit),
            limit=SEARCH_DEPTH,
            depth="deep" if self.uses_model else "shallow",
        )
        hits = [self._hit(m.record) for m in matches]
        return Retrieved(hits, render(hits), insufficient=not hits)

    def _hit(self, record: Any) -> Hit:
        sources = self.attribution.of(record.id)
        at = (record.metadata or {}).get("at")
        if self.turn_level:
            return Hit(
                record.content, sources, tuple(sorted({turn_session(s) for s in sources})), at=at
            )
        return Hit(record.content, (), sources, at=at)

    async def close(self) -> None:
        self.memory.drain_writes()
        self.attribution.save()
        self.memory.close()


class CrewAIMemoryLLM(CrewAIMemory):
    name = SystemName.CREWAI_LLM
    uses_model = True
    turn_level = False

    def _ids(self, unit: Unit) -> set[str]:
        return {r.id for r in self.memory.list_records(scope=scope(unit), limit=LIST_LIMIT)}

    async def write(self, unit: Unit, session: Session) -> None:
        before = self._ids(unit)
        transcript = "\n".join(turn_text(t) for t in session.turns)
        if session.at:
            transcript = f"Conversation on {session.at}:\n{transcript}"
        extracted = self.memory.extract_memories(transcript)
        if extracted:
            self.memory.remember_many(
                extracted,
                scope=scope(unit),
                metadata={"session": session.session_id, "at": session.at},
            )
            self.memory.drain_writes()
        self.attribution.credit(sorted(self._ids(unit) - before), session.session_id)
        self.attribution.save()
