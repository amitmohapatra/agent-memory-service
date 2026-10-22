"""The neutral workload and result records every system reads and writes (standard library only).

A workload is a list of units. A unit is one owner's memory: the sessions written into it and
the queries asked of it afterwards (a LongMemEval question and its haystack, a LoCoMo
conversation, one capability scenario in one language). Turn ids are the gold the evidence
recall is measured against, so every system has to carry them from write to read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SystemName(StrEnum):
    """Every memory system the benchmark drives."""

    TRELLIS = "trellis"  # this service, through agent-harness (h.wrap)
    TRELLIS_REACT = "trellis_react"  # this service, through agent-harness's ReAct agent
    LANGGRAPH_STORE = "langgraph_store"  # LangGraph BaseStore (SqliteStore, vector index)
    LANGMEM = "langmem"  # LangMem's memory manager over a LangGraph store (LLM extraction)
    DEEPAGENTS = "deepagents"  # Deep Agents' StoreBackend files (/memories/)
    OPENAI_SESSION = "openai_session"  # OpenAI Agents SDK SQLiteSession
    CREWAI = "crewai"  # CrewAI unified Memory, fields given, no LLM analysis
    CREWAI_LLM = "crewai_llm"  # CrewAI unified Memory, its default LLM analysis
    MEM0 = "mem0"  # Mem0 OSS, raw messages (infer=False)
    MEM0_LLM = "mem0_llm"  # Mem0 OSS, its default LLM extraction (infer=True)
    GRAPHITI = "graphiti"  # Graphiti (Zep's open core) on embedded Kuzu


class DatasetName(StrEnum):
    LME18 = "lme18"  # LongMemEval-S, the 18-question seed-7 sample
    LME_HELDOUT = "lme_heldout"  # LongMemEval-S, a disjoint seed-11 sample
    LOCOMO = "locomo"  # LoCoMo, all ten conversations
    LOCOMO_C0 = "locomo_c0"  # LoCoMo conversation 0 (the LLM-extraction tier)
    CAPABILITY = "capability"  # the capability gate's core set, every language
    ISOLATION = "isolation"  # two tenants x two users, each with secrets


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Phase(StrEnum):
    """The run's phases; each one is its own process, so ``query`` reads after a restart."""

    INGEST = "ingest"
    QUERY = "query"
    ANSWER = "answer"


@dataclass(frozen=True)
class Turn:
    turn_id: str
    role: Role
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class Session:
    session_id: str
    #: ISO-8601, the session's time (``None`` when the dataset has none)
    at: str | None
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    #: ISO-8601: when the question is asked
    at: str | None = None
    answer: str | None = None
    category: str = ""
    language: str = "en"
    evidence_turns: tuple[str, ...] = ()
    evidence_sessions: tuple[str, ...] = ()
    #: the question is part of the answer-accuracy sample
    answer_sample: bool = False


@dataclass(frozen=True)
class Unit:
    """One owner's memory: what is written, then what is asked."""

    unit_id: str
    tenant: str
    user: str
    sessions: tuple[Session, ...]
    queries: tuple[Query, ...]


@dataclass(frozen=True)
class Workload:
    dataset: DatasetName
    units: tuple[Unit, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> Workload:
        raw = json.loads(path.read_text())
        units = tuple(
            Unit(
                unit_id=u["unit_id"],
                tenant=u["tenant"],
                user=u["user"],
                sessions=tuple(
                    Session(
                        session_id=s["session_id"],
                        at=s["at"],
                        turns=tuple(
                            Turn(t["turn_id"], Role(t["role"]), t["text"], t.get("speaker"))
                            for t in s["turns"]
                        ),
                    )
                    for s in u["sessions"]
                ),
                queries=tuple(
                    Query(
                        **{
                            **q,
                            "evidence_turns": tuple(q["evidence_turns"]),
                            "evidence_sessions": tuple(q["evidence_sessions"]),
                        }
                    )
                    for q in u["queries"]
                ),
            )
            for u in raw["units"]
        )
        return cls(DatasetName(raw["dataset"]), units, raw.get("meta", {}))


@dataclass(frozen=True)
class Hit:
    """One retrieved item, ranked, with the turns (and sessions) it was read out of."""

    text: str
    turns: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    #: the system marks the item as no longer current (superseded, invalidated)
    historical: bool = False
    #: when what it says was said (ISO-8601), as the system returned it
    at: str | None = None


@dataclass
class QueryRow:
    """One question asked of one system: what came back, how fast, and (later) the answer."""

    system: str
    dataset: str
    unit_id: str
    query_id: str
    category: str
    language: str
    hits: list[dict[str, Any]]
    #: the memory text a model reads for this question (the system's own rendering)
    context: str
    context_tokens: int
    latency_ms: float
    #: the system said the memory holds nothing for the question (INSUFFICIENT, no hit)
    insufficient: bool = False
    answer: str | None = None
    answer_prompt_tokens: int | None = None
    answer_completion_tokens: int | None = None
    answer_latency_ms: float | None = None
    judge: bool | None = None
    error: str | None = None


def turn_session(turn_id: str) -> str:
    """The session a turn id belongs to (``<session>:<k>``)."""
    return turn_id.rsplit(":", 1)[0]
