"""What a memory system is to the benchmark, and what every adapter shares (stdlib + httpx).

An adapter drives one library through that library's own write and read calls. It keeps the
dataset's turn (or session) ids next to what it writes, in the library's own metadata, so a
retrieved item can be traced back to its source; where a library rewrites what it is given
(an LLM extraction), the items each write created or changed are attributed to that write's
session (``Attribution``) - provenance is observed, never asked of the library.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from benchmark.memvs.llm import MEMORY_TOKEN_BUDGET, Completion, Gateway, within_budget
from benchmark.memvs.llm import answer as llm_answer
from benchmark.memvs.model import DatasetName, Hit, Query, Session, SystemName, Turn, Unit

ENV_EMBED_URL = "MEMVS_EMBED_URL"  # the shared encoder's OpenAI-compatible root, ".../v1"
#: the service's multilingual encoder, served to every library that takes an embedder
EMBED_MODEL = "bekko-embedding-v1-a8m"
EMBED_DIMENSION = 384
EMBED_TIMEOUT_S = 120.0
#: the deepest rank recall is measured at
SEARCH_DEPTH = 50


class MemorySystemError(RuntimeError):
    """A memory system failed in a way the run reports instead of scoring."""


@dataclass(frozen=True)
class Embedder:
    """The shared encoder, for libraries that take a callable or an OpenAI-compatible URL."""

    base_url: str
    model: str = EMBED_MODEL
    dimension: int = EMBED_DIMENSION

    @classmethod
    def from_env(cls) -> Embedder:
        url = os.environ.get(ENV_EMBED_URL)
        if not url:
            raise MemorySystemError(f"{ENV_EMBED_URL} is not set")
        return cls(url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": texts},
            timeout=EMBED_TIMEOUT_S,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]


@dataclass(frozen=True)
class Retrieved:
    """What a system returned for one question: ranked hits and the text a model reads."""

    hits: list[Hit]
    context: str
    insufficient: bool = False
    #: the read the product makes, when it is not the whole of ``search`` (this service: the
    #: harness's prompt-form context, not the full-form read the recall is measured on)
    latency_ms: float | None = None


@dataclass
class Environment:
    store_dir: Path
    embedder: Embedder
    gateway: Gateway | None
    dataset: DatasetName


#: the speaker of a capability event turn (something that happened, not something said)
EVENT_SPEAKER = "event"


def turn_text(turn: Turn) -> str:
    """The turn as a text-only store keeps it: who said it, then what was said."""
    return f"{turn.speaker or turn.role.value}: {turn.text}"


def spoken(turn: Turn) -> str:
    """The turn as a message API takes it (the role says who): a named speaker of a
    conversation between people (LoCoMo) is kept in the text, the only place it can go."""
    if turn.speaker and turn.speaker != EVENT_SPEAKER:
        return f"{turn.speaker}: {turn.text}"
    return turn.text


def render(hits: list[Hit], budget: int = MEMORY_TOKEN_BUDGET) -> str:
    """The hits, ranked, one line each with their date, within the memory budget."""
    return within_budget([f"- [{h.at}] {h.text}" if h.at else f"- {h.text}" for h in hits], budget)


@dataclass
class Attribution:
    """Item id -> the sources (turn or session ids) whose writes created or changed it, kept
    in a file beside the store so a later process reads what an earlier one observed."""

    path: Path
    sources: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Attribution:
        raw = json.loads(path.read_text()) if path.exists() else {}
        return cls(path, {k: set(v) for k, v in raw.items()})

    def credit(self, item_ids: list[str], source: str) -> None:
        for item in item_ids:
            self.sources.setdefault(item, set()).add(source)

    def of(self, item_id: str) -> tuple[str, ...]:
        return tuple(sorted(self.sources.get(item_id, ())))

    def save(self) -> None:
        self.path.write_text(json.dumps({k: sorted(v) for k, v in self.sources.items()}))


class MemorySystem(ABC):
    """One memory system. ``open`` in each phase's process; the store persists in between."""

    name: SystemName
    #: provenance the system keeps: turn-level (verbatim items) or only session-level
    turn_level: bool = True

    def __init__(self, env: Environment) -> None:
        self.env = env

    async def open(self) -> None:  # noqa: B027 - a no-op for in-process stores
        """Connect, or load what an earlier process wrote."""

    @abstractmethod
    async def write(self, unit: Unit, session: Session) -> None:
        """Remember one session of the unit's owner."""

    async def after_write(self, unit: Unit) -> None:  # noqa: B027 - synchronous stores
        """Wait until the session just written is processed (not counted as write latency)."""

    async def settle(self) -> None:  # noqa: B027 - synchronous stores are ready at once
        """Return once everything written can be read."""

    @abstractmethod
    async def search(self, unit: Unit, query: Query) -> Retrieved:
        """The owner's memory for the question, ranked, at most ``SEARCH_DEPTH`` items."""

    async def answer(self, unit: Unit, query: Query, context: str, gateway: Gateway) -> Completion:
        """The model's answer from the memory this system returned (``context``). A system
        whose product path builds the prompt itself (agent-harness) overrides this."""
        return llm_answer(gateway, context, query.text, query.at)

    async def close(self) -> None:  # noqa: B027
        """Release connections; what was written stays."""
