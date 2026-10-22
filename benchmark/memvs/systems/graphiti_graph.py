"""Graphiti (the open core of Zep) on embedded Kuzu: each session is an episode the model turns
into entities and time-stamped facts; ``search`` returns facts (hybrid: meaning, BM25, graph
distance, reciprocal-rank fused). A fact names the episodes it came from, and each episode is
the session it was made of. Group id: the owner. Zep's own server (Community Edition) is no
longer maintained; Graphiti is what runs locally.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from graphiti_core import Graphiti as GraphitiCore
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType

from benchmark.memvs.model import Hit, Query, Session, SystemName, Unit
from benchmark.memvs.systems.base import (
    SEARCH_DEPTH,
    Attribution,
    MemorySystem,
    MemorySystemError,
    Retrieved,
    render,
    turn_text,
)

DB_FILE = "graph.kuzu"
ATTRIBUTION_FILE = "attribution.json"
SOURCE = "conversation"
UNUSED_KEY = "unused"
#: Graphiti's group ids take letters, digits, dashes and underscores only
_GROUP_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def group(unit: Unit) -> str:
    return _GROUP_UNSAFE.sub("_", f"{unit.tenant}__{unit.user}")


class Graphiti(MemorySystem):
    name = SystemName.GRAPHITI
    turn_level = False

    async def open(self) -> None:
        gateway = self.env.gateway
        if gateway is None:
            raise MemorySystemError("Graphiti's pipeline needs the model (MEMVS_LLM_*)")
        llm_config = LLMConfig(
            api_key=gateway.key,
            model=gateway.model,
            small_model=gateway.model,
            base_url=gateway.base_url,
            temperature=0,
        )
        embedder = self.env.embedder
        self.graph = GraphitiCore(
            graph_driver=KuzuDriver(db=str(self.env.store_dir / DB_FILE)),
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=OpenAIEmbedder(
                OpenAIEmbedderConfig(
                    api_key=UNUSED_KEY,
                    embedding_model=embedder.model,
                    embedding_dim=embedder.dimension,
                    base_url=embedder.base_url,
                )
            ),
            # never called by the default search; built so Graphiti does not build its own
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )
        await self.graph.build_indices_and_constraints()
        self.attribution = Attribution.load(self.env.store_dir / ATTRIBUTION_FILE)

    async def write(self, unit: Unit, session: Session) -> None:
        when = datetime.fromisoformat(session.at) if session.at else datetime.now(UTC)
        result = await self.graph.add_episode(
            name=session.session_id,
            episode_body="\n".join(turn_text(t) for t in session.turns),
            source_description=SOURCE,
            reference_time=when,
            source=EpisodeType.message,
            group_id=group(unit),
        )
        self.attribution.credit([result.episode.uuid], session.session_id)
        self.attribution.save()

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        edges = await self.graph.search(
            query.text, group_ids=[group(unit)], num_results=SEARCH_DEPTH
        )
        hits = [self._hit(e) for e in edges]
        return Retrieved(hits, render(hits), insufficient=not hits)

    def _hit(self, edge: Any) -> Hit:
        sessions = sorted({s for ep in edge.episodes or () for s in self.attribution.of(ep)})
        valid = edge.valid_at.isoformat() if edge.valid_at else None
        return Hit(edge.fact, (), tuple(sessions), historical=edge.invalid_at is not None, at=valid)

    async def close(self) -> None:
        self.attribution.save()
        await self.graph.close()
