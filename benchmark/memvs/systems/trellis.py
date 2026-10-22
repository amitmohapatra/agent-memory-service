"""This service, the way agent-harness uses it (runs in agent-harness's environment).

Writes are what the harness records after a run: the transcript through ``history.add``
(``POST /v1/messages``), each session a thread of the owner's, its date on every message, the
dataset's turn id as ``source_message_id``. Each session's processing jobs finish before the
next session is written (a conversation days apart), outside the write's latency.

Reads: the harness's push - ``/v1/context`` in the prompt form within the harness's budget
(2,000 tokens for an 8,192-token window), asked in a thread of its own - is the context and the
read latency. Recall is measured on the full form of the same question at the service's
default budget, whose memories carry the messages they were read out of.

Answers: ``Trellis`` - ``h.wrap`` of a function agent: the harness pushes the context and the
function asks the model with the same instruction every system gets. ``TrellisReAct`` -
``h.wrap(ReAct(...))``: the harness's ReAct agent with its memory tools, on the same model.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

from benchmark.capabilities.runner import (
    INDEX_QUEUES,
    SETTLE_QUEUES,
    Service,
    _calm,
    _now,
)
from benchmark.capabilities.scoring import items_from_bundle
from benchmark.memvs.llm import (
    ANSWER_MAX_TOKENS,
    ANSWER_SYSTEM,
    MEMORY_TOKEN_BUDGET,
    NOTHING_ON_FILE,
    Completion,
    Gateway,
)
from benchmark.memvs.model import (
    DatasetName,
    Hit,
    Query,
    Role,
    Session,
    SystemName,
    Turn,
    Unit,
    turn_session,
)
from benchmark.memvs.systems.base import (
    EVENT_SPEAKER,
    Attribution,
    MemorySystem,
    MemorySystemError,
    Retrieved,
    spoken,
)

ENV_URL = "MEMORY_URL"
ENV_ADMIN_KEY = "CAPEVAL_ADMIN_KEY"
ENV_WORKER_METRICS = "MEMORY_WORKER_METRICS_URL"
KEYS_FILE = "keys.json"
ATTRIBUTION_FILE = "attribution.json"
SOURCE_SYSTEM = "memvs"
#: the runner's batch: one ``history.add`` call per 25 messages
BATCH = 25
CLIENT_TIMEOUT_S = 300.0
JOB_WAIT_S = 1800.0
#: graph enrichment and summaries trail indexing; the capability gate waits this long for them
ENRICH_SECONDS = 900.0
SETTLE_SECONDS = 7200.0
RUN_TIMEOUT_S = 1800.0
AGENT_ID = "memvs"
#: the ReAct agent's model name on the gateway
REACT_MODEL_ENV = "MEMVS_REACT_MODEL"
_ROLE = {Role.USER: "USER", Role.ASSISTANT: "ASSISTANT"}


def _role(turn: Turn) -> str:
    return "EVENT" if turn.speaker == EVENT_SPEAKER else _ROLE[turn.role]


def _ask_thread(unit: Unit, query: Query) -> str:
    return f"{unit.user}-ask-{hashlib.sha1(query.query_id.encode()).hexdigest()[:10]}"


class Trellis(MemorySystem):
    name = SystemName.TRELLIS

    async def open(self) -> None:
        from trellis.memory import MemoryClient

        try:
            self.url, self.admin_key = os.environ[ENV_URL], os.environ[ENV_ADMIN_KEY]
        except KeyError as missing:
            raise MemorySystemError(
                f"{missing} is not set (run inside isolated_stack.sh)"
            ) from None
        self._client_cls = MemoryClient
        self.keys_path = self.env.store_dir / KEYS_FILE
        self.keys: dict[str, str] = (
            json.loads(self.keys_path.read_text()) if self.keys_path.exists() else {}
        )
        self.clients: dict[str, Any] = {}
        self.attribution = Attribution.load(self.env.store_dir / ATTRIBUTION_FILE)
        self.pending: list[tuple[Any, list[str]]] = []
        self.first_write: float | None = None

    async def _client(self, tenant: str) -> Any:
        if tenant not in self.keys:
            self.keys[tenant] = await self._provision(tenant)
            self.keys_path.write_text(json.dumps(self.keys))
        if tenant not in self.clients:
            self.clients[tenant] = self._client_cls(
                self.url, api_key=self.keys[tenant], timeout=CLIENT_TIMEOUT_S
            )
        return self.clients[tenant]

    async def _provision(self, tenant: str) -> str:
        """A tenant of its own with no model anywhere (the gate's nollm policy) and a service key."""
        async with self._client_cls(self.url, api_key=self.admin_key) as platform:
            created = await platform.admin.create_tenant(f"memvs {tenant}", tenant_id=tenant)
            admin_key = created.admin_key.token or self.admin_key
        async with self._client_cls(self.url, api_key=admin_key) as admin:
            handle = admin.administer(tenant)
            await handle.set_model_policy([], read_assist=False, models={})
            key = await handle.keys.issue("service", f"memvs-{tenant}")
        if not key.token:
            raise MemorySystemError(f"no service key issued for {tenant}")
        return key.token

    async def write(self, unit: Unit, session: Session) -> None:
        from trellis.memory import Message

        client = await self._client(unit.tenant)
        ctx = client.bind(user_id=unit.user, thread_id=f"{unit.user}-{session.session_id}")
        start = datetime.fromisoformat(session.at) if session.at else None
        jobs: list[str] = []
        self.first_write = self.first_write or time.time()
        for i in range(0, len(session.turns), BATCH):
            batch = session.turns[i : i + BATCH]
            acks = await ctx.history.add(
                [
                    Message(
                        role=_role(t),  # type: ignore[arg-type]
                        content=spoken(t),
                        # one second apart: the order of a session survives the shared date
                        occurred_at=start + timedelta(seconds=i + k) if start else None,
                        source_system=SOURCE_SYSTEM,
                        source_message_id=t.turn_id,
                    )
                    for k, t in enumerate(batch)
                ]
            )
            for ack, turn in zip(acks, batch, strict=True):
                self.attribution.credit([ack.message_id], turn.turn_id)
                jobs += ack.job_ids
        self.pending.append((ctx, jobs))

    async def after_write(self, unit: Unit) -> None:
        client = await self._client(unit.tenant)
        service = Service(client, os.environ.get(ENV_WORKER_METRICS))
        for ctx, jobs in self.pending:
            failed = await service.wait_jobs(ctx, jobs, within=JOB_WAIT_S)
            if failed:
                raise MemorySystemError(f"{failed} processing jobs did not succeed")
        self.pending.clear()
        self.attribution.save()

    async def settle(self) -> None:
        """Indexing drained, then graph enrichment and summaries (bounded), as the gate waits."""
        if not self.clients:
            return
        service = Service(next(iter(self.clients.values())), os.environ.get(ENV_WORKER_METRICS))
        since = self.first_write or time.time()
        if service.worker_metrics_url:
            await _calm(service, INDEX_QUEUES, since, _now() + SETTLE_SECONDS)
            await _calm(service, SETTLE_QUEUES, since, _now() + ENRICH_SECONDS)

    async def search(self, unit: Unit, query: Query) -> Retrieved:
        client = await self._client(unit.tenant)
        ctx = client.bind(user_id=unit.user, thread_id=_ask_thread(unit, query))
        t0 = time.perf_counter()
        pushed = await ctx.context(query.text, token_budget=MEMORY_TOKEN_BUDGET)
        latency = (time.perf_counter() - t0) * 1000
        if self.env.dataset is DatasetName.CAPABILITY:
            # the capability gate's surface: every part of the context it scores, in the order
            # it renders them, within the same budget the other systems' hits are held to
            bundle = await ctx.context(query.text, token_budget=MEMORY_TOKEN_BUDGET, format="full")
            hits = [
                Hit(i.text, historical=i.historical)
                for i in items_from_bundle(bundle.model_dump(mode="json"))
            ]
        else:
            full = await ctx.context(query.text, format="full")
            hits = [self._hit(m) for m in full.memories]
        return Retrieved(
            hits,
            pushed.rendered,
            insufficient=pushed.evidence_status == "INSUFFICIENT",
            latency_ms=latency,
        )

    def _hit(self, memory: Any) -> Hit:
        sources = (memory.model_extra or {}).get("sources") or []
        turns = sorted(
            {
                t
                for s in sources
                if s.get("message_id")
                for t in self.attribution.of(s["message_id"])
            }
        )
        return Hit(
            memory.text,
            tuple(turns),
            tuple(sorted({turn_session(t) for t in turns})),
            historical=bool(memory.superseded),
            at=memory.observed_at,
        )

    async def answer(self, unit: Unit, query: Query, context: str, gateway: Gateway) -> Completion:
        """``h.wrap`` of a function agent: the harness pushes the memory, the function answers."""
        from trellis import Harness, Runtime  # pyright: ignore[reportAttributeAccessIssue]
        from trellis import (
            Settings as HarnessSettings,  # pyright: ignore[reportAttributeAccessIssue]
        )

        client = await self._client(unit.tenant)
        done: list[Completion] = []

        async def respond(question: str, agent: Runtime) -> str:
            system = ANSWER_SYSTEM.format(
                now=query.at or "unknown", memory=agent.context or NOTHING_ON_FILE
            )
            reply = gateway.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": question}],
                ANSWER_MAX_TOKENS,
            )
            done.append(reply)
            return reply.text

        settings = HarnessSettings.from_env().model_copy(
            update={
                "memory_url": self.url,
                "api_key": self.keys[unit.tenant],
                "grounding_sample": 0.0,
            }
        )
        async with Harness(config=settings, memory=client) as h:
            agent = h.wrap(respond, id=AGENT_ID, timeout=RUN_TIMEOUT_S)
            t0 = time.perf_counter()
            result = await agent.run(
                query.text, user=unit.user, thread=_ask_thread(unit, query), without={"records"}
            )
        if result.error is not None or not done:
            raise MemorySystemError(f"harness run ended {result.status}: {result.error}")
        last = done[-1]
        return Completion(
            last.text, last.prompt_tokens, last.completion_tokens, (time.perf_counter() - t0) * 1000
        )

    async def close(self) -> None:
        self.attribution.save()
        for client in self.clients.values():
            await client.aclose()


class TrellisReAct(Trellis):
    """The same store (written and read by ``Trellis``); answers through ``h.wrap(ReAct)``."""

    name = SystemName.TRELLIS_REACT

    async def answer(self, unit: Unit, query: Query, context: str, gateway: Gateway) -> Completion:
        from trellis import Harness, ReAct  # pyright: ignore[reportAttributeAccessIssue]
        from trellis import (
            Settings as HarnessSettings,  # pyright: ignore[reportAttributeAccessIssue]
        )

        client = await self._client(unit.tenant)
        system = ANSWER_SYSTEM.split("\n\n# Memory")[0].format(now=query.at or "unknown")
        target = ReAct(
            system=system,
            model=os.environ.get(REACT_MODEL_ENV, gateway.model),
            context_window=int(os.environ.get("MEMVS_CONTEXT_WINDOW", "8192")),
        )
        settings = HarnessSettings.from_env().model_copy(
            update={
                "memory_url": self.url,
                "api_key": self.keys[unit.tenant],
                "grounding_sample": 0.0,
            }
        )
        async with Harness(config=settings, memory=client) as h:
            agent = h.wrap(target, id=f"{AGENT_ID}-react", timeout=RUN_TIMEOUT_S)
            t0 = time.perf_counter()
            result = await agent.run(
                query.text, user=unit.user, thread=_ask_thread(unit, query), without={"records"}
            )
        if result.error is not None:
            raise MemorySystemError(f"harness run ended {result.status}: {result.error}")
        # the ReAct loop's tokens are counted by the run's token meter (answer.json), per run
        return Completion(str(result.answer or ""), 0, 0, (time.perf_counter() - t0) * 1000)
