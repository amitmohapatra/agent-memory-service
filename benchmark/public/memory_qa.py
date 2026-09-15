"""LongMemEval and LoCoMo through the memory pipeline.

Every conversation session becomes a thread whose turns are appended as messages with the
benchmark's date as ``occurred_at`` (the pipeline keeps it as the memory's ``observed_at``
and derives ``valid_from`` from the content); the observation pipeline runs to completion
after each session. Each question is answered from the in-process ``/v1/context`` bundle by
the strong model behind Bifrost and scored with the benchmark's own grader: LongMemEval's
GPT-judge prompt repeated ``judge_runs`` times (mean ± sd), LoCoMo's F1/BLEU-1 rules.

Configurations: ``native`` (no LLM uses inside the pipeline), ``bifrost`` (every LLM use
enabled) and the third-party memory providers (``mem0``, ``langmem``, ``cognee``), each
skipped with a reason when it cannot run here.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, get_args

from sqlalchemy import text

from benchmark.public.data import Conversation, MemoryDataset, Question
from benchmark.public.judge import anscheck_prompt, generate_answer, judge_once
from benchmark.public.metrics import grouped_judge_summary, locomo_score, stemmer_name
from benchmark.retrieval import TABLES, _pct, _settings
from memory_service.__about__ import __version__
from memory_service.application.container import build_container
from memory_service.config.settings import LLMUse, Settings
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.enums import MessageRole
from memory_service.domain.errors import DependencyUnavailable, ProviderNotConfigured
from memory_service.domain.ids import new_id
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.llm.cost import LLMTokens, start_llm_accounting

TENANT = "public"
NATIVE_CONFIGS = ("native", "bifrost")
PROVIDER_CONFIGS = ("mem0", "langmem", "cognee")
LLM_DISABLED = (
    "models.llm.enabled=false: answers are generated and judged through the Bifrost gateway "
    "(set MEMORY__MODELS__LLM__ENABLED=true, MEMORY__MODELS__LLM__BASE_URL and "
    "MEMORY__MODELS__LLM__API_KEY)"
)


def config_settings(base: Settings, config: str) -> Settings:
    data = base.model_dump()
    llm = dict(data["models"]["llm"])
    if config == "native":
        llm["uses"] = []
    elif config == "bifrost":
        llm["uses"] = list(get_args(LLMUse))
    elif config in PROVIDER_CONFIGS:
        llm["uses"] = []
        data["memory_intelligence"] = {**data["memory_intelligence"], "provider": config}
    else:
        raise ValueError(f"unknown configuration {config!r}")
    data["models"] = {**data["models"], "llm": llm}
    return Settings(**data)


@contextmanager
def _accounting(costs: dict[str, Any], phase: str) -> Iterator[LLMTokens]:
    counter = start_llm_accounting()
    try:
        yield counter
    finally:
        costs[phase] = {"input": counter.input, "output": counter.output, "total": counter.total}


def _role(turn_role: str) -> MessageRole:
    return MessageRole.ASSISTANT if turn_role.lower() == "assistant" else MessageRole.USER


def _content(conv: Conversation, speaker: str | None, content: str) -> str:
    return f"{speaker}: {content}" if speaker else content


async def _ingest_conversation(container: Any, conv: Conversation) -> dict[str, int]:
    uow_factory = container.services["uow_factory"]
    conversation = container.services["conversation"]
    user_ctx = MemoryExecutionContext(
        tenant_id=TENANT, user_id=conv.conversation_id, workspace_id="bench"
    )
    messages = 0
    for session in conv.sessions:
        thread_id = new_id("thread")
        session_id = new_id("session")
        turn_id = new_id("turn")
        for turn in session.turns:
            role = _role(turn.role)
            if role is MessageRole.USER and messages:
                turn_id = new_id("turn")
            ctx = user_ctx.model_copy(
                update={"thread_id": thread_id, "session_id": session_id, "turn_id": turn_id}
            )
            async with uow_factory() as uow:
                await conversation.append_message(
                    uow,
                    ctx,
                    role=role,
                    content=_content(conv, turn.speaker, turn.content),
                    occurred_at=session.date,
                    source_system=f"bench:{conv.conversation_id}",
                    source_message_id=f"{session.session_id}:{messages}",
                )
                await uow.commit()
            messages += 1
        await container.tasks.drain()
    await container.tasks.drain()
    return {"sessions": len(conv.sessions), "messages": messages}


async def _answer(
    container: Any, conv: Conversation, question: Question, *, token_budget: int
) -> tuple[str, float, int]:
    builder = container.services["context_builder"]
    ctx = MemoryExecutionContext(
        tenant_id=TENANT, user_id=conv.conversation_id, workspace_id="bench"
    )
    t = time.perf_counter()
    bundle = await builder.build(ctx, question.text, token_budget=token_budget)
    context_ms = (time.perf_counter() - t) * 1000
    answer = await generate_answer(
        container.llm, context=bundle.render(), question=question.text, date=question.date
    )
    return answer, context_ms, bundle.token_estimate


async def _judge_longmemeval(
    container: Any, items: list[tuple[Question, str]], *, runs: int
) -> dict[str, Any]:
    labels: list[list[bool]] = []
    for _ in range(runs):
        labels.append(
            [
                await judge_once(
                    container.llm,
                    anscheck_prompt(
                        q.category, q.text, q.answer, answer, abstention=q.abstention
                    ),
                )
                for q, answer in items
            ]
        )
    return grouped_judge_summary(labels, [q.category for q, _ in items])


def score_locomo(items: list[tuple[Question, str]]) -> dict[str, Any]:
    per_category: dict[str, list[dict[str, float]]] = {}
    for q, answer in items:
        per_category.setdefault(q.category, []).append(locomo_score(answer, q.answer, q.category))
    out: dict[str, Any] = {}
    everything: list[dict[str, float]] = []
    for category, rows in sorted(per_category.items()):
        everything.extend(rows)
        out[f"category_{category}"] = {
            "f1": round(statistics.fmean(r["f1"] for r in rows), 4),
            "bleu1": round(statistics.fmean(r["bleu1"] for r in rows), 4),
            "questions": len(rows),
        }
    out["overall"] = {
        "f1": round(statistics.fmean(r["f1"] for r in everything), 4) if everything else 0.0,
        "bleu1": round(statistics.fmean(r["bleu1"] for r in everything), 4) if everything else 0.0,
        "questions": len(everything),
    }
    out["stemmer"] = stemmer_name()
    return out


async def run_config(
    dataset: MemoryDataset,
    config: str,
    *,
    judge_runs: int = 5,
    token_budget: int = 6000,
    keep_answers: bool = False,
) -> dict[str, Any]:
    settings = config_settings(_settings(), config)
    if not settings.models.llm.enabled:
        return {"skipped": LLM_DISABLED, "config": config}
    try:
        container = await build_container(settings, __version__)
    except (DependencyUnavailable, NotImplementedError, ProviderNotConfigured, ImportError) as exc:
        return {"skipped": f"{type(exc).__name__}: {exc}", "config": config}
    costs: dict[str, Any] = {}
    try:
        async with container.database.engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
        register_handlers(container)
        ingest = {"sessions": 0, "messages": 0}
        t0 = time.perf_counter()
        with _accounting(costs, "ingest"):
            for conv in dataset.conversations:
                stats = await _ingest_conversation(container, conv)
                ingest["sessions"] += stats["sessions"]
                ingest["messages"] += stats["messages"]
        ingest_seconds = round(time.perf_counter() - t0, 2)
        async with container.database.engine.connect() as conn:
            memories = (await conn.execute(text("SELECT count(*) FROM memories"))).scalar_one()

        items: list[tuple[Question, str]] = []
        context_ms: list[float] = []
        bundle_tokens: list[int] = []
        with _accounting(costs, "answer"):
            for conv, question in dataset.questions:
                answer, ms, tokens = await _answer(
                    container, conv, question, token_budget=token_budget
                )
                items.append((question, answer))
                context_ms.append(ms)
                bundle_tokens.append(tokens)

        if dataset.name == "longmemeval":
            with _accounting(costs, "judge"):
                scores = await _judge_longmemeval(container, items, runs=judge_runs)
            judge: dict[str, Any] = {
                "runs": judge_runs,
                "model": container.llm.model,
                "grader": "LongMemEval evaluate_qa.py prompts (yes/no), repeated",
            }
        else:
            scores = score_locomo(items)
            judge = {"runs": 0, "grader": "LoCoMo task_eval/evaluation.py F1 rules + BLEU-1"}
        costs["total"] = sum(v["total"] for v in costs.values() if isinstance(v, dict))
        row: dict[str, Any] = {
            "config": config,
            "llm_uses": list(settings.models.llm.uses),
            "memory_provider": settings.memory_intelligence.provider,
            "ingest": {**ingest, "seconds": ingest_seconds, "memories": memories},
            "questions": len(items),
            "scores": scores,
            "judge": judge,
            "context": {
                "p50_ms": _pct(context_ms, 50),
                "p95_ms": _pct(context_ms, 95),
                "mean_bundle_tokens": round(statistics.fmean(bundle_tokens), 1)
                if bundle_tokens
                else 0.0,
            },
            "llm_tokens": costs,
            "answer_model": container.llm.model,
        }
        if keep_answers:
            row["answers"] = [
                {"question_id": q.question_id, "answer": a, "gold": q.answer} for q, a in items
            ]
        return row
    finally:
        await container.close()


def format_table(name: str, configs: dict[str, Any]) -> str:
    lines = [f"{name:12} {'config':9} {'score':>16} {'questions':>9} {'tokens':>9} {'ctx p95':>8}"]
    for cname, row in configs.items():
        if "skipped" in row:
            lines.append(f"{name:12} {cname:9} skipped: {row['skipped'][:70]}")
            continue
        overall = row["scores"]["overall"]
        if "mean" in overall:
            score = f"{overall['mean']:.4f} ± {overall['sd']:.4f}"
        else:
            score = f"F1 {overall['f1']:.4f} B1 {overall['bleu1']:.3f}"
        lines.append(
            f"{name:12} {cname:9} {score:>16} {row['questions']:9d} "
            f"{row['llm_tokens']['total']:9d} {row['context']['p95_ms']:8.1f}"
        )
    return "\n".join(lines)
