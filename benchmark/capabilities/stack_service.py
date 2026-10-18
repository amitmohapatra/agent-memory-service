"""The isolated stack's memory-api and memory-worker, started with the eval's tuning overrides.

    python -c "from benchmark.capabilities.stack_service import main; main('api')"     # or 'worker'

``isolated_stack.sh`` starts the service through this module in llm mode. It is the service's
own entry point (``run_api`` / ``run_worker``) with ``Overrides.llm`` - the frozen-tuning
override the container keeps for tests and benchmarks - set from the environment:

``CAPEVAL_LLM_TIMEOUT_SECONDS``
    the model call's timeout. ``constants.LLM.timeout_seconds`` (30 s) is sized for a hosted
    gateway model. The eval's gateway serves Granite 3.3 2B on the measuring host's CPU at
    about 8 tokens a second: a 160-token summary takes 20-30 s alone and 50-65 s beside three
    others, so at 30 s nearly every long call timed out, was retried twice and fell back, and
    the breaker opened for everyone (main's llm_core baseline: >1,000 fallbacks). The eval sets
    a timeout the model can meet, and records it with the run (``meta.stack``).

``CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS``
    the read path's deadline for model-assisted query expansion
    (``RetrievalSettings.query_expansion_timeout_ms``, 250 ms). The CPU model needs seconds to
    answer: at 250 ms every expansion was abandoned (llama-server cancelled 1,007 requests in
    one llm_core run), so the llm baseline's read path never used the model and its context
    latency was the deadline. The eval sets one the model can meet (recorded in ``meta.stack``);
    its context latency then includes the CPU model and is reported, not gated.

Unset, this is exactly ``run_api()`` / ``run_worker()``.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from types import ModuleType
from typing import Any


def overrides() -> Any:
    """The eval's ``Overrides``, or None when nothing is overridden."""
    timeout = os.environ.get("CAPEVAL_LLM_TIMEOUT_SECONDS")
    expansion = os.environ.get("CAPEVAL_QUERY_EXPANSION_TIMEOUT_MS")
    if not timeout and not expansion:
        return None
    from memory_service.application.container import Overrides
    from memory_service.config.constants import LLM, RETRIEVAL

    return Overrides(
        llm=dataclasses.replace(LLM, timeout_seconds=float(timeout)) if timeout else None,
        # model_copy: the setting's own bound (5 s) is a production read path's, not a CPU model's
        retrieval=RETRIEVAL.model_copy(update={"query_expansion_timeout_ms": int(expansion)})
        if expansion
        else None,
    )


def _with_overrides(module: ModuleType, tuned: Any) -> None:
    build = module.build_container

    async def build_container(
        settings: Any, version: str, *, overrides: Any = None, **kw: Any
    ) -> Any:
        return await build(settings, version, overrides=overrides or tuned, **kw)

    module.build_container = build_container  # type: ignore[attr-defined]


def main(role: str) -> None:
    tuned = overrides()
    if role == "api":
        import memory_service.api.app as app

        if tuned is not None:
            _with_overrides(app, tuned)
        from memory_service.__main__ import run_api

        run_api()
    elif role == "worker":
        import memory_service.worker as worker

        if tuned is not None:
            _with_overrides(worker, tuned)
        from memory_service.__main__ import run_worker

        run_worker()
    else:
        sys.exit(f"role must be api or worker, not {role!r}")
