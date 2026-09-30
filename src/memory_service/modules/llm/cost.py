"""Per-request and per-job LLM token accounting.

The API middleware opens an accounting scope around each request and the job registry around
each job; every gateway call made inside adds its usage to that scope's counter (the object is
shared across the tasks Starlette spawns, so child-task context copies still see it). A
request's total is exposed as ``X-Trellis-LLM-Tokens`` and, per report, as ``llm_tokens``; a
job's is logged. Scopes nest: a job run inline inside a request counts on its own and leaves
the request's counter as it was.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class LLMTokens:
    input: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output


_tokens: ContextVar[LLMTokens | None] = ContextVar("memory_llm_tokens", default=None)


@contextmanager
def llm_accounting() -> Iterator[LLMTokens]:
    """A fresh counter for the enclosed execution (request, job or test)."""
    counter = LLMTokens()
    token = _tokens.set(counter)
    try:
        yield counter
    finally:
        _tokens.reset(token)


def record_llm_tokens(input_tokens: int | None, output_tokens: int | None) -> None:
    counter = _tokens.get()
    if counter is None:
        return
    counter.input += int(input_tokens or 0)
    counter.output += int(output_tokens or 0)


def llm_tokens_used() -> int:
    counter = _tokens.get()
    return counter.total if counter is not None else 0
