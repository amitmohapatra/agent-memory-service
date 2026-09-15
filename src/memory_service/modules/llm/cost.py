"""Per-request LLM token accounting.

The API middleware opens an accounting scope before handing the request to the app; every
gateway call made while serving it adds its usage to the same mutable counter (the object is
shared across the tasks Starlette spawns, so child-task context copies still see it). The
total is exposed as ``X-Memory-LLM-Tokens`` and, per report, as ``llm_tokens``.
"""

from __future__ import annotations

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


def start_llm_accounting() -> LLMTokens:
    """Open a fresh counter for the current execution (request, job or test)."""
    counter = LLMTokens()
    _tokens.set(counter)
    return counter


def record_llm_tokens(input_tokens: int | None, output_tokens: int | None) -> None:
    counter = _tokens.get()
    if counter is None:
        return
    counter.input += int(input_tokens or 0)
    counter.output += int(output_tokens or 0)


def llm_tokens_used() -> int:
    counter = _tokens.get()
    return counter.total if counter is not None else 0
