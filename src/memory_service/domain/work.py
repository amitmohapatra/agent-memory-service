"""Whether the running code is background work (a job) or a request someone waits on.

The in-process models (``adapters.models._runner``) are one caller at a time, and requests
and jobs share them: the verify path and the statement labeller one NLI head, a query's
encode and the indexing jobs one encoder. A request waiting for a model enters before any
job still waiting (``SerialRunner``); a job handler runs inside :func:`background`
(``modules.jobs.registry``). Nothing else reads it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_BACKGROUND: ContextVar[bool] = ContextVar("memory_background_work", default=False)


def in_background() -> bool:
    """Whether the current task is background work (no caller waits on it)."""
    return _BACKGROUND.get()


@contextmanager
def background() -> Iterator[None]:
    """Run the block as background work."""
    token = _BACKGROUND.set(True)
    try:
        yield
    finally:
        _BACKGROUND.reset(token)
