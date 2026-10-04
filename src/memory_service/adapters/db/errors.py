"""What a database failure means to a client: the driver's exceptions as domain errors.

A route that meets PostgreSQL going away should answer the 503 a client retries, not the
500 a client gives up on; a statement the server stopped at ``statement_timeout`` is a 504
the same request may pass on a quieter database. The API layer may not import the driver
(tests/unit/test_architecture.py), so the reading lives here and ``api/errors.py`` installs
it as the handler for :data:`DATABASE_ERRORS`. Anything else the driver raises - an
integrity error a repository did not translate, a programming error - stays a 500: it is a
bug, and retrying it changes nothing.

The problem's ``detail`` never carries the driver's message: it can quote SQL and
parameter values. The message is logged instead.
"""

from __future__ import annotations

from typing import Final

from psycopg.errors import QueryCanceled
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeout

from memory_service.domain.errors import (
    DependencyUnavailable,
    MemoryServiceError,
    OperationTimedOut,
)
from memory_service.observability.logging import get_logger

log = get_logger(__name__)

#: The exception classes :func:`database_error` reads; the API installs one handler for them.
DATABASE_ERRORS: Final[tuple[type[Exception], ...]] = (SQLAlchemyError,)


def database_error(exc: BaseException) -> MemoryServiceError | None:
    """The domain error a driver exception stands for, or ``None`` when it is a bug.

    - a statement cancelled by the server (``statement_timeout``): ``OperationTimedOut`` (504);
    - no connection within the pool timeout: ``DependencyUnavailable`` (503);
    - the server unreachable, the connection lost or invalidated mid-statement
      (``OperationalError``, ``InterfaceError``, ``connection_invalidated``): 503.
    """
    if isinstance(exc, DBAPIError) and isinstance(exc.orig, QueryCanceled):
        failure: MemoryServiceError = OperationTimedOut(
            "a database statement exceeded its time budget"
        )
    elif isinstance(exc, PoolTimeout):
        failure = DependencyUnavailable("PostgreSQL connection pool exhausted")
    elif isinstance(exc, OperationalError | InterfaceError) or (
        isinstance(exc, DBAPIError) and exc.connection_invalidated
    ):
        failure = DependencyUnavailable("PostgreSQL unavailable")
    else:
        return None
    log.warning(
        "db.unavailable", code=failure.code, error_type=type(exc).__name__, error=str(exc)[:500]
    )
    return failure
