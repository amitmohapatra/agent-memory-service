"""Document parsers, registered rather than switched on.

``config/registry.py`` documents the intent — *"adding a provider means registering a
factory, never editing a switch statement"* — but nothing ever registered one, and wiring
grew the switch statement anyway. This is that registry actually used.

A new parser is added by writing an adapter and appending one entry to ``PARSERS`` below,
plus its name in ``DocumentSettings.parser`` so the value is typed and appears in the
OpenAPI schema. A test asserts those two lists agree, so they cannot drift.

Each factory returns ``None`` when its dependency is absent, which is how a parser reports
"I cannot run here" without the caller knowing why. The caller decides what to do about it:
fall back to the builtin, and say so in ``/version.degraded``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

from memory_service.observability.logging import get_logger

log = get_logger(__name__)


def _builtin() -> Any:
    from memory_service.adapters.parsers.builtin import BuiltinParser

    return BuiltinParser()


def _docling() -> Any | None:
    # DoclingParser imports docling lazily on its first parse, so constructing it succeeds in
    # an image built without the extra and the downgrade would surface much later as poorly
    # parsed documents. Check the dependency here, where it can still be reported.
    if importlib.util.find_spec("docling") is None:
        log.warning(
            "parser.unavailable",
            parser="docling",
            reason="docling is not installed",
            hint='rebuild with --build-arg EXTRAS="gcp models docling"',
        )
        return None
    try:
        from memory_service.adapters.parsers.docling_parser import DoclingParser

        return DoclingParser()
    except Exception as exc:  # an import that resolves but cannot initialise
        log.warning("parser.unavailable", parser="docling", error=str(exc))
        return None


#: name -> factory. The one place a parser is added.
PARSERS: dict[str, Callable[[], Any | None]] = {
    "builtin": _builtin,
    "docling": _docling,
}


def register_parsers(registry: Any) -> None:
    for name, factory in PARSERS.items():
        registry.register(name, factory)
