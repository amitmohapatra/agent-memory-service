"""Wider memory recall remains bounded and cannot change an explicit caller limit."""

from unittest.mock import AsyncMock

import pytest

from memory_service.config.constants import RETRIEVAL
from memory_service.modules.retrieval.engine import Candidate
from tests.unit import test_llm_retrieval as base

parts = base.parts


@pytest.mark.parametrize(
    ("kinds", "explicit", "expected"),
    [
        (("memory",), None, 4),
        (("memory",), 1, 1),
        (("chunk",), None, 2),
        (("memory", "chunk"), None, 2),
    ],
)
async def test_wider_recall_is_only_for_memory_only_pools(parts, kinds, explicit, expected):
    engine = base._engine(parts)
    engine.cfg = engine.cfg.model_copy(update={"final_k": 2, "memory_recall_k": 4, "rerank": False})

    async def search(*args, kind, **kwargs):
        return [
            Candidate(f"{kind}_{i}", kind, f"{kind} evidence {i}", 1 / (i + 1)) for i in range(8)
        ]

    engine._search_kind = AsyncMock(side_effect=search)
    result = await engine.retrieve(
        base.CTX, base.QUERY, kinds=kinds, visibility=base.VISIBILITY, limit=explicit
    )
    assert len(result.candidates) == expected
    assert result.diagnostics.get("memory_recall_k") == (
        4 if kinds == ("memory",) and explicit is None else None
    )


async def test_memory_search_depth_preserves_store_authorization_and_document_depth(parts):
    engine = base._engine(parts)
    engine.cfg = engine.cfg.model_copy(update={"final_k": 2, "memory_recall_k": 4})
    engine.store.search_hybrid = AsyncMock(return_value=[])
    for kind in ("memory", "chunk"):
        await engine._hybrid(
            base.QUERY, base.VISIBILITY, kind=kind, document_ids=None, encoded=(None, None)
        )
        kwargs = engine.store.search_hybrid.call_args.kwargs
        assert kwargs["limit"] == kwargs["prefetch_limit"] == (8 if kind == "memory" else 4)
        assert kwargs["flt"].tenant_id == base.VISIBILITY.tenant_id
        assert kwargs["flt"].must_any["visibility_keys"] == list(base.VISIBILITY.keys)
        if kind == "memory":
            assert kwargs["flt"].must["current"] is True


@pytest.mark.parametrize(
    ("kinds", "explicit", "expected"),
    [
        (("memory",), None, 100),
        (("memory",), 12, 12),
        (("chunk",), None, 50),
        (("memory", "chunk"), None, 50),
    ],
)
async def test_promoted_defaults_apply_through_retrieval(parts, kinds, explicit, expected):
    engine = base._engine(parts)
    engine.cfg = RETRIEVAL

    async def search(*args, kind, **kwargs):
        return [
            Candidate(f"{kind}_{i}", kind, f"{kind} evidence {i}", 1 / (i + 1)) for i in range(120)
        ]

    engine._search_kind = AsyncMock(side_effect=search)
    result = await engine.retrieve(
        base.CTX, base.QUERY, kinds=kinds, visibility=base.VISIBILITY, limit=explicit
    )
    assert len(result.candidates) == expected
