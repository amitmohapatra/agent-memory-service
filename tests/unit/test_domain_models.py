from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from memory_service.domain import (
    CanonicalMemory,
    ContextBundle,
    ConversationWindow,
    EvidenceRef,
    EvidenceReport,
    EvidenceStatus,
    Lifetime,
    MemoryType,
    QueryType,
    Scope,
    ScopeLevel,
    TemporalState,
    TemporalStatus,
    Visibility,
)
from memory_service.domain.ids import content_hash, is_valid_id, new_id, stable_key


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        source_type="message", source_id="msg_1", message_id="msg_1", observed_at=datetime.now(UTC)
    )


def test_new_id_prefixes_and_validity() -> None:
    assert new_id("thread").startswith("thr_")
    assert is_valid_id(new_id("memory"))
    assert not is_valid_id("")
    assert not is_valid_id("has space")
    with pytest.raises(ValueError):
        new_id("nope")


def test_content_hash_and_stable_key_are_deterministic() -> None:
    assert content_hash("a") == content_hash(b"a")
    assert stable_key("a", "b") == stable_key("a", "b")
    assert stable_key("a", "b") != stable_key("ab", "")


def test_scope_requires_anchor_for_level() -> None:
    with pytest.raises(ValidationError, match="requires user_id"):
        Scope(level=ScopeLevel.USER, tenant_id="t")
    s = Scope(level=ScopeLevel.THREAD, tenant_id="t", thread_id="thr_1")
    assert s.key() == "t=t/l=THREAD/thread=thr_1"


def test_canonical_memory_requires_evidence_and_matching_tenant() -> None:
    scope = Scope(level=ScopeLevel.USER, tenant_id="t", user_id="u")
    with pytest.raises(ValidationError):
        CanonicalMemory(
            tenant_id="t",
            scope=scope,
            visibility=Visibility.USER,
            owner_principal="user:u",
            lifetime=Lifetime.LONG_TERM,
            memory_type=MemoryType.PREFERENCE,
            content="likes tea",
            normalized_hash="x",
            temporal=TemporalState(observed_at=datetime.now(UTC)),
            evidence=[],
        )
    with pytest.raises(ValidationError, match="scope.tenant_id"):
        CanonicalMemory(
            tenant_id="other",
            scope=scope,
            visibility=Visibility.USER,
            owner_principal="user:u",
            lifetime=Lifetime.LONG_TERM,
            memory_type=MemoryType.PREFERENCE,
            content="likes tea",
            normalized_hash="x",
            temporal=TemporalState(observed_at=datetime.now(UTC)),
            evidence=[_evidence()],
        )
    with pytest.raises(ValidationError, match="custom_type"):
        CanonicalMemory(
            tenant_id="t",
            scope=scope,
            visibility=Visibility.USER,
            owner_principal="user:u",
            lifetime=Lifetime.LONG_TERM,
            memory_type=MemoryType.CUSTOM,
            content="x",
            normalized_hash="x",
            temporal=TemporalState(observed_at=datetime.now(UTC)),
            evidence=[_evidence()],
        )


def test_temporal_state_validity_window() -> None:
    now = datetime.now(UTC)
    ts = TemporalState(
        observed_at=now, valid_from=now - timedelta(days=1), valid_to=now + timedelta(days=1)
    )
    assert ts.is_current_at(now)
    assert not ts.is_current_at(now + timedelta(days=2))
    assert not TemporalState(observed_at=now, status=TemporalStatus.SUPERSEDED).is_current_at(now)


def test_evidence_citation_key_prefers_most_specific() -> None:
    e = EvidenceRef(
        source_type="chunk",
        source_id="s",
        document_id="doc_1",
        chunk_id="chk_1",
        observed_at=datetime.now(UTC),
    )
    assert e.citation_key() == "chunk_id:chk_1"
    assert _evidence().citation_key() == "message_id:msg_1"


def test_context_bundle_render_marks_insufficient_evidence() -> None:
    bundle = ContextBundle(
        query="q",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        evidence=EvidenceReport(status=EvidenceStatus.INSUFFICIENT, missing_groups=["PAGE11"]),
        token_budget=100,
        token_estimate=0,
    )
    assert "INSUFFICIENT" in bundle.render()
