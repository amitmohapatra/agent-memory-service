import pytest
from pydantic import ValidationError

from memory_service.config.settings import Settings


def test_defaults_are_cpu_first_and_llm_disabled() -> None:
    s = Settings(_env_file=None)
    assert s.models.llm.enabled is False
    assert s.models.embedding.model == "ibm-granite/granite-embedding-small-english-r2"
    assert s.retrieval.bm25 and s.retrieval.dense
    # `splade` is the one remaining benchmark-gated retrieval experiment. The other seven
    # (colbert, pageindex, raptor, graph_ppr, late_chunking, minicoil, graphrag_global) were
    # removed rather than left off: each named a capability something already-on provides,
    # so keeping them meant carrying code and configuration that could only ever be verified
    # to do nothing new.
    assert s.retrieval.splade is False, "splade must be benchmark-gated (off by default)"
    # `rerank` is off on measured evidence, not on caution — see RetrievalSettings.rerank.
    assert s.retrieval.rerank is False


def test_env_overrides_with_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY__DATABASE__POOL_SIZE", "3")
    monkeypatch.setenv("MEMORY__RETRIEVAL__RRF_K", "42")
    s = Settings(_env_file=None)
    assert s.database.pool_size == 3
    assert s.retrieval.rrf_k == 42


def test_the_test_stand_ins_are_not_settings() -> None:
    """A deployment cannot be pointed at an in-memory queue or a dict cache by an env file.

    ``cache.provider=memory``, ``search.provider=memory`` and ``tasks.provider=inline`` were
    settings, so the suite's stand-ins were part of the operator surface. They are
    ``build_container(overrides=...)`` now, reachable from code only.
    """
    from memory_service.config.settings import CacheSettings, SearchSettings, TaskSettings

    for section in (CacheSettings, SearchSettings, TaskSettings):
        assert "provider" not in section.model_fields, section.__name__
    assert "qdrant_local_path" not in SearchSettings.model_fields


def test_prod_guards_reject_dev_only_providers() -> None:
    with pytest.raises(ValueError, match="trusted_dev"):
        Settings(_env_file=None, service={"environment": "prod"})
    with pytest.raises(ValueError, match="authorization.provider=memory"):
        Settings(
            _env_file=None,
            service={"environment": "prod"},
            authentication={"mode": "jwt"},
            authorization={"provider": "memory"},
        )
    with pytest.raises(ValueError, match="blob.provider"):
        Settings(
            _env_file=None,
            service={"environment": "prod"},
            authentication={"mode": "jwt"},
            blob={"provider": "filesystem"},
        )


def test_llm_enabled_requires_a_model() -> None:
    """An LLM that is on and has no model name is a startup failure, not a runtime one.

    The companion check — that `enabled` and a separate `provider` field agreed — is gone
    with the field. Two settings that had to be kept in agreement were one setting.
    """
    with pytest.raises(ValueError, match="models.llm.model"):
        Settings(_env_file=None, models={"llm": {"enabled": True, "model": None}})
    assert Settings(_env_file=None, models={"llm": {"enabled": False}}).models.llm.enabled is False


def test_redacted_hides_secrets() -> None:
    s = Settings(_env_file=None, authorization={"openfga_api_token": "supersecret"})
    dumped = s.redacted()
    assert "supersecret" not in str(dumped)
    assert "trusted_dev_api_keys" not in dumped["authentication"]


# --------------------------------------------------- an LLM that would call nothing


def test_enabling_the_llm_without_naming_any_uses_is_refused() -> None:
    """Opting in per use is the design; the silence was not.

    ``uses`` defaults to empty and ``wants()`` requires membership, so
    ``enabled=true`` on its own started cleanly, reported ``"llm": "bifrost"`` on
    /version, and sent the gateway nothing at all. All ten paths quietly took their
    native fallback, and the only way to notice was that the token metrics never moved.
    """
    with pytest.raises(ValidationError, match="nothing would call the model"):
        Settings(
            _env_file=None,
            models={
                "llm": {
                    "enabled": True,
                    "model": "gemini/gemini-3.6-flash",
                }
            },
        )


def test_naming_a_use_is_enough_to_be_accepted() -> None:
    settings = Settings(
        _env_file=None,
        models={
            "llm": {
                "enabled": True,
                "model": "gemini/gemini-3.6-flash",
                "fast_model": "gemini/gemini-3.6-flash",
                "uses": ["query_expansion", "summaries"],
            }
        },
    )
    llm = settings.models.llm
    assert llm.wants("query_expansion") and llm.wants("summaries")
    assert not llm.wants("reflection"), "a use you did not name stays deterministic"


def test_an_enabled_fast_use_without_a_fast_model_is_refused() -> None:
    """fast_model is a separate credential in practice — it defaults to a different
    provider than model does, and ``fast_model or model`` in the adapter cannot rescue a
    mismatch because that default is truthy. Setting model and forgetting fast_model sends
    exactly the fast_uses somewhere the operator never chose."""
    with pytest.raises(ValidationError, match="fast_model is unset"):
        Settings(
            _env_file=None,
            models={
                "llm": {
                    "enabled": True,
                    "model": "gemini/gemini-3.6-flash",
                    "fast_model": None,
                    "uses": ["query_expansion"],
                }
            },
        )


def test_a_slow_only_use_list_does_not_need_a_fast_model() -> None:
    """The check is about overlap, not about fast_model always being set."""
    settings = Settings(
        _env_file=None,
        models={
            "llm": {
                "enabled": True,
                "model": "gemini/gemini-3.6-flash",
                "fast_model": None,
                "uses": ["summaries"],
            }
        },
    )
    assert settings.models.llm.wants("summaries")
