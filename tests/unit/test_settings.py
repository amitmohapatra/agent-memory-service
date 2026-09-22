import pytest
from pydantic import ValidationError

from memory_service.config.constants import CONTEXT, FROZEN_MODELS, RETRIEVAL
from memory_service.config.settings import Settings


def test_defaults_are_cpu_first_and_llm_disabled() -> None:
    s = Settings(_env_file=None)
    assert s.models.llm.enabled is False
    assert FROZEN_MODELS.dense.id == "ibm-granite/granite-embedding-small-english-r2"
    assert FROZEN_MODELS.dense.dimension == 384 and FROZEN_MODELS.dense.backend == "torch"
    assert FROZEN_MODELS.reranker is None, "no reranker ships (SciFact -5.2 nDCG, p=0.012)"
    assert RETRIEVAL.bm25 and RETRIEVAL.dense and RETRIEVAL.exact and RETRIEVAL.graph
    # `rerank` is off on measured evidence, not on caution — see RetrievalSettings.rerank.
    assert RETRIEVAL.rerank is False


def test_todays_depth_is_the_frozen_depth() -> None:
    """One knob: ``final_k``, with prefetch and fusion derived from it. The depth itself is
    unchanged - 100/100/50 is what every shipped-depth artifact on disk was produced at, and
    moving it is a measured change, not a refactor. See ``test_retrieval_depth.py``."""
    assert (RETRIEVAL.prefetch_k, RETRIEVAL.fused_k, RETRIEVAL.final_k) == (100, 100, 50)
    assert (CONTEXT.memories_max, CONTEXT.token_budget) == (50, 8000)


def test_env_overrides_with_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY__DATABASE__POOL_SIZE", "3")
    monkeypatch.setenv("MEMORY__SERVICE__LOG_LEVEL", "WARNING")
    s = Settings(_env_file=None)
    assert s.database.pool_size == 3
    assert s.service.log_level == "WARNING"


def test_env_beats_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real environment variable outranks ``.env``: compose sets the topology through
    ``environment:`` and an operator's copied-in line must not be able to shadow it."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("MEMORY__DATABASE__POOL_SIZE=7\nMEMORY__SERVICE__PORT=9999\n")
    monkeypatch.delenv("MEMORY__DATABASE__POOL_SIZE", raising=False)
    monkeypatch.setenv("MEMORY__SERVICE__PORT", "8181")
    s = Settings(_env_file=str(dotenv))
    assert s.database.pool_size == 7, ".env is still read"
    assert s.service.port == 8181, "the environment wins over .env"


def test_the_test_stand_ins_are_not_settings() -> None:
    """A deployment cannot be pointed at an in-memory queue, a dict cache or a hash embedding
    by an env file.

    ``cache.provider=memory``, ``search.provider=memory``, ``tasks.provider=inline`` and
    ``models.embedding.provider=hash`` were settings, so the suite's stand-ins were part of
    the operator surface. They are ``build_container(overrides=...)`` now, reachable from
    code only.
    """
    from memory_service.config.settings import (
        AuthorizationSettings,
        CacheSettings,
        EmbeddingSettings,
        SearchSettings,
        TaskSettings,
    )

    for section in (CacheSettings, SearchSettings, TaskSettings, AuthorizationSettings):
        assert "provider" not in section.model_fields, section.__name__
    assert "qdrant_local_path" not in SearchSettings.model_fields
    assert set(EmbeddingSettings.model_fields) == {"threads"}


def _leaves(model: type, prefix: str = "") -> list[str]:
    import typing

    from pydantic import BaseModel

    out: list[str] = []
    for name, field in model.model_fields.items():
        ann = field.annotation
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        base = args[0] if args and typing.get_origin(ann) is typing.Union else ann
        if isinstance(base, type) and issubclass(base, BaseModel):
            out += _leaves(base, f"{prefix}{name}.")
        else:
            out.append(f"{prefix}{name}")
    return out


def test_the_environment_surface_is_topology_and_credentials_only() -> None:
    """206 leaf fields became ~40: every model, depth, budget, timeout and threshold is a
    constant now (config/constants.py). Growing this number needs a reason that is about a
    deployment, not about tuning."""
    leaves = _leaves(Settings)
    assert len(leaves) <= 40, f"{len(leaves)} env fields: {leaves}"
    for forbidden in ("prefetch_k", "final_k", "token_budget", "dimension", "model_path"):
        assert not [leaf for leaf in leaves if leaf.endswith(forbidden)], forbidden


def test_prod_guards_reject_dev_only_providers() -> None:
    with pytest.raises(ValueError, match="trusted_dev"):
        Settings(_env_file=None, service={"environment": "prod"})
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


def test_secrets_are_masked() -> None:
    """Every credential is a SecretStr and ``redacted()`` shows none of them. database.url
    was a plain str that redacted() did not mask: a /version response carried the password."""
    scheme = "postgresql+psycopg://"
    s = Settings(
        _env_file=None,
        database={"url": scheme + "memory:hunter2@db:5432/memory"},
        cache={"url": "redis://:cachepass@cache:6379/0"},
        search={"qdrant_api_key": "qdrantsecret"},
        authorization={"openfga_api_token": "supersecret"},
        authentication={"trusted_dev_api_keys": ["devsecret"]},
        models={"llm": {"api_key": "virtualkey"}},
    )
    dumped = str(s.redacted())
    for secret in (
        "hunter2",
        "cachepass",
        "qdrantsecret",
        "supersecret",
        "devsecret",
        "virtualkey",
    ):
        assert secret not in dumped, secret
    # the adapters still get the real value, in both spellings
    assert "hunter2" in s.database.dsn and s.database.dsn.startswith(scheme)
    assert (
        "hunter2" in s.database.procrastinate_dsn and "+psycopg" not in s.database.procrastinate_dsn
    )


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
