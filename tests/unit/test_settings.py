import pytest

from memory_service.config.settings import Settings


def test_defaults_are_cpu_first_and_llm_disabled() -> None:
    s = Settings()
    assert s.models.llm.enabled is False
    assert s.models.embedding.model == "ibm-granite/granite-embedding-small-english-r2"
    assert s.retrieval.bm25 and s.retrieval.dense and s.retrieval.fusion == "rrf"
    for flag in (
        "splade",
        "colbert",
        "pageindex",
        "raptor",
        "graph_ppr",
        "late_chunking",
        "minicoil",
        "graphrag_global",
    ):
        assert getattr(s.retrieval, flag) is False, (
            f"{flag} must be benchmark-gated (off by default)"
        )


def test_env_overrides_with_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY__CACHE__PROVIDER", "valkey")
    monkeypatch.setenv("MEMORY__RETRIEVAL__RRF_K", "42")
    s = Settings()
    assert s.cache.provider == "valkey"
    assert s.retrieval.rrf_k == 42


def test_yaml_file_source(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # _env_file=None isolates the test from a developer's .env, which legitimately outranks
    # the YAML source and would otherwise decide the assertions below.
    cfg = tmp_path / "memory.yaml"
    cfg.write_text("search:\n  provider: memory\nmodels:\n  llm:\n    enabled: false\n")
    monkeypatch.setenv("MEMORY_CONFIG_FILE", str(cfg))
    monkeypatch.delenv("MEMORY__SEARCH__PROVIDER", raising=False)
    s = Settings(_env_file=None)
    assert s.search.provider == "memory"
    # env still wins over yaml
    monkeypatch.setenv("MEMORY__SEARCH__PROVIDER", "qdrant")
    assert Settings(_env_file=None).search.provider == "qdrant"


def test_prod_guards_reject_dev_only_providers() -> None:
    with pytest.raises(ValueError, match="trusted_dev"):
        Settings(service={"environment": "prod"})
    with pytest.raises(ValueError, match="authorization.provider=memory"):
        Settings(
            service={"environment": "prod"},
            authentication={"mode": "jwt"},
            authorization={"provider": "memory"},
        )
    with pytest.raises(ValueError, match="blob.provider"):
        Settings(
            service={"environment": "prod"},
            authentication={"mode": "jwt"},
            blob={"provider": "filesystem"},
        )


def test_llm_enabled_requires_provider() -> None:
    with pytest.raises(ValueError, match="provider=bifrost"):
        Settings(models={"llm": {"enabled": True, "provider": "disabled"}})
    with pytest.raises(ValueError, match="models.llm.model"):
        Settings(models={"llm": {"enabled": True, "provider": "bifrost", "model": None}})


def test_redacted_hides_secrets() -> None:
    s = Settings(authorization={"openfga_api_token": "supersecret"})
    dumped = s.redacted()
    assert "supersecret" not in str(dumped)
    assert "trusted_dev_api_keys" not in dumped["authentication"]
