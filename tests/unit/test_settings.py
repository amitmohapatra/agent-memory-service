import pytest
from pydantic import ValidationError

from memory_service.config.constants import CONTEXT, FROZEN_MODELS, RETRIEVAL
from memory_service.config.settings import Settings


def test_defaults_are_cpu_first_and_the_model_is_off_without_a_gateway() -> None:
    s = Settings(_env_file=None)
    assert not s.llm.enabled and s.llm.api_key is None
    assert not s.llm.wants("contextual_extraction")
    assert s.otel_endpoint is None and s.authentication.mode == "api_key"
    assert FROZEN_MODELS.dense.id == "ibm-granite/granite-embedding-small-english-r2"
    assert FROZEN_MODELS.dense.dimension == 384 and FROZEN_MODELS.dense.backend == "torch"
    assert RETRIEVAL.bm25 and RETRIEVAL.dense and RETRIEVAL.exact and RETRIEVAL.graph


def test_default_depth_matches_the_promoted_memory_configuration() -> None:
    """Promote measured memory depth without widening document or mixed final pools."""
    assert (RETRIEVAL.prefetch_k, RETRIEVAL.fused_k, RETRIEVAL.final_k) == (100, 100, 50)
    assert RETRIEVAL.memory_recall_k == CONTEXT.memories_max == 100
    assert CONTEXT.token_budget == 8000


def test_env_overrides_with_nested_delimiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY__DATABASE__CONNECTION_BUDGET", "30")
    monkeypatch.setenv("MEMORY__SERVICE__LOG_LEVEL", "WARNING")
    s = Settings(_env_file=None)
    assert s.database.connection_budget == 30
    assert s.service.log_level == "WARNING"


def test_env_beats_dotenv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real environment variable outranks ``.env``: compose sets the topology through
    ``environment:`` and an operator's copied-in line must not be able to shadow it."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("MEMORY__DATABASE__CONNECTION_BUDGET=70\nMEMORY__SERVICE__PORT=9999\n")
    monkeypatch.delenv("MEMORY__DATABASE__CONNECTION_BUDGET", raising=False)
    monkeypatch.setenv("MEMORY__SERVICE__PORT", "8181")
    s = Settings(_env_file=str(dotenv))
    assert s.database.connection_budget == 70, ".env is still read"
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
        SearchSettings,
        TaskSettings,
    )

    for section in (CacheSettings, SearchSettings, TaskSettings, AuthorizationSettings):
        assert "provider" not in section.model_fields, section.__name__
    assert "qdrant_local_path" not in SearchSettings.model_fields


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
    deployment, not about tuning.

    39 -> 41 in Phase 2, for two facts about a host rather than two tunings: how many worker
    processes this machine's cores are worth (``service.workers``) and which port the Qdrant
    beside it publishes gRPC on (``search.qdrant_grpc_port``).

    41 -> 42 for ``authentication.tenant_claim``, which is a credential fact and not a
    tuning: it names the claim a credential must carry to act for a tenant. Without it the
    tenant is asserted in a header and compared to nothing, so one credential reaches every
    tenant on the deployment by changing that header - sound when a gateway stamps the tenant
    as a constant (one deployment per customer), not sound when several teams share one. The
    two deployments genuinely differ, which is what earns an environment field.
    """
    leaves = _leaves(Settings)
    # Integrated extraction adds endpoint/auth/bank selection and its deployment's
    # timeout/concurrency quota. Retrieval tuning stays frozen; LLM/use gates still apply.
    # Two more credential facts: the active envelope-key version and its secret keyring.
    # They permit tenant/agent-owned VKs without storing provider credentials in plaintext.
    # 49 -> 50 for ``authentication.bootstrap_admin_key``: the one secret that onboards tenants
    # in api_key mode. A deployment fact, and the only one a shared deployment adds.
    # 51 -> 35 in the final overhaul: the webhooks flag went with the webhooks; the model's
    # tuning (model names, fast uses, the use allow-list, tokens, timeout, retries), the
    # Hindsight quota, the encoder threads, the service-wide rate limit and the exporter
    # kind became constants or are derived (the model is on when BIFROST_URL is set, tracing
    # when OTEL_EXPORTER_OTLP_ENDPOINT is, the authentication mode from the credentials).
    # 35 -> 36 for ``retail_calendar``: which fiscal calendar the customer reports in is a
    # fact about the customer, like ``tenant_claim``, not a tuning - two retailers on 4-5-4
    # and 4-4-5 calendars mean different days by the same "last week".
    # 36 -> 38 (ADR 0031), all topology: ``database.pool_size`` and ``max_overflow`` became
    # one ``connection_budget`` per pod (the pools are derived from it); ``direct_url`` and
    # ``transaction_pooler`` say a PgBouncer sits in front of the request path and where the
    # session work goes instead; ``tasks.metrics_port`` is a port, like ``service.port``. The
    # overload limits and deadlines that came with them are constants (``OVERLOAD``).
    assert len(leaves) <= 38, f"{len(leaves)} env fields: {leaves}"
    for forbidden in (
        "prefetch_k",
        "final_k",
        "token_budget",
        "dimension",
        "model_path",
        "max_tokens",
        "timeout_seconds",
        "max_retries",
        "uses",
        "rate_limit_per_minute",
        "threads",
        "mode",
    ):
        assert not [leaf for leaf in leaves if leaf.endswith(forbidden)], forbidden


def test_prod_guards_reject_dev_only_providers() -> None:
    with pytest.raises(ValueError, match="trusted_dev"):
        Settings(
            _env_file=None,
            service={"environment": "prod"},
            authentication={"trusted_dev_api_keys": ["dev-key"]},
        )
    with pytest.raises(ValueError, match="blob.provider"):
        Settings(
            _env_file=None,
            service={"environment": "prod"},
            authentication={"jwt_jwks_url": "https://issuer/jwks"},
            blob={"provider": "filesystem"},
        )


def test_the_authentication_mode_follows_from_the_credentials_configured() -> None:
    """A mode that had to agree with the credentials beside it was two settings for one
    fact: the issuer's JWKS means jwt, the bootstrap key or no development keys means the
    service's own issued keys, development keys alone mean trusted_dev."""
    from memory_service.config.settings import AuthenticationSettings

    assert AuthenticationSettings().mode == "api_key"
    assert AuthenticationSettings(trusted_dev_api_keys=["k"]).mode == "trusted_dev"
    assert (
        AuthenticationSettings(trusted_dev_api_keys=["k"], bootstrap_admin_key="b" * 32).mode
        == "api_key"
    )
    assert AuthenticationSettings(jwt_jwks_url="https://issuer/jwks").mode == "jwt"


def test_the_platform_names_are_read_unprefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model gateway and the trace endpoint are the platform's variables, read under the
    names the harness and agent-runs read them by."""
    monkeypatch.setenv("BIFROST_URL", "http://bifrost:8080/v1")
    monkeypatch.setenv("BIFROST_VIRTUAL_KEY", "sk-bf-operator")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/v1/traces")
    s = Settings(_env_file=None)
    assert s.llm.enabled and s.llm.base_url == "http://bifrost:8080/v1"
    assert s.llm.operator_pays and "sk-bf-operator" not in str(s.redacted())
    assert s.otel_endpoint == "http://collector:4318/v1/traces"


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
        bifrost_virtual_key="virtualkey",
        hindsight={"api_key": "hindsightsecret"},
    )
    dumped = str(s.redacted())
    for secret in (
        "hunter2",
        "cachepass",
        "qdrantsecret",
        "supersecret",
        "devsecret",
        "virtualkey",
        "hindsightsecret",
    ):
        assert secret not in dumped, secret
    # the adapters still get the real value, in both spellings
    assert "hunter2" in s.database.dsn and s.database.dsn.startswith(scheme)
    assert (
        "hunter2" in s.database.procrastinate_dsn and "+psycopg" not in s.database.procrastinate_dsn
    )


def test_a_short_bootstrap_key_is_refused_in_deployed_environments() -> None:
    """The one credential that onboards tenants and can administer any of them; a short
    operator-chosen value is guessable online. Laptops may use anything."""
    import pytest

    base = {
        "service": {"environment": "prod"},
        "authentication": {"bootstrap_admin_key": "short"},
        "blob": {"provider": "gcs"},
    }
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(_env_file=None, **base)
    ok = {**base, "authentication": {"bootstrap_admin_key": "x" * 32}}
    assert Settings(_env_file=None, **ok).authentication.mode == "api_key"
    dev = {"authentication": {"bootstrap_admin_key": "short"}}
    assert Settings(_env_file=None, **dev).authentication.bootstrap_admin_key is not None
