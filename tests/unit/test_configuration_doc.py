"""docs/configuration.md is the reference for every setting; these keep it the code's.

Every field of ``Settings`` has a row there and a line in ``.env.example``; every default the
page states is the default the code has; and the page's **Example** column, set all at once,
is one valid production configuration (so no example is a value the service would refuse).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, SecretStr

from memory_service.config.settings import Settings

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PAGE = (REPO / "docs" / "configuration.md").read_text(encoding="utf-8")
ENV_EXAMPLE = (REPO / ".env.example").read_text(encoding="utf-8")
ROW = re.compile(
    r"^\| `(?P<var>[A-Z_]+)` \| (?P<default>[^|]+) \| (?P<example>[^|]+) \| (?P<auto>[^|]+) \|"
)


def _rows() -> dict[str, dict[str, str]]:
    rows = {}
    for line in PAGE.splitlines():
        match = ROW.match(line)
        if match:
            rows[match["var"]] = {k: v.strip() for k, v in match.groupdict().items()}
    return rows


ROWS = _rows()


def _env_name(path: tuple[str, ...], field: Any) -> str:
    alias = field.validation_alias
    if alias is not None:  # the platform's names (BIFROST_URL, OTEL_EXPORTER_OTLP_ENDPOINT)
        return next(a for a in alias.choices if a.isupper())
    return "MEMORY__" + "__".join(p.upper() for p in path)


def _leaves(model: type[BaseModel], path: tuple[str, ...] = ()) -> list[tuple[str, Any, Any]]:
    """(environment variable, field, default) for every leaf setting."""
    out: list[tuple[str, Any, Any]] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            out += _leaves(annotation, (*path, name))
        else:
            out.append((_env_name((*path, name), field), field, field.default))
    return out


LEAVES = _leaves(Settings)


def _documented(value: Any) -> str:
    """A default as the page writes it."""
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (list, dict)):
        return f"`{json.dumps(value)}`"
    return f"`{value}`"


def test_every_setting_has_a_row_and_a_line_in_the_env_example() -> None:
    assert len(LEAVES) >= 42
    missing_rows = [env for env, _, _ in LEAVES if env not in ROWS]
    missing_env = [env for env, _, _ in LEAVES if env not in ENV_EXAMPLE]
    assert not missing_rows, f"docs/configuration.md has no row for {missing_rows}"
    assert not missing_env, f".env.example does not list {missing_env}"


def test_every_documented_default_is_the_codes() -> None:
    wrong = {}
    for env, field, declared in LEAVES:
        default = declared
        if field.default_factory is not None:
            # sized from the machine at start (workers, job concurrency): the page says so
            if field.default_factory in (list, dict):
                default = field.default_factory()
            else:
                assert ROWS[env]["default"] == "auto", env
                continue
        if ROWS[env]["default"] != _documented(default):
            wrong[env] = (ROWS[env]["default"], _documented(default))
    assert not wrong, f"page default != code default: {wrong}"


def test_the_examples_together_are_a_valid_production_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in [k for k in __import__("os").environ if k.startswith("MEMORY__")]:
        monkeypatch.delenv(name)
    for env, row in ROWS.items():
        monkeypatch.setenv(env, row["example"].strip("`"))
    settings = Settings(_env_file=None)
    assert settings.service.environment == "prod"
    assert settings.authentication.mode == "jwt"
    assert settings.blob.provider == "gcs" and settings.llm.enabled
    assert settings.search.write_consistency_factor <= settings.search.replication_factor
