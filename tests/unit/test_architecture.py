"""Architecture tests: hexagonal boundaries are enforced, not just documented."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "memory_service"

BANNED_IN_CORE = {
    "qdrant_client",
    "redis",
    "google",
    "mem0",
    "graphiti_core",
    "openfga_sdk",
    "procrastinate",
    "langgraph",
    "docling",
    "cognee",
    "langmem",
    "sentence_transformers",
    "fastembed",
    "sqlalchemy",
    "psycopg",
}
CORE_PACKAGES = ("domain", "application", "modules", "ports", "api")

# LLM provider SDKs are banned in *every* package, adapters included: the only LLM path is the
# Bifrost gateway adapter speaking plain HTTP (adapters/models/llm.py).
LLM_SDKS = {
    "openai",
    "anthropic",
    "google.generativeai",
    "google.genai",
    "vertexai",
    "litellm",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_google_genai",
    "mistralai",
    "cohere",
    "ollama",
    "groq",
    "together",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_packages_do_not_import_provider_sdks() -> None:
    offenders: list[str] = []
    for pkg in CORE_PACKAGES:
        for path in (SRC / pkg).rglob("*.py"):
            bad = _imports(path) & BANNED_IN_CORE
            if bad:
                offenders.append(f"{path.relative_to(SRC)}: {sorted(bad)}")
    assert not offenders, "provider SDKs leaked into core:\n" + "\n".join(offenders)


def _dotted_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_llm_provider_sdk_anywhere_under_src() -> None:
    root = SRC.parent
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        for name in _dotted_imports(path):
            if name in LLM_SDKS or any(name.startswith(f"{sdk}.") for sdk in LLM_SDKS):
                offenders.append(f"{path.relative_to(root)}: {name}")
    assert not offenders, "LLM SDK imported outside Bifrost:\n" + "\n".join(offenders)


def test_llm_settings_only_allow_bifrost() -> None:
    from typing import get_args

    from memory_service.config.settings import LLMSettings

    assert set(get_args(LLMSettings.model_fields["provider"].annotation)) == {"disabled", "bifrost"}


def test_domain_does_not_import_application_adapters_or_frameworks() -> None:
    offenders: list[str] = []
    for path in (SRC / "domain").rglob("*.py"):
        for name in _imports(path):
            if name in {"fastapi", "starlette"}:
                offenders.append(f"{path.name}: {name}")
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            "memory_service.adapters",
            "memory_service.application",
            "memory_service.api",
            "memory_service.modules",
        ):
            if forbidden in text:
                offenders.append(f"{path.name}: {forbidden}")
    assert not offenders, "\n".join(offenders)


def test_ports_are_protocols_not_implementations() -> None:
    for path in (SRC / "ports").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        names = _imports(path)
        assert not (names & BANNED_IN_CORE), f"{path.name} imports {names & BANNED_IN_CORE}"


def test_sdk_does_not_depend_on_service_internals() -> None:
    sdk = Path(__file__).resolve().parents[2] / "sdk" / "python" / "src" / "universal_memory"
    for path in sdk.rglob("*.py"):
        assert "memory_service" not in _imports(path), f"{path.name} imports memory_service"
