"""Architecture tests: hexagonal boundaries are enforced, not just documented."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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


def test_llm_settings_offer_no_way_to_name_a_provider() -> None:
    """The gateway is the only way out, and there is no field that could say otherwise.

    This used to assert that `provider` was a Literal["disabled", "bifrost"], which made the
    guarantee a matter of keeping a two-value enum two-valued. The field is gone: a
    deployment can point `base_url` at a different gateway, which is the operator's business,
    but nothing in the settings can select a provider SDK — there is no key for it, so there
    is nothing to widen by accident. `test_no_llm_provider_sdk_anywhere_under_src` covers the
    other half: none of them is importable either.
    """
    from memory_service.config.settings import LLMSettings

    fields = set(LLMSettings.model_fields)
    assert "provider" not in fields
    assert not {f for f in fields if "provider" in f or "vendor" in f}


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


def test_the_test_suite_has_exactly_one_database_url() -> None:
    """Only ``tests/conftest.DB_URL`` may spell a Postgres DSN; everything derives from it.

    This is a ratchet over a real incident, not a style rule. ``tests/conftest.py`` grew a
    *second* independent default — ``_test_settings`` built its own
    ``os.environ.get("MEMORY__DATABASE__URL", ".../memory")`` — so pointing ``DB_URL`` at a
    suite-owned database changed nothing: every container still opened the dev database and
    the suite truncated it out from under a running ``docker compose`` stack. One definition
    is the invariant; a second one is invisible until the dev data is gone.
    """
    tests_root = Path(__file__).resolve().parents[1]
    allowed = {tests_root / "conftest.py": {"DB_URL", "ADMIN_URL"}}
    offenders: list[str] = []
    for path in sorted(tests_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        exempt: list[range] = []
        for node in tree.body:
            names = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else []
            )
            if set(names) & allowed.get(path, set()):
                exempt.append(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            # a DSN, not a bare scheme: ``.replace("postgresql+psycopg://", "postgresql://")``
            # derives from DB_URL and is fine — a string that names a host does not.
            if not node.value.startswith("postgresql") or "://" not in node.value:
                continue
            if not node.value.split("://", 1)[1]:
                continue
            if any(node.lineno in span for span in exempt):
                continue
            offenders.append(f"{path.relative_to(tests_root)}:{node.lineno}: {node.value}")
    assert not offenders, (
        "a Postgres DSN outside tests/conftest.DB_URL — derive it from DB_URL instead:\n  "
        + "\n  ".join(offenders)
    )


def test_the_suite_never_reads_the_services_own_database_variable() -> None:
    """``MEMORY__DATABASE__URL`` must not decide where tests write.

    It is the *service's* variable and ``.env`` sets it to the dev database. The deepeval
    pytest plugin loads ``.env`` into ``os.environ`` before conftest is imported, so reading
    it here is not "a default with an escape hatch" — it resolves to the dev database on
    every run, and the suite truncated it. The suite's own variable is
    ``MEMORY_TEST_DATABASE_URL``.
    """
    tests_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted(tests_root.rglob("*.py")):
        if "__pycache__" in path.parts or path == Path(__file__):
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            target = node.func
            reads_env = (
                isinstance(target, ast.Attribute)
                and target.attr in {"get", "getenv"}
                and ast.unparse(target).startswith(("os.environ", "os.getenv", "environ"))
            )
            first = node.args[0]
            if (
                reads_env
                and isinstance(first, ast.Constant)
                and first.value == "MEMORY__DATABASE__URL"
            ):
                offenders.append(f"{path.relative_to(tests_root)}:{node.lineno}")
    assert not offenders, (
        "tests read MEMORY__DATABASE__URL — use MEMORY_TEST_DATABASE_URL:\n  "
        + "\n  ".join(offenders)
    )


def test_every_declared_provider_value_has_a_wiring_branch() -> None:
    """The configuration surface may not advertise a provider that cannot be built.

    ``models.embedding.provider`` accepted "vertex", "openai" and "disabled". All three
    passed validation and then killed startup with ``NotImplementedError`` from
    ``wire_models`` — a setting the schema says is legal and the process cannot honour. With
    2,073,600 provider combinations declared, "it is in the Literal" has to mean "it works".
    """
    import typing

    from pydantic import BaseModel

    from memory_service.config.settings import Settings

    declared: dict[str, list[str]] = {}

    def walk(model: type[BaseModel], prefix: str = "") -> None:
        for name, field in model.model_fields.items():
            annotation = field.annotation
            path = f"{prefix}{name}"
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            base = args[0] if args and typing.get_origin(annotation) is typing.Union else annotation
            if isinstance(base, type) and issubclass(base, BaseModel):
                walk(base, f"{path}.")
                continue
            for candidate in [annotation, *args]:
                if typing.get_origin(candidate) is typing.Literal:
                    declared[path] = list(typing.get_args(candidate))
                    break

    walk(Settings)

    wiring = ast.parse((SRC / "adapters" / "wiring.py").read_text())
    handled: set[str] = set()
    fallthrough = False
    for node in ast.walk(wiring):
        # `cfg.provider == "x"` and `cfg.provider in ("x", "y")`
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute):
            if node.left.attr != "provider":
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant):
                    handled.add(str(comparator.value))
                elif isinstance(comparator, ast.Tuple | ast.List):
                    handled.update(
                        str(e.value) for e in comparator.elts if isinstance(e, ast.Constant)
                    )
        # an `else` that raises is not a branch; an `else` that builds something is
        if isinstance(node, ast.If) and node.orelse:
            fallthrough = True
    assert fallthrough, "sanity: wiring is expected to use else-branches"

    # A value is fine if wiring names it OR the section's else-branch builds a real provider.
    # Only the sections whose else-branch *raises* are strict, so list them explicitly.
    strict = {"models.embedding.provider"}
    offenders: list[str] = []
    for path, values in declared.items():
        if path not in strict:
            continue
        offenders += [f"{path}={v!r}" for v in values if v not in handled]
    assert not offenders, (
        "declared in config but not buildable by wiring (its else-branch raises):\n  "
        + "\n  ".join(offenders)
    )


def test_the_suite_configures_itself_and_never_the_developers_shell() -> None:
    """A test run must mean the same thing on every machine.

    Two doors let a local ``.env`` into the suite, and both had to be shut: the deepeval
    pytest plugin loads that file into ``os.environ`` before conftest is imported, and
    pydantic-settings reads ``./.env`` directly as a settings source. Four model paths were
    arriving that way, pointing at directories that do not exist — invisible only because the
    stand-in providers never load a model.
    """
    import json
    import subprocess
    import sys
    import tempfile

    root = Path(__file__).resolve().parents[2]
    probe = (
        "import json,sys;"
        f"sys.path.insert(0,{str(root / 'src')!r});sys.path.insert(0,{str(root)!r});"
        "from tests.conftest import _test_settings;"
        'print("@@"+json.dumps(_test_settings().model_dump(mode="json")))'
    )

    def settings_from(cwd: str) -> dict:
        out = subprocess.run(  # noqa: S603 - our own interpreter, literal argv
            [sys.executable, "-c", probe], capture_output=True, text=True, cwd=cwd, check=False
        )
        line = next((x for x in out.stdout.splitlines() if x.startswith("@@")), None)
        assert line, f"probe failed in {cwd}: {out.stderr[-400:]}"
        return json.loads(line[2:])

    def flat(d: dict, prefix: str = "") -> dict:
        out: dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                out |= flat(value, f"{prefix}{key}.")
            else:
                out[f"{prefix}{key}"] = value
        return out

    if not (root / ".env").is_file():
        pytest.skip("no local .env to leak from")
    with_env = flat(settings_from(str(root)))
    without_env = flat(settings_from(tempfile.mkdtemp()))
    drift = sorted(k for k in with_env | without_env if with_env.get(k) != without_env.get(k))
    assert not drift, f"test settings inherited from the local .env: {drift}"


def test_hosts_and_urls_are_configuration_not_literals() -> None:
    """A hostname in logic is a deployment that cannot move.

    Defaults belong in ``config/settings.py``, where an operator overrides them with
    ``MEMORY__*``; anywhere else they are a machine's address compiled into behaviour. This
    passes today — it exists so it keeps passing.
    """
    import re

    # settings.py holds the defaults an operator overrides. A server entrypoint is the other
    # legitimate case: binding 0.0.0.0 is how a process listens, not a peer address compiled
    # into behaviour — and it is overridable (MEMORY_MODEL_HOST) besides. Its docstring also
    # carries the client-side URLs, which is documentation of the wire contract rather than a
    # hostname in logic.
    allowed = {SRC / "config" / "settings.py", SRC / "tools" / "model_server.py"}
    network = re.compile(r"https?://[A-Za-z0-9.:_-]+|\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0)\b")
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            # a bare scheme ("http://") carries no address and is fine
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and network.search(node.value)
                and not node.value.rstrip(":/").endswith("http")
            ):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}: {node.value[:60]}")
    assert not offenders, (
        "hosts/URLs outside config/settings.py — make them settings:\n  " + "\n  ".join(offenders)
    )


def test_the_parser_enum_and_the_parser_registry_agree() -> None:
    """Two lists that must not drift: what configuration accepts, and what can be built.

    ``documents.parser`` is a Literal so the value is typed and appears in the OpenAPI
    schema; ``adapters/parsers.PARSERS`` is what wiring can actually construct. A name in one
    and not the other is either a setting that fails at startup or a parser nobody can select
    — the same class of defect as an embedding provider declared with no wiring branch.
    """
    import typing

    from memory_service.adapters.parsers import PARSERS
    from memory_service.config.settings import DocumentSettings

    declared = set(typing.get_args(DocumentSettings.model_fields["parser"].annotation))
    assert declared == set(PARSERS), (
        f"documents.parser accepts {sorted(declared)} but the registry builds {sorted(PARSERS)}"
    )


def test_wiring_selects_providers_through_the_registry_not_a_switch() -> None:
    """``config/registry.py`` calls itself the Open/Closed extension point: "adding a provider
    means registering a factory, never editing a switch statement". It was dead code —
    nothing registered, nothing created — while wiring grew the switch it forbids.

    This pins the parser port, the first one converted. Extending the assertion to another
    port is the intended way to make this true of the rest.
    """
    source = (SRC / "adapters" / "wiring.py").read_text()
    assert "registries.document_parser.create(" in source, (
        "the document parser must be built through the registry"
    )
    assert 'cfg.parser == "docling"' not in source, "the parser switch statement is back"


def test_the_memory_type_docs_match_what_the_pipeline_actually_produces() -> None:
    """The hint description tells callers which values the service writes for them.

    It claimed SUMMARY, DERIVED and KNOWLEDGE_RAG were "written by the pipeline". They are
    produced at zero sites — reachable only by an explicit caller hint. A description that
    tells a caller not to set a value nothing else sets makes that value unreachable.
    """
    import re

    from memory_service.api.schemas.conversation import ProcessingHintsIn
    from memory_service.domain.enums import MemoryType

    blob = "\n".join(path.read_text() for path in SRC.rglob("*.py"))
    produced = set(re.findall(r"memory_type\s*=\s*MemoryType\.(\w+)", blob))
    description = ProcessingHintsIn.model_fields["memory_type"].description or ""

    claimed = {
        name
        for name in (t.name for t in MemoryType)
        if re.search(rf"(?<![A-Z_]){name}\b[^)]*?written by the pipeline", description)
    }
    assert not (claimed - produced), (
        f"documented as pipeline-written but never produced: {sorted(claimed - produced)}"
    )


def test_the_default_image_can_honour_the_default_configuration() -> None:
    """The build and the settings must not disagree.

    ``documents.parser`` defaulted to "docling" while the default build shipped without the
    docling extra, so a default deployment could not honour its own default setting. It did
    not fail either — it fell back to a text parser that decoded PDFs as UTF-8 and indexed
    the file structure as document text.
    """
    import re

    from memory_service.config.settings import DocumentSettings

    dockerfile = (SRC.parents[1] / "deploy" / "Dockerfile").read_text()
    match = re.search(r'^ARG EXTRAS="([^"]*)"', dockerfile, re.MULTILINE)
    assert match, "deploy/Dockerfile no longer declares a default EXTRAS"
    extras = set(match.group(1).split())
    default_parser = DocumentSettings().parser
    if default_parser != "builtin":
        assert default_parser in extras, (
            f"documents.parser defaults to {default_parser!r} but the default image builds "
            f"extras {sorted(extras)} — the deployment cannot honour its own default"
        )
