"""Start a single-process Memory Service for the examples: the API with its jobs run inline.

Needs PostgreSQL (migrated here before the server starts) and a Redis/Dragonfly, both at the
``MEMORY__*`` defaults unless the environment says otherwise. Without a Qdrant server the
search index runs inside this process (qdrant-client local mode), which is why the jobs run
inline: a separate worker process would index into its own local Qdrant and the API would
never see the vectors. With the compose stack (``make dev-up``) the service itself is the
server - ``uv run memory-api`` and ``uv run memory-worker`` - and nothing here applies.

None of the stand-ins below is reachable from the environment. They are ``Overrides`` on
``create_app`` - the same object the test suite and the benchmarks use - because the shipped
service has one implementation per port and a deployment must not be able to reach an
in-memory queue or a hash encoder through an env file.

Models: with the frozen weights under ``./models`` (``make models``) this runs the real
encoder and NLI head, so the examples exercise retrieval quality and not only the plumbing.
Without them it runs the deterministic stand-ins and says so: the tour still passes, and any
retrieval quality it shows is meaningless.

    uv run python examples/serve.py              # http://localhost:8080/docs, API key "dev-key"
    uv run python examples/serve.py --port 9000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from memory_service.application.container import Overrides
from memory_service.config.constants import FROZEN_MODELS, local_model_path

REPO = Path(__file__).resolve().parents[1]

#: what the launcher sets when the environment does not: a readable log, the dev key
DEFAULT_ENV = {
    "MEMORY__SERVICE__ENVIRONMENT": "dev",
    "MEMORY__SERVICE__LOG_JSON": "false",
    "MEMORY__SERVICE__LOG_LEVEL": "WARNING",
    "MEMORY__AUTHENTICATION__MODE": "trusted_dev",
    "MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS": '["dev-key"]',
    "MEMORY__MODELS__LLM__ENABLED": "false",
}


def weights_present() -> bool:
    """Both the dense encoder and the NLI head are under a model root."""
    return all(
        local_model_path(model.local_dir) is not None
        for model in (FROZEN_MODELS.dense, FROZEN_MODELS.nli)
    )


def stand_ins(*, weights: bool) -> Overrides:
    """The stores this launcher never talks to, plus the model stand-ins when the weights
    are absent. The in-memory authorization model replaces OpenFGA, local-mode Qdrant the
    server, inline jobs the worker, the builtin parser docling; with no weights the hash
    encoder (labelled non-representative), the lexical reranker and the lexical NLI."""
    return Overrides(
        search="memory",
        tasks="inline",
        authorization="memory",
        document_parser="builtin",
        embedding=None if weights else "hash",
        reranker=None if weights else "lexical",
        nli=None if weights else "lexical",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="default MEMORY__SERVICE__PORT")
    parser.add_argument("--skip-migrate", action="store_true", help="do not run alembic first")
    args = parser.parse_args(argv)

    # ``./models``, ``./.blob`` and alembic.ini are all relative to the checkout
    os.chdir(REPO)
    for name, value in DEFAULT_ENV.items():
        os.environ.setdefault(name, value)

    import uvicorn

    from memory_service.api.app import create_app
    from memory_service.config.settings import Settings

    settings = Settings()
    weights = weights_present()
    if weights:
        print(f"models: frozen weights ({FROZEN_MODELS.dense.source})")
    else:
        print("models: NOT FOUND under a model root - running the deterministic stand-ins.")
        print(
            "        The examples will run, but retrieval quality is meaningless. Run 'make models'."
        )

    if not args.skip_migrate:
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)

    port = args.port or settings.service.port
    print(f"serving http://{args.host}:{port}/docs  (API key: dev-key, jobs inline, Qdrant local)")
    uvicorn.run(
        create_app(settings, overrides=stand_ins(weights=weights)),
        host=args.host,
        port=port,
        log_level=settings.service.log_level.lower(),
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
