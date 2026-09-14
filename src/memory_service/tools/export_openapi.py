"""Export the OpenAPI schema to a file (used by the CI schema-diff check)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from memory_service.api.app import create_app
from memory_service.config.settings import Settings


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("docs/openapi.json")
    app = create_app(Settings())
    schema = app.openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {out} ({len(schema.get('paths', {}))} paths)\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
