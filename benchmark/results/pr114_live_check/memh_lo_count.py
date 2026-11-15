"""memh_lo_count.py <log>...: per model use, the calls that answered (llm.call) and the calls that
failed (llm.assist.fallback, by error), and the billed tokens of the answered ones, from the
service's own log lines."""

import collections
import json
import re
import sys
from pathlib import Path

for path in sys.argv[1:]:
    ok: collections.Counter[str] = collections.Counter()
    failed: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[tuple[str, str]] = collections.Counter()
    tokens_in = tokens_out = 0
    for line in Path(path).read_text(errors="ignore").splitlines():
        use = re.search(r"\buse=(\w+)", line)
        if not use:
            continue
        if " llm.call " in line:
            ok[use.group(1)] += 1
            tokens_in += int(re.search(r"input_tokens=(\d+)", line).group(1))
            tokens_out += int(re.search(r"output_tokens=(\d+)", line).group(1))
        elif "llm.assist.fallback" in line:
            error = re.search(r"error=(\w+)", line)
            failed[use.group(1)] += 1
            errors[(use.group(1), error.group(1) if error else "?")] += 1
    table = {
        u: {"calls": ok[u] + failed[u], "failed": failed[u]} for u in sorted(set(ok) | set(failed))
    }
    print(
        json.dumps(
            {
                "log": path,
                "uses": table,
                "errors": {f"{u}:{e}": n for (u, e), n in sorted(errors.items())},
                "answered_tokens": [tokens_in, tokens_out],
                "answered_usd": round(tokens_in * 0.1e-6 + tokens_out * 0.5e-6, 4),
            },
            indent=1,
        )
    )
