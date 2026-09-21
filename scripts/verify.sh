#!/usr/bin/env bash
# One command that has to pass before anyone says "it works".
#
# Exists because "done" kept meaning "the thing I just changed passes". Each stage below is
# something that has actually broken in this repo at least once: a config surface that
# advertised providers wiring could not build, a suite that wrote to the dev database, a
# checked-in OpenAPI file that no longer matched the code, a stack that needed a manual step
# after `docker compose up`.
#
#   ./scripts/verify.sh           non-destructive: uses the stack that is already running
#   ./scripts/verify.sh --fresh   tears the stack down *including volumes* and rebuilds it
#   ./scripts/verify.sh --quick   static checks and the fast suites only
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="${HARNESS_DIR:-$ROOT/../agent-harness}"
PY="${PY:-$ROOT/.venv/bin/python}"
BASE_URL="${BASE_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-dev-key}"
COMPOSE="${COMPOSE:-docker compose}"

MODE="normal"
for arg in "$@"; do
  case "$arg" in
    --fresh) MODE="fresh" ;;
    --quick) MODE="quick" ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

declare -a NAMES=() RESULTS=() DETAILS=()
FAILED=0

stage() {  # stage <name> <command...>
  local name="$1"; shift
  printf '\n\033[1m── %s\033[0m\n' "$name"
  # Output is streamed *and* captured: a gate whose stages take minutes has to show progress,
  # and piping it through tail made a 30-minute run look like a hang. PIPESTATUS keeps the
  # command's own exit code rather than tee's.
  local log status
  log="$(mktemp)"
  "$@" 2>&1 | sed 's/^/   │ /' | tee "$log" >&2
  status=${PIPESTATUS[0]}
  local last; last="$(tail -1 "$log" | sed 's/^   │ //')"
  if [ "$status" -eq 0 ]; then
    NAMES+=("$name"); RESULTS+=("PASS"); DETAILS+=("$last")
    printf '   \033[32mPASS\033[0m  %s\n' "$last"
  else
    NAMES+=("$name"); RESULTS+=("FAIL"); DETAILS+=("$(tail -3 "$log" | tr '\n' ' ')")
    printf '   \033[31mFAIL\033[0m  (exit %d)\n' "$status"
    FAILED=1
  fi
  rm -f "$log"
}

# ---------------------------------------------------------------- static checks
lint()      { cd "$ROOT" && "$ROOT/.venv/bin/ruff" check . && "$ROOT/.venv/bin/ruff" format --check .; }
artifacts() {
  # docs/openapi.json is generated; a stale one is a lie the SDK and clients read
  cd "$ROOT" && "$PY" -m memory_service.tools.export_openapi /tmp/openapi.verify.json >/dev/null \
    && "$PY" - <<'PYEOF'
import json, sys
from pathlib import Path
a = json.loads(Path("docs/openapi.json").read_text())
b = json.loads(Path("/tmp/openapi.verify.json").read_text())
if a != b:
    sys.exit("docs/openapi.json is stale — run `make openapi` and commit it")
print("openapi.json matches the code")
PYEOF
}

# ---------------------------------------------------------------- suites
svc_tests()     { cd "$ROOT" && "$PY" -m pytest tests -q -p no:randomly; }
svc_fast()      { cd "$ROOT" && "$PY" -m pytest tests/unit tests/contract -q -p no:randomly; }
harness_tests() {
  [ -d "$HARNESS_DIR" ] || { echo "no harness checkout at $HARNESS_DIR — skipped"; return 0; }
  cd "$HARNESS_DIR" && "$HARNESS_DIR/.venv/bin/python" -m pytest -q -p no:randomly
}

# ---------------------------------------------------------------- the stack
stack_up() {
  cd "$ROOT"
  if [ "$MODE" = "fresh" ]; then
    echo "tearing the stack down, volumes included"
    $COMPOSE down -v || true
  fi
  $COMPOSE up -d --build
}
stack_ready() {
  cd "$ROOT"
  for _ in $(seq 1 120); do
    curl -sf "$BASE_URL/health/ready" >/dev/null 2>&1 && { echo "ready at $BASE_URL"; return 0; }
    sleep 5
  done
  echo "never became ready at $BASE_URL"; $COMPOSE ps; return 1
}
smoke() {  # one real turn through the running API: write, then read it back
  cd "$ROOT" && BASE_URL="$BASE_URL" API_KEY="$API_KEY" "$PY" scripts/smoke.py
}

# ---------------------------------------------------------------- run
printf '\033[1mverify (%s)\033[0m  root=%s\n' "$MODE" "$ROOT"
stage "lint + format"        lint
stage "generated artifacts"  artifacts
if [ "$MODE" = "quick" ]; then
  stage "service tests (fast)" svc_fast
else
  stage "stack up"            stack_up
  stage "stack ready"         stack_ready
  stage "smoke through API"   smoke
  stage "service tests"       svc_tests
  stage "harness tests"       harness_tests
fi

printf '\n\033[1m── summary\033[0m\n'
for i in "${!NAMES[@]}"; do
  colour=$([ "${RESULTS[$i]}" = "PASS" ] && echo 32 || echo 31)
  printf '   \033[%sm%-4s\033[0m %-22s %s\n' "$colour" "${RESULTS[$i]}" "${NAMES[$i]}" "${DETAILS[$i]}"
done
[ $FAILED -eq 0 ] && printf '\n\033[32mverify passed\033[0m\n' || printf '\n\033[31mverify FAILED\033[0m\n'
exit $FAILED
