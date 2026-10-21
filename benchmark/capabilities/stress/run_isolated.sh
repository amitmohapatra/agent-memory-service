#!/usr/bin/env bash
# One stress_accumulated run (nollm) of a service checkout, on a store of its own.
#
#   benchmark/capabilities/stress/run_isolated.sh <name> <service checkout> <out dir> [run.py args...]
#
# The harness code is this checkout's; the service (memory-api + memory-worker, migrations) is
# <service checkout>'s src/ - so one harness measures any branch. isolated_stack.sh creates a
# private PostgreSQL database, Qdrant, OpenFGA and Redis db, and drops them when the run ends; it
# refuses the live ports (Qdrant 6333/6334), Redis 0 and the reserved dbs, and a database not
# named capeval*. Paired runs of two trees: the same --run-date (default below) and seed.
#
# Writes <out dir>/<name>.json (records + summary), <name>.md, <name>.log.
set -euo pipefail
NAME=${1:?name}; ROOT=$(realpath "${2:?service checkout}"); OUT=$(realpath -m "${3:?out dir}"); shift 3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
mkdir -p "$OUT"
export CAPEVAL_ROOT=$ROOT
export CAPEVAL_PY=${CAPEVAL_PY:-/home/user/agent-memory-service/.venv/bin/python}
export CAPEVAL_DB=${CAPEVAL_DB:-capeval_stress}
export CAPEVAL_WORK=${CAPEVAL_WORK:-/var/tmp/capeval_stress}
export CAPEVAL_LOGS=${CAPEVAL_LOGS:-/var/tmp/capeval_stress_logs_$NAME}
export CAPEVAL_REDIS_DB=${CAPEVAL_REDIS_DB:-6}
export CAPEVAL_API_PORT=${CAPEVAL_API_PORT:-8491} CAPEVAL_METRICS_PORT=${CAPEVAL_METRICS_PORT:-9491}
export CAPEVAL_QDRANT_HTTP=${CAPEVAL_QDRANT_HTTP:-6833} CAPEVAL_QDRANT_GRPC=${CAPEVAL_QDRANT_GRPC:-6834}
export CAPEVAL_FGA_HTTP=${CAPEVAL_FGA_HTTP:-8591} CAPEVAL_FGA_GRPC=${CAPEVAL_FGA_GRPC:-8593}
RUN_DATE=${RUN_DATE:-2026-10-18}
cd "$HERE"
t0=$(date +%s)
set +e
benchmark/capabilities/isolated_stack.sh "$CAPEVAL_PY" -m benchmark.capabilities.stress.run \
  --run-date "$RUN_DATE" --out "$OUT/$NAME.json" "$@" > "$OUT/$NAME.log" 2>&1
rc=$?
set -e
echo "$NAME rc=$rc rev=$(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo +dirty) wall=$(( $(date +%s) - t0 ))s $(grep -E '^wrote .*pass' "$OUT/$NAME.log" | tail -1)"
exit $rc
