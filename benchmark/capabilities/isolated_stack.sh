#!/usr/bin/env bash
# Run a command against a memory service of its own, then remove every trace of it.
#
#   benchmark/capabilities/isolated_stack.sh <command...>
#   (make eval-capabilities-isolated MODE=nollm SET=core)
#
# The capability eval writes tens of thousands of memories per run. On a shared stack they
# accumulate: the service has no tenant purge, and forgetting them one by one through the API
# costs an index job each. So a gate run gets its own store and drops it when it ends:
#
#   PostgreSQL   a database of its own (CAPEVAL_DB, default capeval), created, migrated, dropped
#   Qdrant       an instance of its own on its own storage path (the service has no collection
#                prefix setting); its collections are deleted through the API, then the
#                instance is stopped and the path removed
#   OpenFGA      an in-memory instance of its own
#   Redis        a database number of its own (CAPEVAL_REDIS_DB), flushed at the end
#   blobs        a directory of its own, removed
#   the service  memory-api + memory-worker from this checkout's src/, on their own ports
#
# The command runs with MEMORY_URL, MEMORY_WORKER_METRICS_URL and CAPEVAL_ADMIN_KEY pointing
# at it. Teardown runs on any exit (a failed or interrupted run leaves nothing either). The
# disk is recorded before the stack starts and after it is gone; they should match.
#
# Machine paths are settings: QDRANT_BIN, OPENFGA_BIN, PG* (host, user, password), and the
# ports below. The service runs as `nollm` (no model gateway): the llm and harness modes keep
# running against a stack with a gateway.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${CAPEVAL_PY:-$ROOT/.venv/bin/python}"
QDRANT_BIN="${QDRANT_BIN:-/var/tmp/qdrant}"
OPENFGA_BIN="${OPENFGA_BIN:-/var/tmp/fga/openfga}"
DB="${CAPEVAL_DB:-capeval}"
REDIS_DB="${CAPEVAL_REDIS_DB:-14}"
API_PORT="${CAPEVAL_API_PORT:-8191}"
METRICS_PORT="${CAPEVAL_METRICS_PORT:-9481}"
QDRANT_HTTP="${CAPEVAL_QDRANT_HTTP:-6533}"
QDRANT_GRPC="${CAPEVAL_QDRANT_GRPC:-6534}"
FGA_HTTP="${CAPEVAL_FGA_HTTP:-8281}"
FGA_GRPC="${CAPEVAL_FGA_GRPC:-8283}"
WORK="${CAPEVAL_WORK:-/var/tmp/capeval_stack}"
export PGHOST="${PGHOST:-localhost}" PGUSER="${PGUSER:-memory}" PGPASSWORD="${PGPASSWORD:-memory}"

log() { echo "[isolated-stack $(date +%T)] $*" >&2; }
disk() {
  echo "df_avail_mb=$(df -BM --output=avail / | tail -1 | tr -d ' M') work_mb=$(du -sm "$WORK" 2>/dev/null | cut -f1 || echo 0)"\
" db=$(psql -d postgres -Atc "select coalesce(pg_database_size('$DB')/1048576, 0) from pg_database where datname='$DB'" 2>/dev/null || true)"\
" redis_keys=$(redis-cli -n "$REDIS_DB" dbsize 2>/dev/null || echo ?)"
}
wait_for() {  # url, seconds
  for _ in $(seq 1 "$2"); do curl -sf -m 2 "$1" >/dev/null && return 0; sleep 1; done
  log "not ready: $1"; return 1
}

pids=()
teardown() {
  set +e
  log "teardown"
  for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  pids=()
  qdrant_pid="$(cat "$WORK/qdrant.pid" 2>/dev/null)"
  if curl -sf -m 2 "http://127.0.0.1:$QDRANT_HTTP/collections" >/dev/null; then
    for c in $(curl -s "http://127.0.0.1:$QDRANT_HTTP/collections" \
        | "$PY" -c "import json,sys; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))"); do
      curl -s -X DELETE "http://127.0.0.1:$QDRANT_HTTP/collections/$c" >/dev/null
    done
  fi
  [ -n "$qdrant_pid" ] && kill "$qdrant_pid" 2>/dev/null && sleep 2
  psql -d postgres -qc "drop database if exists $DB with (force)" 2>/dev/null
  redis-cli -n "$REDIS_DB" flushdb >/dev/null 2>&1
  rm -rf "$WORK"
  log "disk after: $(disk)"
}

if [ -e "$WORK" ] || psql -d postgres -Atc "select 1 from pg_database where datname='$DB'" | grep -q 1; then
  log "a previous isolated stack was left behind ($WORK or database $DB): removing it first"
  teardown
fi
log "disk before: $(disk)"
# the API and the worker each load the encoders (about 2 GB apiece): wait, bounded, for the
# room rather than start beside a job that would push the machine into the OOM killer
for _ in $(seq 1 60); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  [ "$avail" -ge "${CAPEVAL_MIN_MEM_MB:-5000}" ] && break
  log "waiting for memory: ${avail} MB available"; sleep 10
done
trap teardown EXIT
mkdir -p "$WORK/qdrant" "$WORK/blob"

# Qdrant and OpenFGA of its own
(QDRANT__STORAGE__STORAGE_PATH="$WORK/qdrant" QDRANT__STORAGE__SNAPSHOTS_PATH="$WORK/qdrant/snapshots" \
  QDRANT__SERVICE__HTTP_PORT="$QDRANT_HTTP" QDRANT__SERVICE__GRPC_PORT="$QDRANT_GRPC" \
  exec "$QDRANT_BIN" --disable-telemetry > "$WORK/qdrant.log" 2>&1) &
echo $! > "$WORK/qdrant.pid"
"$OPENFGA_BIN" run --datastore-engine memory --http-addr "127.0.0.1:$FGA_HTTP" \
  --grpc-addr "127.0.0.1:$FGA_GRPC" --playground-enabled=false --metrics-enabled=false \
  --log-level warn > "$WORK/openfga.log" 2>&1 &
pids+=($!)
wait_for "http://127.0.0.1:$QDRANT_HTTP/readyz" 60
wait_for "http://127.0.0.1:$FGA_HTTP/healthz" 60

# the service, from this checkout
psql -d postgres -qc "create database $DB"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MEMORY__SERVICE__ENVIRONMENT=dev MEMORY__SERVICE__PORT="$API_PORT" MEMORY__SERVICE__WORKERS=1
export MEMORY__SERVICE__LOG_JSON=false MEMORY__SERVICE__LOG_LEVEL=WARNING
export MEMORY__AUTHENTICATION__TRUSTED_DEV_API_KEYS='["capeval-dev-key"]'
export MEMORY__DATABASE__URL="postgresql+psycopg://$PGUSER:$PGPASSWORD@$PGHOST:5432/$DB"
export MEMORY__CACHE__URL="redis://localhost:6379/$REDIS_DB"
export MEMORY__BLOB__FILESYSTEM_ROOT="$WORK/blob" MEMORY__TASKS__METRICS_PORT="$METRICS_PORT"
export MEMORY__SEARCH__QDRANT_URL="http://127.0.0.1:$QDRANT_HTTP" MEMORY__SEARCH__QDRANT_GRPC_PORT="$QDRANT_GRPC"
export MEMORY__AUTHORIZATION__OPENFGA_API_URL="http://127.0.0.1:$FGA_HTTP"
unset BIFROST_URL BIFROST_VIRTUAL_KEY MEMORY__AUTHORIZATION__OPENFGA_STORE_ID MEMORY__AUTHORIZATION__OPENFGA_MODEL_ID
(cd "$ROOT" && "$PY" -m alembic upgrade head > "$WORK/migrate.log" 2>&1)
(cd "$ROOT" && "$PY" -m memory_service.adapters.tasks.procrastinate_schema >> "$WORK/migrate.log" 2>&1) || true
(cd "$ROOT" && exec "$PY" -c "from memory_service.__main__ import run_api; run_api()" > "$WORK/api.log" 2>&1) &
pids+=($!)
(cd "$ROOT" && exec "$PY" -c "from memory_service.__main__ import run_worker; run_worker()" > "$WORK/worker.log" 2>&1) &
pids+=($!)
wait_for "http://127.0.0.1:$API_PORT/health/ready" 300
wait_for "http://127.0.0.1:$METRICS_PORT/metrics" 300
log "ready: memory on :$API_PORT, worker metrics on :$METRICS_PORT"

export MEMORY_URL="http://127.0.0.1:$API_PORT"
export MEMORY_WORKER_METRICS_URL="http://127.0.0.1:$METRICS_PORT/metrics"
export CAPEVAL_ADMIN_KEY=capeval-dev-key
unset TRELLIS_ADMIN_KEY TRELLIS_API_KEY
set +e
"$@"
rc=$?
exit $rc
