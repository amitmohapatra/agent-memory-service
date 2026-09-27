"""Own queues only; never touch a shared benchmark database or use paid providers."""
import os
from pathlib import Path
import subprocess
import sys
import time

wait_pid = 23152
print('Waiting for owned OCR screen', wait_pid, flush=True)
for _ in range(2160):
    try:
        os.kill(wait_pid, 0)
    except ProcessLookupError:
        break
    time.sleep(10)
else:
    raise RuntimeError('Preceding queue exceeded six hours')

def run(name, command, env=None):
    print('Starting', name, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    with (Path('.bench_data') / (name + '.log')).open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    print('Finished', name, result.returncode, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

run('multilingual-final-complete', ['bash', '-c', 'set -euo pipefail\nsource .bench_data/test-env.sh\n/Users/ricky/usage_data/agent-memory-service/.venv/bin/ruff check src/memory_service tests benchmark sdk/python/src\n/Users/ricky/usage_data/agent-memory-service/.venv/bin/python -m pytest -q -m "not models and not bifrost and not docker" --junitxml=.bench_data/multilingual-final-complete.xml'])
env = dict(os.environ)
for key in list(env):
    if key.startswith(('MEMORY__', 'BENCH_')):
        del env[key]
env.update(
    PYTHONPATH='src:.sdk-test-deps:sdk/python/src:/Users/ricky/usage_data/ams-hindsight-benchmark/.hindsight-venv/lib/python3.12/site-packages',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1',
    BENCH_SEARCH='qdrant', BENCH_AUTHZ='memory', BENCH_CACHE='memory',
    BENCH_NLI='lexical', BENCH_GRAPH_ENRICHMENT='native', BENCH_DEPTH='shipped',
    MEMORY__SEARCH__QDRANT_URL='http://localhost:16333',
    MEMORY__SEARCH__QDRANT_GRPC_PORT='16334',
    MEMORY__MODELS__LLM__ENABLED='false', MEMORY__MODELS__LLM__USES='[]',
)
# These databases are newly allocated to these arms. No existing baseline database is reused.
import psycopg
from psycopg import sql
arms = (
    ('granite_off', '/Users/ricky/usage_data/ams-hindsight-integration/.bench_data/models/granite-baseline.json', False),
    ('granite_on', '/Users/ricky/usage_data/ams-hindsight-integration/.bench_data/models/granite-baseline.json', True),
    ('bekko8_on', '.bench_data/models/bekko-a8m.json', True),
)
for name, spec, graph in arms:
    database = 'memory_hi_ml_' + name + '_20260927'
    with psycopg.connect('postgresql://memory:memory@localhost:5432/postgres', autocommit=True) as conn:
        exists = conn.execute('SELECT 1 FROM pg_database WHERE datname=%s', (database,)).fetchone()
        if exists:
            raise RuntimeError('Refusing to overwrite an already allocated arm database: ' + database)
        conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(database)))
    env['MEMORY__DATABASE__URL'] = 'postgresql+psycopg://memory:memory@localhost:5432/' + database
    run('locomo-ml-' + name + '-migration', [sys.executable, '-m', 'alembic', 'upgrade', 'head'], env)
    run('locomo-ml-' + name, [sys.executable, '-m', 'benchmark.native_source_retrieval',
        '--data', '/Users/ricky/usage_data/agent-memory-service/benchmark/data/locomo10.json',
        '--spec', spec, '--output', 'benchmark/results/locomo_multilingual_current_' + name + '.json',
        '--semantic-graph' if graph else '--no-semantic-graph'], env)
