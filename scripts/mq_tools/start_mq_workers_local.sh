#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [ -f .env ]; then
  # Load .env as defaults only: already-exported env vars take precedence.
  eval "$(
    python3 - <<'PY'
import os
import shlex

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

values = {}
if dotenv_values is not None:
    values = dotenv_values(".env") or {}

for key, value in values.items():
    if value is None:
        continue
    if key in os.environ and os.environ.get(key) not in (None, ""):
        continue
    print(f"export {key}={shlex.quote(str(value))}")
PY
  )"
fi

export TASK_QUEUE_MODE="${TASK_QUEUE_MODE:-celery}"

PID_FILE="log/mq_workers.pids"
LOG_DIR="log/mq_workers"
mkdir -p "${LOG_DIR}"

if [ -f "${PID_FILE}" ]; then
  alive=0
  while read -r pid name queue; do
    if [ -z "${pid}" ]; then
      continue
    fi
    if kill -0 "${pid}" >/dev/null 2>&1; then
      alive=1
      break
    fi
  done < "${PID_FILE}" || true
  if [ "${alive}" -eq 1 ]; then
    echo "PID file exists and workers are still running: ${PID_FILE} (run scripts/mq_tools/stop_mq_workers_local.sh first)"
    exit 1
  fi
  echo "Stale PID file detected (no live workers). Removing ${PID_FILE}."
  rm -f "${PID_FILE}"
fi

if [ "${MQ_WORKER_FORCE_START:-0}" != "1" ]; then
  if command -v pgrep >/dev/null 2>&1; then
    if pgrep -f "celery -A encapsulation.message_queue.celery_app worker" >/dev/null 2>&1; then
      echo "Detected existing celery worker processes (use MQ_WORKER_FORCE_START=1 to override)."
      pgrep -af "celery -A encapsulation.message_queue.celery_app worker" || true
      exit 1
    fi
  fi
fi

POOL="${CELERY_WORKER_POOL:-prefork}"
LOGLEVEL="${CELERY_LOGLEVEL:-info}"
CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-2}"

INDEXING_QUEUE="${CELERY_QUEUE_INDEXING:-indexing}"
DEEPSEARCH_QUEUE="${CELERY_QUEUE_DEEPSEARCH:-deepsearch}"
EXPORT_QUEUE="${CELERY_QUEUE_EXPORT:-export}"

start_worker () {
  local queue="$1"
  local name="$2"
  local logfile="${LOG_DIR}/${name}.log"
  echo "Starting worker name=${name} queue=${queue} pool=${POOL} concurrency=${CONCURRENCY} log=${logfile}"
  # Use a dedicated process group so stop script can terminate the full worker tree
  # (uv wrapper + celery python child) deterministically.
  nohup setsid uv run celery -A encapsulation.message_queue.celery_app worker \
    --pool "${POOL}" \
    --loglevel "${LOGLEVEL}" \
    --concurrency "${CONCURRENCY}" \
    --hostname "${name}@%h" \
    --queues "${queue}" \
    >"${logfile}" 2>&1 &
  echo "$! ${name} ${queue}" >> "${PID_FILE}"
}

start_worker "${INDEXING_QUEUE}" "rag-arc-indexing"
start_worker "${DEEPSEARCH_QUEUE}" "rag-arc-deepsearch"
start_worker "${EXPORT_QUEUE}" "rag-arc-export"

echo "Workers started. PIDs recorded in ${PID_FILE}"
