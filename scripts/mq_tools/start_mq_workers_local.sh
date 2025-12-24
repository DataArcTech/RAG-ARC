#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export TASK_QUEUE_MODE="${TASK_QUEUE_MODE:-celery}"

PID_FILE="log/mq_workers.pids"
LOG_DIR="log/mq_workers"
mkdir -p "${LOG_DIR}"

if [ -f "${PID_FILE}" ]; then
  echo "PID file exists: ${PID_FILE} (run scripts/mq_tools/stop_mq_workers_local.sh first)"
  exit 1
fi

POOL="${CELERY_WORKER_POOL:-solo}"
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
  nohup uv run celery -A encapsulation.message_queue.celery_app worker \
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
