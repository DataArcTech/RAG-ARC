#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PID_FILE="log/mq_workers.pids"
if [ ! -f "${PID_FILE}" ]; then
  echo "No PID file: ${PID_FILE}"
  exit 0
fi

while read -r pid name queue; do
  if [ -z "${pid}" ]; then
    continue
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "Stopping pid=${pid} name=${name} queue=${queue}"
    # Prefer killing the process group (spawned by `setsid`) so uv wrapper + celery child exit together.
    kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
    kill -TERM "${pid}" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -KILL -- "-${pid}" >/dev/null 2>&1 || true
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  else
    echo "Already stopped pid=${pid} name=${name} queue=${queue}"
  fi
done < "${PID_FILE}"

rm -f "${PID_FILE}"
echo "Done."
